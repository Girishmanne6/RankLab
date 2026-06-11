"""Train all RankLab models and cache evaluation results.

Usage:
    python -m api.train                # train everything, overwrite
    python -m api.train --if-missing   # skip models with existing checkpoints
    python -m api.train --models popularity lightgcn
    python -m api.train --epochs 10 --max-eval-users 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from evaluation.fairness import evaluate_fairness, exposure_scatter
from evaluation.metrics import evaluate_model
from models import (
    MODEL_REGISTRY,
    ItemCFModel,
    LightGCNModel,
    PopularityModel,
    RecencyModel,
    Recommender,
    XSimGCLModel,
)


def build_model(
    name: str,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    patience: int,
) -> Recommender:
    if name == "popularity":
        return PopularityModel()
    if name == "recency":
        return RecencyModel()
    if name == "item_cf":
        return ItemCFModel()
    gnn_kwargs = {
        "embedding_dim": embedding_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "device": config.DEVICE,
    }
    if name == "lightgcn":
        return LightGCNModel(**gnn_kwargs)
    if name == "xsimgcl":
        return XSimGCLModel(**gnn_kwargs)
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RankLab models")
    parser.add_argument("--models", nargs="*", default=list(MODEL_REGISTRY))
    parser.add_argument("--if-missing", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=config.EMBEDDING_DIM)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-eval-users", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else config.EPOCHS

    data_dir, model_dir = config.DATA_DIR, config.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(data_dir / "train.parquet")
    val = pd.read_parquet(data_dir / "val.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    articles = pd.read_parquet(data_dir / "articles.parquet")
    stats = json.loads((data_dir / "stats.json").read_text())
    n_users, n_items = stats["n_users"], stats["n_items"]

    metrics_path = model_dir / "metrics.json"
    scatter_path = model_dir / "scatter.json"
    all_metrics: dict = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_scatter: dict = json.loads(scatter_path.read_text()) if scatter_path.exists() else {}

    for name in args.models:
        ckpt = model_dir / f"{name}.pkl"
        if args.if_missing and ckpt.exists() and name in all_metrics:
            print(f"[skip] {name}: checkpoint + metrics already exist")
            continue

        print(f"\n=== Training {name} ===")
        model = build_model(
            name,
            embedding_dim=args.embedding_dim,
            epochs=epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )

        t0 = time.perf_counter()
        if isinstance(model, LightGCNModel):
            model.fit(train, articles, n_users, n_items, val=val)
        else:
            model.fit(train, articles, n_users, n_items)
        train_s = time.perf_counter() - t0
        model.save(ckpt)
        print(f"[{name}] trained in {train_s:.1f}s -> {ckpt}")

        def rec_fn(user_idx: int, k: int) -> list[int]:
            return [i for i, _ in model.recommend(user_idx, k=k)]

        quality = evaluate_model(
            rec_fn, test, k=args.k, max_users=args.max_eval_users
        )
        fairness = evaluate_fairness(
            rec_fn, train, test, articles, n_items,
            k=args.k, max_users=args.max_eval_users,
        )
        all_metrics[name] = {
            **quality,
            **fairness,
            "train_seconds": round(train_s, 2),
            "data_source": stats.get("raw_source", str(data_dir)),
            "dataset_users": stats["n_users"],
            "dataset_items": stats["n_items"],
            "dataset_interactions": stats["n_interactions"],
        }
        if hasattr(model, "history_") and model.history_:
            all_metrics[name]["epochs_trained"] = len(model.history_)
        all_scatter[name] = exposure_scatter(
            rec_fn, train, test, n_items, k=args.k
        )
        print(f"[{name}] {json.dumps(all_metrics[name], indent=2)}")

        metrics_path.write_text(json.dumps(all_metrics, indent=2))
        scatter_path.write_text(json.dumps(all_scatter, indent=2))

    print(f"\nMetrics cached to {metrics_path}")


if __name__ == "__main__":
    main()
