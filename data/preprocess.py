"""Preprocess raw EB-NeRD data into model-ready parquet files.

Reads the official EB-NeRD small layout (articles.parquet at the root,
behaviors/history under train/ and validation/) from ``EBNERD_RAW_PATH``.

Outputs (in DATA_DIR):
    train.parquet / val.parquet / test.parquet
        Columns: user_idx (int), item_idx (int), ts (int64, unix seconds).
        Chronological 70/10/20 split per user.
    articles.parquet
        Columns: item_idx, article_id, title, category, published_ts, text,
        pageviews. `text` concatenates title/subtitle/body for TF-IDF.
    stats.json
        n_users, n_items, n_interactions and split sizes.

Usage:
    python -m data.preprocess
    EBNERD_RAW_PATH=C:\\path\\to\\ebnerd python -m data.preprocess
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config


def resolve_raw_root() -> Path:
    """Resolve the EB-NeRD raw root directory.

    Prefers ``EBNERD_RAW_PATH``. If that path does not exist, falls back to
    its parent (common when ``ebnerd_small.zip`` extracts to ``.../ebnerd``).
    """
    candidates = [
        config.EBNERD_RAW_PATH,
        config.EBNERD_RAW_PATH.parent,
        config.RAW_DIR / "ebnerd_small",
        config.RAW_DIR / "ebnerd_demo",
    ]
    for cand in candidates:
        if (cand / "train" / "behaviors.parquet").exists():
            return cand
        if (cand / "behaviors.parquet").exists():
            return cand
    raise FileNotFoundError(
        "No EB-NeRD raw data found. Set EBNERD_RAW_PATH to the folder that "
        f"contains train/behaviors.parquet (tried: {candidates})."
    )


def _load_split_interactions(split_dir: Path) -> pd.DataFrame:
    """Flatten click events from one split's behaviors (+history if present)."""
    behaviors = pd.read_parquet(split_dir / "behaviors.parquet")
    frames: list[pd.DataFrame] = []

    clicked = behaviors[["user_id", "article_ids_clicked", "impression_time"]].copy()
    clicked = clicked.explode("article_ids_clicked").dropna()
    clicked = clicked.rename(columns={"article_ids_clicked": "article_id"})
    frames.append(clicked)

    history_path = split_dir / "history.parquet"
    if history_path.exists():
        history = pd.read_parquet(history_path)
        if {"article_id_fixed", "impression_time_fixed"} <= set(history.columns):
            hist = history[["user_id", "article_id_fixed", "impression_time_fixed"]]
            hist = hist.explode(["article_id_fixed", "impression_time_fixed"]).dropna()
            hist = hist.rename(
                columns={
                    "article_id_fixed": "article_id",
                    "impression_time_fixed": "impression_time",
                }
            )
            frames.append(hist)

    inter = pd.concat(frames, ignore_index=True)
    inter["ts"] = pd.to_datetime(inter["impression_time"]).astype("int64") // 10**9
    inter["article_id"] = inter["article_id"].astype("int64")
    return (
        inter[["user_id", "article_id", "ts"]]
        .drop_duplicates(subset=["user_id", "article_id"], keep="first")
        .reset_index(drop=True)
    )


def load_interactions(raw_root: Path) -> pd.DataFrame:
    """Load interactions from all available EB-NeRD splits."""
    if (raw_root / "train" / "behaviors.parquet").exists():
        frames = []
        for split in ("train", "validation", "test"):
            split_dir = raw_root / split
            if (split_dir / "behaviors.parquet").exists():
                frames.append(_load_split_interactions(split_dir))
        return pd.concat(frames, ignore_index=True)

    return _load_split_interactions(raw_root)


def load_articles(raw_root: Path) -> pd.DataFrame:
    articles_path = raw_root / "articles.parquet"
    if not articles_path.exists():
        raise FileNotFoundError(f"Missing articles metadata at {articles_path}")

    articles = pd.read_parquet(articles_path)
    title = articles.get("title", pd.Series([""] * len(articles))).fillna("")
    subtitle = articles.get("subtitle", pd.Series([""] * len(articles))).fillna("")
    body = articles.get("body", pd.Series([""] * len(articles))).fillna("")
    text = title + " " + subtitle + " " + body.str.slice(0, 500)

    category = articles.get("category_str", articles.get("category", "unknown"))
    if hasattr(category, "dtype") and category.dtype != object:
        category = category.astype(str)

    out = pd.DataFrame(
        {
            "article_id": articles["article_id"].astype("int64"),
            "title": title,
            "category": category,
            "published_ts": pd.to_datetime(articles["published_time"]).astype("int64")
            // 10**9,
            "text": text,
            "pageviews": articles.get(
                "total_pageviews", pd.Series(np.ones(len(articles)))
            ).fillna(0.0),
        }
    )
    return out


def chronological_split(
    inter: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user chronological 70/10/20 split."""
    inter = inter.sort_values(["user_idx", "ts"], kind="mergesort")
    counts = inter.groupby("user_idx")["item_idx"].transform("size")
    rank = inter.groupby("user_idx").cumcount()
    frac = (rank + 1) / counts

    too_few = counts < 5
    train_mask = too_few | (frac <= train_frac)
    val_mask = ~train_mask & (frac <= train_frac + val_frac)
    test_mask = ~train_mask & ~val_mask

    return (
        inter[train_mask].reset_index(drop=True),
        inter[val_mask].reset_index(drop=True),
        inter[test_mask].reset_index(drop=True),
    )


def main() -> None:
    raw_root = resolve_raw_root()
    print(f"Reading raw EB-NeRD data from {raw_root}")

    inter = load_interactions(raw_root)
    articles = load_articles(raw_root)

    inter = inter[inter["article_id"].isin(set(articles["article_id"]))]

    user_counts = inter["user_id"].value_counts()
    inter = inter[inter["user_id"].isin(user_counts[user_counts >= 3].index)]
    item_counts = inter["article_id"].value_counts()
    inter = inter[inter["article_id"].isin(item_counts[item_counts >= 2].index)]

    user_ids = np.sort(inter["user_id"].unique())
    item_ids = np.sort(inter["article_id"].unique())
    user_map = {int(u): i for i, u in enumerate(user_ids)}
    item_map = {int(a): i for i, a in enumerate(item_ids)}

    inter["user_idx"] = inter["user_id"].map(user_map)
    inter["item_idx"] = inter["article_id"].map(item_map)
    inter = inter[["user_idx", "item_idx", "ts"]]

    articles = articles[articles["article_id"].isin(item_map)].copy()
    articles["item_idx"] = articles["article_id"].map(item_map)
    articles = articles.sort_values("item_idx").reset_index(drop=True)

    train, val, test = chronological_split(inter)

    out = config.DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    articles.to_parquet(out / "articles.parquet", index=False)

    stats = {
        "n_users": len(user_ids),
        "n_items": len(item_ids),
        "n_interactions": len(inter),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "raw_source": str(raw_root),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Processed data written to {out}")


if __name__ == "__main__":
    main()
