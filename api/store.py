"""Application state: processed data, trained models, cached metrics."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

import config
from evaluation.ab_testing import ABTestFramework
from models import MODEL_REGISTRY, Recommender


class ModelStore:
    """Lazily loads processed data and trained model checkpoints."""

    def __init__(self, data_dir: Path, model_dir: Path) -> None:
        self.data_dir = data_dir
        self.model_dir = model_dir
        self._models: dict[str, Recommender] = {}
        self._data: dict[str, pd.DataFrame] = {}

    # -- data ----------------------------------------------------------
    def df(self, split: str) -> pd.DataFrame:
        if split not in self._data:
            self._data[split] = pd.read_parquet(self.data_dir / f"{split}.parquet")
        return self._data[split]

    @property
    def articles(self) -> pd.DataFrame:
        return self.df("articles")

    @property
    def stats(self) -> dict:
        return json.loads((self.data_dir / "stats.json").read_text())

    def data_ready(self) -> bool:
        return (self.data_dir / "stats.json").exists()

    # -- models --------------------------------------------------------
    def checkpoint_path(self, name: str) -> Path:
        return self.model_dir / f"{name}.pkl"

    def available_models(self) -> list[dict]:
        out = []
        for name in MODEL_REGISTRY:
            path = self.checkpoint_path(name)
            out.append(
                {
                    "name": name,
                    "trained": path.exists(),
                    "loaded": name in self._models,
                }
            )
        return out

    def get_model(self, name: str) -> Recommender:
        if name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
        if name not in self._models:
            path = self.checkpoint_path(name)
            if not path.exists():
                raise FileNotFoundError(
                    f"Model '{name}' has no checkpoint at {path}. "
                    "Run `python -m api.train` first."
                )
            self._models[name] = MODEL_REGISTRY[name].load(path)
        return self._models[name]

    # -- cached evaluation artifacts ------------------------------------
    def metrics(self) -> dict:
        path = self.model_dir / "metrics.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def scatter(self) -> dict:
        path = self.model_dir / "scatter.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    # -- article metadata for the dashboard ------------------------------
    def article_info(self, item_idx: int) -> dict:
        row = self.articles[self.articles["item_idx"] == item_idx]
        if row.empty:
            return {"item_idx": item_idx, "title": "unknown", "category": "unknown"}
        r = row.iloc[0]
        return {
            "item_idx": int(item_idx),
            "article_id": int(r["article_id"]),
            "title": str(r["title"]),
            "category": str(r["category"]),
            "published_ts": int(r["published_ts"]),
        }


@lru_cache(maxsize=1)
def get_store() -> ModelStore:
    return ModelStore(config.DATA_DIR, config.MODEL_DIR)


@lru_cache(maxsize=1)
def get_ab_framework() -> ABTestFramework:
    return ABTestFramework(config.DATABASE_URL)
