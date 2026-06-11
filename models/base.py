"""Common recommender interface shared by all RankLab models."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class Recommender(ABC):
    """Base class: fit on the train split, score every item for a user.

    Cold start contract: `score_all` must return a valid score vector for
    *any* user index, including unseen users (models fall back to a
    popularity-style prior in that case).
    """

    name: str = "base"

    def __init__(self) -> None:
        self.n_users: int = 0
        self.n_items: int = 0
        self._seen: dict[int, set[int]] = {}

    # ------------------------------------------------------------------
    @abstractmethod
    def fit(
        self,
        train: pd.DataFrame,
        articles: pd.DataFrame,
        n_users: int,
        n_items: int,
    ) -> "Recommender":
        """Train on interactions (user_idx, item_idx, ts) + article metadata."""

    @abstractmethod
    def score_all(self, user_idx: int) -> np.ndarray:
        """Return a score for every item (shape: [n_items])."""

    # ------------------------------------------------------------------
    def _index_seen(self, train: pd.DataFrame) -> None:
        self._seen = (
            train.groupby("user_idx")["item_idx"].agg(set).to_dict()
        )

    def seen(self, user_idx: int) -> set[int]:
        return self._seen.get(user_idx, set())

    def is_known_user(self, user_idx: int) -> bool:
        return user_idx in self._seen

    def recommend(
        self, user_idx: int, k: int = 10, exclude_seen: bool = True
    ) -> list[tuple[int, float]]:
        """Top-k (item_idx, score) pairs, highest score first."""
        scores = self.score_all(user_idx).astype(np.float64).copy()
        if exclude_seen:
            seen = list(self.seen(user_idx))
            if seen:
                scores[seen] = -np.inf
        k = min(k, self.n_items)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def explain(self, user_idx: int, item_idx: int) -> str:
        """Human-readable reason an item was recommended."""
        return f"Ranked highly by {self.name} for user {user_idx}."

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path) -> "Recommender":
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return model
