"""Recency-weighted baseline: rank items by publish-time freshness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Recommender


class RecencyModel(Recommender):
    """Score items by exponential decay over article age.

    score(i) = exp(-age_hours(i) / tau) * (1 + alpha * log1p(popularity(i)))

    The small popularity term breaks ties between articles published at
    similar times; ``recommend`` (from the base class) excludes items the
    user has already seen, so this never re-serves read articles.
    """

    name = "recency"

    def __init__(self, tau_hours: float = 48.0, alpha: float = 0.1) -> None:
        super().__init__()
        self.tau_hours = tau_hours
        self.alpha = alpha
        self.scores_: np.ndarray | None = None

    def fit(
        self,
        train: pd.DataFrame,
        articles: pd.DataFrame,
        n_users: int,
        n_items: int,
    ) -> "RecencyModel":
        self.n_users, self.n_items = n_users, n_items
        self._index_seen(train)

        now = float(train["ts"].max())
        published = np.zeros(n_items, dtype=np.float64)
        published[articles["item_idx"].to_numpy()] = articles[
            "published_ts"
        ].to_numpy()
        self._age_hours = np.maximum((now - published) / 3600.0, 0.0)

        clicks = np.zeros(n_items, dtype=np.float64)
        np.add.at(clicks, train["item_idx"].to_numpy(), 1.0)

        self.scores_ = np.exp(-self._age_hours / self.tau_hours) * (
            1.0 + self.alpha * np.log1p(clicks)
        )
        return self

    def score_all(self, user_idx: int) -> np.ndarray:
        assert self.scores_ is not None, "fit() must be called first"
        return self.scores_

    def explain(self, user_idx: int, item_idx: int) -> str:
        age = self._age_hours[item_idx]
        return f"Fresh article published {age:.1f}h before the end of training data."
