"""Popularity baseline: rank items by (time-decayed) global click count."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Recommender


class PopularityModel(Recommender):
    """Score items by global click count with optional exponential time decay.

    With ``half_life_hours`` set, a click that happened ``t`` hours before the
    most recent training interaction contributes ``0.5 ** (t / half_life)``
    instead of 1, so recently popular items rank higher.
    """

    name = "popularity"

    def __init__(self, half_life_hours: float | None = 72.0) -> None:
        super().__init__()
        self.half_life_hours = half_life_hours
        self.scores_: np.ndarray | None = None

    def fit(
        self,
        train: pd.DataFrame,
        articles: pd.DataFrame,
        n_users: int,
        n_items: int,
    ) -> "PopularityModel":
        self.n_users, self.n_items = n_users, n_items
        self._index_seen(train)

        if self.half_life_hours:
            now = train["ts"].max()
            age_h = (now - train["ts"].to_numpy()) / 3600.0
            weights = np.power(0.5, age_h / self.half_life_hours)
        else:
            weights = np.ones(len(train))

        scores = np.zeros(n_items, dtype=np.float64)
        np.add.at(scores, train["item_idx"].to_numpy(), weights)
        self.scores_ = scores
        return self

    def score_all(self, user_idx: int) -> np.ndarray:
        assert self.scores_ is not None, "fit() must be called first"
        return self.scores_

    def explain(self, user_idx: int, item_idx: int) -> str:
        assert self.scores_ is not None
        rank = int((self.scores_ > self.scores_[item_idx]).sum()) + 1
        return (
            f"Globally popular article (#{rank} by time-decayed click count "
            f"across all users)."
        )
