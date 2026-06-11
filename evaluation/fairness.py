"""Fairness and responsibility metrics for news recommendation.

These quantify *who gets exposure*, not just ranking accuracy:

  * intra-list diversity   — category spread within a recommendation list
  * freshness              — average age (hours) of recommended articles
  * long-tail exposure     — share of recommendations outside the most
                             popular 20% of items
  * popularity bias score  — Spearman correlation between item popularity
                             and how often the item gets recommended
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def intra_list_diversity(
    ranked: Sequence[int], item_categories: dict[int, str]
) -> float:
    """Fraction of item pairs in the list that belong to different categories.

    1.0 = every pair differs (maximally diverse), 0.0 = single category.
    """
    cats = [item_categories.get(i, "unknown") for i in ranked]
    n = len(cats)
    if n < 2:
        return 0.0
    diff_pairs = sum(
        1 for a in range(n) for b in range(a + 1, n) if cats[a] != cats[b]
    )
    return diff_pairs / (n * (n - 1) / 2)


def freshness_hours(
    ranked: Sequence[int], published_ts: dict[int, float], now_ts: float
) -> float:
    """Mean age in hours of the recommended articles at time `now_ts`."""
    ages = [
        max(now_ts - published_ts.get(i, now_ts), 0.0) / 3600.0 for i in ranked
    ]
    return float(np.mean(ages)) if ages else 0.0


def long_tail_exposure(
    ranked: Sequence[int], head_items: set[int]
) -> float:
    """Share of recommendations that are NOT in the popular head.

    `head_items` should contain the top-20% most popular items.
    """
    if not ranked:
        return 0.0
    return sum(1 for i in ranked if i not in head_items) / len(ranked)


def popularity_head(train: pd.DataFrame, n_items: int, head_frac: float = 0.2) -> set[int]:
    """Items in the top `head_frac` of training popularity."""
    counts = np.zeros(n_items, dtype=np.int64)
    np.add.at(counts, train["item_idx"].to_numpy(), 1)
    n_head = max(int(n_items * head_frac), 1)
    return set(np.argsort(-counts)[:n_head].tolist())


def popularity_bias_score(
    rec_frequency: np.ndarray, train_popularity: np.ndarray
) -> float:
    """Pearson correlation between item popularity and recommendation rate.

    1.0 means the model purely amplifies popularity (head items soak up all
    exposure); values near 0 mean exposure is decoupled from historical
    popularity.
    """
    if rec_frequency.std() == 0 or train_popularity.std() == 0:
        return 0.0
    corr, _ = stats.pearsonr(train_popularity, rec_frequency)
    return float(corr) if np.isfinite(corr) else 0.0


def evaluate_fairness(
    recommend_fn: Callable[[int, int], list[int]],
    train: pd.DataFrame,
    test: pd.DataFrame,
    articles: pd.DataFrame,
    n_items: int,
    k: int = 10,
    max_users: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Aggregate fairness metrics over (a sample of) test users.

    Also returns `exposure_scatter`-ready arrays via rec_frequency.
    """
    item_categories = dict(zip(articles["item_idx"], articles["category"]))
    published = dict(
        zip(articles["item_idx"], articles["published_ts"].astype(float))
    )
    now_ts = float(train["ts"].max())
    head = popularity_head(train, n_items)

    users = test["user_idx"].unique()
    if max_users is not None and len(users) > max_users:
        users = np.random.default_rng(seed).choice(users, max_users, replace=False)

    rec_freq = np.zeros(n_items, dtype=np.float64)
    diversities, freshnesses, tails = [], [], []
    for u in users:
        ranked = recommend_fn(int(u), k)
        np.add.at(rec_freq, np.asarray(ranked, dtype=int), 1.0)
        diversities.append(intra_list_diversity(ranked, item_categories))
        freshnesses.append(freshness_hours(ranked, published, now_ts))
        tails.append(long_tail_exposure(ranked, head))

    train_pop = np.zeros(n_items, dtype=np.float64)
    np.add.at(train_pop, train["item_idx"].to_numpy(), 1.0)

    return {
        "diversity": float(np.mean(diversities)) if diversities else 0.0,
        "freshness_hours": float(np.mean(freshnesses)) if freshnesses else 0.0,
        "long_tail_exposure": float(np.mean(tails)) if tails else 0.0,
        "popularity_bias": popularity_bias_score(rec_freq, train_pop),
        "n_users_evaluated": float(len(users)),
    }


def exposure_scatter(
    recommend_fn: Callable[[int, int], list[int]],
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_items: int,
    k: int = 10,
    max_users: int = 300,
    max_points: int = 400,
    seed: int = 42,
) -> list[dict[str, float]]:
    """(popularity, recommendation frequency) points for the bias scatter plot."""
    users = test["user_idx"].unique()
    rng = np.random.default_rng(seed)
    if len(users) > max_users:
        users = rng.choice(users, max_users, replace=False)

    rec_freq = np.zeros(n_items, dtype=np.float64)
    for u in users:
        ranked = recommend_fn(int(u), k)
        np.add.at(rec_freq, np.asarray(ranked, dtype=int), 1.0)

    train_pop = np.zeros(n_items, dtype=np.float64)
    np.add.at(train_pop, train["item_idx"].to_numpy(), 1.0)

    idx = np.arange(n_items)
    if n_items > max_points:
        idx = rng.choice(idx, max_points, replace=False)
    return [
        {"popularity": float(train_pop[i]), "rec_frequency": float(rec_freq[i])}
        for i in idx
    ]
