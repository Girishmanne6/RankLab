"""Ranking quality metrics, implemented from scratch.

All functions take a recommended ranking (list of item indices, best first)
and a set of relevant (held-out) items, and return a float in [0, 1]
(except latency, measured in milliseconds).
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """|top-k ∩ relevant| / min(|relevant|, k).

    The min() denominator caps the achievable score at 1.0 even when a user
    has more relevant items than the list length.
    """
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain with binary relevance.

    DCG = sum over hit positions p (1-indexed) of 1 / log2(p + 1),
    normalised by the ideal DCG for this user.
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / np.log2(pos + 2)
        for pos, item in enumerate(ranked[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(pos + 2) for pos in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked: Sequence[int], relevant: set[int]) -> float:
    """Reciprocal rank of the first relevant item (0 if none found)."""
    for pos, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (pos + 1)
    return 0.0


def p95_latency_ms(latencies_ms: Iterable[float]) -> float:
    """95th percentile of recorded inference latencies (linear interpolation)."""
    arr = np.asarray(list(latencies_ms), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 95))


def evaluate_model(
    recommend_fn: Callable[[int, int], list[int]],
    test: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate a recommender over all (or a sample of) test users.

    Args:
        recommend_fn: (user_idx, k) -> ranked list of item indices.
        test: dataframe with user_idx / item_idx columns (held-out truth).
        max_users: optional cap to subsample users for speed.

    Returns averaged recall@k, ndcg@k, mrr and p95 latency in ms.
    """
    truth = test.groupby("user_idx")["item_idx"].agg(set)
    users = truth.index.to_numpy()
    if max_users is not None and len(users) > max_users:
        users = np.random.default_rng(seed).choice(users, max_users, replace=False)

    recalls, ndcgs, mrrs, latencies = [], [], [], []
    for u in users:
        start = time.perf_counter()
        ranked = recommend_fn(int(u), k)
        latencies.append((time.perf_counter() - start) * 1000.0)
        rel = truth.loc[u]
        recalls.append(recall_at_k(ranked, rel, k))
        ndcgs.append(ndcg_at_k(ranked, rel, k))
        mrrs.append(mrr(ranked, rel))

    return {
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "p95_latency_ms": p95_latency_ms(latencies),
        "n_users_evaluated": float(len(users)),
    }
