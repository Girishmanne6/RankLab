"""Shared fixtures: a small deterministic interaction dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

N_USERS = 40
N_ITEMS = 30


@pytest.fixture(scope="session")
def dataset() -> dict:
    """Deterministic synthetic dataset with popularity skew + categories."""
    rng = np.random.default_rng(7)
    base_ts = 1_700_000_000

    # Popularity-skewed item sampling.
    item_weights = 1.0 / (np.arange(N_ITEMS) + 1)
    item_weights /= item_weights.sum()

    rows = []
    for u in range(N_USERS):
        n_clicks = rng.integers(6, 15)
        items = rng.choice(N_ITEMS, size=n_clicks, replace=False, p=item_weights)
        times = base_ts + rng.integers(0, 14 * 24 * 3600, size=n_clicks)
        for i, t in zip(items, np.sort(times)):
            rows.append((u, int(i), int(t)))

    inter = pd.DataFrame(rows, columns=["user_idx", "item_idx", "ts"])
    inter = inter.sort_values(["user_idx", "ts"]).reset_index(drop=True)

    rank = inter.groupby("user_idx").cumcount()
    size = inter.groupby("user_idx")["item_idx"].transform("size")
    frac = (rank + 1) / size
    train = inter[frac <= 0.7].reset_index(drop=True)
    val = inter[(frac > 0.7) & (frac <= 0.8)].reset_index(drop=True)
    test = inter[frac > 0.8].reset_index(drop=True)

    cats = ["sport", "news", "tech", "music", "crime"]
    articles = pd.DataFrame(
        {
            "item_idx": np.arange(N_ITEMS),
            "article_id": np.arange(N_ITEMS) + 1000,
            "title": [f"{cats[i % 5]} article {i}" for i in range(N_ITEMS)],
            "category": [cats[i % 5] for i in range(N_ITEMS)],
            "published_ts": base_ts
            + rng.integers(0, 14 * 24 * 3600, size=N_ITEMS),
            "text": [
                f"{cats[i % 5]} story about topic {i} with words "
                f"{cats[i % 5]} {cats[(i + 1) % 5]}"
                for i in range(N_ITEMS)
            ],
            "pageviews": rng.integers(1, 1000, size=N_ITEMS).astype(float),
        }
    )

    return {
        "train": train,
        "val": val,
        "test": test,
        "articles": articles,
        "n_users": N_USERS,
        "n_items": N_ITEMS,
    }
