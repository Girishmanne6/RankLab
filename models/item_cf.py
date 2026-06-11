"""Item-item collaborative filtering with TF-IDF content similarity.

For each candidate item, the score is the similarity-weighted sum over the
user's clicked items, where similarity is the cosine similarity between
TF-IDF vectors of article text, restricted to each item's top-k neighbours
(sparse kNN graph keeps inference fast).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import Recommender


class ItemCFModel(Recommender):
    name = "item_cf"

    def __init__(self, top_k_neighbors: int = 50, max_features: int = 20_000) -> None:
        super().__init__()
        self.top_k_neighbors = top_k_neighbors
        self.max_features = max_features
        self.sim_: sparse.csr_matrix | None = None
        self.fallback_: np.ndarray | None = None

    def fit(
        self,
        train: pd.DataFrame,
        articles: pd.DataFrame,
        n_users: int,
        n_items: int,
    ) -> "ItemCFModel":
        self.n_users, self.n_items = n_users, n_items
        self._index_seen(train)
        self._titles = dict(
            zip(articles["item_idx"].to_numpy(), articles["title"])
        )

        texts = [""] * n_items
        for idx, text in zip(articles["item_idx"], articles["text"]):
            texts[int(idx)] = str(text)

        vectorizer = TfidfVectorizer(
            max_features=self.max_features, sublinear_tf=True, stop_words=None
        )
        tfidf = vectorizer.fit_transform(texts)

        # Dense cosine in blocks, then sparsify to top-k per row.
        k = min(self.top_k_neighbors, n_items - 1)
        rows, cols, vals = [], [], []
        block = 512
        for start in range(0, n_items, block):
            sims = cosine_similarity(tfidf[start : start + block], tfidf)
            for local_i in range(sims.shape[0]):
                i = start + local_i
                sims[local_i, i] = 0.0  # remove self-similarity
                top = np.argpartition(-sims[local_i], k - 1)[:k]
                top = top[sims[local_i, top] > 0]
                rows.extend([i] * len(top))
                cols.extend(top.tolist())
                vals.extend(sims[local_i, top].tolist())

        self.sim_ = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_items, n_items)
        )

        # Popularity prior for cold-start users.
        pop = np.zeros(n_items, dtype=np.float64)
        np.add.at(pop, train["item_idx"].to_numpy(), 1.0)
        self.fallback_ = pop / max(pop.max(), 1.0)
        return self

    def score_all(self, user_idx: int) -> np.ndarray:
        assert self.sim_ is not None and self.fallback_ is not None
        history = self.seen(user_idx)
        if not history:
            return self.fallback_
        user_vec = np.zeros(self.n_items, dtype=np.float64)
        user_vec[list(history)] = 1.0
        # candidate scores = sum of similarities to clicked items
        scores = np.asarray(user_vec @ self.sim_).ravel()
        return scores

    def explain(self, user_idx: int, item_idx: int) -> str:
        assert self.sim_ is not None
        history = self.seen(user_idx)
        if not history:
            return "Popular fallback (no click history for this user)."
        sims = self.sim_[list(history), item_idx].toarray().ravel()
        if sims.max() <= 0:
            return "Weakly similar to your overall reading history."
        best = list(history)[int(np.argmax(sims))]
        title = self._titles.get(best, f"article {best}")
        return f"Similar content to '{title}' which you read (cos={sims.max():.2f})."
