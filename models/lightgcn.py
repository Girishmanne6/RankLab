"""LightGCN (He et al., SIGIR 2020), implemented from scratch in PyTorch.

Faithful to the paper:
  * embeddings only — no feature transformation matrices
  * no nonlinear activation
  * light graph convolution: E^(l+1) = Â E^(l) with the symmetrically
    normalised bipartite adjacency Â = D^{-1/2} A D^{-1/2}
  * layer combination: final embedding is the mean of E^(0..L)
  * BPR (Bayesian Personalised Ranking) loss with L2 regularisation

Defaults: 3 convolution layers, embedding dim 64, up to 50 epochs with
early stopping on validation Recall@10.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .base import Recommender


def build_norm_adj(
    users: np.ndarray, items: np.ndarray, n_users: int, n_items: int
) -> torch.Tensor:
    """Symmetrically normalised adjacency of the bipartite user-item graph.

    Node ordering: [users (0..n_users-1), items (n_users..n_users+n_items-1)].
    """
    n = n_users + n_items
    rows = np.concatenate([users, items + n_users])
    cols = np.concatenate([items + n_users, users])
    vals = np.ones(len(rows), dtype=np.float32)

    deg = np.zeros(n, dtype=np.float32)
    np.add.at(deg, rows, 1.0)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    norm_vals = vals * d_inv_sqrt[rows] * d_inv_sqrt[cols]

    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    values = torch.tensor(norm_vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


class LightGCNNet(torch.nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, n_layers: int) -> None:
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.n_layers = n_layers
        self.embedding = torch.nn.Embedding(n_users + n_items, dim)
        torch.nn.init.normal_(self.embedding.weight, std=0.1)

    def propagate(self, adj: torch.Tensor) -> list[torch.Tensor]:
        """Return per-layer embeddings [E^0, E^1, ..., E^L]."""
        emb = self.embedding.weight
        layers = [emb]
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(adj, emb)
            layers.append(emb)
        return layers

    def forward(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        final = torch.stack(self.propagate(adj), dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]


def bpr_loss(
    user_e: torch.Tensor,
    pos_e: torch.Tensor,
    neg_e: torch.Tensor,
    reg_embs: list[torch.Tensor],
    reg_weight: float,
    batch_size: int,
) -> torch.Tensor:
    pos_scores = (user_e * pos_e).sum(dim=1)
    neg_scores = (user_e * neg_e).sum(dim=1)
    loss = torch.nn.functional.softplus(neg_scores - pos_scores).mean()
    reg = sum((e**2).sum() for e in reg_embs) / batch_size
    return loss + reg_weight * reg


class LightGCNModel(Recommender):
    name = "lightgcn"

    def __init__(
        self,
        embedding_dim: int = 64,
        n_layers: int = 3,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 2048,
        reg_weight: float = 1e-4,
        patience: int = 5,
        device: str = "cpu",
        seed: int = 42,
        max_train_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.reg_weight = reg_weight
        self.patience = patience
        self.device = torch.device(device)
        self.seed = seed
        self.max_train_samples = max_train_samples

        self.user_emb_: np.ndarray | None = None
        self.item_emb_: np.ndarray | None = None
        self.fallback_: np.ndarray | None = None
        self.history_: list[dict] = []

    # ------------------------------------------------------------------
    def _sample_negatives(
        self, users: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Uniform negative sampling with rejection against seen items."""
        negs = rng.integers(0, self.n_items, size=len(users))
        for i, u in enumerate(users):
            seen = self._seen.get(int(u), set())
            while int(negs[i]) in seen:
                negs[i] = rng.integers(0, self.n_items)
        return negs

    def _val_recall(self, val: pd.DataFrame, k: int = 10, max_users: int = 500) -> float:
        """Recall@k on a sample of validation users (for early stopping)."""
        if val is None or len(val) == 0:
            return 0.0
        truth = val.groupby("user_idx")["item_idx"].agg(set)
        users = truth.index.to_numpy()
        if len(users) > max_users:
            users = np.random.default_rng(self.seed).choice(
                users, size=max_users, replace=False
            )
        hits, total = 0.0, 0.0
        for u in users:
            recs = [i for i, _ in self.recommend(int(u), k=k)]
            rel = truth.loc[u]
            hits += len(set(recs) & rel)
            total += min(len(rel), k)
        return hits / max(total, 1.0)

    # ------------------------------------------------------------------
    def fit(
        self,
        train: pd.DataFrame,
        articles: pd.DataFrame,
        n_users: int,
        n_items: int,
        val: pd.DataFrame | None = None,
    ) -> "LightGCNModel":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self.n_users, self.n_items = n_users, n_items
        self._index_seen(train)

        users = train["user_idx"].to_numpy()
        items = train["item_idx"].to_numpy()
        adj = build_norm_adj(users, items, n_users, n_items).to(self.device)

        n_all = len(users)
        n_train = n_all
        if self.max_train_samples is not None:
            n_train = min(n_all, self.max_train_samples)
        elif n_all > 500_000:
            n_train = 500_000
            print(
                f"[{self.name}] subsampling BPR to {n_train:,}/{n_all:,} "
                "interactions per epoch (full graph still used)"
            )

        net = LightGCNNet(n_users, n_items, self.embedding_dim, self.n_layers).to(
            self.device
        )
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)

        best_recall, best_state, epochs_no_improve = -1.0, None, 0

        for epoch in range(1, self.epochs + 1):
            net.train()
            idx = rng.choice(n_all, size=n_train, replace=False)
            total_loss = 0.0
            for start in range(0, n_train, self.batch_size):
                batch = idx[start : start + self.batch_size]
                bu = users[batch]
                bp = items[batch]
                bn = self._sample_negatives(bu, rng)

                opt.zero_grad()
                ue, ie = net(adj)
                u_t = torch.as_tensor(bu, dtype=torch.long, device=self.device)
                p_t = torch.as_tensor(bp, dtype=torch.long, device=self.device)
                n_t = torch.as_tensor(bn, dtype=torch.long, device=self.device)

                ego = net.embedding.weight
                loss = bpr_loss(
                    ue[u_t],
                    ie[p_t],
                    ie[n_t],
                    [ego[u_t], ego[p_t + n_users], ego[n_t + n_users]],
                    self.reg_weight,
                    len(batch),
                )
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(batch)

            cl_loss = self._epoch_aux_loss(net, adj)
            if cl_loss is not None:
                opt.zero_grad()
                cl_loss.backward()
                opt.step()
                total_loss += cl_loss.item() * n_train

            # Refresh inference embeddings, then validate.
            self._materialize(net, adj)
            recall = self._val_recall(val) if val is not None else 0.0
            self.history_.append(
                {"epoch": epoch, "loss": total_loss / n_train, "val_recall@10": recall}
            )
            print(
                f"[{self.name}] epoch {epoch:02d} "
                f"loss={total_loss / n_train:.4f} val_recall@10={recall:.4f}",
                flush=True,
            )

            if val is not None:
                if recall > best_recall + 1e-5:
                    best_recall = recall
                    best_state = {
                        k: v.detach().clone() for k, v in net.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(f"[{self.name}] early stopping at epoch {epoch}")
                        break

        if best_state is not None:
            net.load_state_dict(best_state)
        self._materialize(net, adj)

        pop = np.zeros(n_items, dtype=np.float64)
        np.add.at(pop, items, 1.0)
        self.fallback_ = pop / max(pop.max(), 1.0)
        return self

    def _epoch_aux_loss(
        self, net: LightGCNNet, adj: torch.Tensor
    ) -> torch.Tensor | None:
        """Optional once-per-epoch auxiliary loss (XSimGCL contrastive term)."""
        return None

    def _aux_loss(
        self,
        net: LightGCNNet,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos: torch.Tensor,
        ue: torch.Tensor | None = None,
        ie: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hook for subclasses (XSimGCL adds a contrastive term here)."""
        return torch.zeros((), device=self.device)

    def _materialize(self, net: LightGCNNet, adj: torch.Tensor) -> None:
        net.eval()
        with torch.no_grad():
            ue, ie = net(adj)
        self.user_emb_ = ue.cpu().numpy()
        self.item_emb_ = ie.cpu().numpy()

    # ------------------------------------------------------------------
    def score_all(self, user_idx: int) -> np.ndarray:
        assert self.user_emb_ is not None and self.item_emb_ is not None
        if user_idx < 0 or user_idx >= self.n_users or not self.is_known_user(user_idx):
            assert self.fallback_ is not None
            return self.fallback_
        return self.item_emb_ @ self.user_emb_[user_idx]

    def explain(self, user_idx: int, item_idx: int) -> str:
        if not self.is_known_user(user_idx):
            return "Popular fallback (cold-start user with no training history)."
        return (
            "High affinity in the learned user-item graph: users with similar "
            "click patterns also read this article."
        )

    # ------------------------------------------------------------------
    # Checkpointing: store numpy state rather than pickling torch modules.
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "class": type(self).__name__,
            "params": {
                "embedding_dim": self.embedding_dim,
                "n_layers": self.n_layers,
            },
            "n_users": self.n_users,
            "n_items": self.n_items,
            "user_emb": self.user_emb_,
            "item_emb": self.item_emb_,
            "fallback": self.fallback_,
            "seen": self._seen,
            "history": self.history_,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load(cls, path: Path) -> "LightGCNModel":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        model = cls(**state["params"])
        model.n_users = state["n_users"]
        model.n_items = state["n_items"]
        model.user_emb_ = state["user_emb"]
        model.item_emb_ = state["item_emb"]
        model.fallback_ = state["fallback"]
        model._seen = state["seen"]
        model.history_ = state.get("history", [])
        return model
