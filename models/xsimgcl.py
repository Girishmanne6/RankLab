"""XSimGCL (Yu et al., 2023): LightGCN + cross-layer contrastive learning."""

from __future__ import annotations

import torch

from .lightgcn import LightGCNModel, LightGCNNet


def info_nce(
    view_a: torch.Tensor, view_b: torch.Tensor, temperature: float
) -> torch.Tensor:
    """InfoNCE between two views of the same nodes (in-batch negatives)."""
    a = torch.nn.functional.normalize(view_a, dim=1)
    b = torch.nn.functional.normalize(view_b, dim=1)
    pos = (a * b).sum(dim=1) / temperature
    logits = (a @ b.T) / temperature
    return (torch.logsumexp(logits, dim=1) - pos).mean()


class XSimGCLModel(LightGCNModel):
    name = "xsimgcl"

    def __init__(
        self,
        eps: float = 0.2,
        temperature: float = 0.2,
        cl_lambda: float = 0.1,
        contrast_layer: int = 1,
        noise_type: str = "uniform",
        cl_sample_size: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.eps = eps
        self.temperature = temperature
        self.cl_lambda = cl_lambda
        self.contrast_layer = contrast_layer
        self.noise_type = noise_type
        self.cl_sample_size = cl_sample_size

    def _noisy_propagate(
        self, net: LightGCNNet, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = net.embedding.weight
        layers = []
        for _ in range(net.n_layers):
            emb = torch.sparse.mm(adj, emb)
            if self.noise_type == "gaussian":
                noise = torch.randn_like(emb)
            else:
                noise = torch.rand_like(emb)
            noise = torch.nn.functional.normalize(noise, dim=1)
            emb = emb + self.eps * torch.sign(emb) * noise.abs()
            layers.append(emb)
        final = torch.stack(layers, dim=0).mean(dim=0)
        l_star = layers[self.contrast_layer - 1]
        return final, l_star

    def _epoch_aux_loss(
        self, net: LightGCNNet, adj: torch.Tensor
    ) -> torch.Tensor | None:
        """One contrastive step per epoch (much faster than per-batch CL)."""
        final, l_star = self._noisy_propagate(net, adj)
        n = min(self.cl_sample_size, self.n_users, self.n_items)
        user_idx = torch.randperm(self.n_users, device=self.device)[:n]
        item_idx = torch.randperm(self.n_items, device=self.device)[:n]

        u_final, i_final = final[: self.n_users], final[self.n_users :]
        u_star, i_star = l_star[: self.n_users], l_star[self.n_users :]

        cl = info_nce(
            u_final[user_idx], u_star[user_idx], self.temperature
        ) + info_nce(i_final[item_idx], i_star[item_idx], self.temperature)
        return self.cl_lambda * cl

    def explain(self, user_idx: int, item_idx: int) -> str:
        if not self.is_known_user(user_idx):
            return "Popular fallback (cold-start user with no training history)."
        return (
            "Matched via debiased graph embeddings: contrastive training "
            "spreads exposure to long-tail articles similar users engaged with."
        )
