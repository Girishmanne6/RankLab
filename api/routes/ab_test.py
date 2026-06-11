"""A/B testing endpoints: start tests, collect observations, view results."""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.store import get_ab_framework, get_store
from evaluation.ab_testing import assign_variant
from evaluation.metrics import ndcg_at_k

router = APIRouter(prefix="/ab-test", tags=["ab-testing"])


class StartTestRequest(BaseModel):
    name: str = Field(..., examples=["lightgcn vs xsimgcl"])
    variant_a: str = Field(..., examples=["lightgcn"])
    variant_b: str = Field(..., examples=["xsimgcl"])
    primary_metric: str = "ndcg@10"
    simulate_users: int = Field(
        400,
        ge=0,
        le=5000,
        description=(
            "Replay this many held-out test users through both arms to "
            "seed the experiment with observations (0 = start empty)."
        ),
    )


@router.post("/start")
def start_test(req: StartTestRequest) -> dict:
    store = get_store()
    ab = get_ab_framework()

    for variant in (req.variant_a, req.variant_b):
        try:
            store.get_model(variant)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    test_id = ab.start_test(req.name, req.variant_a, req.variant_b, req.primary_metric)

    n_seeded = 0
    if req.simulate_users > 0:
        n_seeded = _simulate(test_id, req.variant_a, req.variant_b, req.simulate_users)

    return {"test_id": test_id, "status": "running", "observations_seeded": n_seeded}


def _simulate(test_id: int, variant_a: str, variant_b: str, n_users: int) -> int:
    """Replay held-out test users through their assigned arm.

    Each user is hashed into arm A or B; we serve recommendations with the
    arm's model and log NDCG@10 against the user's held-out clicks plus the
    observed inference latency.
    """
    store = get_store()
    ab = get_ab_framework()

    test_df = store.df("test")
    truth = test_df.groupby("user_idx")["item_idx"].agg(set)
    users = truth.index.to_numpy()
    if len(users) > n_users:
        users = np.random.default_rng(0).choice(users, n_users, replace=False)

    models = {"A": store.get_model(variant_a), "B": store.get_model(variant_b)}
    rows = []
    for u in users:
        variant = assign_variant(test_id, int(u))
        model = models[variant]
        start = time.perf_counter()
        ranked = [i for i, _ in model.recommend(int(u), k=10)]
        latency_ms = (time.perf_counter() - start) * 1000.0
        rows.append(
            {
                "variant": variant,
                "user_idx": int(u),
                "metric_value": ndcg_at_k(ranked, truth.loc[u], k=10),
                "latency_ms": latency_ms,
            }
        )
    ab.log_observations_bulk(test_id, rows)
    return len(rows)


@router.get("/results")
def results() -> dict:
    """Results for the most recent active test (or 404 if none)."""
    ab = get_ab_framework()
    res = ab.results()
    if res is None:
        raise HTTPException(status_code=404, detail="No active A/B test.")
    return res


@router.get("/results/{test_id}")
def results_by_id(test_id: int) -> dict:
    ab = get_ab_framework()
    res = ab.results(test_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"No test with id {test_id}.")
    return res
