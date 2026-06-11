"""RankLab API — responsible news recommendation service.

Run locally:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import ab_test, metrics, recommendations
from api.store import get_store

app = FastAPI(
    title="RankLab",
    description=(
        "Responsible news recommendation on the EB-NeRD benchmark: "
        "five models, fairness analysis, and live A/B testing."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendations.router)
app.include_router(metrics.router)
app.include_router(ab_test.router)


@app.get("/health", tags=["system"])
def health() -> dict:
    store = get_store()
    return {
        "status": "ok",
        "data_ready": store.data_ready(),
        "stats": store.stats if store.data_ready() else None,
    }


@app.get("/models", tags=["system"])
def models() -> list[dict]:
    """Available models, whether each is trained and currently loaded."""
    return get_store().available_models()


@app.get("/users/sample", tags=["system"])
def sample_users(n: int = 20) -> list[int]:
    """A few valid user indices, handy for trying out /recommend."""
    store = get_store()
    test = store.df("test")
    users = test["user_idx"].unique()[:n]
    return [int(u) for u in users]
