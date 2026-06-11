"""POST /recommend — serve ranked recommendations for a user."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.store import get_store

router = APIRouter(tags=["recommendations"])


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="User index (user_idx)")
    model_name: str = Field("xsimgcl", description="One of /models")
    k: int = Field(10, ge=1, le=100)


class RecommendationItem(BaseModel):
    rank: int
    item_idx: int
    article_id: int | None = None
    title: str
    category: str
    published_ts: int | None = None
    score: float
    explanation: str


class RecommendResponse(BaseModel):
    user_id: int
    model_name: str
    cold_start: bool
    latency_ms: float
    recommendations: list[RecommendationItem]


@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    store = get_store()
    try:
        model = store.get_model(req.model_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    start = time.perf_counter()
    recs = model.recommend(req.user_id, k=req.k)
    latency_ms = (time.perf_counter() - start) * 1000.0

    items = []
    for rank, (item_idx, score) in enumerate(recs, start=1):
        info = store.article_info(item_idx)
        items.append(
            RecommendationItem(
                rank=rank,
                item_idx=item_idx,
                article_id=info.get("article_id"),
                title=info.get("title", "unknown"),
                category=info.get("category", "unknown"),
                published_ts=info.get("published_ts"),
                score=score,
                explanation=model.explain(req.user_id, item_idx),
            )
        )

    return RecommendResponse(
        user_id=req.user_id,
        model_name=req.model_name,
        cold_start=not model.is_known_user(req.user_id),
        latency_ms=round(latency_ms, 3),
        recommendations=items,
    )
