"""GET /metrics/{model_name}, /metrics, /scatter — cached evaluation results."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.store import get_store

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def all_metrics() -> dict:
    """Evaluation metrics for every trained model."""
    return get_store().metrics()


@router.get("/metrics/{model_name}")
def model_metrics(model_name: str) -> dict:
    metrics = get_store().metrics()
    if model_name not in metrics:
        raise HTTPException(
            status_code=404,
            detail=f"No cached metrics for '{model_name}'. Run `python -m api.train`.",
        )
    return {"model": model_name, **metrics[model_name]}


@router.get("/scatter/{model_name}")
def model_scatter(model_name: str) -> dict:
    """(popularity, recommendation frequency) points for the bias scatter."""
    scatter = get_store().scatter()
    if model_name not in scatter:
        raise HTTPException(
            status_code=404, detail=f"No scatter data for '{model_name}'."
        )
    return {"model": model_name, "points": scatter[model_name]}
