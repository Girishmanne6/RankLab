"""RankLab evaluation: ranking metrics, fairness analysis, A/B testing."""

from .metrics import evaluate_model, mrr, ndcg_at_k, p95_latency_ms, recall_at_k
from .fairness import (
    evaluate_fairness,
    exposure_scatter,
    freshness_hours,
    intra_list_diversity,
    long_tail_exposure,
    popularity_bias_score,
    popularity_head,
)
from .ab_testing import ABTestFramework, assign_variant, welch_t_test

__all__ = [
    "evaluate_model",
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "p95_latency_ms",
    "evaluate_fairness",
    "exposure_scatter",
    "intra_list_diversity",
    "freshness_hours",
    "long_tail_exposure",
    "popularity_bias_score",
    "popularity_head",
    "ABTestFramework",
    "assign_variant",
    "welch_t_test",
]
