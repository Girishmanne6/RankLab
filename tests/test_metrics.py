"""Metric correctness tests on hand-computed inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.ab_testing import ABTestFramework, assign_variant, welch_t_test
from evaluation.fairness import (
    freshness_hours,
    intra_list_diversity,
    long_tail_exposure,
    popularity_bias_score,
    popularity_head,
)
from evaluation.metrics import mrr, ndcg_at_k, p95_latency_ms, recall_at_k


# ---------------------------------------------------------------- recall
def test_recall_perfect():
    assert recall_at_k([1, 2, 3], {1, 2, 3}, k=10) == 1.0


def test_recall_partial():
    # 2 of 4 relevant items in top-10 -> 2/4
    assert recall_at_k([1, 2, 5, 6], {1, 2, 3, 4}, k=10) == 0.5


def test_recall_capped_by_k():
    # 15 relevant items but k=10: hitting 10 of them is a perfect score.
    ranked = list(range(10))
    relevant = set(range(15))
    assert recall_at_k(ranked, relevant, k=10) == 1.0


def test_recall_empty_relevant():
    assert recall_at_k([1, 2, 3], set(), k=10) == 0.0


def test_recall_no_hits():
    assert recall_at_k([7, 8, 9], {1, 2, 3}, k=10) == 0.0


# ---------------------------------------------------------------- ndcg
def test_ndcg_perfect_ranking():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=10) == pytest.approx(1.0)


def test_ndcg_hit_at_position_2():
    # Single relevant item at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2).
    expected = (1 / np.log2(3)) / (1 / np.log2(2))
    assert ndcg_at_k([9, 1, 8], {1}, k=10) == pytest.approx(expected)


def test_ndcg_order_matters():
    relevant = {1}
    early = ndcg_at_k([1, 8, 9], relevant, k=10)
    late = ndcg_at_k([8, 9, 1], relevant, k=10)
    assert early > late


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k([7, 8], {1}, k=10) == 0.0


def test_ndcg_known_value():
    # Hits at ranks 1 and 3 with 2 relevant items.
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert ndcg_at_k([1, 9, 2], {1, 2}, k=10) == pytest.approx(dcg / idcg)


# ---------------------------------------------------------------- mrr
def test_mrr_first_position():
    assert mrr([5, 6, 7], {5}) == 1.0


def test_mrr_third_position():
    assert mrr([8, 9, 5], {5}) == pytest.approx(1 / 3)


def test_mrr_uses_first_hit():
    assert mrr([9, 5, 6], {5, 6}) == pytest.approx(1 / 2)


def test_mrr_no_hit():
    assert mrr([1, 2, 3], {99}) == 0.0


# ---------------------------------------------------------------- latency
def test_p95_latency():
    values = list(range(1, 101))  # 1..100 ms
    assert p95_latency_ms(values) == pytest.approx(95.05)


def test_p95_latency_empty():
    assert p95_latency_ms([]) == 0.0


# ---------------------------------------------------------------- fairness
def test_diversity_single_category():
    cats = {1: "sport", 2: "sport", 3: "sport"}
    assert intra_list_diversity([1, 2, 3], cats) == 0.0


def test_diversity_all_distinct():
    cats = {1: "a", 2: "b", 3: "c"}
    assert intra_list_diversity([1, 2, 3], cats) == 1.0


def test_diversity_mixed():
    # Pairs: (a,a) same, (a,b) diff, (a,b) diff -> 2/3
    cats = {1: "a", 2: "a", 3: "b"}
    assert intra_list_diversity([1, 2, 3], cats) == pytest.approx(2 / 3)


def test_freshness_hours():
    published = {1: 0.0, 2: 3600.0}
    # now = 7200s: ages are 2h and 1h -> mean 1.5h
    assert freshness_hours([1, 2], published, 7200.0) == pytest.approx(1.5)


def test_long_tail_exposure():
    head = {1, 2}
    assert long_tail_exposure([1, 2, 3, 4], head) == 0.5


def test_popularity_head_selects_top_items():
    train = pd.DataFrame(
        {"user_idx": [0] * 10, "item_idx": [0] * 5 + [1] * 3 + [2, 3], "ts": [0] * 10}
    )
    head = popularity_head(train, n_items=10, head_frac=0.2)
    assert head == {0, 1}


def test_popularity_bias_perfect_correlation():
    pop = np.array([1.0, 2.0, 3.0, 4.0])
    freq = np.array([10.0, 20.0, 30.0, 40.0])
    assert popularity_bias_score(freq, pop) == pytest.approx(1.0)


def test_popularity_bias_constant_input():
    assert popularity_bias_score(np.ones(5), np.arange(5).astype(float)) == 0.0


# ---------------------------------------------------------------- a/b test
def test_welch_t_test_significant():
    a = [0.9] * 30 + [0.8] * 30
    b = [0.1] * 30 + [0.2] * 30
    t, p = welch_t_test(a, b)
    assert p < 0.01
    assert t > 0


def test_welch_t_test_identical_groups():
    rng = np.random.default_rng(0)
    samples = rng.normal(0.5, 0.1, 100).tolist()
    _, p = welch_t_test(samples, samples)
    assert p > 0.9


def test_welch_t_test_tiny_samples():
    assert welch_t_test([1.0], [2.0]) == (0.0, 1.0)


def test_assign_variant_deterministic_and_balanced():
    assignments = [assign_variant(1, u) for u in range(2000)]
    assert assignments == [assign_variant(1, u) for u in range(2000)]
    share_a = assignments.count("A") / len(assignments)
    assert 0.45 < share_a < 0.55


def test_ab_framework_end_to_end(tmp_path):
    ab = ABTestFramework(f"sqlite:///{tmp_path / 'ab.db'}")
    test_id = ab.start_test("pop vs gcn", "popularity", "lightgcn")

    rng = np.random.default_rng(1)
    rows = []
    for u in range(100):
        variant = assign_variant(test_id, u)
        mean = 0.3 if variant == "A" else 0.6
        rows.append(
            {
                "variant": variant,
                "user_idx": u,
                "metric_value": float(rng.normal(mean, 0.05)),
                "latency_ms": float(rng.uniform(1, 5)),
            }
        )
    ab.log_observations_bulk(test_id, rows)

    res = ab.results(test_id)
    assert res["variant_b"]["mean"] > res["variant_a"]["mean"]
    assert res["significant"] is True
    assert res["winner"] == "lightgcn"
    assert ab.observation_count(test_id) == 100


def test_ab_framework_starting_new_test_stops_previous(tmp_path):
    ab = ABTestFramework(f"sqlite:///{tmp_path / 'ab2.db'}")
    first = ab.start_test("t1", "popularity", "recency")
    second = ab.start_test("t2", "lightgcn", "xsimgcl")
    active = ab.active_test()
    assert active["id"] == second
    assert ab.results(first)["status"] == "stopped"
