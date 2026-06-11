"""A/B testing framework backed by PostgreSQL (SQLite fallback for dev).

Workflow:
  1. `start_test` registers a test between two model variants.
  2. Users are deterministically hashed into variant A or B.
  3. Each served recommendation logs an observation (per-user quality +
     latency) via `log_observation`.
  4. `results` aggregates both arms and runs Welch's t-test to decide
     whether the difference in the primary metric is significant.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from scipy import stats
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class ABTest(Base):
    __tablename__ = "ab_tests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    variant_a = Column(String(64), nullable=False)
    variant_b = Column(String(64), nullable=False)
    primary_metric = Column(String(64), nullable=False, default="ndcg@10")
    status = Column(String(16), nullable=False, default="running")
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class ABObservation(Base):
    __tablename__ = "ab_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    test_id = Column(Integer, ForeignKey("ab_tests.id"), nullable=False, index=True)
    variant = Column(String(8), nullable=False)  # "A" or "B"
    user_idx = Column(Integer, nullable=False)
    metric_value = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False, default=0.0)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


def assign_variant(test_id: int, user_idx: int) -> str:
    """Deterministic 50/50 assignment via hashing (stable across requests)."""
    digest = hashlib.sha256(f"{test_id}:{user_idx}".encode()).hexdigest()
    return "A" if int(digest, 16) % 2 == 0 else "B"


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Welch's two-sample t-test. Returns (t_statistic, p_value)."""
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_ind(a_arr, b_arr, equal_var=False)
    if not (np.isfinite(t_stat) and np.isfinite(p_val)):
        return 0.0, 1.0
    return float(t_stat), float(p_val)


class ABTestFramework:
    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)

    # ------------------------------------------------------------------
    def start_test(
        self,
        name: str,
        variant_a: str,
        variant_b: str,
        primary_metric: str = "ndcg@10",
    ) -> int:
        with Session(self.engine) as session:
            # Pause any currently running test: one active test at a time.
            for running in session.scalars(
                select(ABTest).where(ABTest.status == "running")
            ):
                running.status = "stopped"
            test = ABTest(
                name=name,
                variant_a=variant_a,
                variant_b=variant_b,
                primary_metric=primary_metric,
            )
            session.add(test)
            session.commit()
            return int(test.id)

    def active_test(self) -> dict | None:
        with Session(self.engine) as session:
            test = session.scalar(
                select(ABTest)
                .where(ABTest.status == "running")
                .order_by(ABTest.created_at.desc())
            )
            if test is None:
                return None
            return {
                "id": test.id,
                "name": test.name,
                "variant_a": test.variant_a,
                "variant_b": test.variant_b,
                "primary_metric": test.primary_metric,
                "status": test.status,
                "created_at": test.created_at.isoformat(),
            }

    def log_observation(
        self,
        test_id: int,
        variant: str,
        user_idx: int,
        metric_value: float,
        latency_ms: float = 0.0,
    ) -> None:
        with Session(self.engine) as session:
            session.add(
                ABObservation(
                    test_id=test_id,
                    variant=variant,
                    user_idx=user_idx,
                    metric_value=metric_value,
                    latency_ms=latency_ms,
                )
            )
            session.commit()

    def log_observations_bulk(
        self, test_id: int, rows: list[dict]
    ) -> None:
        with Session(self.engine) as session:
            session.add_all(
                ABObservation(test_id=test_id, **row) for row in rows
            )
            session.commit()

    # ------------------------------------------------------------------
    def results(self, test_id: int | None = None, alpha: float = 0.05) -> dict | None:
        """Aggregate stats per arm + Welch's t-test on the primary metric."""
        with Session(self.engine) as session:
            if test_id is None:
                info = self.active_test()
                if info is None:
                    return None
                test_id = int(info["id"])
            test = session.get(ABTest, test_id)
            if test is None:
                return None

            def arm(variant: str) -> dict:
                values = [
                    float(v)
                    for v in session.scalars(
                        select(ABObservation.metric_value).where(
                            ABObservation.test_id == test_id,
                            ABObservation.variant == variant,
                        )
                    )
                ]
                lats = [
                    float(v)
                    for v in session.scalars(
                        select(ABObservation.latency_ms).where(
                            ABObservation.test_id == test_id,
                            ABObservation.variant == variant,
                        )
                    )
                ]
                return {
                    "n": len(values),
                    "mean": float(np.mean(values)) if values else 0.0,
                    "std": float(np.std(values)) if values else 0.0,
                    "p95_latency_ms": float(np.percentile(lats, 95)) if lats else 0.0,
                    "values": values,
                }

            arm_a, arm_b = arm("A"), arm("B")
            t_stat, p_val = welch_t_test(arm_a.pop("values"), arm_b.pop("values"))

            return {
                "test_id": test_id,
                "name": test.name,
                "status": test.status,
                "primary_metric": test.primary_metric,
                "created_at": test.created_at.isoformat(),
                "variant_a": {"model": test.variant_a, **arm_a},
                "variant_b": {"model": test.variant_b, **arm_b},
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant": p_val < alpha,
                "winner": (
                    test.variant_a if arm_a["mean"] >= arm_b["mean"] else test.variant_b
                )
                if p_val < alpha
                else None,
            }

    def observation_count(self, test_id: int) -> int:
        with Session(self.engine) as session:
            return int(
                session.scalar(
                    select(func.count(ABObservation.id)).where(
                        ABObservation.test_id == test_id
                    )
                )
                or 0
            )
