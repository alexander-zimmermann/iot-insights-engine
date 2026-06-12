"""Unit tests for IsolationForest helpers — registry shape,
classification, warmup demote, training threshold computation.

The full train/score path needs a TimescaleDB instance with toolkit
and the source-CAGGs populated; that path is covered by the cluster
smoke tests after deploy.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from iot_insights_engine import iforest_common, registry, score_iforest, train_iforest
from iot_insights_engine.config import Settings
from iot_insights_engine.iforest_common import ModelEnvelope
from iot_insights_engine.registry import IsolationForestUseCase


def _settings() -> Settings:
    return Settings(
        db_host="localhost",
        db_name="x",
        db_username="x",
        db_password="x",  # noqa: S106 — test stub
        db_write_username="x",
        db_write_password="x",  # noqa: S106 — test stub
    )


def test_iforest_registry_slugs_unique_and_disjoint_from_univariate() -> None:
    """Slugs across UC families must be globally unique — they share
    the NATS subject namespace and the knx-nats-bridge writer-rules."""
    if_slugs = [u.uc for u in registry.IFOREST_USECASES]
    uni_slugs = [m.uc for m in registry.UNIVARIATE_METRICS]
    assert len(if_slugs) == len(set(if_slugs)), f"duplicate IF slugs: {if_slugs}"
    overlap = set(if_slugs) & set(uni_slugs)
    assert not overlap, f"slugs collide across families: {overlap}"


def test_iforest_registry_features_have_expected_shape() -> None:
    for uc in registry.IFOREST_USECASES:
        assert uc.source_cagg.endswith("_1h"), uc.source_cagg
        assert uc.feature_cols, f"{uc.uc} has no features"
        assert all("_avg" in c or "_samples" in c for c in uc.feature_cols), uc.feature_cols
        assert uc.contamination > 0, uc.uc


def test_feature_names_includes_hour_of_day_when_enabled() -> None:
    pv = next(u for u in registry.IFOREST_USECASES if u.uc == "pv_iforest")
    assert "hour_of_day" in iforest_common.feature_names(pv)
    heating = next(u for u in registry.IFOREST_USECASES if u.uc == "heating_iforest")
    assert "hour_of_day" not in iforest_common.feature_names(heating)


def test_group_key_no_groups() -> None:
    uc = registry.IFOREST_USECASES[0]
    assert iforest_common.group_key(uc, ()) == uc.uc


def test_group_key_with_groups() -> None:
    pv = next(u for u in registry.IFOREST_USECASES if u.uc == "pv_iforest")
    assert iforest_common.group_key(pv, (0,)) == "pv_iforest/0"


def _envelope(
    threshold_warning: float, threshold_critical: float, trained_at: datetime
) -> ModelEnvelope:
    return ModelEnvelope(
        pipeline=MagicMock(),
        feature_names=("a", "b"),
        threshold_warning=threshold_warning,
        threshold_critical=threshold_critical,
        n_train_samples=1000,
        trained_at=trained_at.isoformat(),
    )


def test_classify_thresholds() -> None:
    env = _envelope(
        threshold_warning=-0.5, threshold_critical=-0.8, trained_at=datetime.now(tz=UTC)
    )
    assert score_iforest._classify(env, 0.0) is None
    assert score_iforest._classify(env, -0.4) is None
    assert score_iforest._classify(env, -0.5) is None
    assert score_iforest._classify(env, -0.51) == "warning"
    assert score_iforest._classify(env, -0.79) == "warning"
    assert score_iforest._classify(env, -0.81) == "critical"


def test_classify_severity_floor_suppresses_below() -> None:
    env = _envelope(
        threshold_warning=-0.5, threshold_critical=-0.8, trained_at=datetime.now(tz=UTC)
    )
    assert score_iforest._classify(env, -0.51, "critical") is None
    assert score_iforest._classify(env, -0.81, "critical") == "critical"
    assert score_iforest._classify(env, -0.51, "warning") == "warning"


def test_warmup_demote_within_window() -> None:
    fresh = datetime.now(tz=UTC) - timedelta(days=3)
    env = _envelope(-0.5, -0.8, trained_at=fresh)
    assert score_iforest._warmup_demote(env, "critical", warmup_days=7) == "info"
    assert score_iforest._warmup_demote(env, "warning", warmup_days=7) == "info"


def test_warmup_demote_after_window() -> None:
    old = datetime.now(tz=UTC) - timedelta(days=10)
    env = _envelope(-0.5, -0.8, trained_at=old)
    assert score_iforest._warmup_demote(env, "critical", warmup_days=7) == "critical"
    assert score_iforest._warmup_demote(env, "warning", warmup_days=7) == "warning"


def test_train_threshold_quantiles_match_constants() -> None:
    """Sanity-check the quantile constants: critical must be a tighter
    cutoff than warning (more negative = more anomalous in score_samples)."""
    assert train_iforest.CRITICAL_QUANTILE < train_iforest.WARNING_QUANTILE
    rng = np.random.default_rng(seed=0)
    scores = rng.normal(loc=0.0, scale=0.1, size=10_000)
    warn_q = float(np.quantile(scores, train_iforest.WARNING_QUANTILE))
    crit_q = float(np.quantile(scores, train_iforest.CRITICAL_QUANTILE))
    assert crit_q < warn_q


def test_run_isolates_per_uc_score_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient S3/rustfs hiccup scoring one UC must not abort the sweep —
    the remaining UCs are still scored and the job exits 0."""

    @contextmanager
    def fake_conn(_settings: Settings) -> Iterator[MagicMock]:
        yield MagicMock()

    monkeypatch.setattr(score_iforest, "write_connection", fake_conn)
    monkeypatch.setattr(score_iforest, "_load_last_bucket", lambda *_: [{"bucket": 1}])
    calls: list[str] = []

    def fake_score(
        settings: Settings, conn: object, uc: IsolationForestUseCase, rows: object
    ) -> tuple[int, int]:
        calls.append(uc.uc)
        if len(calls) == 1:
            raise RuntimeError("rustfs read timeout")
        return 0, 0

    monkeypatch.setattr(score_iforest, "_score_group", fake_score)

    rc = score_iforest.run(_settings(), [])
    assert rc == 0
    # All UCs attempted despite the first one raising.
    assert len(calls) == len(registry.IFOREST_USECASES)
