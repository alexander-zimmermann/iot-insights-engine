"""Unit tests for IsolationForest helpers — registry shape,
classification, warmup demote, training threshold computation.

The full train/score path needs a TimescaleDB instance with toolkit
and the source-CAGGs populated; that path is covered by the cluster
smoke tests after deploy.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np

from iot_insights_engine import iforest_common, registry, score_iforest, train_iforest
from iot_insights_engine.iforest_common import ModelEnvelope


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
