"""Unit tests for the seasonal scorer's pure-Python helpers.

Full StatsForecast.fit() + DB integration runs in the cluster smoke
after deploy — too expensive for CI (real fit needs >=2 weeks of
hourly data; ~30s on 8k samples).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from iot_insights_engine import registry, score_seasonal


def test_registry_seasonal_slugs_unique_across_families() -> None:
    """Slugs share the NATS subject namespace with the univariate +
    iforest + knx-join UCs."""
    seasonal_slugs = [s.uc for s in registry.SEASONAL_MODELS]
    assert len(seasonal_slugs) == len(set(seasonal_slugs))
    other = (
        {m.uc for m in registry.UNIVARIATE_METRICS}
        | {u.uc for u in registry.IFOREST_USECASES}
        | {k.uc for k in registry.KNX_JOIN_USECASES}
    )
    overlap = set(seasonal_slugs) & other
    assert not overlap, f"seasonal slugs collide with other families: {overlap}"


def test_classify_z_score_tiers() -> None:
    assert score_seasonal._classify(0.0, "info") is None
    assert score_seasonal._classify(0.9, "info") is None
    assert score_seasonal._classify(1.0, "info") == "info"
    assert score_seasonal._classify(-1.0, "info") == "info"
    assert score_seasonal._classify(1.5, "info") == "warning"
    assert score_seasonal._classify(-2.4, "info") == "warning"
    assert score_seasonal._classify(2.5, "info") == "critical"
    assert score_seasonal._classify(-5.0, "info") == "critical"


def test_classify_severity_floor_suppresses_below() -> None:
    """severity_floor='warning' drops info-level findings."""
    assert score_seasonal._classify(3.0, "warning") == "critical"
    assert score_seasonal._classify(1.6, "warning") == "warning"
    assert score_seasonal._classify(1.0, "warning") is None
    assert score_seasonal._classify(2.5, "critical") == "critical"
    assert score_seasonal._classify(1.5, "critical") is None


def test_warmup_active_within_window() -> None:
    fresh = datetime.now(tz=UTC) - timedelta(days=3)
    assert score_seasonal._warmup_active(fresh, warmup_days=14) is True


def test_warmup_active_after_window() -> None:
    old = datetime.now(tz=UTC) - timedelta(days=21)
    assert score_seasonal._warmup_active(old, warmup_days=14) is False


def test_min_train_samples_bigger_than_max_season() -> None:
    """MSTL requires more than 2 full cycles of the longest season."""
    longest = max(max(s.season_length) for s in registry.SEASONAL_MODELS)
    assert 2 * longest <= score_seasonal.MIN_TRAIN_SAMPLES
