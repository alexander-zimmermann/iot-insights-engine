"""Unit tests for the seasonal (statsforecast) train + score helpers.

Full StatsForecast.fit() is exercised by the cluster smoke test after
deploy — it needs a real Postgres + 2+ weeks of hourly data and is too
expensive for CI. Here we cover the pure-Python classification,
warmup, registry invariants, and the SQL-string shape.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from iot_insights_engine import registry, score_seasonal, train_seasonal
from iot_insights_engine.seasonal_common import ModelEnvelope


def test_registry_seasonal_slugs_unique_across_families() -> None:
    """Like other UC families, slugs share the NATS subject namespace —
    a duplicate would create ambiguous knx-nats-bridge writer-rule
    routing."""
    seasonal_slugs = [s.uc for s in registry.SEASONAL_MODELS]
    assert len(seasonal_slugs) == len(set(seasonal_slugs))
    other_slugs = (
        {m.uc for m in registry.UNIVARIATE_METRICS}
        | {u.uc for u in registry.IFOREST_USECASES}
        | {k.uc for k in registry.KNX_JOIN_USECASES}
    )
    overlap = set(seasonal_slugs) & other_slugs
    assert not overlap, f"seasonal slugs collide with other families: {overlap}"


def test_min_train_samples_bigger_than_max_season() -> None:
    """MSTL requires more than 2 full cycles of the longest season —
    otherwise the decomposition is undefined."""
    longest_season = max(
        max(s.season_length) for s in registry.SEASONAL_MODELS
    )
    assert 2 * longest_season <= train_seasonal.MIN_TRAIN_SAMPLES


def _envelope(
    residual_stddev: float = 1.0,
    sigma_threshold: float = 3.0,
    trained_at: datetime | None = None,
) -> ModelEnvelope:
    return ModelEnvelope(
        sf=MagicMock(),
        season_length=(24, 168),
        forecast_horizon_hours=24,
        sigma_threshold=sigma_threshold,
        residual_stddev=residual_stddev,
        n_train_samples=2000,
        trained_at=(trained_at or datetime.now(tz=UTC)).isoformat(),
    )


def test_classify_z_score_tiers() -> None:
    """severity_floor='info' lets all tiers through."""
    assert score_seasonal._classify(0.0, "info") is None
    assert score_seasonal._classify(0.9, "info") is None
    assert score_seasonal._classify(1.0, "info") == "info"
    assert score_seasonal._classify(-1.0, "info") == "info"
    assert score_seasonal._classify(1.5, "info") == "warning"
    assert score_seasonal._classify(-2.4, "info") == "warning"
    assert score_seasonal._classify(2.5, "info") == "critical"
    assert score_seasonal._classify(-5.0, "info") == "critical"


def test_classify_severity_floor_suppresses_below() -> None:
    """severity_floor='warning' drops info-level findings, keeps
    warning + critical."""
    assert score_seasonal._classify(3.0, "warning") == "critical"
    assert score_seasonal._classify(1.6, "warning") == "warning"
    assert score_seasonal._classify(1.0, "warning") is None
    assert score_seasonal._classify(2.5, "critical") == "critical"
    assert score_seasonal._classify(1.5, "critical") is None


def test_warmup_demote_inside_window() -> None:
    fresh = datetime.now(tz=UTC) - timedelta(days=3)
    env = _envelope(trained_at=fresh)
    assert score_seasonal._warmup_demote(env, "critical", warmup_days=14) == "info"
    assert score_seasonal._warmup_demote(env, "warning", warmup_days=14) == "info"


def test_warmup_demote_after_window() -> None:
    old = datetime.now(tz=UTC) - timedelta(days=21)
    env = _envelope(trained_at=old)
    assert score_seasonal._warmup_demote(env, "critical", warmup_days=14) == "critical"
    assert score_seasonal._warmup_demote(env, "warning", warmup_days=14) == "warning"


def test_envelope_round_trip() -> None:
    """Trained_at is stored as ISO string — make sure parsing it back
    gives the original UTC instant."""
    now = datetime.now(tz=UTC).replace(microsecond=0)
    env = _envelope(trained_at=now)
    parsed = datetime.fromisoformat(env.trained_at)
    assert parsed == now
