"""Unit tests for the univariate detector's classification logic.

The SQL path (rollup() over timescaledb_toolkit stats_agg) needs the
toolkit extension which the plain `timescale/timescaledb:latest-pg17`
test image doesn't ship — that path is covered by the cluster smoke
tests after deploy. Here we only exercise the pure-Python helpers.
"""

from __future__ import annotations

from iot_insights_engine import detect_univariate, registry


def test_classify_thresholds() -> None:
    assert detect_univariate._classify(0.0) is None
    assert detect_univariate._classify(2.9) is None
    assert detect_univariate._classify(3.0) == "info"
    assert detect_univariate._classify(-3.5) == "info"
    assert detect_univariate._classify(4.0) == "warning"
    assert detect_univariate._classify(-5.9) == "warning"
    assert detect_univariate._classify(6.0) == "critical"
    assert detect_univariate._classify(-10.0) == "critical"


def test_registry_slugs_unique() -> None:
    """A duplicate slug would break NATS routing (same subject, ambiguous
    KNX-GA mapping). Catch the typo at import time."""
    slugs = [m.uc for m in registry.UNIVARIATE_METRICS]
    assert len(slugs) == len(set(slugs)), f"duplicate slugs: {slugs}"


def test_registry_metrics_match_known_caggs() -> None:
    """Every UnivariateMetric points at a baseline-CAGG named
    `<source>_baseline_30d`. Defensive — easy to typo, hard to catch."""
    for m in registry.UNIVARIATE_METRICS:
        assert m.baseline_cagg.endswith("_baseline_30d"), m.baseline_cagg
        expected_prefix = m.source_cagg.removesuffix("_1h")
        assert m.baseline_cagg == f"{expected_prefix}_baseline_30d", (
            f"{m.uc}: baseline {m.baseline_cagg} does not match source {m.source_cagg}"
        )
