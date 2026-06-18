"""Unit tests for the univariate detector's classification logic.

The SQL path (rollup() over timescaledb_toolkit stats_agg) needs the
toolkit extension which the plain `timescale/timescaledb:latest-pg17`
test image doesn't ship — that path is covered by the cluster smoke
tests after deploy. Here we only exercise the pure-Python helpers.
"""

from __future__ import annotations

from iot_insights_engine import detect_univariate, registry
from iot_insights_engine.registry import UnivariateMetric


def _metric(
    *,
    min_stddev_rel: float = 0.0,
    min_stddev_abs: float = 0.0,
    deadband_abs: float = 0.0,
    deadband_rel: float = 0.0,
) -> UnivariateMetric:
    return UnivariateMetric(
        uc="t",
        source_cagg="x_1h",
        baseline_cagg="x_baseline_30d",
        metric="v",
        stats_field="v_stats",
        min_stddev_rel=min_stddev_rel,
        min_stddev_abs=min_stddev_abs,
        deadband_abs=deadband_abs,
        deadband_rel=deadband_rel,
    )


def test_entity_for_grouped_metric_uses_slug() -> None:
    metric = UnivariateMetric(
        uc="pv_production",
        source_cagg="solaredge_powerflow_1h",
        baseline_cagg="solaredge_powerflow_baseline_30d",
        metric="pv_production_avg",
        stats_field="pv_production_stats",
        group_cols=("inverter_id",),
    )
    assert detect_univariate._entity_for(metric, (1,)) == "inv1"
    assert detect_univariate._entity_for(metric, (2,)) == "inv2"


def test_entity_for_emit_entity_false_is_none() -> None:
    # House-level metric that is grouped only to keep the baseline join
    # correct — it must route to a single GA, so entity is None despite the
    # group_cols/group_values being present.
    metric = UnivariateMetric(
        uc="grid_power",
        source_cagg="solaredge_powerflow_1h",
        baseline_cagg="solaredge_powerflow_baseline_30d",
        metric="grid_power_avg",
        stats_field="grid_power_stats",
        group_cols=("inverter_id",),
        source_filter="inverter_id = 1",
        emit_entity=False,
    )
    assert detect_univariate._entity_for(metric, (1,)) is None


def test_registry_house_level_pv_metrics_opt_out_of_entity() -> None:
    # Lock the intent: the two duplicated-under-both-inverters metrics route
    # to one house-level KNX-GA each (anomaly.<uc>, no entity suffix).
    by_uc = {m.uc: m for m in registry.UNIVARIATE_METRICS}
    assert by_uc["grid_power"].emit_entity is False
    assert by_uc["consumer_total"].emit_entity is False
    assert by_uc["pv_production"].emit_entity is True


def test_classify_thresholds() -> None:
    assert detect_univariate._classify(0.0) is None
    assert detect_univariate._classify(2.9) is None
    assert detect_univariate._classify(3.0) == "info"
    assert detect_univariate._classify(-3.5) == "info"
    assert detect_univariate._classify(4.0) == "warning"
    assert detect_univariate._classify(-5.9) == "warning"
    assert detect_univariate._classify(6.0) == "critical"
    assert detect_univariate._classify(-10.0) == "critical"


def test_classify_severity_floor_suppresses_below() -> None:
    assert detect_univariate._classify(3.0, "warning") is None
    assert detect_univariate._classify(4.0, "warning") == "warning"
    assert detect_univariate._classify(4.0, "critical") is None
    assert detect_univariate._classify(6.0, "critical") == "critical"


def test_zscore_plain_default() -> None:
    """With default knobs (0.0) the score is the textbook z-score."""
    m = _metric()
    assert detect_univariate._zscore(10.0, 4.0, 2.0, m) == 3.0
    # Degenerate variance with no floor → skipped (no division by zero).
    assert detect_univariate._zscore(0.14, 0.10, 0.0, m) is None


def test_zscore_stddev_floor_tames_near_constant() -> None:
    """The lux 1e15 case: tiny stddev + a relative floor → sane, sub-threshold."""
    m = _metric(min_stddev_rel=0.15, deadband_rel=0.25)
    z = detect_univariate._zscore(0.14, 0.10, 1e-17, m)
    # eff_std = max(1e-17, 0.15*0.10) = 0.015 → z = 0.04/0.015 ≈ 2.67, not 4e15.
    assert z is not None
    assert abs(z) < detect_univariate.SEVERITY_INFO_THRESHOLD
    assert detect_univariate._classify(z) is None


def test_zscore_relative_deadband_drops_trivial_deviation() -> None:
    m = _metric(min_stddev_rel=0.15, deadband_rel=0.25)
    # 0.11 vs 0.10 → |dev| 0.01 < 0.25*0.10 = 0.025 → dropped outright.
    assert detect_univariate._zscore(0.11, 0.10, 1e-17, m) is None


def test_zscore_absolute_floor_and_deadband_appliance_standby() -> None:
    m = _metric(min_stddev_rel=0.10, min_stddev_abs=5.0, deadband_abs=20.0)
    # Standby creep 43 → 300 fires hard (phantom load).
    z = detect_univariate._zscore(300.0, 43.0, 0.5, m)
    assert z is not None and detect_univariate._classify(z) == "critical"
    # A 43 → 50 wobble is below the 20-unit deadband → dropped.
    assert detect_univariate._zscore(50.0, 43.0, 0.5, m) is None


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
