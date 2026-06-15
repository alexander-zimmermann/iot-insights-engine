"""Unit tests for the KNX-join detector helpers.

SQL JOINs need a populated knx_1h + ga_catalog and live in the cluster
smoke tests after deploy. Here we exercise the pure-Python classification
+ registry invariants only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from iot_insights_engine import detect_knx_join, registry


def _hours(*active: bool) -> list[tuple[datetime, bool]]:
    """Build newest→oldest hourly (bucket, is_active) rows from `active[0]`
    as the most recent hour."""
    base = datetime(2026, 6, 8, 20, tzinfo=UTC)
    return [(base - timedelta(hours=i), a) for i, a in enumerate(active)]


def test_classify_fbh_thresholds() -> None:
    assert detect_knx_join._classify_fbh(0.5) is None
    assert detect_knx_join._classify_fbh(1.0) == "info"
    assert detect_knx_join._classify_fbh(1.9) == "info"
    assert detect_knx_join._classify_fbh(2.0) == "warning"
    assert detect_knx_join._classify_fbh(2.9) == "warning"
    assert detect_knx_join._classify_fbh(3.0) == "critical"
    assert detect_knx_join._classify_fbh(10.0) == "critical"


def test_classify_window_thresholds() -> None:
    assert detect_knx_join._classify_window(0.0) is None
    assert detect_knx_join._classify_window(0.1) == "info"
    assert detect_knx_join._classify_window(24.9) == "info"
    assert detect_knx_join._classify_window(25.0) == "warning"
    assert detect_knx_join._classify_window(49.9) == "warning"
    assert detect_knx_join._classify_window(50.0) == "critical"
    assert detect_knx_join._classify_window(100.0) == "critical"


def test_registry_slugs_unique_across_families() -> None:
    """KNX-join slugs share the NATS subject namespace with the
    univariate + iforest UCs — a duplicate would create ambiguous
    knx-nats-bridge writer-rule routing."""
    knx_slugs = [u.uc for u in registry.KNX_JOIN_USECASES]
    assert len(knx_slugs) == len(set(knx_slugs))
    other_slugs = (
        {m.uc for m in registry.UNIVARIATE_METRICS}
        | {u.uc for u in registry.IFOREST_USECASES}
        | {s.uc for s in registry.SEASONAL_MODELS}
    )
    overlap = set(knx_slugs) & other_slugs
    assert not overlap, f"KNX-join slugs collide with other families: {overlap}"


def test_every_registered_uc_has_implementation() -> None:
    """A registry entry without a matching `_DETECTORS` key is dead — the
    dispatcher logs `uc_not_implemented` and silently skips it."""
    for uc in registry.KNX_JOIN_USECASES:
        assert uc.uc in detect_knx_join._DETECTORS, (
            f"{uc.uc} listed in registry but missing from _DETECTORS"
        )


def test_classify_fbh_constants_monotonic() -> None:
    """Severity thresholds must be strictly increasing — otherwise
    classify silently degrades back to a lower tier."""
    assert (
        detect_knx_join.FBH_GAP_INFO_C
        < detect_knx_join.FBH_GAP_WARNING_C
        < detect_knx_join.FBH_GAP_CRITICAL_C
    )


def test_classify_window_constants_monotonic() -> None:
    assert (
        detect_knx_join.WIN_STELLWERT_INFO_PCT
        < detect_knx_join.WIN_STELLWERT_WARNING_PCT
        < detect_knx_join.WIN_STELLWERT_CRITICAL_PCT
    )


def test_classify_runtime_thresholds() -> None:
    assert detect_knx_join._classify_runtime(2) is None
    assert detect_knx_join._classify_runtime(3) == "info"
    assert detect_knx_join._classify_runtime(5) == "info"
    assert detect_knx_join._classify_runtime(6) == "warning"
    assert detect_knx_join._classify_runtime(8) == "warning"
    assert detect_knx_join._classify_runtime(9) == "critical"


def test_classify_runtime_constants_monotonic() -> None:
    assert (
        detect_knx_join.APPLIANCE_RUNTIME_INFO_HOURS
        < detect_knx_join.APPLIANCE_RUNTIME_WARNING_HOURS
        < detect_knx_join.APPLIANCE_RUNTIME_CRITICAL_HOURS
    )


def test_classify_icing_thresholds() -> None:
    assert detect_knx_join._classify_icing(37.9) is None
    assert detect_knx_join._classify_icing(38.0) == "info"
    assert detect_knx_join._classify_icing(44.9) == "info"
    assert detect_knx_join._classify_icing(45.0) == "warning"
    assert detect_knx_join._classify_icing(54.9) == "warning"
    assert detect_knx_join._classify_icing(55.0) == "critical"
    assert detect_knx_join._classify_icing(120.0) == "critical"


def test_classify_icing_constants_monotonic() -> None:
    # Baseline must sit below the first alert tier, tiers strictly increasing.
    assert (
        detect_knx_join.FREEZER_ICING_BASELINE_MIN
        < detect_knx_join.FREEZER_ICING_INFO_MIN
        < detect_knx_join.FREEZER_ICING_WARNING_MIN
        < detect_knx_join.FREEZER_ICING_CRITICAL_MIN
    )
    # The door-event floor must exceed the heaviest icing tier, else genuine
    # icing runs would be discarded as door-ajar outliers before the median.
    assert detect_knx_join.FREEZER_ICING_CRITICAL_MIN < detect_knx_join.FREEZER_ICING_DOOR_EVENT_MIN


def test_hour_is_active() -> None:
    # >=50% of samples above the standby valley → active.
    assert detect_knx_join._hour_is_active(30, 60) is True
    assert detect_knx_join._hour_is_active(29, 60) is False
    assert detect_knx_join._hour_is_active(0, 60) is False
    # No samples / null → not active (never bridges a streak).
    assert detect_knx_join._hour_is_active(0, 0) is False
    assert detect_knx_join._hour_is_active(None, 0) is False


def test_trailing_active_streak_counts_from_newest() -> None:
    assert detect_knx_join._trailing_active_streak(_hours(True, True, True)) == 3
    # Stops at the first inactive hour, even with active hours behind it.
    assert detect_knx_join._trailing_active_streak(_hours(True, False, True)) == 1
    # Not currently active → no streak.
    assert detect_knx_join._trailing_active_streak(_hours(False, True, True)) == 0
    assert detect_knx_join._trailing_active_streak([]) == 0


def test_trailing_active_streak_breaks_on_gap() -> None:
    base = datetime(2026, 6, 8, 20, tzinfo=UTC)
    # 20:00 active, then a 2h gap to 18:00 active → streak is just the newest.
    rows = [(base, True), (base - timedelta(hours=2), True)]
    assert detect_knx_join._trailing_active_streak(rows) == 1
