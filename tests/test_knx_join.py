"""Unit tests for the KNX-join detector helpers.

SQL JOINs need a populated knx_1h + ga_catalog and live in the cluster
smoke tests after deploy. Here we exercise the pure-Python classification
+ registry invariants only.
"""

from __future__ import annotations

from iot_insights_engine import detect_knx_join, registry


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
