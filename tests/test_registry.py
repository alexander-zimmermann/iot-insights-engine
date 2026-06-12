"""Registry-wide invariants.

All identifier-typed registry fields (CAGG names, column names) are
interpolated into SQL verbatim — safe only as long as they are plain
identifiers defined in code. This test pins that invariant so a future
"load UCs from the DB/config" refactor can't silently introduce SQL
injection.
"""

from __future__ import annotations

import re

from iot_insights_engine import registry
from iot_insights_engine.severity import SEVERITY_ORDER

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def _assert_identifier(value: str, context: str) -> None:
    assert _IDENTIFIER.fullmatch(value), f"{context}: {value!r} is not a plain SQL identifier"


def test_univariate_fields_are_safe_identifiers() -> None:
    for m in registry.UNIVARIATE_METRICS:
        _assert_identifier(m.source_cagg, m.uc)
        _assert_identifier(m.baseline_cagg, m.uc)
        _assert_identifier(m.metric, m.uc)
        _assert_identifier(m.stats_field, m.uc)
        for c in m.group_cols:
            _assert_identifier(c, m.uc)


def test_iforest_fields_are_safe_identifiers() -> None:
    for u in registry.IFOREST_USECASES:
        _assert_identifier(u.source_cagg, u.uc)
        for c in (*u.feature_cols, *u.group_cols):
            _assert_identifier(c, u.uc)


def test_seasonal_fields_are_safe_identifiers() -> None:
    for s in registry.SEASONAL_MODELS:
        _assert_identifier(s.source_cagg, s.uc)
        _assert_identifier(s.metric, s.uc)


def test_severity_floors_are_valid() -> None:
    floors = (
        [m.severity_floor for m in registry.UNIVARIATE_METRICS]
        + [u.severity_floor for u in registry.IFOREST_USECASES]
        + [s.severity_floor for s in registry.SEASONAL_MODELS]
    )
    for floor in floors:
        assert floor in SEVERITY_ORDER, floor
