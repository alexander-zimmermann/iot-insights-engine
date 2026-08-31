"""Unit tests for the shared severity helpers."""

from __future__ import annotations

from iot_insights_engine.severity import severity_level


def test_severity_level() -> None:
    # 0 is the clear value written to the KNX GA when nothing fires.
    assert severity_level(None) == 0
    assert severity_level("info") == 1
    assert severity_level("warning") == 2
    assert severity_level("critical") == 3
