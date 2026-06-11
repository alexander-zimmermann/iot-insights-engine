"""Unit tests for the shared severity helpers."""

from __future__ import annotations

from iot_insights_engine.severity import escalated, meets_floor


def test_meets_floor() -> None:
    assert meets_floor("info", "info") is True
    assert meets_floor("warning", "info") is True
    assert meets_floor("info", "warning") is False
    assert meets_floor("warning", "critical") is False
    assert meets_floor("critical", "critical") is True


def test_escalated_on_fresh_insert_is_false() -> None:
    """old_severity is NULL on a fresh insert — the insert itself already
    triggers the publish, escalation must not double-fire."""
    assert escalated(None, "critical") is False


def test_escalated_upgrade_fires() -> None:
    assert escalated("info", "warning") is True
    assert escalated("info", "critical") is True
    assert escalated("warning", "critical") is True


def test_escalated_unchanged_or_downgrade_is_false() -> None:
    assert escalated("info", "info") is False
    assert escalated("critical", "critical") is False
    assert escalated("critical", "warning") is False
    assert escalated("warning", "info") is False
