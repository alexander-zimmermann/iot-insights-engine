"""Unit tests for the PV-underperformance detector — severity mapping + the
fire/clear decision. The SQL (actual vs forecast) is covered by the cluster smoke."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from iot_insights_engine import detect_pv_underperformance as d
from iot_insights_engine.config import Settings


def _settings() -> Settings:
    return Settings(
        db_host="localhost",
        db_name="x",
        db_username="x",
        db_password="x",  # noqa: S106 — test stub
    )


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (1.0, None),
        (0.65, None),
        (0.64, "info"),
        (0.50, "info"),
        (0.49, "warning"),
        (0.34, "critical"),
    ],
)
def test_severity_thresholds(ratio: float, expected: str | None) -> None:
    assert d._severity(ratio) == expected


def _run_with(actual: float, expected: float) -> MagicMock:
    s = _settings()
    with (
        patch.object(d, "read_connection"),
        patch.object(d, "_actual_and_expected", return_value=(actual, expected)),
        patch.object(d.nats_publisher, "publish_anomaly") as pub,
    ):
        d.run(s, [])
    return pub


def test_run_clears_when_expected_below_floor() -> None:
    # Dawn / missing forecast: expected < MIN_EXPECTED_KWH → clear, no judgement.
    pub = _run_with(actual=0.5, expected=1.0)
    _, kwargs = pub.call_args
    assert kwargs["firing"] is False
    assert kwargs["payload"]["ratio"] is None


def test_run_clears_when_yield_healthy() -> None:
    # 40/45 = 0.89 → healthy → firing False.
    pub = _run_with(actual=40.0, expected=45.0)
    _, kwargs = pub.call_args
    assert kwargs["firing"] is False


def test_run_fires_on_shortfall() -> None:
    # 10/40 = 0.25 → critical.
    pub = _run_with(actual=10.0, expected=40.0)
    _, kwargs = pub.call_args
    assert kwargs["firing"] is True
    assert kwargs["severity"] == "critical"
    assert kwargs["payload"]["ratio"] == 0.25
