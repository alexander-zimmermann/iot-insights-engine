"""Unit tests for weekly_report. SQL paths exercised by the cluster
smoke after deploy; here we cover the Markdown renderer (pure function)
and SMTP sender wiring."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from iot_insights_engine import weekly_report
from iot_insights_engine.config import Settings


def _settings() -> Settings:
    return Settings(
        db_host="localhost",
        db_name="x",
        db_username="x",
        db_password="x",  # noqa: S106 — test stub
        smtp_host="mail.local",
        smtp_port=25,
        smtp_from="bot@example",
        smtp_to="admin@example",
    )


def test_render_markdown_includes_all_sections() -> None:
    until = datetime(2026, 6, 9, 8, 0, tzinfo=UTC)
    since = datetime(2026, 6, 2, 8, 0, tzinfo=UTC)
    body = weekly_report._render_markdown(
        since=since,
        until=until,
        severity_now={"critical": 2, "warning": 7, "info": 14},
        severity_prev={"critical": 1, "warning": 5, "info": 20},
        uc_counts=[("heating_activity_seasonal", 8), ("pv_iforest", 4)],
        top=[
            {
                "time": datetime(2026, 6, 5, 12, 0, tzinfo=UTC),
                "severity": "critical",
                "uc": "heating_activity_seasonal",
                "metric": "heatingactive_samples",
                "source": "ems_esp_boiler_1h",
                "score": 4.21,
                "actual": 47.0,
                "expected": 9.0,
            },
        ],
    )
    assert "# Weekly Anomaly Digest — 2026-06-09" in body
    assert "Window: 2026-06-02 → 2026-06-09" in body
    assert "**Total this week: 23**" in body
    # 23 now vs 26 prev → down 3
    assert "↓" in body and "3" in body
    assert "## By severity" in body
    assert "| critical | 2 |" in body
    assert "## By use-case" in body
    assert "| heating_activity_seasonal | 8 |" in body
    assert "## Top 10 anomalies" in body
    assert "2026-06-05 12:00" in body
    assert "| critical | heating_activity_seasonal | heatingactive_samples | 4.21 |" in body


def test_render_markdown_handles_nulls() -> None:
    """score / actual / expected can be NULL for rule-based detectors
    that don't compute a numeric score."""
    body = weekly_report._render_markdown(
        since=datetime(2026, 6, 2, tzinfo=UTC),
        until=datetime(2026, 6, 9, tzinfo=UTC),
        severity_now={"info": 1},
        severity_prev={},
        uc_counts=[("fbh_cold", 1)],
        top=[
            {
                "time": datetime(2026, 6, 5, 12, 0, tzinfo=UTC),
                "severity": "info",
                "uc": "fbh_cold",
                "metric": "fbh_cold[Wohnzimmer]",
                "source": "knx_1h+ga_catalog",
                "score": None,
                "actual": None,
                "expected": None,
            },
        ],
    )
    # Three empty cells between metric and the closing pipe (score, actual, expected all NULL)
    assert "| fbh_cold | fbh_cold[Wohnzimmer] |  |  |  |" in body


def test_render_markdown_zero_zero_no_arrow() -> None:
    body = weekly_report._render_markdown(
        since=datetime(2026, 6, 2, tzinfo=UTC),
        until=datetime(2026, 6, 9, tzinfo=UTC),
        severity_now={},
        severity_prev={},
        uc_counts=[],
        top=[],
    )
    assert "**Total this week: 0**" in body
    assert "→" in body  # no change arrow when delta=0


def test_send_constructs_message_correctly() -> None:
    s = _settings()
    with patch("iot_insights_engine.weekly_report.smtplib.SMTP") as smtp_cls:
        instance = MagicMock()
        smtp_cls.return_value.__enter__.return_value = instance
        weekly_report._send(s, "Subject Line", "body text")
        smtp_cls.assert_called_once_with("mail.local", 25, timeout=30)
        instance.send_message.assert_called_once()
        msg = instance.send_message.call_args[0][0]
        assert msg["Subject"] == "Subject Line"
        assert msg["From"] == "bot@example"
        assert msg["To"] == "admin@example"
        assert msg.get_content().strip() == "body text"


def test_run_returns_1_when_smtp_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings()
    monkeypatch.setattr(weekly_report, "write_connection", MagicMock())
    monkeypatch.setattr(weekly_report, "_counts_by_severity", lambda *_: {})
    monkeypatch.setattr(weekly_report, "_counts_by_uc", lambda *_: [])
    monkeypatch.setattr(weekly_report, "_top_anomalies", lambda *_: [])
    monkeypatch.setattr(
        weekly_report,
        "_send",
        MagicMock(side_effect=OSError("smtprelay unreachable")),
    )
    rc = weekly_report.run(s, [])
    assert rc == 1
