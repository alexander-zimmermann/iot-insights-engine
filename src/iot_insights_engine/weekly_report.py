"""Weekly anomaly digest — Markdown e-mail summarising the last 7d of
`mcp_anomalies` activity. Sent via the cluster-internal smtprelay
(no auth, BGP-advertised LoadBalancer).

Scheduled Mondays 08:00. Focused on actionable signal:
* counts by severity for the week
* top-N most severe anomalies
* per-UC activity (which detectors fired, how often)
* trend vs the previous 7d (more or fewer anomalies)

Rich domain summaries (heating runtime, PV daily totals, etc.) stay
out — those are interactive Claude.ai queries via MCP tools, not a
fixed digest.
"""

from __future__ import annotations

import smtplib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from typing import Any

import psycopg

from .config import Settings
from .db_write import write_connection
from .logging_setup import get_logger

log = get_logger(__name__)

TOP_N = 10


def _counts_by_severity(
    conn: psycopg.Connection[Any], since: datetime, until: datetime
) -> dict[str, int]:
    sql = """
        SELECT severity, count(*) AS n
        FROM mcp_anomalies
        WHERE time >= %s AND time < %s
        GROUP BY severity
    """
    with conn.cursor() as cur:
        cur.execute(sql, (since, until))
        return {row["severity"]: int(row["n"]) for row in cur.fetchall()}


def _counts_by_uc(
    conn: psycopg.Connection[Any], since: datetime, until: datetime
) -> list[tuple[str, int]]:
    sql = """
        SELECT uc, count(*) AS n
        FROM mcp_anomalies
        WHERE time >= %s AND time < %s
        GROUP BY uc
        ORDER BY n DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (since, until))
        return [(row["uc"], int(row["n"])) for row in cur.fetchall()]


def _top_anomalies(
    conn: psycopg.Connection[Any], since: datetime, until: datetime, limit: int
) -> list[dict[str, Any]]:
    sql = """
        SELECT time, severity, uc, metric, source, score, actual, expected
        FROM mcp_anomalies
        WHERE time >= %s AND time < %s
        ORDER BY
          CASE severity WHEN 'critical' THEN 3 WHEN 'warning' THEN 2 ELSE 1 END DESC,
          abs(score) DESC NULLS LAST
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (since, until, limit))
        return list(cur.fetchall())


def _render_markdown(
    *,
    since: datetime,
    until: datetime,
    severity_now: dict[str, int],
    severity_prev: dict[str, int],
    uc_counts: list[tuple[str, int]],
    top: list[dict[str, Any]],
) -> str:
    total_now = sum(severity_now.values())
    total_prev = sum(severity_prev.values())
    delta = total_now - total_prev
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    lines: list[str] = [
        f"# Weekly Anomaly Digest — {until.date().isoformat()}",
        "",
        f"Window: {since.date().isoformat()} → {until.date().isoformat()} (7 days)",
        "",
        f"**Total this week: {total_now}** ({arrow} {abs(delta)} vs prior week, {total_prev})",
        "",
        "## By severity",
        "",
        "| severity | count |",
        "|---|---:|",
    ]
    for sev in ("critical", "warning", "info"):
        lines.append(f"| {sev} | {severity_now.get(sev, 0)} |")
    lines += ["", "## By use-case", "", "| uc | count |", "|---|---:|"]
    for uc, n in uc_counts:
        lines.append(f"| {uc} | {n} |")
    lines += [
        "",
        f"## Top {TOP_N} anomalies",
        "",
        "| time | severity | uc | metric | score | actual | expected |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in top:
        score = "" if row["score"] is None else f"{row['score']:.2f}"
        actual = "" if row["actual"] is None else f"{row['actual']:.2f}"
        expected = "" if row["expected"] is None else f"{row['expected']:.2f}"
        ts = row["time"].astimezone(UTC).strftime("%Y-%m-%d %H:%M")
        sev = row["severity"]
        uc = row["uc"]
        metric = row["metric"]
        lines.append(
            f"| {ts} | {sev} | {uc} | {metric} | {score} | {actual} | {expected} |"
        )
    return "\n".join(lines) + "\n"


def _send(settings: Settings, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from
    msg["To"] = settings.smtp_to
    msg.set_content(body)
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
        smtp.send_message(msg)


def run(settings: Settings, _argv: Sequence[str]) -> int:
    until = datetime.now(tz=UTC).replace(microsecond=0)
    since = until - timedelta(days=7)
    prev_since = since - timedelta(days=7)
    with write_connection(settings) as conn:
        severity_now = _counts_by_severity(conn, since, until)
        severity_prev = _counts_by_severity(conn, prev_since, since)
        uc_counts = _counts_by_uc(conn, since, until)
        top = _top_anomalies(conn, since, until, TOP_N)
    body = _render_markdown(
        since=since,
        until=until,
        severity_now=severity_now,
        severity_prev=severity_prev,
        uc_counts=uc_counts,
        top=top,
    )
    subject = f"[Lares] Weekly Anomaly Digest — {until.date().isoformat()}"
    try:
        _send(settings, subject, body)
    except (smtplib.SMTPException, OSError):
        log.exception("weekly_report_send_failed")
        return 1
    log.info(
        "weekly_report_sent",
        recipient=settings.smtp_to,
        total=sum(severity_now.values()),
        critical=severity_now.get("critical", 0),
        warning=severity_now.get("warning", 0),
        info=severity_now.get("info", 0),
    )
    return 0
