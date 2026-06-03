"""Univariate anomaly detector: z-score of the last 1h-bucket against the
matching `<source>_baseline_30d` hour×weekday profile.

Each registry entry maps a 1h-CAGG column to its baseline `stats_agg` field;
we `rollup()` across the last 60 days to compute the mean and stddev for
the current bucket's hour-of-day × weekday, then score the actual value.

z-score tiers: `|z| >= 6 → critical`, `>= 4 → warning`, `>= 3 → info`.
Tighter than the textbook 2σ on purpose — these run every 15 min on 12
hot metrics each with multiple groups, so even 3σ produces a few alerts
per day worth looking at.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import psycopg

from . import nats_publisher
from .config import Settings
from .db_write import write_connection
from .logging_setup import get_logger
from .registry import UNIVARIATE_METRICS, UnivariateMetric

log = get_logger(__name__)

BASELINE_LOOKBACK_DAYS = 60
SEVERITY_CRITICAL_THRESHOLD = 6.0
SEVERITY_WARNING_THRESHOLD = 4.0
SEVERITY_INFO_THRESHOLD = 3.0


@dataclass
class _Hit:
    bucket: Any  # datetime
    group_values: tuple[Any, ...]
    actual: float
    mean: float
    stddev: float
    zscore: float
    severity: str


def _classify(zscore: float) -> str | None:
    abs_z = abs(zscore)
    if abs_z >= SEVERITY_CRITICAL_THRESHOLD:
        return "critical"
    if abs_z >= SEVERITY_WARNING_THRESHOLD:
        return "warning"
    if abs_z >= SEVERITY_INFO_THRESHOLD:
        return "info"
    return None


def _scan_metric(
    conn: psycopg.Connection[Any], metric: UnivariateMetric
) -> list[_Hit]:
    # Inner CTE uses bare column names (single table). Outer SELECT joins
    # `last_bucket lb` with `<baseline_cagg> b`, both of which carry the
    # group columns — so the outer SELECT/GROUP BY must qualify with `lb.`
    # to disambiguate.
    inner_group_select = ", ".join(metric.group_cols) if metric.group_cols else ""
    inner_group_select_prefix = f", {inner_group_select}" if inner_group_select else ""
    outer_group_select = (
        ", ".join(f"lb.{c}" for c in metric.group_cols) if metric.group_cols else ""
    )
    outer_group_select_prefix = f", {outer_group_select}" if outer_group_select else ""
    baseline_join = (
        " AND ".join(f"b.{c} = lb.{c}" for c in metric.group_cols)
        if metric.group_cols
        else "TRUE"
    )
    outer_group_by_clause = f", {outer_group_select}" if outer_group_select else ""

    sql = f"""
        WITH last_bucket AS (
            SELECT bucket{inner_group_select_prefix}, {metric.metric} AS actual
            FROM {metric.source_cagg}
            WHERE bucket = (
                SELECT max(bucket) FROM {metric.source_cagg}
                WHERE bucket <= date_trunc('hour', now()) - interval '1 hour'
            )
        )
        SELECT
            lb.bucket{outer_group_select_prefix},
            lb.actual,
            average(rollup(b.{metric.stats_field}))            AS mean,
            stddev(rollup(b.{metric.stats_field}), 'sample')   AS stddev
        FROM last_bucket lb
        JOIN {metric.baseline_cagg} b
          ON b.hour_of_day = EXTRACT(hour FROM lb.bucket)::smallint
          AND b.weekday    = EXTRACT(isodow FROM lb.bucket)::smallint
          AND b.day        > now() - interval '{BASELINE_LOOKBACK_DAYS} days'
          AND {baseline_join}
        GROUP BY lb.bucket, lb.actual{outer_group_by_clause}
    """

    hits: list[_Hit] = []
    with conn.cursor() as cur:
        cur.execute(sql)
        for row in cur.fetchall():
            actual = row["actual"]
            mean = row["mean"]
            stddev = row["stddev"]
            if actual is None or mean is None or stddev is None or stddev == 0:
                continue
            if math.isnan(actual) or math.isnan(mean) or math.isnan(stddev):
                continue
            zscore = (actual - mean) / stddev
            severity = _classify(zscore)
            if severity is None:
                continue
            group_values = tuple(row[c] for c in metric.group_cols)
            hits.append(
                _Hit(
                    bucket=row["bucket"],
                    group_values=group_values,
                    actual=float(actual),
                    mean=float(mean),
                    stddev=float(stddev),
                    zscore=float(zscore),
                    severity=severity,
                )
            )
    return hits


def _insert_anomaly(
    conn: psycopg.Connection[Any],
    metric: UnivariateMetric,
    hit: _Hit,
) -> bool:
    """Returns True if a new row was inserted (xmax=0); False on no-op or
    severity-equal update. The publisher fires only on True (or escalation,
    which we don't track yet — covered in a follow-up)."""
    payload = {
        "group": dict(zip(metric.group_cols, hit.group_values, strict=True)),
        "mean": hit.mean,
        "stddev": hit.stddev,
        "zscore": hit.zscore,
    }
    sql = """
        INSERT INTO mcp_anomalies (
            time, source, metric, detector, severity, uc,
            actual, expected, score, payload
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (time, source, metric, detector) DO UPDATE
        SET severity = EXCLUDED.severity,
            actual   = EXCLUDED.actual,
            expected = EXCLUDED.expected,
            score    = EXCLUDED.score,
            payload  = EXCLUDED.payload,
            uc       = EXCLUDED.uc
        RETURNING xmax = 0 AS inserted
    """
    metric_with_group = (
        f"{metric.metric}[{','.join(str(v) for v in hit.group_values)}]"
        if hit.group_values
        else metric.metric
    )
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                hit.bucket,
                metric.source_cagg,
                metric_with_group,
                "zscore",
                hit.severity,
                metric.uc,
                hit.actual,
                hit.mean,
                hit.zscore,
                json.dumps(payload),
            ),
        )
        result = cur.fetchone()
    return bool(result and result["inserted"])


def run(settings: Settings, _argv: Sequence[str]) -> int:
    total_hits = 0
    total_published = 0
    with write_connection(settings) as conn:
        for metric in UNIVARIATE_METRICS:
            if metric.silenced:
                log.info("metric_silenced", uc=metric.uc)
                continue
            try:
                hits = _scan_metric(conn, metric)
            except psycopg.Error as exc:
                log.exception("metric_scan_failed", uc=metric.uc, source=metric.source_cagg)
                # Keep going — one bad CAGG must not stop the whole sweep.
                _ = exc
                continue
            for hit in hits:
                inserted = _insert_anomaly(conn, metric, hit)
                total_hits += 1
                if not inserted:
                    continue
                try:
                    nats_publisher.publish_anomaly(
                        settings,
                        uc=metric.uc,
                        severity=hit.severity,
                        payload={
                            "source": metric.source_cagg,
                            "metric": metric.metric,
                            "actual": hit.actual,
                            "expected": hit.mean,
                            "zscore": hit.zscore,
                            "group": dict(
                                zip(metric.group_cols, hit.group_values, strict=True)
                            ),
                            "bucket": hit.bucket.isoformat(),
                        },
                    )
                    total_published += 1
                except Exception:
                    # NATS-outage must not fail the job — the row is in TSDB
                    # and the next run retries.
                    log.exception("nats_publish_failed", uc=metric.uc)
    log.info(
        "detect_univariate_done",
        scanned_metrics=len(UNIVARIATE_METRICS),
        hits=total_hits,
        published=total_published,
    )
    return 0
