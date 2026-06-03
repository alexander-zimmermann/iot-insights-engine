"""Hourly forecast + anomaly-check for each registered seasonal UC.

Loads the persisted statsforecast model, generates the next
`forecast_horizon_hours` of point estimates + sigma-based confidence
bounds, upserts them into `mcp_forecasts`, then compares the most
recent completed bucket against its predicted value. A deviation
beyond `sigma_threshold * residual_stddev` becomes an anomaly row
on `mcp_anomalies` and a `anomaly.<uc>.<severity>` NATS publish.

Cold-start safe — until train_seasonal runs once, load_model returns
None and the UC is silently skipped.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg

from . import artifacts, nats_publisher
from .config import Settings
from .db_write import write_connection
from .logging_setup import get_logger
from .registry import SEASONAL_MODELS, SeasonalModel
from .seasonal_common import DETECTOR_NAME, ModelEnvelope

log = get_logger(__name__)


def _classify(z: float, severity_floor: str) -> str | None:
    """Severity from |z|:
    * >=2.5 → critical
    * >=1.5 → warning
    * >=1.0 → info
    Lower than info is no-anomaly.

    `severity_floor` is the minimum severity that gets emitted — UCs
    that flap at info can set floor='warning' to suppress noise without
    losing the genuinely-actionable warning/critical events.
    """
    abs_z = abs(z)
    severity: str | None
    if abs_z >= 2.5:
        severity = "critical"
    elif abs_z >= 1.5:
        severity = "warning"
    elif abs_z >= 1.0:
        severity = "info"
    else:
        return None
    order = ["info", "warning", "critical"]
    if order.index(severity) < order.index(severity_floor):
        return None
    return severity


def _warmup_demote(envelope: ModelEnvelope, severity: str, warmup_days: int) -> str:
    trained_at = datetime.fromisoformat(envelope.trained_at)
    if datetime.now(tz=UTC) - trained_at < timedelta(days=warmup_days):
        return "info"
    return severity


def _upsert_forecast_row(
    conn: psycopg.Connection[Any],
    *,
    uc: SeasonalModel,
    forecast_for: Any,
    value: float,
    lower: float | None,
    upper: float | None,
) -> None:
    sql = """
        INSERT INTO mcp_forecasts (
            forecast_for, source, metric, model,
            forecast_value, forecast_lower, forecast_upper
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (forecast_for, source, metric, model) DO UPDATE
        SET forecast_value = EXCLUDED.forecast_value,
            forecast_lower = EXCLUDED.forecast_lower,
            forecast_upper = EXCLUDED.forecast_upper,
            created_at     = now()
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                forecast_for,
                uc.source_cagg,
                uc.metric,
                f"{DETECTOR_NAME}:{uc.uc}",
                value,
                lower,
                upper,
            ),
        )


def _last_actual(
    conn: psycopg.Connection[Any], uc: SeasonalModel
) -> tuple[Any, float] | None:
    sql = f"""
        SELECT bucket, {uc.metric}::float AS y
        FROM {uc.source_cagg}
        WHERE bucket = (
            SELECT max(bucket) FROM {uc.source_cagg}
            WHERE bucket <= date_trunc('hour', now()) - interval '1 hour'
              AND {uc.metric} IS NOT NULL
        )
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
    if not row or row["y"] is None:
        return None
    return row["bucket"], float(row["y"])


def _insert_anomaly(
    conn: psycopg.Connection[Any],
    *,
    uc: SeasonalModel,
    bucket: Any,
    actual: float,
    expected: float,
    z: float,
    severity: str,
    payload: dict[str, Any],
) -> bool:
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
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                bucket,
                uc.source_cagg,
                uc.metric,
                DETECTOR_NAME,
                severity,
                uc.uc,
                actual,
                expected,
                z,
                json.dumps(payload),
            ),
        )
        result = cur.fetchone()
    return bool(result and result["inserted"])


def _score_uc(
    settings: Settings, conn: psycopg.Connection[Any], uc: SeasonalModel
) -> tuple[int, int, int]:
    """Returns (forecast_rows, anomaly_inserts, publishes)."""
    envelope: ModelEnvelope | None = artifacts.load_model(settings, DETECTOR_NAME, uc.uc)
    if envelope is None:
        return 0, 0, 0
    forecast_df = envelope.sf.predict(h=envelope.forecast_horizon_hours)
    # statsforecast returns columns: unique_id, ds, MSTL
    forecast_rows = 0
    for _, row in forecast_df.iterrows():
        forecast_for = row["ds"]
        value = float(row["MSTL"])
        sigma = envelope.sigma_threshold * envelope.residual_stddev
        _upsert_forecast_row(
            conn,
            uc=uc,
            forecast_for=forecast_for,
            value=value,
            lower=value - sigma,
            upper=value + sigma,
        )
        forecast_rows += 1

    # Now compare the last actual against its forecast (which we just wrote).
    actual_pair = _last_actual(conn, uc)
    if actual_pair is None:
        return forecast_rows, 0, 0
    bucket, actual = actual_pair
    actual_ts = bucket.astimezone(UTC).replace(tzinfo=None)
    match = forecast_df[forecast_df["ds"] == actual_ts]
    if match.empty:
        log.info(
            "seasonal_no_forecast_for_actual",
            uc=uc.uc,
            actual_bucket=bucket.isoformat(),
        )
        return forecast_rows, 0, 0
    expected = float(match["MSTL"].iloc[0])
    stddev = envelope.residual_stddev
    if stddev <= 0 or math.isnan(stddev):
        return forecast_rows, 0, 0
    z = (actual - expected) / stddev
    severity = _classify(z, uc.severity_floor)
    if severity is None:
        return forecast_rows, 0, 0
    severity = _warmup_demote(envelope, severity, uc.warmup_days)
    payload = {
        "actual": actual,
        "expected": expected,
        "z": z,
        "residual_stddev": stddev,
        "sigma_threshold": envelope.sigma_threshold,
        "bucket": bucket.isoformat(),
    }
    inserted = _insert_anomaly(
        conn,
        uc=uc,
        bucket=bucket,
        actual=actual,
        expected=expected,
        z=z,
        severity=severity,
        payload=payload,
    )
    if not inserted:
        return forecast_rows, 0, 0
    try:
        nats_publisher.publish_anomaly(
            settings,
            uc=uc.uc,
            severity=severity,
            payload={"source": uc.source_cagg, "metric": uc.metric, **payload},
        )
        return forecast_rows, 1, 1
    except Exception:
        log.exception("nats_publish_failed", uc=uc.uc)
        return forecast_rows, 1, 0


def run(settings: Settings, _argv: Sequence[str]) -> int:
    total_forecasts = total_inserts = total_publishes = 0
    with write_connection(settings) as conn:
        for uc in SEASONAL_MODELS:
            if uc.silenced:
                log.info("uc_silenced", uc=uc.uc)
                continue
            try:
                fc, ins, pub = _score_uc(settings, conn, uc)
            except psycopg.Error:
                log.exception("seasonal_score_failed", uc=uc.uc)
                continue
            total_forecasts += fc
            total_inserts += ins
            total_publishes += pub
    log.info(
        "score_seasonal_done",
        scanned_ucs=len(SEASONAL_MODELS),
        forecasts=total_forecasts,
        inserted=total_inserts,
        published=total_publishes,
    )
    return 0
