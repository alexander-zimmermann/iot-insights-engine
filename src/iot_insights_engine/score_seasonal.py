"""Hourly fit + forecast + anomaly-check per registered seasonal UC.

Replaces the older train/score split that pickled a fitted
StatsForecast model to rustfs — statsforecast 2.x has two API quirks
that made the pickle path brittle (see fix history of v0.1.1 + 0.1.2).
Now everything runs inline each hour:

1. Load the last `lookback_days` of (bucket, metric) from the
   source-CAGG into a long-form DataFrame.
2. Fit `MSTL(season_length=[24,168]) + AutoARIMA` against it.
3. Generate `forecast_horizon_hours` of point + sigma bounds, upsert
   into `mcp_forecasts`.
4. Compute residual_stddev from in-sample fitted values.
5. Compare the most recent completed bucket against its forecast.
   |z| >= 1σ info, >= 1.5σ warning, >= 2.5σ critical (clamped by
   `severity_floor`); during the first `warmup_days` after a UC was
   added everything is demoted to info.

Cost: ~30s per UC per hour (MSTL+AutoARIMA fit on ~8k hourly samples).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA

from . import nats_publisher
from .config import Settings
from .db_write import write_connection
from .logging_setup import get_logger
from .registry import SEASONAL_MODELS, SeasonalModel

log = get_logger(__name__)

DETECTOR_NAME = "seasonal"
MIN_TRAIN_SAMPLES = 24 * 14


def _classify(z: float, severity_floor: str) -> str | None:
    abs_z = abs(z)
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


def _warmup_active(uc_added_at: datetime, warmup_days: int) -> bool:
    return datetime.now(tz=UTC) - uc_added_at < timedelta(days=warmup_days)


def _load_training_frame(
    conn: psycopg.Connection[Any], uc: SeasonalModel
) -> pd.DataFrame:
    sql = f"""
        SELECT bucket AS ds, {uc.metric}::float AS y
        FROM {uc.source_cagg}
        WHERE bucket > now() - interval '{uc.lookback_days} days'
          AND {uc.metric} IS NOT NULL
        ORDER BY bucket
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])
    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_convert(None)
    df["unique_id"] = uc.uc
    return df[["unique_id", "ds", "y"]]


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
    df = _load_training_frame(conn, uc)
    n_samples = int(df.shape[0])
    if n_samples < MIN_TRAIN_SAMPLES:
        log.info(
            "seasonal_insufficient_samples",
            uc=uc.uc,
            n=n_samples,
            min_required=MIN_TRAIN_SAMPLES,
        )
        return 0, 0, 0

    sf = StatsForecast(
        models=[MSTL(season_length=list(uc.season_length), trend_forecaster=AutoARIMA())],
        freq="h",
    )
    # forecast(fitted=True) fits AND returns the prediction we'll upsert
    # below — single API call, no second predict() round-trip needed.
    forecast_df = sf.forecast(
        df=df, h=uc.forecast_horizon_hours, fitted=True
    )
    fitted = sf.forecast_fitted_values()
    residuals = fitted["y"].to_numpy() - fitted["MSTL"].to_numpy()
    residual_stddev = float(np.nanstd(residuals, ddof=1))
    if residual_stddev <= 0 or math.isnan(residual_stddev):
        log.warning("seasonal_zero_stddev", uc=uc.uc)
        return 0, 0, 0

    log.info(
        "seasonal_fit",
        uc=uc.uc,
        n_samples=n_samples,
        residual_stddev=residual_stddev,
        season_length=list(uc.season_length),
    )

    sigma = uc.sigma_threshold * residual_stddev
    forecast_rows = 0
    for _, row in forecast_df.iterrows():
        forecast_for = row["ds"]
        value = float(row["MSTL"])
        _upsert_forecast_row(
            conn,
            uc=uc,
            forecast_for=forecast_for,
            value=value,
            lower=value - sigma,
            upper=value + sigma,
        )
        forecast_rows += 1

    actual_pair = _last_actual(conn, uc)
    if actual_pair is None:
        return forecast_rows, 0, 0
    bucket, actual = actual_pair
    actual_ts = bucket.astimezone(UTC).replace(tzinfo=None)
    match = forecast_df[forecast_df["ds"] == actual_ts]
    if match.empty:
        return forecast_rows, 0, 0
    expected = float(match["MSTL"].iloc[0])
    z = (actual - expected) / residual_stddev
    severity = _classify(z, uc.severity_floor)
    if severity is None:
        return forecast_rows, 0, 0

    earliest = df["ds"].min()
    uc_added_at = earliest.tz_localize(UTC) if earliest.tzinfo is None else earliest
    if _warmup_active(uc_added_at, uc.warmup_days):
        severity = "info"

    payload = {
        "actual": actual,
        "expected": expected,
        "z": z,
        "residual_stddev": residual_stddev,
        "sigma_threshold": uc.sigma_threshold,
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
            except (psycopg.Error, ValueError):
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
