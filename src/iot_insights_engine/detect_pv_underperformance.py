"""Detect PV underperformance: today's actual yield vs the weather-adjusted
forecast.solar expectation, published on ``anomaly.pv_underperformance`` → KNX
15/4/11.

Low PV production is normally just clouds — so the only meaningful house-level
anomaly is "less than expected *for this weather*". forecast.solar is already
weather-adjusted, so a sustained shortfall against it isolates real causes
(soiling, shading, a dead string, inverter derating). This complements the
per-inverter iforest detectors (which watch electrical behaviour) with a
yield/performance signal.

Compares **cumulative energy over today's elapsed hours** (robust to single
cloudy hours): actual = Σ inverter ``energytotal`` deltas; expected = the
time-weighted integral of the forecast power curve over today's elapsed time
(forecast.solar is 15-min resolution → integrate, don't sum the samples). Only
judges once enough is expected that a shortfall is meaningful.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg

from . import nats_publisher
from .config import Settings
from .db_write import read_connection
from .logging_setup import get_logger

log = get_logger(__name__)

UC = "pv_underperformance"

# mcp_forecasts keys written by the forecast-solar job.
FORECAST_SOURCE = "forecast_solar"
FORECAST_METRIC = "pv_production"
FORECAST_MODEL = "forecast_solar"

# Don't judge until this much is expected today — avoids firing at dawn, on a
# near-zero forecast, or when forecast.solar has no data.
MIN_EXPECTED_KWH = 3.0


def _severity(ratio: float) -> str | None:
    """Cumulative actual/expected ratio → severity tier (None = ok). Generous,
    because the forecast is a model — only a clear, sustained shortfall fires."""
    if ratio < 0.35:
        return "critical"
    if ratio < 0.50:
        return "warning"
    if ratio < 0.65:
        return "info"
    return None


def _actual_and_expected(conn: psycopg.Connection[Any], tz: str) -> tuple[float, float]:
    """(actual_kwh, expected_kwh) produced/forecast today up to now, in ``tz``."""
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('timezone', %s, false)", (tz,))
        cur.execute("""
            WITH per AS (
                SELECT inverter_id,
                       last(energytotal, time) - first(energytotal, time) AS wh
                FROM solaredge_inverter
                WHERE time >= date_trunc('day', now())
                GROUP BY inverter_id
            )
            SELECT COALESCE(sum(GREATEST(wh, 0)), 0) / 1000.0 AS actual_kwh FROM per
        """)
        actual = float((cur.fetchone() or {}).get("actual_kwh") or 0.0)

        # Expected energy so far = time-weighted integral of the forecast power
        # curve over today's elapsed time. forecast.solar is 15-min resolution,
        # so summing the watt samples would over-count ~4x — integrate instead.
        # The forecast slice is tiny (tens of rows), so time_weight is cheap here.
        cur.execute(
            """
            SELECT COALESCE(
                       integral(time_weight('Linear', forecast_for, forecast_value), 'hours'),
                       0
                   ) / 1000.0 AS expected_kwh
            FROM mcp_forecasts
            WHERE source = %s AND metric = %s AND model = %s
              AND forecast_for >= date_trunc('day', now())
              AND forecast_for <= now()
            """,
            (FORECAST_SOURCE, FORECAST_METRIC, FORECAST_MODEL),
        )
        expected = float((cur.fetchone() or {}).get("expected_kwh") or 0.0)
    return actual, expected


def run(settings: Settings, _argv: Sequence[str]) -> int:
    with read_connection(settings) as conn:
        actual, expected = _actual_and_expected(conn, settings.energy_timezone)

    payload: dict[str, Any] = {
        "actual_kwh": round(actual, 2),
        "expected_kwh": round(expected, 2),
    }

    if expected < MIN_EXPECTED_KWH:
        # Too little expected yet (night/dawn/missing forecast) → no judgement, clear.
        nats_publisher.publish_anomaly(
            settings, uc=UC, severity="info", payload={**payload, "ratio": None}, firing=False
        )
        log.info("pv_underperformance_skip", **payload)
        return 0

    ratio = actual / expected
    severity = _severity(ratio)
    payload["ratio"] = round(ratio, 3)
    nats_publisher.publish_anomaly(
        settings, uc=UC, severity=severity or "info", payload=payload, firing=severity is not None
    )
    log.info("pv_underperformance_done", severity=severity or "ok", **payload)
    return 0
