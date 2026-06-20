"""Pull hour-by-hour PV-production forecast from api.forecast.solar.

Personal-Plus tier supports multiple planes in a single request, so the
homelab's east+west roof fits one hourly call. Forecast values land in
``mcp_forecasts`` with model=`forecast_solar`, metric=`pv_production`
(watts). Each row is keyed on ``(forecast_for, source, metric, model)``
so a re-pull during the same hour overwrites the existing forecast
instead of creating a duplicate.

The API returns naive local timestamps (account timezone, see
``MCP_FORECAST_SOLAR_TIMEZONE``); they are converted to UTC-aware
datetimes before insert so ``forecast_for`` (TIMESTAMPTZ) lines up
with the UTC-based seasonal forecasts in the same table.

The companion comparison job — does actual production match what was
forecast? — lives in ``score_solar_actual`` (TBD) and reads back from
this same table.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import psycopg

from . import nats_publisher
from .config import Settings
from .db_write import write_connection
from .logging_setup import get_logger

log = get_logger(__name__)

SOURCE = "forecast_solar"
METRIC = "pv_production"
MODEL = "forecast_solar"

HTTP_TIMEOUT_S = 30.0

# Bridge writer-rules (knx-nats-bridge) map these onto KNX 15/4/70..73:
# today_kwh→70, remaining_kwh→71, tomorrow_kwh→72 (DPT 13.013 kWh),
# now_watts→73 (DPT 14.056 W). Keep the keys stable — the rules pin them.
FORECAST_SUBJECT_PREFIX = "forecast.pv"


def _build_url(settings: Settings) -> str:
    if not settings.forecast_solar_api_key:
        raise ValueError("MCP_FORECAST_SOLAR_API_KEY (or *_FILE) is required")
    if settings.forecast_solar_lat is None or settings.forecast_solar_lon is None:
        raise ValueError("MCP_FORECAST_SOLAR_LAT / _LON are required")
    planes = json.loads(settings.forecast_solar_planes)
    if not planes:
        raise ValueError("MCP_FORECAST_SOLAR_PLANES is empty — expected JSON array")

    parts: list[str] = [
        settings.forecast_solar_base_url.rstrip("/"),
        settings.forecast_solar_api_key,
        "estimate",
        f"{settings.forecast_solar_lat}",
        f"{settings.forecast_solar_lon}",
    ]
    for plane in planes:
        parts.extend(
            (
                f"{plane['dec']}",
                f"{plane['az']}",
                f"{plane['kwp']}",
            )
        )
    return "/".join(parts)


def _fetch_result(url: str) -> dict[str, Any]:
    """Returns the forecast.solar ``result`` object in one request (the
    Personal tier is rate-limited, so we never call twice). It carries
    ``watts`` (W per instant), ``watt_hours_period`` (Wh per interval) and
    ``watt_hours_day`` (Wh per day); all instant/interval keys are
    ``"YYYY-MM-DD HH:MM:SS"`` strings in the account-local timezone, day
    keys are ``"YYYY-MM-DD"``."""
    response = httpx.get(url, timeout=HTTP_TIMEOUT_S)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or "result" not in body:
        raise ValueError(f"forecast.solar response missing 'result': {body!r}")
    result = body["result"]
    if not isinstance(result, dict):
        raise ValueError(f"forecast.solar 'result' not a dict: {result!r}")
    return result


def _watts_from_result(result: dict[str, Any]) -> dict[str, float]:
    """Extract the ``result.watts`` mapping (instant → W) for TSDB insert."""
    watts = result.get("watts") or {}
    if not isinstance(watts, dict):
        raise ValueError(f"forecast.solar 'result.watts' not a dict: {watts!r}")
    return {k: float(v) for k, v in watts.items()}


def _to_utc(ts_str: str, tz_name: str) -> datetime:
    """forecast.solar emits naive ``YYYY-MM-DD HH:MM:SS`` strings in the
    account's local timezone — attach it, then convert to UTC so the
    TIMESTAMPTZ insert is unambiguous regardless of the session TZ."""
    return datetime.fromisoformat(ts_str).replace(tzinfo=ZoneInfo(tz_name)).astimezone(UTC)


def _insert_forecasts(conn: psycopg.Connection[Any], watts: dict[str, float], tz_name: str) -> int:
    sql = """
        INSERT INTO mcp_forecasts (
            forecast_for, source, metric, model,
            forecast_value, forecast_lower, forecast_upper
        )
        VALUES (%s, %s, %s, %s, %s, NULL, NULL)
        ON CONFLICT (forecast_for, source, metric, model) DO UPDATE
        SET forecast_value = EXCLUDED.forecast_value,
            created_at     = now()
    """
    rows = 0
    with conn.cursor() as cur:
        for ts_str, value in watts.items():
            forecast_for = _to_utc(ts_str, tz_name)
            cur.execute(sql, (forecast_for, SOURCE, METRIC, MODEL, value))
            rows += 1
    return rows


def _compute_scalars(
    result: dict[str, Any], tz_name: str, *, now_local: datetime | None = None
) -> dict[str, float]:
    """Derive the four Basalte-facing scalars from the forecast curve:
    ``today_kwh`` / ``tomorrow_kwh`` (whole-day energy), ``remaining_kwh``
    (energy still to come today) and ``now_watts`` (the most recent
    at-or-before-now power point today, 0 outside daylight). Only keys the
    API actually returned are included so a partial response degrades
    gracefully. ``now_local`` is injectable for tests."""
    tz = ZoneInfo(tz_name)
    now = now_local or datetime.now(tz)
    now_naive = now.replace(tzinfo=None)
    today = now.date()
    tomorrow = today + timedelta(days=1)

    wh_day: dict[str, Any] = result.get("watt_hours_day") or {}
    wh_period: dict[str, Any] = result.get("watt_hours_period") or {}
    watts: dict[str, Any] = result.get("watts") or {}

    scalars: dict[str, float] = {}

    today_wh = wh_day.get(today.isoformat())
    if today_wh is not None:
        scalars["today_kwh"] = round(float(today_wh) / 1000.0, 3)
    tomorrow_wh = wh_day.get(tomorrow.isoformat())
    if tomorrow_wh is not None:
        scalars["tomorrow_kwh"] = round(float(tomorrow_wh) / 1000.0, 3)

    if wh_period:
        remaining_wh = sum(
            float(v)
            for ts_str, v in wh_period.items()
            if (ts := datetime.fromisoformat(ts_str)).date() == today and ts > now_naive
        )
        scalars["remaining_kwh"] = round(remaining_wh / 1000.0, 3)

    if watts:
        # Power forecast for the hour we're in. forecast.solar only emits
        # daylight points, so an hour with no entry (night) is 0 W — match the
        # current hour, not "last point so far", or 22:00 reports the midday peak.
        this_hour = [
            (ts, float(v))
            for ts_str, v in watts.items()
            if (ts := datetime.fromisoformat(ts_str)).date() == today and ts.hour == now_naive.hour
        ]
        scalars["now_watts"] = round(max(this_hour, key=lambda p: p[0])[1], 1) if this_hour else 0.0

    return scalars


def _publish_scalars(settings: Settings, scalars: dict[str, float]) -> None:
    """Best-effort publish of each scalar to ``forecast.pv.<key>`` as
    ``{"value": …}``. A NATS outage (or NATS not configured for this job)
    must not fail the run — the forecast is already in TSDB."""
    if not settings.nats_servers:
        log.info("forecast_solar_nats_skip", reason="MCP_NATS_SERVERS not set")
        return
    for key, value in scalars.items():
        subject = f"{FORECAST_SUBJECT_PREFIX}.{key}"
        try:
            nats_publisher.publish(settings, subject, {"value": value})
        except Exception:
            log.exception("forecast_solar_publish_failed", subject=subject)


def run(settings: Settings, _argv: Sequence[str]) -> int:
    try:
        url = _build_url(settings)
    except (ValueError, KeyError, json.JSONDecodeError):
        log.exception("forecast_solar_config_invalid")
        return 2
    # Strip the API-key from the logged URL — keep host + plane path only.
    safe_url = url.replace(settings.forecast_solar_api_key, "<key>")
    try:
        result = _fetch_result(url)
    except httpx.HTTPError:
        log.exception("forecast_solar_fetch_failed", url=safe_url)
        return 1
    watts = _watts_from_result(result)
    if not watts:
        log.warning("forecast_solar_empty_response", url=safe_url)
        return 0
    with write_connection(settings) as conn:
        inserted = _insert_forecasts(conn, watts, settings.forecast_solar_timezone)
    scalars = _compute_scalars(result, settings.forecast_solar_timezone)
    _publish_scalars(settings, scalars)
    log.info(
        "forecast_solar_done",
        url=safe_url,
        points=len(watts),
        upserted=inserted,
        scalars=scalars,
    )
    return 0
