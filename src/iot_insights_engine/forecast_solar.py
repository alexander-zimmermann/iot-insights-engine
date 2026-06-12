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
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import psycopg

from .config import Settings
from .db_write import write_connection
from .logging_setup import get_logger

log = get_logger(__name__)

SOURCE = "forecast_solar"
METRIC = "pv_production"
MODEL = "forecast_solar"

HTTP_TIMEOUT_S = 30.0


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


def _fetch_watts(url: str) -> dict[str, float]:
    """Returns the ``result.watts`` mapping verbatim: keys are
    ``"YYYY-MM-DD HH:MM:SS"`` strings (Europe-local timezone per
    forecast.solar's account default), values are watts at that
    instant."""
    response = httpx.get(url, timeout=HTTP_TIMEOUT_S)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or "result" not in body:
        raise ValueError(f"forecast.solar response missing 'result': {body!r}")
    watts = body["result"].get("watts") or {}
    if not isinstance(watts, dict):
        raise ValueError(f"forecast.solar 'result.watts' not a dict: {watts!r}")
    return {k: float(v) for k, v in watts.items()}


def _to_utc(ts_str: str, tz_name: str) -> datetime:
    """forecast.solar emits naive ``YYYY-MM-DD HH:MM:SS`` strings in the
    account's local timezone — attach it, then convert to UTC so the
    TIMESTAMPTZ insert is unambiguous regardless of the session TZ."""
    return datetime.fromisoformat(ts_str).replace(tzinfo=ZoneInfo(tz_name)).astimezone(UTC)


def _insert_forecasts(
    conn: psycopg.Connection[Any], watts: dict[str, float], tz_name: str
) -> int:
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


def run(settings: Settings, _argv: Sequence[str]) -> int:
    try:
        url = _build_url(settings)
    except (ValueError, KeyError, json.JSONDecodeError):
        log.exception("forecast_solar_config_invalid")
        return 2
    # Strip the API-key from the logged URL — keep host + plane path only.
    safe_url = url.replace(settings.forecast_solar_api_key, "<key>")
    try:
        watts = _fetch_watts(url)
    except httpx.HTTPError:
        log.exception("forecast_solar_fetch_failed", url=safe_url)
        return 1
    if not watts:
        log.warning("forecast_solar_empty_response", url=safe_url)
        return 0
    with write_connection(settings) as conn:
        inserted = _insert_forecasts(conn, watts, settings.forecast_solar_timezone)
    log.info(
        "forecast_solar_done",
        url=safe_url,
        points=len(watts),
        upserted=inserted,
    )
    return 0
