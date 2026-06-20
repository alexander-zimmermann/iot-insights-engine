"""Pull hour-by-hour weather forecast from api.open-meteo.com.

Open-Meteo is keyless and public. We request the DWD ICON model
(``icon_seamless``: ICON-D2 ~2 km over Central Europe, seamlessly
falling back to ICON-EU/global) which covers the homelab (Eupen, BE,
~15 km from the German border) at high resolution. Five hourly metrics
land in ``mcp_forecasts`` with model=`open_meteo`, source=`open_meteo`,
one row per ``(forecast_for, metric)``, keyed on
``(forecast_for, source, metric, model)`` so an hourly re-pull
overwrites the existing forecast instead of duplicating it.

Unlike forecast.solar, Open-Meteo returns parallel arrays: ``hourly.time``
plus one array per metric. We request ``timezone=UTC`` so the timestamps
are already UTC — no local-tz conversion needed before the TIMESTAMPTZ
insert.

The persisted weather (vs an in-process cache) is what later lets the
MCP server's get_weather_forecast tool read back a stable 48 h horizon,
and gives the seasonal heating model a weather exogenous variable.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import httpx
import psycopg

from .config import Settings
from .db_write import write_connection
from .logging_setup import get_logger

log = get_logger(__name__)

SOURCE = "open_meteo"
MODEL = "open_meteo"

# Open-Meteo hourly field -> normalised metric stored in mcp_forecasts.
# The mapping is code-level (it drives parsing), not config — the set of
# metrics is stable and tied to the request below.
METRIC_MAP: dict[str, str] = {
    "temperature_2m": "temperature",
    "cloud_cover": "cloud_cover",
    "precipitation": "precipitation",
    "shortwave_radiation": "solar_radiation",
    "wind_speed_10m": "wind_speed",
}

HTTP_TIMEOUT_S = 30.0


def _build_params(settings: Settings) -> dict[str, str]:
    if settings.forecast_weather_lat is None or settings.forecast_weather_lon is None:
        raise ValueError("MCP_FORECAST_WEATHER_LAT / _LON are required")
    return {
        "latitude": f"{settings.forecast_weather_lat}",
        "longitude": f"{settings.forecast_weather_lon}",
        "hourly": ",".join(METRIC_MAP),
        "models": settings.forecast_weather_model,
        "timezone": "UTC",
        "forecast_hours": f"{settings.forecast_weather_forecast_hours}",
    }


def _fetch_hourly(base_url: str, params: dict[str, str]) -> dict[str, list[Any]]:
    """Returns the ``hourly`` block: ``time`` plus one array per requested
    metric (parallel, index-aligned). Raises if a requested metric is
    absent — that means the chosen model doesn't serve it and the config
    needs fixing."""
    response = httpx.get(base_url, params=params, timeout=HTTP_TIMEOUT_S)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or "hourly" not in body:
        raise ValueError(f"open-meteo response missing 'hourly': {body!r}")
    hourly = body["hourly"]
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError(f"open-meteo 'hourly' missing 'time': {hourly!r}")
    for field in METRIC_MAP:
        if field not in hourly:
            raise ValueError(f"open-meteo 'hourly' missing requested metric {field!r}")
    return hourly


def _parse_rows(hourly: dict[str, list[Any]]) -> list[tuple[datetime, str, float]]:
    """Zip ``time`` with each metric array → one ``(forecast_for, metric,
    value)`` tuple per (hour, metric). Null values (model gaps) are skipped."""
    times = [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in hourly["time"]]
    rows: list[tuple[datetime, str, float]] = []
    for field, metric in METRIC_MAP.items():
        for forecast_for, value in zip(times, hourly[field], strict=True):
            if value is None:
                continue
            rows.append((forecast_for, metric, float(value)))
    return rows


def _insert_forecasts(
    conn: psycopg.Connection[Any], rows: list[tuple[datetime, str, float]]
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
    inserted = 0
    with conn.cursor() as cur:
        for forecast_for, metric, value in rows:
            cur.execute(sql, (forecast_for, SOURCE, metric, MODEL, value))
            inserted += 1
    return inserted


def run(settings: Settings, _argv: Sequence[str]) -> int:
    try:
        params = _build_params(settings)
    except ValueError:
        log.exception("forecast_weather_config_invalid")
        return 2
    base_url = settings.forecast_weather_base_url
    try:
        hourly = _fetch_hourly(base_url, params)
    except httpx.HTTPError:
        log.exception("forecast_weather_fetch_failed", url=base_url)
        return 1
    rows = _parse_rows(hourly)
    if not rows:
        log.warning("forecast_weather_empty_response", url=base_url)
        return 0
    with write_connection(settings) as conn:
        inserted = _insert_forecasts(conn, rows)
    log.info(
        "forecast_weather_done",
        url=base_url,
        model=settings.forecast_weather_model,
        hours=len(hourly["time"]),
        metrics=len(METRIC_MAP),
        upserted=inserted,
    )
    return 0
