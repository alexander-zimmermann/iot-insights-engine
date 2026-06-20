"""Unit tests for the Open-Meteo weather forecast job.

Network calls are mocked with respx; DB writes are not exercised here
(integration via the cluster smoke test after deploy).
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

from iot_insights_engine import forecast_weather
from iot_insights_engine.config import Settings


def _settings() -> Settings:
    return Settings(
        db_host="localhost",
        db_name="x",
        db_username="x",
        db_password="x",  # noqa: S106 — test stub
        forecast_weather_lat=50.626,
        forecast_weather_lon=6.024,
    )


def _hourly_payload() -> dict[str, object]:
    return {
        "hourly": {
            "time": ["2026-06-20T00:00", "2026-06-20T01:00"],
            "temperature_2m": [15.5, 14.0],
            "cloud_cover": [40, 90],
            "precipitation": [0.0, 1.2],
            "shortwave_radiation": [0.0, 0.0],
            "wind_speed_10m": [3.4, 4.1],
        }
    }


def test_build_params_requires_coords() -> None:
    s = Settings(db_host="localhost", db_name="x", db_username="x", db_password="x")  # noqa: S106
    with pytest.raises(ValueError, match="LAT / _LON"):
        forecast_weather._build_params(s)


def test_build_params_shape() -> None:
    params = forecast_weather._build_params(_settings())
    assert params["latitude"] == "50.626"
    assert params["timezone"] == "UTC"
    assert params["models"] == "icon_seamless"
    # All five requested metrics in the hourly list.
    assert params["hourly"].split(",") == [
        "temperature_2m",
        "cloud_cover",
        "precipitation",
        "shortwave_radiation",
        "wind_speed_10m",
    ]


def test_fetch_hourly_parses() -> None:
    base = "https://api.open-meteo.com/v1/forecast"
    with respx.mock(assert_all_called=True) as router:
        router.get(base).mock(return_value=httpx.Response(200, json=_hourly_payload()))
        hourly = forecast_weather._fetch_hourly(base, forecast_weather._build_params(_settings()))
    assert hourly["temperature_2m"] == [15.5, 14.0]


def test_fetch_hourly_rejects_missing_metric() -> None:
    base = "https://api.open-meteo.com/v1/forecast"
    payload = _hourly_payload()
    del payload["hourly"]["cloud_cover"]  # type: ignore[attr-defined]
    with respx.mock(assert_all_called=True) as router:
        router.get(base).mock(return_value=httpx.Response(200, json=payload))
        with pytest.raises(ValueError, match="missing requested metric 'cloud_cover'"):
            forecast_weather._fetch_hourly(base, forecast_weather._build_params(_settings()))


def test_parse_rows_pivots_to_one_row_per_metric_hour() -> None:
    rows = forecast_weather._parse_rows(_hourly_payload()["hourly"])  # type: ignore[arg-type]
    # 5 metrics x 2 hours = 10 rows.
    assert len(rows) == 10
    # UTC timestamps attached verbatim (timezone=UTC requested).
    assert (datetime(2026, 6, 20, 0, 0, tzinfo=UTC), "temperature", 15.5) in rows
    assert (datetime(2026, 6, 20, 1, 0, tzinfo=UTC), "cloud_cover", 90.0) in rows


def test_parse_rows_skips_nulls() -> None:
    payload = _hourly_payload()
    payload["hourly"]["temperature_2m"] = [None, 14.0]  # type: ignore[index]
    rows = forecast_weather._parse_rows(payload["hourly"])  # type: ignore[arg-type]
    temps = [r for r in rows if r[1] == "temperature"]
    assert temps == [(datetime(2026, 6, 20, 1, 0, tzinfo=UTC), "temperature", 14.0)]
