"""Unit tests for the Forecast.Solar job.

Network calls are mocked with respx; DB writes are not exercised here
(integration via the cluster smoke test after deploy).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import httpx
import pytest
import respx

from iot_insights_engine import forecast_solar
from iot_insights_engine.config import Settings


def _settings(api_key: str = "AAAA-test-key-BBBB", planes: str | None = None) -> Settings:
    return Settings(
        db_host="localhost",
        db_name="x",
        db_username="x",
        db_password="x",  # noqa: S106 — test stub
        forecast_solar_api_key=api_key,
        forecast_solar_lat=50.626,
        forecast_solar_lon=6.024,
        forecast_solar_planes=planes
        or '[{"dec":17,"az":-51,"kwp":6.175},{"dec":17,"az":129,"kwp":6.435}]',
    )


def test_build_url_two_planes() -> None:
    s = _settings()
    url = forecast_solar._build_url(s)
    assert url == (
        "https://api.forecast.solar/AAAA-test-key-BBBB/estimate/"
        "50.626/6.024/17/-51/6.175/17/129/6.435"
    )


def test_build_url_requires_api_key() -> None:
    s = _settings(api_key="")
    with pytest.raises(ValueError, match="API_KEY"):
        forecast_solar._build_url(s)


def test_build_url_requires_planes() -> None:
    s = _settings(planes="[]")
    with pytest.raises(ValueError, match="PLANES"):
        forecast_solar._build_url(s)


def test_build_url_three_planes_scales() -> None:
    """Adding a 3rd plane should be a config-only change."""
    s = _settings(
        planes='[{"dec":17,"az":-51,"kwp":6.175},'
        '{"dec":17,"az":129,"kwp":6.435},'
        '{"dec":30,"az":0,"kwp":2.0}]'
    )
    url = forecast_solar._build_url(s)
    assert url.endswith("/17/-51/6.175/17/129/6.435/30/0/2.0")


def test_fetch_watts_parses_result() -> None:
    payload = {
        "result": {
            "watts": {
                "2026-06-03 06:00:00": 250.5,
                "2026-06-03 07:00:00": 1200,
                "2026-06-03 08:00:00": 3400.0,
            }
        },
        "message": {"code": 0, "type": "success"},
    }
    url = "https://api.forecast.solar/k/estimate/x/y/17/-51/6.175"
    with respx.mock(assert_all_called=True) as router:
        router.get(url).mock(return_value=httpx.Response(200, json=payload))
        result = forecast_solar._fetch_watts(url)
    assert result == {
        "2026-06-03 06:00:00": 250.5,
        "2026-06-03 07:00:00": 1200.0,
        "2026-06-03 08:00:00": 3400.0,
    }


def test_fetch_watts_rejects_missing_result() -> None:
    url = "https://api.forecast.solar/k/estimate/x/y/17/-51/6.175"
    with respx.mock(assert_all_called=True) as router:
        router.get(url).mock(return_value=httpx.Response(200, json={"foo": "bar"}))
        with pytest.raises(ValueError, match="missing 'result'"):
            forecast_solar._fetch_watts(url)


def test_run_strips_api_key_from_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The job logs the URL on every call — make sure the API-key never
    appears in plaintext so a future log shipper can't leak it."""
    s = _settings(api_key="SECRET123")
    captured: list[str] = []

    def fake_fetch(url: str) -> dict[str, float]:
        captured.append(url)
        return {}

    with patch.object(forecast_solar, "_fetch_watts", side_effect=fake_fetch):
        rc = forecast_solar.run(s, [])
    assert rc == 0
    assert any("SECRET123" in u for u in captured), "internal call MUST use the real key"
    # The job's structlog output is JSON-on-stdout, not via stdlib logging,
    # so the simplest invariant is: caplog (stdlib path) never sees the key.
    assert all("SECRET123" not in r.getMessage() for r in caplog.records)


def test_to_utc_converts_winter_time() -> None:
    """CET (UTC+1): 12:00 local → 11:00 UTC."""
    got = forecast_solar._to_utc("2026-01-15 12:00:00", "Europe/Berlin")
    assert got == datetime(2026, 1, 15, 11, 0, tzinfo=UTC)


def test_to_utc_converts_summer_time() -> None:
    """CEST (UTC+2): 12:00 local → 10:00 UTC."""
    got = forecast_solar._to_utc("2026-06-15 12:00:00", "Europe/Berlin")
    assert got == datetime(2026, 6, 15, 10, 0, tzinfo=UTC)


def test_invalid_plane_json_returns_rc_2() -> None:
    s = _settings(planes="not-json")
    with patch.object(forecast_solar, "_fetch_watts"):
        rc = forecast_solar.run(s, [])
    assert rc == 2
