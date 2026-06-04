# iot-insights-engine

TSDB-backed background jobs for the homelab — anomaly detection +
forecast pulls that write to `mcp_anomalies` and `mcp_forecasts`.
Companion to [iot-mcp-bridge](https://github.com/alexander-zimmermann/iot-mcp-bridge)
(the MCP server, read-only) and [knx-nats-bridge](https://github.com/alexander-zimmermann/knx-nats-bridge)
(KNX ↔ NATS, owns the GA catalog).

## Architecture

```
TSDB (knx_1h, ems_esp_*, solaredge_*, warp_meter_1h) ─┐
api.forecast.solar (Personal Plus)                    ├─► iot-insights-engine
api.open-meteo.com (future)                           │     │
api.awattar.de / tibber (future)                      ┘     ▼
                                                    TSDB (mcp_anomalies, mcp_forecasts)
                                                    NATS (anomaly.<uc>.<severity>)
                                                            │
                                                            ▼
                                                       knx-nats-bridge ─► KNX-GA ─► Basalte push
```

## Subcommands

Run via the single entrypoint:

```
iot-insights-engine <subcommand>
```

| Subcommand              | Schedule (Kubernetes CronJob) | What it does |
|-------------------------|-------------------------------|--------------|
| `detect-univariate`     | `*/15 * * * *`                | Per-metric z-score vs `<source>_baseline_30d` |
| `detect-knx-join`       | `*/15 * * * *`                | Rule-based per-room joins (FBH-kalt, window+heating) |
| `train-iforest`         | `30 2 * * *`                  | Daily fit of IsolationForest per (uc, group) — pickled to rustfs |
| `score-iforest`         | `5,20,35,50 * * * *`          | Score last hour against the persisted IF model |
| `score-seasonal`        | `25 * * * *`                  | Fit MSTL+AutoARIMA inline, forecast 24h, anomaly-check last bucket |
| `forecast-solar`        | `15 * * * *`                  | Pull PV forecast → `mcp_forecasts` |
| `weekly-report`         | `0 8 * * 1`                   | Weekly Markdown digest via SMTP |

## Configuration

All `MCP_*` env vars (kept for compatibility with the existing
SealedSecret + Kyverno-clone topology shared with iot-mcp-bridge).
See `src/iot_insights_engine/config.py` for the full list.

## Local dev

```
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src
```
