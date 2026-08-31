# iot-insights-engine

TSDB-backed background jobs for the homelab — external forecast pulls
and the daily energy balance, writing to `mcp_forecasts`.
Companion to [iot-mcp-bridge](https://github.com/alexander-zimmermann/iot-mcp-bridge)
(the MCP server, read-only) and [knx-nats-bridge](https://github.com/alexander-zimmermann/knx-nats-bridge)
(KNX ↔ NATS, owns the GA catalog).

## Architecture

```
TSDB (solaredge_*, warp_meter_1h) ─┐
api.forecast.solar (Personal Plus) ├─► iot-insights-engine
api.open-meteo.com                 ┘     │
                                         ▼
                                 TSDB (mcp_forecasts)
                                 NATS (forecast.pv.*, energy.pv.*)
                                         │
                                         ▼
                                  knx-nats-bridge ─► KNX-GA ─► Basalte
```

The detection chain is being rebuilt: the old detectors, their
baselines and the weekly report are switched off, and the new fault
definitions land here as declared entries, not as subcommands.

## Subcommands

Run via the single entrypoint:

```
iot-insights-engine <subcommand>
```

| Subcommand              | Schedule (Kubernetes CronJob) | What it does |
|-------------------------|-------------------------------|--------------|
| `forecast-solar`        | `15 * * * *`                  | Pull PV forecast → `mcp_forecasts`, publish `forecast.pv.*` |
| `forecast-weather`      | `20 * * * *`                  | Pull Open-Meteo (ICON) forecast → `mcp_forecasts` |
| `energy-balance`        | `*/15 * * * *`                | Today's kWh counters → `energy.pv.*` |

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
