# iot-insights-engine

TSDB-backed background jobs for the homelab: external forecast pulls,
the daily energy balance, and the declared fault list — history faults
measured over the hourly aggregates and delivered as a severity 0–3 on a
KNX group address.
Companion to [iot-mcp-bridge](https://github.com/alexander-zimmermann/iot-mcp-bridge)
(the MCP server, read-only — where verdicts are given) and
[knx-nats-bridge](https://github.com/alexander-zimmermann/knx-nats-bridge)
(KNX ↔ NATS, owns the GA catalog and the writer rules).

The vocabulary — fault, kind, subject, episode, severity, frontier,
delivery — is defined in [CONTEXT.md](CONTEXT.md); module docstrings,
log records and tests use those names.

## Architecture

```
TSDB (hourly aggregates, ga_catalog) ─┐
faults.yaml (lares ConfigMap)         ├─► iot-insights-engine
api.forecast.solar / api.open-meteo   ┘     │
                                            ├─► TSDB (mcp_forecasts, episodes)
                                            ▼
                             NATS (forecast.pv.*, energy.pv.*, anomaly.*)
                                            │
                                            ▼
                             knx-nats-bridge (writer rules) ─► KNX GA ─► Basalte
                                             3 = push · 1–2 = indicator · 0 = clear · e-mail for all
```

Downstream of the episodes: iot-mcp-bridge lists them and takes the
binary verdict ("real" / "nonsense") per episode; the Grafana
`knx-episodes` dashboard replaced the weekly mail.

## Subcommands

Run via the single entrypoint:

```
iot-insights-engine <subcommand>
```

| Subcommand         | Schedule (Kubernetes CronJob) | What it does |
|--------------------|-------------------------------|--------------|
| `forecast-solar`   | `15 * * * *`                  | Pull PV forecast → `mcp_forecasts`, publish `forecast.pv.*` |
| `forecast-weather` | `20 * * * *`                  | Pull Open-Meteo (ICON) forecast → `mcp_forecasts` |
| `energy-balance`   | `*/15 * * * *`                | Today's kWh counters → `energy.pv.*` |
| `detect-faults`    | `20 * * * *`                  | Run the fault list: resolve scope, measure, fold into episodes, reconcile, publish `anomaly.*` |

`detect-faults --dry-run` computes and logs everything and touches
neither the database nor NATS. One failing fault does not take the
others down; the job still exits non-zero so the CronJob shows it.

## Fault detection

### The fault file

Faults are declared, not coded. The list lives in lares beside the GA
catalog and the writer rules
(`kubernetes/applications/iot-insights-engine/base/config/faults.yaml`)
and is mounted into the CronJob at `MCP_FAULTS_FILE`. Every entry
carries:

- `sentence` and `unit` — the fault as one readable sentence. If the
  sentence cannot be written, it is not a detector.
- `kind` — one of the measurement kinds below; `drift` also names its
  `signal`, `deviation` may name an `expectation`.
- `parameters` — in the channel's own unit (mA, %, h, × the usual pause).
- `scope` — a catalog query (`name_like`), resolved against `ga_catalog`
  on every run. Never a hand-written address list. Channels that never
  sent are dropped where the scope resolves, logged, not reported.
- `target` — where the severity goes: one `ga`, or `per_main_group`,
  `per_device`, `per_room`. The addresses behind it are the bridge's
  writer rules. A per-device or per-room target may add `name`, the
  address's catalog-name template with `{entity}` standing for the
  entity as the fault's device or room map names it
  (`Raumklima.{entity}.FBH.Aktiv-Anomalie`); the engine validates it, the
  lares generator renders it into those rules.
- `dormant` — optional `reason` and `active_when`; the fault loads and
  validates but does not schedule.

The loader ([faults.py](src/iot_insights_engine/faults.py)) validates
the file against the bundled JSON Schema and freezes it into
dataclasses; a missing sentence, unit or parameter fails at load, naming
the fault and the field. Check an edit before shipping with
`task insights:validate-faults` in lares. Tuning a threshold is a
one-line PR there, not an engine release.

### Measurement kinds

| Kind         | Measures | Reports per |
|--------------|----------|-------------|
| `silence`    | A channel that used to send has been quiet longer than N× its own usual pause (a quantile of its own gaps). A channel new to the window is unproven — unmeasured, not alive — until it has shown enough gaps. | main group |
| `constancy`  | A channel keeps sending but has delivered the exact same value for longer than allowed: the producer works, the register behind it is dead. | main group |
| `duration`   | A device draws current for longer than its declared limit. | device |
| `drift`      | CUSUM against a healthy reference pinned in the entry, never derived from history. `signal` picks the series: `standby` (mA), `duty_cycle` (%), `recovery` (%, the one that walks downward). | device, or one GA |
| `deviation`  | A value sits too far under its reference: a room against its setpoint while a gate holds, or the plant's daily yield against a named `expectation` (`forecast_solar`). | room, or one GA |
| `volume`     | More than N episodes in seven days, over the engine's own episode stream. Declared last, so the count includes what this run just wrote. | one GA |
| `external`   | Basalte detects and delivers itself; the engine reads the severity writes back off the bus archive and only records. Nothing is published. | — |

There is no threshold kind: raw-value limits are Basalte's job.

### Episodes and severity

Repeated observations fold into one episode per fault and subject
(`episodes.fold_observations`, a pure function). An episode keeps its
per-bucket observations as evidence and notifies at most three times:
appeared, escalated, ended. It ends after a few quiet runs, never the
first, so a flickering fault stays one incident.

Severity is 0–3 (clear, info, warning, critical). Within an episode it
is the quantile of the fault's own score distribution, promoted one step
by duration; there is no global ladder. A stored severity is never
lowered.

Episodes (`episodes`, `episode_observations`, `episode_events` in TSDB,
written through the rw role) are the only stored state. Everything else
is recomputed from the last 30 days on every run, so a redeploy cannot
corrupt or lose it. Time is the aggregate's frontier — its newest
bucket, not the wall clock — so a stalled refresh freezes the picture
instead of clearing every open episode. A subject the run cannot measure
up to the frontier is dataless and its episode stays open. Every episode
carries the fingerprint of the rule that last made it — the fault's kind
and parameters — so a rule change does not strand rows on channels it
declines to judge: an open row an earlier rule left on a channel the
current rule holds unproven closes, and its group address is published
anew.

### Delivery

One publish per moved subject on `anomaly.<fault>[.<entity>]` with a
numeric `severity_level`. The knx-nats-bridge writer rules carry it to
the group address; Basalte owns the text and the channel: 3 pushes,
1–2 shows an indicator, 0 clears, and every situation gets an e-mail.
Publishes go out before the database write, so a failed run repeats the
same publish instead of losing it.

Not in this repo: verdicts (iot-mcp-bridge `set_episode_verdict` →
`episode_verdicts`; collected and shown, never acted on), the Grafana
dashboard, and the Basalte Studio faults — those appear here only as
`external` entries.

## Configuration

All `MCP_*` env vars (kept for compatibility with the existing
SealedSecret + Kyverno-clone topology shared with iot-mcp-bridge).
`detect-faults` additionally needs the write credentials
(`MCP_DB_WRITE_*`, episodes only), `MCP_FAULTS_FILE` and a NATS
identity for `anomaly.*`. See
[config.py](src/iot_insights_engine/config.py) for the full list.

## Local dev

```
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src
```

Tests follow the rebuild's seams: the fault loader (`test_faults`), each
measurement kind against invented fixtures (`test_silence`,
`test_constancy`, `test_duration`, `test_drift`, `test_deviation`,
`test_volume`, `test_external`), the episode pipeline (`test_episodes`,
`test_severity`, `test_reconcile`), and the runner lifecycle through fake
store and publisher ends (`test_runner`). Delivery is not tested here —
the writer rules are tested in the bridge repo.
