"""Daily PV energy balance → NATS ``energy.pv.*`` for the KNX bilanz GAs (15/4/60-65).

Computes today's energy balance from **authoritative meter counters**, not the
SolarEdge powerflow grid/consumer fields — those stay invalid until the SolarEdge
meter goes live with the battery + 400 V conversion (export reads 0, the balance
does not close). Sources that work today:

  - generation  = sum of inverter ``energytotal`` deltas (``solaredge_inverter``, Wh→kWh)
  - grid_import = KNX energy meter ``15/1/0`` counter delta (``knx`` table, kWh)
  - grid_export = KNX energy meter ``15/1/7`` counter delta (``knx`` table, kWh)

Derived: consumption, direct self-use, self-consumption/self-sufficiency
(= direct use until the battery is online) and the two ratios. Counter deltas
(last-first over today) are exact and cheap — deliberately **no** ``time_weight``
integral, which OOMs the 1 GiB TSDB.

Published as ``energy.pv.<key>`` ``{"value": …}``; the knx-nats-bridge writer-rules
map them onto KNX 15/4/60-65 (Erzeugung-Energie, Verbrauch-Energie, Eigenverbrauch,
Eigenversorgung, Eigenverbrauchsquote, Eigenversorgungsquote). Best-effort publish —
a NATS outage must not fail the run. When the SolarEdge meter / battery arrive, only
the source query here changes; the published keys and KNX GAs stay the same.
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

SUBJECT_PREFIX = "energy.pv"

# KNX energy-meter (15/1) cumulative readings (DPT 13.013, kWh).
GA_GRID_IMPORT = "15/1/0"  # Netzbezug.Zählerstand-Gesamt
GA_GRID_EXPORT = "15/1/7"  # Einspeisung.Zählerstand-Gesamt-1 (…/8 is the negated mirror)


def _today_deltas(conn: psycopg.Connection[Any], tz: str) -> tuple[float, float, float]:
    """Return (generation_kwh, grid_import_kwh, grid_export_kwh) for today.

    "Today" = since local midnight in ``tz``. Each value is the last-first delta
    of a cumulative counter over the day, floored at 0 (guards a meter reset).
    """
    with conn.cursor() as cur:
        # SET TIME ZONE takes no bind params ("SET TIME ZONE $1" is a syntax
        # error) — set_config() is the parameterised equivalent.
        cur.execute("SELECT set_config('timezone', %s, false)", (tz,))

        # Generation: inverter energytotal is a lifetime Wh counter, per inverter_id.
        # Delta per inverter, then sum (can't nest sum(last(...))).
        cur.execute("""
            WITH per AS (
                SELECT inverter_id,
                       last(energytotal, time) - first(energytotal, time) AS wh
                FROM solaredge_inverter
                WHERE time >= date_trunc('day', now())
                GROUP BY inverter_id
            )
            SELECT COALESCE(sum(GREATEST(wh, 0)), 0) / 1000.0 AS kwh FROM per
        """)
        gen_row = cur.fetchone()
        generation = float((gen_row["kwh"] if gen_row else 0.0) or 0.0)

        # Grid import / export: KNX energy-meter kWh counters.
        cur.execute(
            """
            WITH d AS (
                SELECT ga, GREATEST(last(value, time) - first(value, time), 0) AS delta
                FROM knx
                WHERE ga IN (%s, %s) AND time >= date_trunc('day', now())
                GROUP BY ga
            )
            SELECT
                COALESCE(max(delta) FILTER (WHERE ga = %s), 0) AS grid_import,
                COALESCE(max(delta) FILTER (WHERE ga = %s), 0) AS grid_export
            FROM d
            """,
            (GA_GRID_IMPORT, GA_GRID_EXPORT, GA_GRID_IMPORT, GA_GRID_EXPORT),
        )
        row = cur.fetchone() or {"grid_import": 0.0, "grid_export": 0.0}
        return generation, float(row["grid_import"] or 0.0), float(row["grid_export"] or 0.0)


def _compute(generation: float, grid_import: float, grid_export: float) -> dict[str, float]:
    """Pure energy-balance math (kWh in, kWh + % out). Battery terms are 0 until
    the storage unit is online; self-consumption/self-sufficiency then collapse to
    the direct self-use (generation - grid_export)."""
    consumption = generation + grid_import - grid_export
    direct_use = max(0.0, generation - grid_export)
    self_consumption = direct_use  # + battery charge, once the battery is online
    self_sufficiency = direct_use  # + battery discharge, once the battery is online

    def _rate(numerator: float, denominator: float) -> float:
        return max(0.0, min(100.0, 100.0 * numerator / denominator)) if denominator > 0 else 0.0

    return {
        "generation_kwh": round(generation, 2),
        "consumption_kwh": round(max(0.0, consumption), 2),
        "self_consumption_kwh": round(self_consumption, 2),
        "self_sufficiency_kwh": round(self_sufficiency, 2),
        "self_consumption_rate": round(_rate(self_consumption, generation), 1),
        "self_sufficiency_rate": round(_rate(self_sufficiency, consumption), 1),
    }


def _publish(settings: Settings, values: dict[str, float]) -> None:
    """Best-effort publish of each value to ``energy.pv.<key>``. A NATS outage
    (or NATS not configured) must not fail the run — the data is re-derivable."""
    if not settings.nats_servers:
        log.info("energy_balance_nats_skip", reason="MCP_NATS_SERVERS not set")
        return
    for key, value in values.items():
        subject = f"{SUBJECT_PREFIX}.{key}"
        try:
            nats_publisher.publish(settings, subject, {"value": value})
        except Exception:
            log.exception("energy_balance_publish_failed", subject=subject)


def run(settings: Settings, _argv: Sequence[str]) -> int:
    with read_connection(settings) as conn:
        generation, grid_import, grid_export = _today_deltas(conn, settings.energy_timezone)
    values = _compute(generation, grid_import, grid_export)
    _publish(settings, values)
    log.info(
        "energy_balance_done",
        grid_import_kwh=round(grid_import, 2),
        grid_export_kwh=round(grid_export, 2),
        **values,
    )
    return 0
