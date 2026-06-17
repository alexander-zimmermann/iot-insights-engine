"""Rule-based KNX detectors. Four UCs:

* ``fbh_cold`` — FBH valve open AND room stays below setpoint for >=2h.
  Stuck valve / bled circuit / sensor mismatch. JOINs `knx_1h` × `ga_catalog`.
* ``window_while_heating`` — window open AND FBH heating AND it's cold
  outside. Comfort + energy-waste pattern. JOINs `knx_1h` × `ga_catalog`.
* ``appliance_runtime`` — appliance drawing current above the standby valley
  for several consecutive hours ("left on"), gated to normally-idle loads.
  Reads `knx_appliance_1h` (no catalog join — the CAGG already scopes to
  `%Stromwert` channels).
* ``freezer_icing`` — the freezer compressor's median continuous run time at
  warm kitchen ambient creeps up over weeks as the evaporator frosts over
  (poor heat transfer → longer pull-down). Sessionises compressor runs from
  raw `knx` over a trailing window, keeps only warm-ambient runs, and compares
  their median against a fixed healthy baseline. Slow-drift detector — a
  rolling baseline would absorb the very creep we want to catch.

All insert into ``mcp_anomalies`` idempotently (one row per
``(time, source, metric, detector)``) and publish on
``anomaly.<uc>.<severity>`` only on new inserts or severity
escalations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import psycopg

from . import clear, nats_publisher
from .config import Settings
from .db_write import write_connection
from .logging_setup import get_logger
from .registry import KNX_JOIN_USECASES
from .severity import escalated

log = get_logger(__name__)

DETECTOR_NAME = "knx_join"
SOURCE = "knx_1h+ga_catalog"

# fbh_cold thresholds
FBH_STELLWERT_OPEN_PCT = 50.0
FBH_GAP_INFO_C = 1.0
FBH_GAP_WARNING_C = 2.0
FBH_GAP_CRITICAL_C = 3.0
FBH_LOOKBACK_HOURS = 2

# window_while_heating thresholds
WIN_STELLWERT_INFO_PCT = 0.0
WIN_STELLWERT_WARNING_PCT = 25.0
WIN_STELLWERT_CRITICAL_PCT = 50.0
WIN_OUTDOORTEMP_MAX_C = 12.0

# appliance_runtime thresholds. An hour counts as "active" when at least half
# its samples sit above the standby valley (the `on_samples` FILTER threshold
# baked into knx_appliance_1h). "Left on" escalates with the trailing run of
# consecutive active hours ending at the current bucket.
APPLIANCE_ACTIVE_HOUR_FRACTION = 0.5
APPLIANCE_RUNTIME_LOOKBACK_HOURS = 24
APPLIANCE_PROFILE_DAYS = 30
APPLIANCE_RUNTIME_INFO_HOURS = 3
APPLIANCE_RUNTIME_WARNING_HOURS = 6
APPLIANCE_RUNTIME_CRITICAL_HOURS = 9
# Appliances active in more than this fraction of the last 30d of hours are
# treated as always-on (fridge, network rack, circulation pump) and skipped —
# "left on" is meaningless for a load that is on by design.
APPLIANCE_NORMALLY_IDLE_MAX_RATE = 0.5

# freezer_icing. Channels are resolved by their stable `ga_catalog` *name*, not
# by GA — a GA can be re-addressed in ETS, the semantic name does not move
# (same convention as fbh_cold / window_while_heating). A compressor "run" is a
# continuous stretch with current above the standby valley; the valley sits at
# ~50-67 mA and running draw at 180-580 mA, so 120 mA cleanly separates the two
# (same cut as the real-time KNX "door open" block). Only runs that *started*
# while the kitchen was warm (>= 24 °C) count — below that, iced and healthy run
# lengths are indistinguishable, so the signal only exists under thermal load.
# Runs longer than the door-event floor are dropped: those are door-ajar /
# defrost outliers (the real-time block owns them) and would skew the median.
FREEZER_NAME_PATTERN = "%Gefrierschrank.Stromwert"
KITCHEN_TEMP_NAME_PATTERN = "%Küche.Sensor.Temperatur"
FREEZER_RUN_THRESHOLD_MA = 120.0
FREEZER_WARM_AMBIENT_C = 24.0
FREEZER_ICING_LOOKBACK_DAYS = 21
FREEZER_ICING_DOOR_EVENT_MIN = 180.0
# Need enough warm runs in the window for a stable median; in winter the
# kitchen rarely reaches 24 °C, so the detector simply goes quiet — correct,
# since there is no thermal load to expose icing then.
FREEZER_ICING_MIN_WARM_RUNS = 15
# Healthy warm-ambient median continuous run ≈ 30 min (a full known-good year,
# door events excluded). Thresholds bracket the observed iced state (≈ 46 min).
# Reconfirm against the post-defrost baseline and retune if needed.
FREEZER_ICING_BASELINE_MIN = 30.0
FREEZER_ICING_INFO_MIN = 38.0
FREEZER_ICING_WARNING_MIN = 45.0
FREEZER_ICING_CRITICAL_MIN = 55.0


def _classify_fbh(gap_c: float) -> str | None:
    if gap_c >= FBH_GAP_CRITICAL_C:
        return "critical"
    if gap_c >= FBH_GAP_WARNING_C:
        return "warning"
    if gap_c >= FBH_GAP_INFO_C:
        return "info"
    return None


def _classify_window(stellwert_pct: float) -> str | None:
    if stellwert_pct >= WIN_STELLWERT_CRITICAL_PCT:
        return "critical"
    if stellwert_pct >= WIN_STELLWERT_WARNING_PCT:
        return "warning"
    if stellwert_pct > WIN_STELLWERT_INFO_PCT:
        return "info"
    return None


def _classify_runtime(streak_hours: int) -> str | None:
    if streak_hours >= APPLIANCE_RUNTIME_CRITICAL_HOURS:
        return "critical"
    if streak_hours >= APPLIANCE_RUNTIME_WARNING_HOURS:
        return "warning"
    if streak_hours >= APPLIANCE_RUNTIME_INFO_HOURS:
        return "info"
    return None


def _classify_icing(median_run_min: float) -> str | None:
    if median_run_min >= FREEZER_ICING_CRITICAL_MIN:
        return "critical"
    if median_run_min >= FREEZER_ICING_WARNING_MIN:
        return "warning"
    if median_run_min >= FREEZER_ICING_INFO_MIN:
        return "info"
    return None


def _trailing_active_streak(rows_desc: list[tuple[Any, bool]]) -> int:
    """Count consecutive active, hourly-contiguous buckets from the newest.

    ``rows_desc`` is ``(bucket, is_active)`` ordered newest → oldest. The run
    ends at the first inactive hour or the first gap (buckets not exactly one
    hour apart) — a missing hour breaks "left on" rather than bridging it.
    """
    streak = 0
    prev_bucket = None
    for bucket, is_active in rows_desc:
        if prev_bucket is not None and (prev_bucket - bucket) != timedelta(hours=1):
            break
        if not is_active:
            break
        streak += 1
        prev_bucket = bucket
    return streak


def _insert_anomaly(
    conn: psycopg.Connection[Any],
    *,
    bucket: Any,
    uc: str,
    room: str,
    severity: str,
    score: float,
    payload: dict[str, Any],
    source: str = SOURCE,
) -> tuple[bool, str | None]:
    """Returns (inserted, old_severity) — old_severity is the
    pre-statement value on conflict (NULL on fresh insert) so the
    caller can re-publish on escalation."""
    sql = """
        WITH existing AS (
            SELECT severity FROM mcp_anomalies
            WHERE time = %s AND source = %s AND metric = %s AND detector = %s
        )
        INSERT INTO mcp_anomalies (
            time, source, metric, detector, severity, uc,
            actual, expected, score, payload
        )
        VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL, %s, %s::jsonb)
        ON CONFLICT (time, source, metric, detector) DO UPDATE
        SET severity = EXCLUDED.severity,
            score    = EXCLUDED.score,
            payload  = EXCLUDED.payload,
            uc       = EXCLUDED.uc
        RETURNING xmax = 0 AS inserted,
                  (SELECT severity FROM existing) AS old_severity
    """
    metric = f"{uc}[{room}]"
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                bucket,
                source,
                metric,
                DETECTOR_NAME,
                bucket,
                source,
                metric,
                DETECTOR_NAME,
                severity,
                uc,
                score,
                json.dumps(payload),
            ),
        )
        result = cur.fetchone()
    if not result:
        return False, None
    return bool(result["inserted"]), result["old_severity"]


def _publish(
    settings: Settings,
    uc: str,
    severity: str,
    payload: dict[str, Any],
    *,
    entity: str | None = None,
) -> None:
    try:
        nats_publisher.publish_anomaly(
            settings, uc=uc, severity=severity, payload=payload, entity=entity
        )
    except Exception:
        log.exception("nats_publish_failed", uc=uc)


# --- UC: fbh_cold ----------------------------------------------------------

# Both SQL constants are executed WITHOUT bind parameters, so `%` is a
# plain SQL LIKE wildcard here. If you ever add bind params, escape the
# wildcards as `%%` — psycopg only treats `%` specially when params are
# passed.
_FBH_COLD_SQL = f"""
    WITH last_bucket AS (
        SELECT max(bucket) AS bucket FROM knx_1h
        WHERE bucket <= date_trunc('hour', now()) - interval '1 hour'
    ),
    per_room AS (
        SELECT c.room,
          avg(k.avg_value) FILTER (
            WHERE c.name LIKE '%FBH.Stellwert-Status'
          ) AS stellwert_pct,
          avg(k.avg_value) FILTER (
            WHERE c.name LIKE '%FBH.Soll-Temperatur-Status'
          ) AS soll_c,
          avg(k.avg_value) FILTER (
            WHERE c.name LIKE '%Sensor.Temperatur' AND c.function = 'Sensorik'
          ) AS ist_c
        FROM knx_1h k
        JOIN ga_catalog c ON c.ga = k.ga
        WHERE k.bucket > (SELECT bucket FROM last_bucket) - interval '{FBH_LOOKBACK_HOURS} hours'
          AND k.bucket <= (SELECT bucket FROM last_bucket)
          AND c.room IS NOT NULL
        GROUP BY c.room
    )
    SELECT
        (SELECT bucket FROM last_bucket) AS bucket,
        room, stellwert_pct, soll_c, ist_c,
        (soll_c - ist_c) AS gap_c
    FROM per_room
    WHERE stellwert_pct IS NOT NULL
      AND soll_c IS NOT NULL
      AND ist_c IS NOT NULL
      AND stellwert_pct > {FBH_STELLWERT_OPEN_PCT}
      AND (soll_c - ist_c) > {FBH_GAP_INFO_C}
"""


def _detect_fbh_cold(
    conn: psycopg.Connection[Any], settings: Settings
) -> tuple[int, int]:
    inserted = published = 0
    fired: set[str | None] = set()
    with conn.cursor() as cur:
        cur.execute(_FBH_COLD_SQL)
        rows = cur.fetchall()
    for row in rows:
        gap = float(row["gap_c"])
        severity = _classify_fbh(gap)
        if severity is None:
            continue
        entity = nats_publisher.slugify(str(row["room"]))
        fired.add(entity)
        payload = {
            "entity": entity,
            "room": row["room"],
            "stellwert_pct": float(row["stellwert_pct"]),
            "soll_c": float(row["soll_c"]),
            "ist_c": float(row["ist_c"]),
            "gap_c": gap,
            "lookback_hours": FBH_LOOKBACK_HOURS,
            "bucket": row["bucket"].isoformat(),
        }
        ok, old_severity = _insert_anomaly(
            conn,
            bucket=row["bucket"],
            uc="fbh_cold",
            room=str(row["room"]),
            severity=severity,
            score=gap,
            payload=payload,
        )
        if ok or escalated(old_severity, severity):
            _publish(settings, "fbh_cold", severity, payload, entity=entity)
            published += 1
        if ok:
            inserted += 1
    clear.publish_clears(conn, settings, uc="fbh_cold", fired_entities=fired)
    return inserted, published


# --- UC: window_while_heating ---------------------------------------------

_WINDOW_HEATING_SQL = f"""
    WITH last_bucket AS (
        SELECT max(bucket) AS bucket FROM knx_1h
        WHERE bucket <= date_trunc('hour', now()) - interval '1 hour'
    ),
    outdoor AS (
        SELECT outdoortemp_avg FROM ems_esp_boiler_1h
        WHERE bucket = (
            SELECT max(bucket) FROM ems_esp_boiler_1h
            WHERE bucket <= date_trunc('hour', now()) - interval '1 hour'
        )
    ),
    per_room AS (
        SELECT c.room,
          max(k.avg_value) FILTER (
            WHERE c.name LIKE '%Fenster%Geöffnet-Status'
              AND c.name NOT LIKE '%Stellung-%'
              AND c.function = 'Sicherheit'
          ) AS window_open,
          avg(k.avg_value) FILTER (
            WHERE c.name LIKE '%FBH.Stellwert-Status'
          ) AS stellwert_pct
        FROM knx_1h k
        JOIN ga_catalog c ON c.ga = k.ga
        WHERE k.bucket = (SELECT bucket FROM last_bucket)
          AND c.room IS NOT NULL
        GROUP BY c.room
    )
    SELECT
        (SELECT bucket FROM last_bucket) AS bucket,
        p.room, p.window_open, p.stellwert_pct,
        (SELECT outdoortemp_avg FROM outdoor) AS outdoortemp_c
    FROM per_room p
    WHERE p.window_open IS NOT NULL AND p.window_open > 0
      AND p.stellwert_pct IS NOT NULL AND p.stellwert_pct > {WIN_STELLWERT_INFO_PCT}
      AND (SELECT outdoortemp_avg FROM outdoor) IS NOT NULL
      AND (SELECT outdoortemp_avg FROM outdoor) < {WIN_OUTDOORTEMP_MAX_C}
"""


def _detect_window_while_heating(
    conn: psycopg.Connection[Any], settings: Settings
) -> tuple[int, int]:
    inserted = published = 0
    fired: set[str | None] = set()
    with conn.cursor() as cur:
        cur.execute(_WINDOW_HEATING_SQL)
        rows = cur.fetchall()
    for row in rows:
        stellwert = float(row["stellwert_pct"])
        severity = _classify_window(stellwert)
        if severity is None:
            continue
        entity = nats_publisher.slugify(str(row["room"]))
        fired.add(entity)
        payload = {
            "entity": entity,
            "room": row["room"],
            "window_open": bool(row["window_open"]),
            "stellwert_pct": stellwert,
            "outdoortemp_c": float(row["outdoortemp_c"]),
            "bucket": row["bucket"].isoformat(),
        }
        ok, old_severity = _insert_anomaly(
            conn,
            bucket=row["bucket"],
            uc="window_while_heating",
            room=str(row["room"]),
            severity=severity,
            score=stellwert,
            payload=payload,
        )
        if ok or escalated(old_severity, severity):
            _publish(settings, "window_while_heating", severity, payload, entity=entity)
            published += 1
        if ok:
            inserted += 1
    clear.publish_clears(conn, settings, uc="window_while_heating", fired_entities=fired)
    return inserted, published


# --- UC: appliance_runtime -------------------------------------------------

_APPLIANCE_LAST_BUCKET_SQL = """
    SELECT max(bucket) AS bucket FROM knx_appliance_1h
    WHERE bucket <= date_trunc('hour', now()) - interval '1 hour'
"""

# Recent hours per appliance, newest first — the trailing run is walked in
# Python (clearer than a gaps-and-islands window for a 24h slice).
_APPLIANCE_RECENT_SQL = f"""
    WITH lb AS ({_APPLIANCE_LAST_BUCKET_SQL})
    SELECT ga, knx_name, bucket, on_samples, total_samples
    FROM knx_appliance_1h
    WHERE bucket >  (SELECT bucket FROM lb)
                    - interval '{APPLIANCE_RUNTIME_LOOKBACK_HOURS} hours'
      AND bucket <= (SELECT bucket FROM lb)
    ORDER BY ga, knx_name, bucket DESC
"""

# 30d active-rate per appliance — the always-on gate.
_APPLIANCE_PROFILE_SQL = f"""
    WITH lb AS ({_APPLIANCE_LAST_BUCKET_SQL})
    SELECT ga, knx_name,
      avg((total_samples > 0
           AND on_samples::float / NULLIF(total_samples, 0)
               >= {APPLIANCE_ACTIVE_HOUR_FRACTION})::int::float) AS active_rate
    FROM knx_appliance_1h
    WHERE bucket >  (SELECT bucket FROM lb)
                    - interval '{APPLIANCE_PROFILE_DAYS} days'
      AND bucket <= (SELECT bucket FROM lb)
    GROUP BY ga, knx_name
"""


def _hour_is_active(on_samples: int | None, total_samples: int | None) -> bool:
    if not total_samples or on_samples is None:
        return False
    return (on_samples / total_samples) >= APPLIANCE_ACTIVE_HOUR_FRACTION


def _detect_appliance_runtime(
    conn: psycopg.Connection[Any], settings: Settings
) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute(_APPLIANCE_LAST_BUCKET_SQL)
        lb_row = cur.fetchone()
        last_bucket = lb_row["bucket"] if lb_row else None
        if last_bucket is None:
            return 0, 0
        cur.execute(_APPLIANCE_PROFILE_SQL)
        active_rate = {
            (r["ga"], r["knx_name"]): float(r["active_rate"])
            for r in cur.fetchall()
            if r["active_rate"] is not None
        }
        cur.execute(_APPLIANCE_RECENT_SQL)
        recent = cur.fetchall()

    # Group the (already bucket-DESC ordered) recent rows per appliance.
    by_appliance: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in recent:
        by_appliance.setdefault((row["ga"], row["knx_name"]), []).append(row)

    inserted = published = 0
    fired: set[str | None] = set()
    for (ga, knx_name), rows in by_appliance.items():
        # Must be active in the current bucket to count as "left on".
        if rows[0]["bucket"] != last_bucket:
            continue
        # Always-on loads (high active-rate) are on by design — skip.
        rate = active_rate.get((ga, knx_name))
        if rate is None or rate >= APPLIANCE_NORMALLY_IDLE_MAX_RATE:
            continue
        streak = _trailing_active_streak(
            [(r["bucket"], _hour_is_active(r["on_samples"], r["total_samples"])) for r in rows]
        )
        severity = _classify_runtime(streak)
        if severity is None:
            continue
        entity = nats_publisher.slugify(ga)
        fired.add(entity)
        payload = {
            "entity": entity,
            "ga": ga,
            "knx_name": knx_name,
            "streak_hours": streak,
            "active_rate_30d": round(rate, 3),
            "bucket": last_bucket.isoformat(),
        }
        ok, old_severity = _insert_anomaly(
            conn,
            bucket=last_bucket,
            uc="appliance_runtime",
            room=f"{ga},{knx_name}",
            severity=severity,
            score=float(streak),
            payload=payload,
            source="knx_appliance_1h",
        )
        if ok or escalated(old_severity, severity):
            _publish(settings, "appliance_runtime", severity, payload, entity=entity)
            published += 1
        if ok:
            inserted += 1
    clear.publish_clears(conn, settings, uc="appliance_runtime", fired_entities=fired)
    return inserted, published


# --- UC: freezer_icing -----------------------------------------------------

# No bind params → `%` is a plain LIKE wildcard (see the fbh_cold note). GAs
# are resolved by name from ga_catalog so a re-addressed channel keeps working;
# an unresolved name yields `ga = NULL` → zero rows → the detector goes quiet
# rather than crashing. The gaps-and-islands CTEs sessionise compressor runs
# from raw knx, then the median is taken over warm-ambient, non-door runs.
_FREEZER_ICING_SQL = f"""
    WITH freezer AS (
        SELECT ga, name AS knx_name FROM ga_catalog
        WHERE name LIKE '{FREEZER_NAME_PATTERN}'
        LIMIT 1
    ),
    kitchen_temp AS (
        SELECT ga FROM ga_catalog
        WHERE name LIKE '{KITCHEN_TEMP_NAME_PATTERN}' AND function = 'Sensorik'
        LIMIT 1
    ),
    samples AS (
        SELECT time,
               (value >= {FREEZER_RUN_THRESHOLD_MA}) AS running,
               lead(time) OVER (ORDER BY time) AS next_time
        FROM knx
        WHERE ga = (SELECT ga FROM freezer)
          AND time > now() - interval '{FREEZER_ICING_LOOKBACK_DAYS} days'
    ),
    marked AS (
        SELECT *,
               CASE
                 WHEN running
                      AND NOT coalesce(lag(running) OVER (ORDER BY time), false)
                 THEN 1 ELSE 0
               END AS new_run
        FROM samples
    ),
    grouped AS (
        SELECT *, sum(new_run) OVER (ORDER BY time ROWS UNBOUNDED PRECEDING) AS run_id
        FROM marked
    ),
    runs AS (
        SELECT min(time) AS started_at,
               extract(epoch FROM (max(next_time) - min(time))) / 60.0 AS dur_min
        FROM grouped
        WHERE running
        GROUP BY run_id
    ),
    warm_runs AS (
        SELECT r.dur_min
        FROM runs r
        WHERE r.dur_min < {FREEZER_ICING_DOOR_EVENT_MIN}
          AND (
              SELECT t.value FROM knx t
              WHERE t.ga = (SELECT ga FROM kitchen_temp)
                AND t.time <= r.started_at
              ORDER BY t.time DESC
              LIMIT 1
          ) >= {FREEZER_WARM_AMBIENT_C}
    )
    SELECT
        date_trunc('day', now()) AS bucket,
        (SELECT ga FROM freezer) AS ga,
        (SELECT knx_name FROM freezer) AS knx_name,
        count(*) AS warm_run_count,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY dur_min) AS median_run_min
    FROM warm_runs
"""


def _detect_freezer_icing(
    conn: psycopg.Connection[Any], settings: Settings
) -> tuple[int, int]:
    # 1:1 UC → entity is None (subject `anomaly.freezer_icing`).
    fired: set[str | None] = set()
    inserted = published = 0
    with conn.cursor() as cur:
        cur.execute(_FREEZER_ICING_SQL)
        row = cur.fetchone()

    severity: str | None = None
    # Too few warm runs (winter / unresolved channel) → not enough signal.
    if (
        row is not None
        and row["median_run_min"] is not None
        and int(row["warm_run_count"]) >= FREEZER_ICING_MIN_WARM_RUNS
    ):
        severity = _classify_icing(float(row["median_run_min"]))

    if severity is not None and row is not None:
        fired.add(None)
        median_run_min = float(row["median_run_min"])
        payload = {
            "entity": None,
            "ga": row["ga"],
            "knx_name": row["knx_name"],
            "median_run_min": round(median_run_min, 1),
            "baseline_run_min": FREEZER_ICING_BASELINE_MIN,
            "warm_run_count": int(row["warm_run_count"]),
            "warm_ambient_c": FREEZER_WARM_AMBIENT_C,
            "lookback_days": FREEZER_ICING_LOOKBACK_DAYS,
            "bucket": row["bucket"].isoformat(),
        }
        ok, old_severity = _insert_anomaly(
            conn,
            bucket=row["bucket"],
            uc="freezer_icing",
            room=f"{row['ga']},{row['knx_name']}",
            severity=severity,
            score=median_run_min,
            payload=payload,
            source="knx",
        )
        if ok or escalated(old_severity, severity):
            _publish(settings, "freezer_icing", severity, payload)
            published = 1
        inserted = 1 if ok else 0

    clear.publish_clears(conn, settings, uc="freezer_icing", fired_entities=fired)
    return inserted, published


# --- dispatcher ------------------------------------------------------------

_DETECTORS = {
    "fbh_cold": _detect_fbh_cold,
    "window_while_heating": _detect_window_while_heating,
    "appliance_runtime": _detect_appliance_runtime,
    "freezer_icing": _detect_freezer_icing,
}


def run(settings: Settings, _argv: Sequence[str]) -> int:
    total_inserted = 0
    total_published = 0
    with write_connection(settings) as conn:
        for uc in KNX_JOIN_USECASES:
            if uc.silenced:
                log.info("uc_silenced", uc=uc.uc)
                continue
            fn = _DETECTORS.get(uc.uc)
            if fn is None:
                log.error("uc_not_implemented", uc=uc.uc)
                continue
            try:
                ins, pub = fn(conn, settings)
            except psycopg.Error:
                log.exception("uc_failed", uc=uc.uc)
                continue
            total_inserted += ins
            total_published += pub
    log.info(
        "detect_knx_join_done",
        scanned_ucs=len(KNX_JOIN_USECASES),
        inserted=total_inserted,
        published=total_published,
    )
    return 0
