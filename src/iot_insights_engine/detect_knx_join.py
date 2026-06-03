"""Rule-based detectors that JOIN `knx_1h` × `ga_catalog` to score per
room. Two UCs:

* ``fbh_cold`` — FBH valve open AND room stays below setpoint for >=2h.
  Stuck valve / bled circuit / sensor mismatch.
* ``window_while_heating`` — window open AND FBH heating AND it's cold
  outside. Comfort + energy-waste pattern.

Both insert into ``mcp_anomalies`` idempotently (one row per
``(time, source, metric, detector)``) and publish on
``anomaly.<uc>.<severity>`` only on new inserts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import psycopg

from . import nats_publisher
from .config import Settings
from .db_write import write_connection
from .logging_setup import get_logger
from .registry import KNX_JOIN_USECASES

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


def _insert_anomaly(
    conn: psycopg.Connection[Any],
    *,
    bucket: Any,
    uc: str,
    room: str,
    severity: str,
    score: float,
    payload: dict[str, Any],
) -> bool:
    sql = """
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
        RETURNING xmax = 0 AS inserted
    """
    metric = f"{uc}[{room}]"
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (bucket, SOURCE, metric, DETECTOR_NAME, severity, uc, score, json.dumps(payload)),
        )
        result = cur.fetchone()
    return bool(result and result["inserted"])


def _publish(settings: Settings, uc: str, severity: str, payload: dict[str, Any]) -> None:
    try:
        nats_publisher.publish_anomaly(settings, uc=uc, severity=severity, payload=payload)
    except Exception:
        log.exception("nats_publish_failed", uc=uc)


# --- UC: fbh_cold ----------------------------------------------------------

_FBH_COLD_SQL = f"""
    WITH last_bucket AS (
        SELECT max(bucket) AS bucket FROM knx_1h
        WHERE bucket <= date_trunc('hour', now()) - interval '1 hour'
    ),
    per_room AS (
        SELECT c.room,
          avg(k.avg_value) FILTER (
            WHERE c.name LIKE '%%FBH.Stellwert-Status'
          ) AS stellwert_pct,
          avg(k.avg_value) FILTER (
            WHERE c.name LIKE '%%FBH.Soll-Temperatur-Status'
          ) AS soll_c,
          avg(k.avg_value) FILTER (
            WHERE c.name LIKE '%%Sensor.Temperatur' AND c.function = 'Sensorik'
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
    with conn.cursor() as cur:
        cur.execute(_FBH_COLD_SQL)
        rows = cur.fetchall()
    for row in rows:
        gap = float(row["gap_c"])
        severity = _classify_fbh(gap)
        if severity is None:
            continue
        payload = {
            "room": row["room"],
            "stellwert_pct": float(row["stellwert_pct"]),
            "soll_c": float(row["soll_c"]),
            "ist_c": float(row["ist_c"]),
            "gap_c": gap,
            "lookback_hours": FBH_LOOKBACK_HOURS,
            "bucket": row["bucket"].isoformat(),
        }
        ok = _insert_anomaly(
            conn,
            bucket=row["bucket"],
            uc="fbh_cold",
            room=str(row["room"]),
            severity=severity,
            score=gap,
            payload=payload,
        )
        if not ok:
            continue
        inserted += 1
        _publish(settings, "fbh_cold", severity, payload)
        published += 1
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
            WHERE c.name LIKE '%%Fenster%%Geöffnet-Status'
              AND c.name NOT LIKE '%%Stellung-%%'
              AND c.function = 'Sicherheit'
          ) AS window_open,
          avg(k.avg_value) FILTER (
            WHERE c.name LIKE '%%FBH.Stellwert-Status'
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
    with conn.cursor() as cur:
        cur.execute(_WINDOW_HEATING_SQL)
        rows = cur.fetchall()
    for row in rows:
        stellwert = float(row["stellwert_pct"])
        severity = _classify_window(stellwert)
        if severity is None:
            continue
        payload = {
            "room": row["room"],
            "window_open": bool(row["window_open"]),
            "stellwert_pct": stellwert,
            "outdoortemp_c": float(row["outdoortemp_c"]),
            "bucket": row["bucket"].isoformat(),
        }
        ok = _insert_anomaly(
            conn,
            bucket=row["bucket"],
            uc="window_while_heating",
            room=str(row["room"]),
            severity=severity,
            score=stellwert,
            payload=payload,
        )
        if not ok:
            continue
        inserted += 1
        _publish(settings, "window_while_heating", severity, payload)
        published += 1
    return inserted, published


# --- dispatcher ------------------------------------------------------------

_DETECTORS = {
    "fbh_cold": _detect_fbh_cold,
    "window_while_heating": _detect_window_while_heating,
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
