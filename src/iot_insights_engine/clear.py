"""Auto-clear: drop an anomaly GA back to 0 once its entity stops firing.

Anomalies re-insert every run while active. When an entity that was open for a
UC is absent from the current run, `publish_clears` emits `severity_level=0`
(firing=false) on its subject — so the KNX-GA falls back to 0 — and writes a
clear-marker row (`payload.firing = "false"`) so the next run won't re-clear it
(no NATS spam). The `severity` CHECK forbids a 'cleared' value, hence the
marker lives in the payload, reusing the last valid severity for the column.

Requires `entity` in the stored `mcp_anomalies.payload`; rows without it (older
than this change) are ignored.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import psycopg

from . import nats_publisher
from .config import Settings
from .logging_setup import get_logger

log = get_logger(__name__)

# Lookback for entities that may still need clearing — must exceed any
# detector's re-fire cadence so a just-stopped anomaly is still caught.
_CLEAR_LOOKBACK = "7 days"

# Latest row per (uc, entity), so we can tell a still-open anomaly from one
# already cleared (payload.firing = 'false').
_OPEN_SQL = f"""
    SELECT DISTINCT ON (payload->>'entity')
           payload->>'entity'              AS entity,
           source, metric, detector, severity,
           coalesce(payload->>'firing', 'true') AS firing
    FROM mcp_anomalies
    WHERE uc = %s
      AND time > now() - interval '{_CLEAR_LOOKBACK}'
      AND payload ? 'entity'
    ORDER BY payload->>'entity', time DESC
"""

_MARK_SQL = """
    INSERT INTO mcp_anomalies (time, source, metric, detector, severity, uc, score, payload)
    VALUES (now(), %s, %s, %s, %s, %s, 0, %s::jsonb)
    ON CONFLICT (time, source, metric, detector) DO NOTHING
"""


def publish_clears(
    conn: psycopg.Connection[Any],
    settings: Settings,
    *,
    uc: str,
    fired_entities: Iterable[str | None],
) -> int:
    """Clear every entity that was open for `uc` but did not fire this run.
    Returns the number of clears published."""
    fired = set(fired_entities)
    with conn.cursor() as cur:
        cur.execute(_OPEN_SQL, (uc,))
        rows = cur.fetchall()

    cleared = 0
    for row in rows:
        entity = row["entity"]
        if entity in fired or row["firing"] == "false":
            continue
        try:
            nats_publisher.publish_anomaly(
                settings,
                uc=uc,
                severity=row["severity"],
                payload={"clear": True},
                entity=entity,
                firing=False,
            )
        except Exception:
            log.exception("clear_publish_failed", uc=uc, entity=entity)
            continue
        with conn.cursor() as cur:
            cur.execute(
                _MARK_SQL,
                (
                    row["source"],
                    row["metric"],
                    row["detector"],
                    row["severity"],
                    uc,
                    json.dumps({"entity": entity, "firing": False, "clear": True}),
                ),
            )
        cleared += 1
    if cleared:
        log.info("anomaly_clears", uc=uc, cleared=cleared)
    return cleared
