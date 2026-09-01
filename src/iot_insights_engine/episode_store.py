"""Episode persistence: the thin edge between the pure pipeline and the
episodes tables.

Everything here is idempotent — evidence and events land with ON CONFLICT
DO NOTHING on their natural keys, and the row updates only ever raise
severity and peak, so a rerun after a half-applied failure converges
instead of duplicating.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import psycopg
from psycopg.rows import DictRow

from .episodes import Episode, EventKind

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class OpenEpisodeRow:
    """The open episode the database holds for one fault and subject."""

    id: int
    subject: str
    severity: int


def open_rows(conn: psycopg.Connection[DictRow], fault_name: str) -> list[OpenEpisodeRow]:
    rows = conn.execute(
        "SELECT id, subject, severity FROM episodes"
        " WHERE fault = %(fault)s AND ended_at IS NULL",
        {"fault": fault_name},
    ).fetchall()
    return [OpenEpisodeRow(id=r["id"], subject=r["subject"], severity=r["severity"]) for r in rows]


def history_scores(conn: psycopg.Connection[DictRow], fault_name: str) -> list[float]:
    # Folded episodes are the imported detector era — a different score
    # scale, never this fault's own distribution.
    rows = conn.execute(
        """
        SELECT o.score FROM episode_observations o
        JOIN episodes e ON e.id = o.episode_id
        WHERE e.fault = %(fault)s AND NOT e.folded
        """,
        {"fault": fault_name},
    ).fetchall()
    return [r["score"] for r in rows]


def apply(
    conn: psycopg.Connection[DictRow],
    fault_name: str,
    inserts: Sequence[Episode],
    updates: Sequence[tuple[int, Episode]],
    orphan_closes: Sequence[tuple[int, datetime]],
) -> None:
    for episode in inserts:
        inserted = conn.execute(
            """
            INSERT INTO episodes (fault, subject, started_at, last_seen_at,
                                  ended_at, severity, peak_score)
            VALUES (%(fault)s, %(subject)s, %(started_at)s, %(last_seen_at)s,
                    %(ended_at)s, %(severity)s, %(peak_score)s)
            RETURNING id
            """,
            {
                "fault": fault_name,
                "subject": episode.subject,
                "started_at": episode.started_at,
                "last_seen_at": episode.last_seen_at,
                "ended_at": episode.ended_at,
                "severity": episode.severity,
                "peak_score": episode.peak_score,
            },
        ).fetchone()
        if inserted is None:  # INSERT … RETURNING always yields the row
            raise RuntimeError(f"episode insert for {episode.subject} returned no id")
        _write_details(conn, inserted["id"], episode)
    for episode_id, episode in updates:
        conn.execute(
            """
            UPDATE episodes
            SET last_seen_at = GREATEST(last_seen_at, %(last_seen_at)s),
                severity = GREATEST(severity, %(severity)s),
                peak_score = GREATEST(peak_score, %(peak_score)s),
                ended_at = %(ended_at)s
            WHERE id = %(id)s
            """,
            {
                "id": episode_id,
                "last_seen_at": episode.last_seen_at,
                "severity": episode.severity,
                "peak_score": episode.peak_score,
                "ended_at": episode.ended_at,
            },
        )
        _write_details(conn, episode_id, episode)
    for episode_id, ended_at in orphan_closes:
        conn.execute(
            "UPDATE episodes SET ended_at = %(ended_at)s WHERE id = %(id)s",
            {"id": episode_id, "ended_at": ended_at},
        )
        conn.execute(
            """
            INSERT INTO episode_events (episode_id, kind, time, severity)
            VALUES (%(id)s, %(kind)s, %(time)s, 0)
            ON CONFLICT (episode_id, kind) DO NOTHING
            """,
            {"id": episode_id, "kind": EventKind.ENDED.value, "time": ended_at},
        )


def _write_details(
    conn: psycopg.Connection[DictRow], episode_id: int, episode: Episode
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO episode_observations (episode_id, time, score, severity, value)
            VALUES (%(id)s, %(time)s, %(score)s, %(severity)s, %(value)s)
            ON CONFLICT (episode_id, time) DO NOTHING
            """,
            [
                {
                    "id": episode_id,
                    "time": row.time,
                    "score": row.score,
                    "severity": row.severity,
                    "value": row.value,
                }
                for row in episode.evidence
            ],
        )
        cur.executemany(
            """
            INSERT INTO episode_events (episode_id, kind, time, severity)
            VALUES (%(id)s, %(kind)s, %(time)s, %(severity)s)
            ON CONFLICT (episode_id, kind) DO NOTHING
            """,
            [
                {
                    "id": episode_id,
                    "kind": event.kind.value,
                    "time": event.time,
                    "severity": event.severity,
                }
                for event in episode.events
            ],
        )
