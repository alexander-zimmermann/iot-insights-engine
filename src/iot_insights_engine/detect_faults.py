"""Detect-faults job: run the declared fault list end to end.

For each schedulable fault the runner resolves the scope where the catalog
lives (`ga_catalog` in TSDB — never a hand-written address list), measures,
folds the observations into episodes behind the pure pipeline seam, and
reconciles the result with the episodes the database already holds. What
leaves the engine is a severity 0–3 per main group on
`anomaly.<fault>.<main_group>`; the knx-nats-bridge writer rules carry it
to the group's Zentral diagnosis address, where Basalte owns the text.

Publishes go out before the database writes: a failed run then repeats the
same publish (same value, Basalte's change detector ignores it) instead of
losing it behind an already-updated database.

State is recomputed from history on every run — the only stored artifacts
are the episodes themselves. `--dry-run` computes and logs everything and
touches neither the database nor NATS.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import DictRow

from . import nats_publisher
from .config import Settings
from .db_write import read_connection, write_connection
from .episodes import Episode, EpisodePolicy, EventKind, Observation, fold_observations
from .faults import Fault, FaultList, MeasurementKind, Scope
from .logging_setup import get_logger
from .severity import severity_name
from .silence import (
    BUCKET,
    Channel,
    ChannelState,
    ChannelStats,
    SilenceState,
    classify,
    drop_unmeasurable,
    silence_observations,
)

log = get_logger(__name__)

# Measurement window: pause estimation, observation reconstruction and the
# score history all live inside it. Matches the 30 days the episode fold-in
# started the comparison basis with.
LOOKBACK = timedelta(days=30)


@dataclass(frozen=True, slots=True)
class OpenEpisodeRow:
    """The open episode the database holds for one fault and subject."""

    id: int
    subject: str
    severity: int


@dataclass(frozen=True, slots=True)
class ChannelReport:
    """One open silent channel inside a group publish — the payload names
    the exact channel, silent since when, and how far past its pause.
    """

    ga: str
    name: str
    silent_since: datetime | None
    severity: int
    gap_hours: float | None


@dataclass(frozen=True, slots=True)
class GroupPublish:
    main_group: int
    severity: int
    channels: tuple[ChannelReport, ...]


@dataclass(frozen=True, slots=True)
class RunPlan:
    """What one run changes: new episodes, reconciled open rows, orphaned
    rows to close, and the per-main-group severities whose value moved.
    """

    inserts: tuple[Episode, ...]
    updates: tuple[tuple[int, Episode], ...]
    orphan_closes: tuple[tuple[int, datetime], ...]
    publishes: tuple[GroupPublish, ...]


def _main_group(subject: str) -> int:
    return int(subject.split("/", 1)[0])


def plan_run(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    states_by_ga: Mapping[str, SilenceState],
    frontier: datetime,
) -> RunPlan:
    """Pure reconciliation: computed episodes against the stored open rows.

    Only open computed episodes materialize as new rows; a computed episode
    that already ended matters only to close the open row it reconciles. A
    stored severity is never lowered — the recompute window may have slid
    past the peak. An open row whose subject produced no episode at all
    (recovered long ago, or dropped out of scope) closes at the frontier.
    """
    open_by_subject = {row.subject: row for row in open_rows}

    # Several episodes per subject can fall in the window (flicker beyond
    # max_gap); the stored open row can only correspond to the latest one.
    latest_by_subject: dict[str, Episode] = {}
    for episode in episodes:
        current = latest_by_subject.get(episode.subject)
        if current is None or episode.started_at > current.started_at:
            latest_by_subject[episode.subject] = episode

    inserts: list[Episode] = []
    updates: list[tuple[int, Episode]] = []
    open_episodes: dict[str, tuple[Episode, int]] = {}

    for subject, episode in latest_by_subject.items():
        row = open_by_subject.get(subject)
        if episode.ended_at is None:
            effective = max(episode.severity, row.severity if row else 0)
            open_episodes[subject] = (episode, effective)
            if row is not None:
                updates.append((row.id, episode))
            else:
                inserts.append(episode)
        elif row is not None:
            updates.append((row.id, episode))

    orphan_closes = tuple(
        (row.id, frontier) for row in open_rows if row.subject not in latest_by_subject
    )

    before: dict[int, int] = {}
    for row in open_rows:
        group = _main_group(row.subject)
        before[group] = max(before.get(group, 0), row.severity)
    after: dict[int, int] = {}
    for subject, (_, effective) in open_episodes.items():
        group = _main_group(subject)
        after[group] = max(after.get(group, 0), effective)

    publishes: list[GroupPublish] = []
    for group in sorted(set(before) | set(after)):
        severity = after.get(group, 0)
        if severity == before.get(group, 0):
            continue
        reports = sorted(
            (
                _report(subject, episode, effective, states_by_ga)
                for subject, (episode, effective) in open_episodes.items()
                if _main_group(subject) == group
            ),
            key=lambda r: (-r.severity, -(r.gap_hours or 0.0)),
        )
        publishes.append(GroupPublish(group, severity, tuple(reports)))

    return RunPlan(tuple(inserts), tuple(updates), orphan_closes, tuple(publishes))


def _report(
    subject: str,
    episode: Episode,
    effective: int,
    states_by_ga: Mapping[str, SilenceState],
) -> ChannelReport:
    state = states_by_ga.get(subject)
    return ChannelReport(
        ga=subject,
        name=state.channel.name if state else subject,
        silent_since=state.silent_since if state else None,
        severity=effective,
        gap_hours=episode.evidence[-1].value if episode.evidence else None,
    )


def _publish_groups(
    settings: Settings, fault_name: str, publishes: Iterable[GroupPublish]
) -> None:
    for publish in publishes:
        nats_publisher.publish_anomaly(
            settings,
            fault_name,
            severity_name(publish.severity) if publish.severity > 0 else "info",
            {
                "open_channels": len(publish.channels),
                "channels": [
                    {
                        "ga": report.ga,
                        "name": report.name,
                        "silent_since": report.silent_since,
                        "severity": report.severity,
                        "gap_hours": report.gap_hours,
                    }
                    for report in publish.channels
                ],
            },
            entity=str(publish.main_group),
            firing=publish.severity > 0,
        )


# ------------------------------------------------------------------- database

def _frontier(conn: psycopg.Connection[DictRow]) -> datetime | None:
    row = conn.execute("SELECT max(bucket) AS frontier FROM knx_1h").fetchone()
    return row["frontier"] if row else None


def _resolve_channels(conn: psycopg.Connection[DictRow], scope: Scope) -> list[Channel]:
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if scope.dpt:
        clauses.append("dpt = ANY(%(dpt)s)")
        params["dpt"] = list(scope.dpt)
    if scope.name_like:
        clauses.append("name LIKE ANY(%(name_like)s)")
        params["name_like"] = list(scope.name_like)
    if scope.exclude_name_like:
        clauses.append("NOT (name LIKE ANY(%(exclude_name_like)s))")
        params["exclude_name_like"] = list(scope.exclude_name_like)
    sql = "SELECT ga, name, dpt FROM ga_catalog"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    rows = conn.execute(sql + " ORDER BY ga", params).fetchall()
    return [Channel(ga=r["ga"], name=r["name"], dpt=r["dpt"]) for r in rows]


def _channel_stats(
    conn: psycopg.Connection[DictRow], window_start: datetime
) -> dict[str, ChannelStats]:
    rows = conn.execute(
        """
        SELECT ga, count(*) AS buckets, max(bucket) AS last_bucket,
               min(min_value) AS floor_value, max(max_value) AS ceil_value
        FROM knx_1h WHERE bucket >= %(start)s GROUP BY ga
        """,
        {"start": window_start},
    ).fetchall()
    return {
        r["ga"]: ChannelStats(
            ga=r["ga"],
            buckets=r["buckets"],
            last_bucket=r["last_bucket"],
            floor_value=r["floor_value"],
            ceil_value=r["ceil_value"],
        )
        for r in rows
    }


def _bucket_series(
    conn: psycopg.Connection[DictRow], gas: Sequence[str], window_start: datetime
) -> dict[str, list[datetime]]:
    """Bucket series for the channels that need one, fetched in main-group
    chunks so a single result set stays bounded on the small database.
    """
    by_group: dict[int, list[str]] = defaultdict(list)
    for ga in gas:
        by_group[_main_group(ga)].append(ga)
    series: dict[str, list[datetime]] = defaultdict(list)
    for group_gas in by_group.values():
        rows = conn.execute(
            """
            SELECT ga, bucket FROM knx_1h
            WHERE ga = ANY(%(gas)s) AND bucket >= %(start)s
            ORDER BY ga, bucket
            """,
            {"gas": group_gas, "start": window_start},
        ).fetchall()
        for r in rows:
            series[r["ga"]].append(r["bucket"])
    return dict(series)


def _open_rows(conn: psycopg.Connection[DictRow], fault_name: str) -> list[OpenEpisodeRow]:
    rows = conn.execute(
        "SELECT id, subject, severity FROM episodes"
        " WHERE fault = %(fault)s AND ended_at IS NULL",
        {"fault": fault_name},
    ).fetchall()
    return [OpenEpisodeRow(id=r["id"], subject=r["subject"], severity=r["severity"]) for r in rows]


def _history_scores(conn: psycopg.Connection[DictRow], fault_name: str) -> list[float]:
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


def _apply(conn: psycopg.Connection[DictRow], fault_name: str, plan: RunPlan) -> None:
    for episode in plan.inserts:
        row = conn.execute(
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
        _write_details(conn, row["id"], episode)
    for episode_id, episode in plan.updates:
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
    for episode_id, ended_at in plan.orphan_closes:
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


# ----------------------------------------------------------------- the runner

def _candidates(
    kept: list[Channel],
    stats_by_ga: Mapping[str, ChannelStats],
    open_rows: Sequence[OpenEpisodeRow],
    gap_factor: float,
    frontier: datetime,
    *,
    everything: bool,
) -> list[Channel]:
    """Channels whose bucket series is worth fetching: possibly silent (the
    current gap exceeds the threshold at the tightest possible pause) or
    carrying an open episode. A dry run fetches everything so historical
    episodes are counted too.
    """
    if everything:
        return kept
    open_subjects = {row.subject for row in open_rows}
    return [
        channel
        for channel in kept
        if channel.ga in open_subjects
        or frontier - stats_by_ga[channel.ga].last_bucket > gap_factor * BUCKET
    ]


def _log_drops(drops: Mapping[ChannelState, list[Channel]]) -> None:
    dead = drops[ChannelState.DEAD]
    never_sent = drops[ChannelState.NEVER_SENT]
    if dead or never_sent:
        # Dead registers are the actionable list; never-sent is the normal
        # symmetry-address case and stays a count at info level.
        log.info(
            "scope_drops",
            never_sent=len(never_sent),
            dead=len(dead),
            dead_channels=[c.ga for c in dead],
        )
    if never_sent:
        log.debug("scope_drops_never_sent", channels=[c.ga for c in never_sent])


def _run_silence(settings: Settings, fault: Fault, *, dry_run: bool) -> None:
    gap_factor = float(fault.parameters["gap_factor"])
    now = datetime.now(tz=UTC)

    with read_connection(settings) as conn:
        frontier = _frontier(conn)
        if frontier is None:
            log.warning("no_aggregate_data", fault=fault.name)
            return
        window_start = frontier - LOOKBACK
        channels = _resolve_channels(conn, fault.scope)
        stats_by_ga = _channel_stats(conn, window_start)
        kept, drops = drop_unmeasurable(channels, stats_by_ga)
        _log_drops(drops)
        open_rows = _open_rows(conn, fault.name)
        candidates = _candidates(
            kept, stats_by_ga, open_rows, gap_factor, frontier, everything=dry_run
        )
        series = _bucket_series(conn, [c.ga for c in candidates], window_start)
        history_scores = _history_scores(conn, fault.name)

    states: dict[str, SilenceState] = {}
    observations: list[Observation] = []
    for channel in candidates:
        buckets = series.get(channel.ga, [])
        state = classify(channel, buckets, frontier, gap_factor)
        states[channel.ga] = state
        if state.pause is not None:
            observations.extend(
                silence_observations(channel.ga, buckets, state.pause, gap_factor, frontier)
            )

    episodes = fold_observations(fault.name, observations, history_scores, EpisodePolicy(), now)
    plan = plan_run(
        episodes=episodes, open_rows=open_rows, states_by_ga=states, frontier=frontier
    )

    log.info(
        "channel_silence_run",
        fault=fault.name,
        frontier=frontier.isoformat(),
        channels=len(kept),
        candidates=len(candidates),
        silent=sum(1 for s in states.values() if s.state is ChannelState.SILENT),
        episodes=len(episodes),
        open_episodes=sum(1 for e in episodes if e.ended_at is None),
        inserts=len(plan.inserts),
        updates=len(plan.updates),
        orphan_closes=len(plan.orphan_closes),
        publishes=len(plan.publishes),
        dry_run=dry_run,
    )

    if dry_run:
        per_group = Counter(_main_group(e.subject) for e in episodes)
        log.info(
            "dry_run_episodes",
            fault=fault.name,
            per_main_group={str(g): per_group[g] for g in sorted(per_group)},
            open_subjects=sorted(e.subject for e in episodes if e.ended_at is None),
            would_publish=[
                {"main_group": p.main_group, "severity": p.severity} for p in plan.publishes
            ],
        )
        return

    _publish_groups(settings, fault.name, plan.publishes)
    with write_connection(settings) as conn, conn.transaction():
        _apply(conn, fault.name, plan)


def run(settings: Settings, argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="iot-insights-engine detect-faults")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    fault_list = FaultList.load(Path(settings.faults_file))
    for fault in fault_list:
        if fault.dormant is not None:
            log.info("fault_dormant", fault=fault.name, active_when=fault.dormant.active_when)
            continue
        if fault.kind is MeasurementKind.SILENCE:
            _run_silence(settings, fault, dry_run=args.dry_run)
        else:
            # Arrives with its own ticket; a declared fault must not fail
            # the ones already running.
            log.warning("fault_kind_not_implemented", fault=fault.name, kind=str(fault.kind))
    return 0
