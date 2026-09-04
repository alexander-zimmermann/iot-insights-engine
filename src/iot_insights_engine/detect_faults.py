"""Detect-faults job: run the declared fault list end to end.

For each schedulable fault the runner resolves the scope where the catalog
lives (`ga_catalog` in TSDB — never a hand-written address list), measures,
folds the observations into episodes behind the pure pipeline seam, and
reconciles the result with the episodes the database already holds. What
leaves the engine is a severity 0–3 per main group on
`anomaly.<fault>.<main_group>`; the knx-nats-bridge writer rules carry it
to the group's Zentral diagnosis address, where Basalte owns the text.

External faults run the same loop the other way round: Basalte detects,
writes the severity to the fault address and delivers itself; the engine
reads those writes back from the bus archive, records episodes marked
externally delivered, and publishes nothing. The frontier rule above is
the measured faults': the bus archive has no materialization lag, so
external runs take wall-clock time where an orphaned row needs closing.

Time is the aggregate's frontier throughout — episodes also *end* in
frontier time, so a stalled refresh (or a dead bridge) freezes the picture
instead of clearing every open episode with a severity 0 nobody earned.

Publishes go out before the database writes: a failed run then repeats the
same publish (same value, Basalte's change detector ignores it) instead of
losing it behind an already-updated database.

State is recomputed from history on every run — the only stored artifacts
are the episodes themselves. `--dry-run` computes and logs everything and
touches neither the database nor NATS.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import deviation, duration, episode_store, external, nats_publisher, silence
from .config import Settings
from .db_write import read_connection, write_connection
from .episode_store import OpenEpisodeRow
from .episodes import Episode, EpisodePolicy, Observation, fold_observations
from .faults import Fault, FaultList, MeasurementKind
from .logging_setup import get_logger
from .severity import severity_name
from .silence import BUCKET, Channel, ChannelState, SilenceState, main_group

log = get_logger(__name__)

# Measurement window: pause estimation, observation reconstruction and the
# score history all live inside it. Matches the 30 days the episode fold-in
# started the comparison basis with.
LOOKBACK = timedelta(days=30)


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
    rows to close, rows kept open for want of data, and the main groups
    whose severity or channel set moved.
    """

    inserts: tuple[Episode, ...]
    updates: tuple[tuple[int, Episode], ...]
    orphan_closes: tuple[tuple[int, datetime], ...]
    stale_opens: tuple[str, ...]
    publishes: tuple[GroupPublish, ...]


def plan_run(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    states_by_ga: Mapping[str, SilenceState],
    dataless: frozenset[str],
    frontier: datetime,
) -> RunPlan:
    """Pure reconciliation: computed episodes against the stored open rows.

    Only open computed episodes materialize as new rows; a computed episode
    that already ended matters only to close the open row it reconciles. A
    stored severity is never lowered — the recompute window may have slid
    past the peak. An open row whose subject produced no episode closes at
    the frontier — unless the subject is `dataless` (still in scope, but
    silent for longer than the whole window): with no data to decide a
    recovery, the episode stays open instead of self-clearing.

    A group publishes when its severity moved or its set of open channels
    changed — a second channel going silent at the same tier still gets
    named on the bus.
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
    reports: dict[str, ChannelReport] = {}

    for subject, episode in sorted(latest_by_subject.items()):
        row = open_by_subject.get(subject)
        if episode.ended_at is None:
            effective = max(episode.severity, row.severity if row else 0)
            state = states_by_ga[subject]
            reports[subject] = ChannelReport(
                ga=subject,
                name=state.channel.name,
                silent_since=state.silent_since,
                severity=effective,
                gap_hours=episode.evidence[-1].value if episode.evidence else None,
            )
            if row is not None:
                updates.append((row.id, episode))
            else:
                inserts.append(episode)
        elif row is not None:
            updates.append((row.id, episode))

    orphan_closes: list[tuple[int, datetime]] = []
    stale_opens: list[str] = []
    for row in open_rows:
        if row.subject in latest_by_subject:
            continue
        if row.subject in dataless:
            stale_opens.append(row.subject)
            # Still silent as far as anyone can tell — it keeps counting.
            reports[row.subject] = ChannelReport(
                ga=row.subject,
                name=row.subject,
                silent_since=None,
                severity=row.severity,
                gap_hours=None,
            )
        else:
            orphan_closes.append((row.id, frontier))

    before = _group_state((row.subject, row.severity) for row in open_rows)
    after = _group_state((subject, report.severity) for subject, report in reports.items())

    publishes: list[GroupPublish] = []
    for group in sorted(set(before) | set(after)):
        severity, _ = after.get(group, (0, frozenset()))
        if after.get(group) == before.get(group):
            continue
        channels = sorted(
            (r for r in reports.values() if main_group(r.ga) == group),
            key=lambda r: (-r.severity, -(r.gap_hours or 0.0), r.ga),
        )
        publishes.append(GroupPublish(group, severity, tuple(channels)))

    return RunPlan(
        tuple(inserts),
        tuple(updates),
        tuple(orphan_closes),
        tuple(stale_opens),
        tuple(publishes),
    )


def _group_state(
    subject_severities: Iterable[tuple[str, int]],
) -> dict[int, tuple[int, frozenset[str]]]:
    """Per main group: the maximum severity and the set of open subjects —
    the two things whose change warrants a publish."""
    severities: dict[int, int] = {}
    subjects: dict[int, set[str]] = {}
    for subject, severity in subject_severities:
        group = main_group(subject)
        severities[group] = max(severities.get(group, 0), severity)
        subjects.setdefault(group, set()).add(subject)
    return {g: (severities[g], frozenset(subjects[g])) for g in severities}


def _publish_groups(
    settings: Settings, fault_name: str, publishes: Iterable[GroupPublish]
) -> None:
    for publish in publishes:
        firing = publish.severity > 0
        nats_publisher.publish_anomaly(
            settings,
            fault_name,
            severity_name(publish.severity) if firing else None,
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
            firing=firing,
        )


def _candidates(
    kept: list[Channel],
    stats_by_ga: Mapping[str, silence.ChannelStats],
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
    if fault.target is None or not fault.target.per_main_group:
        raise ValueError(f"fault {fault.name}: silence delivery needs a per_main_group target")
    gap_factor = float(fault.parameters["gap_factor"])
    gap_quantile = float(fault.parameters["gap_quantile"])

    with read_connection(settings) as conn:
        frontier = silence.frontier(conn)
        if frontier is None:
            log.warning("no_aggregate_data", fault=fault.name)
            return
        window_start = frontier - LOOKBACK
        channels = silence.resolve_scope(conn, fault.scope)
        stats_by_ga = silence.channel_stats(conn, window_start)
        kept, drops = silence.drop_unmeasurable(channels, stats_by_ga)
        _log_drops(drops)
        open_rows = episode_store.open_rows(conn, fault.name)
        candidates = _candidates(
            kept, stats_by_ga, open_rows, gap_factor, frontier, everything=dry_run
        )
        series = silence.bucket_series(conn, [c.ga for c in candidates], window_start)
        history_scores = episode_store.history_scores(conn, fault.name)

    states: dict[str, SilenceState] = {}
    observations: list[Observation] = []
    for channel in candidates:
        buckets = series.get(channel.ga, [])
        state = silence.classify(
            channel,
            buckets,
            frontier=frontier,
            gap_factor=gap_factor,
            gap_quantile=gap_quantile,
        )
        states[channel.ga] = state
        if state.pause is not None:
            observations.extend(
                silence.silence_observations(
                    channel.ga, buckets, state.pause, gap_factor, frontier
                )
            )

    # `now` is the frontier: episode ends are decided by aggregate progress,
    # never by wall time racing ahead of a stalled materialization.
    episodes = fold_observations(
        fault.name, observations, history_scores, EpisodePolicy(), frontier
    )
    dataless = frozenset(c.ga for c in channels) - frozenset(stats_by_ga)
    plan = plan_run(
        episodes=episodes,
        open_rows=open_rows,
        states_by_ga=states,
        dataless=dataless,
        frontier=frontier,
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
        stale_opens=list(plan.stale_opens),
        publishes=len(plan.publishes),
        dry_run=dry_run,
    )

    if dry_run:
        per_group = Counter(main_group(e.subject) for e in episodes)
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
        episode_store.apply(conn, fault.name, plan.inserts, plan.updates, plan.orphan_closes)


def _publish_devices(
    settings: Settings, fault_name: str, publishes: Iterable[duration.DevicePublish]
) -> None:
    for publish in publishes:
        firing = publish.severity > 0
        nats_publisher.publish_anomaly(
            settings,
            fault_name,
            severity_name(publish.severity) if firing else None,
            {
                "device": publish.device,
                "ga": publish.ga,
                "name": publish.name,
                "running_since": publish.running_since,
                "run_hours": publish.run_hours,
                "limit_hours": publish.limit_hours,
            },
            entity=publish.ga.replace("/", "-"),
            firing=firing,
        )


def _run_duration(settings: Settings, fault: Fault, *, dry_run: bool) -> None:
    if fault.target is None or not fault.target.per_device:
        raise ValueError(f"fault {fault.name}: duration delivery needs a per_device target")
    active_fraction = float(fault.parameters["active_hour_fraction"])

    with read_connection(settings) as conn:
        frontier = duration.frontier(conn)
        if frontier is None:
            log.warning("no_aggregate_data", fault=fault.name)
            return
        window_start = frontier - LOOKBACK
        channels = silence.resolve_scope(conn, fault.scope)
        # A limit without a channel or a channel without a limit fails the
        # run loudly — never a silently unmeasured device.
        devices = duration.resolve_devices(channels, fault.devices)
        activity = duration.activity(conn, [d.ga for d in devices], window_start, active_fraction)
        open_rows = episode_store.open_rows(conn, fault.name)
        history_scores = episode_store.history_scores(conn, fault.name)

    states: dict[str, duration.DeviceState] = {}
    observations: list[Observation] = []
    for device in devices:
        active = activity.active.get(device.ga, [])
        states[device.ga] = duration.classify(device, active, frontier)
        observations.extend(duration.duration_observations(device.ga, active, device.max_run))

    # `now` is the frontier: episode ends are decided by aggregate progress,
    # never by wall time racing ahead of a stalled materialization.
    episodes = fold_observations(
        fault.name, observations, history_scores, EpisodePolicy(), frontier
    )
    dataless = frozenset(d.ga for d in devices) - activity.present
    plan = duration.plan_run(
        episodes=episodes,
        open_rows=open_rows,
        states_by_ga=states,
        dataless=dataless,
        frontier=frontier,
    )

    log.info(
        "appliance_runtime_run",
        fault=fault.name,
        frontier=frontier.isoformat(),
        devices=len(devices),
        running=sum(1 for s in states.values() if s.running_since is not None),
        over_limit=sum(
            1
            for s in states.values()
            if s.run_hours is not None and s.run_hours * BUCKET > s.device.max_run
        ),
        episodes=len(episodes),
        open_episodes=sum(1 for e in episodes if e.ended_at is None),
        inserts=len(plan.inserts),
        updates=len(plan.updates),
        orphan_closes=len(plan.orphan_closes),
        stale_opens=list(plan.stale_opens),
        publishes=len(plan.publishes),
        dry_run=dry_run,
    )

    if dry_run:
        label_by_ga = {d.ga: d.label for d in devices}
        per_device = Counter(e.subject for e in episodes)
        log.info(
            "dry_run_episodes",
            fault=fault.name,
            per_device={label_by_ga.get(ga, ga): count for ga, count in sorted(per_device.items())},
            open_subjects=sorted(e.subject for e in episodes if e.ended_at is None),
            would_publish=[{"ga": p.ga, "severity": p.severity} for p in plan.publishes],
        )
        return

    _publish_devices(settings, fault.name, plan.publishes)
    with write_connection(settings) as conn, conn.transaction():
        episode_store.apply(conn, fault.name, plan.inserts, plan.updates, plan.orphan_closes)


def _publish_rooms(
    settings: Settings, fault_name: str, publishes: Iterable[deviation.RoomPublish]
) -> None:
    for publish in publishes:
        firing = publish.severity > 0
        nats_publisher.publish_anomaly(
            settings,
            fault_name,
            severity_name(publish.severity) if firing else None,
            {
                "room": publish.room,
                "cold_since": publish.cold_since,
                "gap": publish.gap,
                "value": publish.value,
                "reference": publish.reference,
                "gate": publish.gate,
                "min_gap": publish.min_gap,
            },
            entity=publish.slug,
            firing=firing,
        )


def _run_deviation(settings: Settings, fault: Fault, *, dry_run: bool) -> None:
    if fault.target is None or not fault.target.per_room:
        raise ValueError(f"fault {fault.name}: deviation delivery needs a per_room target")
    if fault.roles is None:
        raise ValueError(f"fault {fault.name}: the deviation kind needs declared roles")
    min_hours = float(fault.parameters["min_hours"])
    gate_min = fault.parameters.get("gate_min_pct")

    with read_connection(settings) as conn:
        frontier = silence.frontier(conn)
        if frontier is None:
            log.warning("no_aggregate_data", fault=fault.name)
            return
        window_start = frontier - LOOKBACK
        channels = silence.resolve_scope(conn, fault.scope)
        # A room without its channels or a channel without its room fails
        # the run loudly — never a silently unmeasured room.
        rooms = deviation.resolve_rooms(channels, fault.rooms, fault.roles)
        gas = [ga for room in rooms for ga in room.gas]
        series = deviation.values(conn, gas, window_start)
        open_rows = episode_store.open_rows(conn, fault.name)
        history_scores = episode_store.history_scores(conn, fault.name)

    dead_values = deviation.dead_value_gas(rooms, series)
    if dead_values:
        # A dead register reading 0.0 would score as a huge gap; the
        # silence fault owns reporting the channel itself.
        log.info("scope_drops", dead=len(dead_values), dead_channels=dead_values)
        for ga in dead_values:
            del series[ga]

    states: dict[str, deviation.RoomState] = {}
    observations: list[Observation] = []
    dataless: set[str] = set()
    for room in rooms:
        buckets = deviation.room_series(room, series, window_start, frontier)
        if not buckets:
            dataless.add(room.slug)
        cold = deviation.cold_buckets(room, buckets, gate_min)
        states[room.slug] = deviation.classify(room, cold, frontier)
        observations.extend(deviation.deviation_observations(room, cold, min_hours))
    if dataless:
        # A role silent for the whole window leaves the room unmeasurable —
        # not a scope error, but never to pass in silence either.
        log.warning("rooms_dataless", fault=fault.name, rooms=sorted(dataless))

    # `now` is the frontier: episode ends are decided by aggregate progress,
    # never by wall time racing ahead of a stalled materialization.
    episodes = fold_observations(
        fault.name, observations, history_scores, EpisodePolicy(), frontier
    )
    plan = deviation.plan_run(
        episodes=episodes,
        open_rows=open_rows,
        states_by_slug=states,
        dataless=frozenset(dataless),
        frontier=frontier,
    )

    log.info(
        "room_deviation_run",
        fault=fault.name,
        frontier=frontier.isoformat(),
        rooms=len(rooms),
        cold=sum(1 for s in states.values() if s.cold_since is not None),
        episodes=len(episodes),
        open_episodes=sum(1 for e in episodes if e.ended_at is None),
        inserts=len(plan.inserts),
        updates=len(plan.updates),
        orphan_closes=len(plan.orphan_closes),
        stale_opens=list(plan.stale_opens),
        publishes=len(plan.publishes),
        dry_run=dry_run,
    )

    if dry_run:
        per_room = Counter(e.subject for e in episodes)
        log.info(
            "dry_run_episodes",
            fault=fault.name,
            per_room=dict(sorted(per_room.items())),
            open_subjects=sorted(e.subject for e in episodes if e.ended_at is None),
            would_publish=[{"slug": p.slug, "severity": p.severity} for p in plan.publishes],
        )
        return

    _publish_rooms(settings, fault.name, plan.publishes)
    with write_connection(settings) as conn, conn.transaction():
        episode_store.apply(conn, fault.name, plan.inserts, plan.updates, plan.orphan_closes)


def _run_external(settings: Settings, fault: Fault, *, dry_run: bool) -> None:
    """Basalte-written severities become episodes: read the fault's severity
    writes back off the bus archive, fold, reconcile — and publish nothing.
    Basalte already delivered; the engine only records.
    """
    now = datetime.now(tz=UTC)
    with read_connection(settings) as conn:
        channels = silence.resolve_scope(conn, fault.scope)
        if not channels:
            # Address not in the catalog yet (ETS work pending) — or a typo.
            log.warning("external_no_subjects", fault=fault.name)
        open_rows = episode_store.open_rows(conn, fault.name)
        processed = episode_store.processed_through(conn, fault.name)
        writes = external.read_writes(conn, [c.ga for c in channels], now - LOOKBACK)

    prior = {row.subject: row.severity for row in open_rows}
    fresh = external.drop_processed(writes, processed)
    episodes = external.fold_severity_writes(fault.name, fresh, prior)
    plan = external.plan_run(
        episodes=episodes,
        open_rows=open_rows,
        in_scope=frozenset(c.ga for c in channels),
        now=now,
    )

    log.info(
        "external_severities_run",
        fault=fault.name,
        subjects=len(channels),
        writes=len(writes),
        fresh_writes=len(fresh),
        episodes=len(episodes),
        open_episodes=sum(1 for e in episodes if e.ended_at is None),
        inserts=len(plan.inserts),
        updates=len(plan.updates),
        orphan_closes=len(plan.orphan_closes),
        still_open=list(plan.still_open),
        dry_run=dry_run,
    )

    if dry_run:
        log.info(
            "dry_run_episodes",
            fault=fault.name,
            open_subjects=sorted(e.subject for e in episodes if e.ended_at is None),
        )
        return

    with write_connection(settings) as conn, conn.transaction():
        episode_store.apply(
            conn,
            fault.name,
            plan.inserts,
            plan.updates,
            plan.orphan_closes,
            externally_delivered=True,
        )


def run(settings: Settings, argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="iot-insights-engine detect-faults")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    fault_list = FaultList.load(Path(settings.faults_file))
    for fault in fault_list:
        if fault.dormant is not None:
            log.info("fault_dormant", fault=fault.name, active_when=fault.dormant.active_when)
    for fault in fault_list.schedulable():
        if fault.kind is MeasurementKind.SILENCE:
            _run_silence(settings, fault, dry_run=args.dry_run)
        elif fault.kind is MeasurementKind.DURATION:
            _run_duration(settings, fault, dry_run=args.dry_run)
        elif fault.kind is MeasurementKind.DEVIATION:
            _run_deviation(settings, fault, dry_run=args.dry_run)
        elif fault.kind is MeasurementKind.EXTERNAL:
            _run_external(settings, fault, dry_run=args.dry_run)
        else:
            # Arrives with its own ticket; a declared fault must not fail
            # the ones already running.
            log.warning("fault_kind_not_implemented", fault=fault.name, kind=str(fault.kind))
    return 0
