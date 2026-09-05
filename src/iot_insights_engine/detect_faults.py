"""Detect-faults job: run the declared fault list end to end.

For each schedulable fault the runner resolves the scope where the catalog
lives (`ga_catalog` in TSDB — never a hand-written address list), measures,
folds the observations into episodes behind the pure pipeline seam, and
reconciles the result with the episodes the database already holds. What
leaves the engine is a severity 0–3 per main group on
`anomaly.<fault>.<main_group>`; the knx-nats-bridge writer rules carry it
to the group's Zentral diagnosis address, where Basalte owns the text.

Every kind that reports per subject runs in one shape: a `SubjectKind`
declares how its series is measured and how its payload is shaped, and
`_run_subjects` owns the rest — the window, the fold, the reconciliation,
the log record, the dry run and the publish-then-write tail. A new
per-subject kind declares those two things and inherits all of it.

Channel silence measures per channel but reports per main group, so it
keeps its own delivery; it reconciles through the same `reconcile` as
everything else.

The volume watchdog runs the loop over the engine's own output: it counts
the incidents of the last seven days out of the episode stream and puts a
severity on one house-wide address, so drift back into noise arrives on the
same bus as everything else.

External faults run the same loop the other way round: Basalte detects,
writes the severity to the fault address and delivers itself; the engine
reads those writes back from the bus archive, records episodes marked
externally delivered, and publishes nothing. The frontier rule below is
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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import (
    deviation,
    drift,
    duration,
    episode_store,
    external,
    nats_publisher,
    silence,
    volume,
)
from .config import Settings
from .db_write import read_connection, write_connection
from .episode_store import OpenEpisodeRow
from .episodes import Episode, EpisodePolicy, Observation, fold_observations
from .faults import DriftSignal, Fault, FaultList, MeasurementKind
from .logging_setup import get_logger
from .reconcile import (
    Measured,
    Plan,
    SubjectPublish,
    Window,
    plan_from,
    reconcile,
    subject_plan,
)
from .severity import severity_name
from .silence import (
    BUCKET,
    MIN_PAUSE_BUCKETS,
    Channel,
    ChannelState,
    SilenceState,
    main_group,
)

if TYPE_CHECKING:
    import psycopg
    from psycopg.rows import DictRow

log = get_logger(__name__)

# Measurement window: pause estimation, observation reconstruction and the
# score history all live inside it. Matches the 30 days the episode fold-in
# started the comparison basis with.
LOOKBACK = timedelta(days=30)


@dataclass(frozen=True, slots=True)
class SubjectKind[S, P: SubjectPublish]:
    """One per-subject fault kind, reduced to what actually differs between
    them: how its series is measured (`frontier`, `measure`) and how its
    payload is shaped (`publish_for`, `publish`). `event` names its log
    record, `delivery` the target form the fault must declare for it.
    """

    event: str
    delivery: str
    frontier: Callable[[psycopg.Connection[DictRow]], datetime | None]
    measure: Callable[[psycopg.Connection[DictRow], Fault, Window], Measured[S]]
    publish_for: Callable[[str, int, S | None], P]
    publish: Callable[[Settings, str, Iterable[P]], None]


def _publish_subjects[P: SubjectPublish](
    settings: Settings,
    fault_name: str,
    publishes: Iterable[P],
    payload: Callable[[P], dict[str, Any]],
) -> None:
    """One publish per moved subject, on the subject's own address: the
    severity decides firing, the kind decides the rest of the payload.
    """
    for publish in publishes:
        firing = publish.severity > 0
        nats_publisher.publish_anomaly(
            settings,
            fault_name,
            severity_name(publish.severity) if firing else None,
            payload(publish),
            entity=publish.entity,
            firing=firing,
        )


def _publish_devices(
    settings: Settings, fault_name: str, publishes: Iterable[duration.DevicePublish]
) -> None:
    """Appliance runtime: the run so far against the device's own limit."""
    _publish_subjects(
        settings,
        fault_name,
        publishes,
        lambda p: {
            "device": p.device,
            "ga": p.ga,
            "name": p.name,
            "running_since": p.running_since,
            "run_hours": p.run_hours,
            "limit_hours": p.limit_hours,
        },
    )


def _publish_standby(
    settings: Settings, fault_name: str, publishes: Iterable[drift.DevicePublish]
) -> None:
    """Standby drift: what the device idles at against what it should."""
    _publish_subjects(
        settings,
        fault_name,
        publishes,
        lambda p: {
            "device": p.device,
            "ga": p.ga,
            "name": p.name,
            "standby_ma": p.level,
            "healthy_ma": p.healthy,
            "excess_ma": p.excess,
            "rising_since": p.rising_since,
        },
    )


def _publish_duty_cycle(
    settings: Settings, fault_name: str, publishes: Iterable[drift.DevicePublish]
) -> None:
    """Duty-cycle drift: how much of the day the compressor runs against how
    much it should — the mail's number for "the freezer is icing up"."""
    _publish_subjects(
        settings,
        fault_name,
        publishes,
        lambda p: {
            "device": p.device,
            "ga": p.ga,
            "name": p.name,
            "duty_pct": p.level,
            "healthy_pct": p.healthy,
            "excess_pct": p.excess,
            "rising_since": p.rising_since,
        },
    )


def _publish_rooms(
    settings: Settings, fault_name: str, publishes: Iterable[deviation.RoomPublish]
) -> None:
    """Room deviation: the gap, and the reference and gate behind it."""
    _publish_subjects(
        settings,
        fault_name,
        publishes,
        lambda p: {
            "room": p.room,
            "cold_since": p.cold_since,
            "gap": p.gap,
            "value": p.value,
            "reference": p.reference,
            "gate": p.gate,
            "min_gap": p.min_gap,
        },
    )


def _run_subjects[S, P: SubjectPublish](
    settings: Settings, fault: Fault, kind: SubjectKind[S, P], *, dry_run: bool
) -> None:
    """The one shape a per-subject kind runs in: guard the declaration, take
    the window off the aggregate, measure, fold, reconcile, log — then
    publish before writing.
    """
    if fault.target is None or fault.target.form != kind.delivery:
        raise ValueError(
            f"fault {fault.name}: {fault.kind} delivery needs a {kind.delivery} target"
        )
    policy = EpisodePolicy()

    with read_connection(settings) as conn:
        frontier = kind.frontier(conn)
        if frontier is None:
            log.warning("no_aggregate_data", fault=fault.name)
            return
        window = Window(start=frontier - LOOKBACK, frontier=frontier, policy=policy)
        measured = kind.measure(conn, fault, window)
        open_rows = episode_store.open_rows(conn, fault.name)
        history_scores = episode_store.history_scores(conn, fault.name)

    if measured.dataless:
        # Never at info level: a subject nobody could measure is the one
        # thing that keeps an open episode from ever clearing itself.
        log.warning(
            "subjects_dataless", fault=fault.name, subjects=sorted(measured.dataless)
        )

    # `now` is the frontier: episode ends are decided by aggregate progress,
    # never by wall time racing ahead of a stalled materialization.
    episodes = fold_observations(
        fault.name, measured.observations, history_scores, policy, frontier
    )

    def payload(subject: str, severity: int) -> P:
        return kind.publish_for(subject, severity, measured.states.get(subject))

    plan = subject_plan(
        episodes=episodes,
        open_rows=open_rows,
        dataless=measured.dataless,
        frontier=frontier,
        publish_for=payload,
    )

    log.info(
        kind.event,
        fault=fault.name,
        frontier=frontier.isoformat(),
        **measured.counts,
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
        _log_dry_run(fault, episodes, plan, measured.labels)
        return

    kind.publish(settings, fault.name, plan.publishes)
    with write_connection(settings) as conn, conn.transaction():
        episode_store.apply(conn, fault.name, plan.inserts, plan.updates, plan.orphan_closes)


def _log_dry_run[P: SubjectPublish](
    fault: Fault,
    episodes: Sequence[Episode],
    plan: Plan[P],
    labels: Mapping[str, str],
) -> None:
    """What the run would have done, by subject — the dry run's whole point,
    so it names each subject the way a human does where the kind knows it.
    """
    per_subject = Counter(e.subject for e in episodes)
    log.info(
        "dry_run_episodes",
        fault=fault.name,
        per_subject={
            labels.get(subject, subject): count
            for subject, count in sorted(per_subject.items())
        },
        open_subjects=sorted(e.subject for e in episodes if e.ended_at is None),
        would_publish=[
            {"subject": p.subject, "severity": p.severity} for p in plan.publishes
        ],
    )


_SUBJECT_KINDS: Mapping[MeasurementKind, SubjectKind[Any, Any]] = {
    MeasurementKind.DURATION: SubjectKind(
        event="appliance_runtime_run",
        delivery="per_device",
        frontier=duration.frontier,
        measure=duration.measure,
        publish_for=duration.publish_for,
        publish=_publish_devices,
    ),
    MeasurementKind.DEVIATION: SubjectKind(
        event="room_deviation_run",
        delivery="per_room",
        frontier=silence.frontier,
        measure=deviation.measure,
        publish_for=deviation.publish_for,
        publish=_publish_rooms,
    ),
}


# The drift kind runs one shape per signal: same CUSUM, different series,
# so the run record and the payload's units differ with the signal the
# fault declares.
_DRIFT_SIGNALS: Mapping[DriftSignal, SubjectKind[Any, Any]] = {
    DriftSignal.STANDBY: SubjectKind(
        event="appliance_standby_run",
        delivery="per_device",
        frontier=duration.frontier,
        measure=drift.measure_standby,
        publish_for=drift.publish_for,
        publish=_publish_standby,
    ),
    DriftSignal.DUTY_CYCLE: SubjectKind(
        event="duty_cycle_drift_run",
        delivery="per_device",
        frontier=duration.frontier,
        measure=drift.measure_duty_cycle,
        publish_for=drift.publish_for,
        publish=_publish_duty_cycle,
    ),
}


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

    @property
    def subject(self) -> str:
        return str(self.main_group)

    @property
    def entity(self) -> str:
        return str(self.main_group)


RunPlan = Plan[GroupPublish]


def plan_run(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    states_by_ga: Mapping[str, SilenceState],
    dataless: frozenset[str],
    frontier: datetime,
) -> RunPlan:
    """The shared reconciliation, delivered per main group — the one thing
    channel silence does not share with the other kinds.

    A group publishes when its severity moved or its set of open channels
    changed: a second channel going silent at the same tier still gets named
    on the bus.
    """
    result = reconcile(
        episodes=episodes, open_rows=open_rows, dataless=dataless, frontier=frontier
    )
    # The fold leaves at most one open episode per subject, and it is the
    # one the reconciliation carried into `after`.
    open_episodes = {e.subject: e for e in episodes if e.ended_at is None}
    reports: dict[str, ChannelReport] = {}
    for subject, severity in result.after.items():
        episode = open_episodes.get(subject)
        if episode is None:
            # Kept open for want of data: nothing measured it this run, so
            # there is no gap to report — only the address and the tier it
            # still carries.
            reports[subject] = ChannelReport(
                ga=subject, name=subject, silent_since=None, severity=severity, gap_hours=None
            )
            continue
        state = states_by_ga[subject]
        reports[subject] = ChannelReport(
            ga=subject,
            name=state.channel.name,
            silent_since=state.silent_since,
            severity=severity,
            gap_hours=episode.evidence[-1].value if episode.evidence else None,
        )

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

    return plan_from(result, publishes)


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
    """Channel silence: how many channels in the group are open, and which."""
    _publish_subjects(
        settings,
        fault_name,
        publishes,
        lambda p: {
            "open_channels": len(p.channels),
            "channels": [
                {
                    "ga": report.ga,
                    "name": report.name,
                    "silent_since": report.silent_since,
                    "severity": report.severity,
                    "gap_hours": report.gap_hours,
                }
                for report in p.channels
            ],
        },
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
    """Channel silence: measured per channel like the other kinds, delivered
    per main group, which is why it runs its own loop.
    """
    if fault.target is None or fault.target.form != "per_main_group":
        raise ValueError(f"fault {fault.name}: silence delivery needs a per_main_group target")
    gap_factor = float(fault.parameters["gap_factor"])
    gap_quantile = float(fault.parameters["gap_quantile"])
    policy = EpisodePolicy()

    with read_connection(settings) as conn:
        frontier = silence.frontier(conn)
        if frontier is None:
            log.warning("no_aggregate_data", fault=fault.name)
            return
        window = Window(start=frontier - LOOKBACK, frontier=frontier, policy=policy)
        channels = silence.resolve_scope(conn, fault.channel_scope())
        stats_by_ga = silence.channel_stats(conn, window.start)
        kept, drops = silence.drop_unmeasurable(channels, stats_by_ga)
        _log_drops(drops)
        open_rows = episode_store.open_rows(conn, fault.name)
        candidates = _candidates(
            kept, stats_by_ga, open_rows, gap_factor, frontier, everything=dry_run
        )
        series = silence.bucket_series(conn, [c.ga for c in candidates], window.start)
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

    # A silence measurement ends at the frontier by construction: the gap
    # walk runs right up to it for every channel whose own pause could be
    # estimated. A channel that sent too little to show one was not measured
    # at all, which is not the same as recovered.
    measured_through = {
        ga: frontier
        for ga, stats in stats_by_ga.items()
        if stats.buckets >= MIN_PAUSE_BUCKETS
    }
    # Deliberately without the per-subject kinds' dataless warning: here the
    # set is every never-sent symmetry address in the catalog — a thousand of
    # them, normal, and already counted by `_log_drops`. The ones it actually
    # holds open are `stale_opens` in the run record below.
    dataless = frozenset(
        channel.ga
        for channel in channels
        if not window.reaches(measured_through.get(channel.ga))
    )

    # `now` is the frontier: episode ends are decided by aggregate progress,
    # never by wall time racing ahead of a stalled materialization.
    episodes = fold_observations(
        fault.name, observations, history_scores, policy, frontier
    )
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
        _log_dry_run(fault, episodes, plan, {c.ga: c.name for c in channels})
        return

    _publish_groups(settings, fault.name, plan.publishes)
    with write_connection(settings) as conn, conn.transaction():
        episode_store.apply(conn, fault.name, plan.inserts, plan.updates, plan.orphan_closes)


def _publish_volume(
    settings: Settings, fault_name: str, publish: volume.VolumePublish
) -> None:
    firing = publish.severity > 0
    nats_publisher.publish_anomaly(
        settings,
        fault_name,
        severity_name(publish.severity) if firing else None,
        {
            "episodes": publish.state.episodes,
            "limit": publish.state.limit,
            "window_days": volume.WINDOW.days,
            "over_since": publish.state.over_since,
            # Which fault is making the noise — the thing a human acts on.
            "by_fault": [
                {"fault": count.fault, "episodes": count.episodes}
                for count in publish.state.by_fault
            ],
        },
        firing=firing,
    )


def _run_volume(settings: Settings, fault: Fault, *, dry_run: bool) -> None:
    """The volume watchdog: the incident count of the last seven days is
    itself a fault, measured over the episode stream and delivered on one
    house-wide address. Declared last in the fault list, so the count
    already includes what this run's other faults just wrote.
    """
    if fault.target is None or fault.target.ga is None:
        raise ValueError(f"fault {fault.name}: volume delivery needs a house-wide ga target")
    limit = float(fault.parameters["max_episodes_per_week"])

    with read_connection(settings) as conn:
        frontier = silence.frontier(conn)
        if frontier is None:
            log.warning("no_aggregate_data", fault=fault.name)
            return
        window_start = frontier - LOOKBACK
        # A week of history before the first bucket, so the oldest count in
        # the window is as complete as the newest.
        starts = volume.episode_starts(conn, window_start - volume.WINDOW)
        open_rows = episode_store.open_rows(conn, fault.name)
        history_scores = episode_store.history_scores(conn, fault.name)

    buckets = volume.count_series(starts, window_start, frontier)
    observations = volume.volume_observations(buckets, limit)
    state = volume.classify(starts, buckets, limit, frontier)

    # `now` is the frontier: episode ends are decided by aggregate progress,
    # never by wall time racing ahead of a stalled materialization.
    episodes = fold_observations(
        fault.name, observations, history_scores, EpisodePolicy(), frontier
    )
    plan = volume.plan_run(
        episodes=episodes, open_rows=open_rows, state=state, frontier=frontier
    )

    log.info(
        "notification_volume_run",
        fault=fault.name,
        frontier=frontier.isoformat(),
        incidents=state.episodes,
        limit=limit,
        over_since=state.over_since.isoformat() if state.over_since else None,
        by_fault={count.fault: count.episodes for count in state.by_fault},
        episodes=len(episodes),
        open_episodes=sum(1 for e in episodes if e.ended_at is None),
        inserts=len(plan.inserts),
        updates=len(plan.updates),
        orphan_closes=len(plan.orphan_closes),
        publishes=1 if plan.publish is not None else 0,
        dry_run=dry_run,
    )

    if dry_run:
        log.info(
            "dry_run_episodes",
            fault=fault.name,
            buckets_over_limit=len(observations),
            peak_incidents=max(b.episodes for b in buckets),
            would_publish=(
                {"severity": plan.publish.severity, "episodes": plan.publish.state.episodes}
                if plan.publish is not None
                else None
            ),
        )
        return

    if plan.publish is not None:
        _publish_volume(settings, fault.name, plan.publish)
    with write_connection(settings) as conn, conn.transaction():
        episode_store.apply(conn, fault.name, plan.inserts, plan.updates, plan.orphan_closes)


def _run_external(settings: Settings, fault: Fault, *, dry_run: bool) -> None:
    """Basalte-written severities become episodes: read the fault's severity
    writes back off the bus archive, fold, reconcile — and publish nothing.
    Basalte already delivered; the engine only records.
    """
    now = datetime.now(tz=UTC)
    with read_connection(settings) as conn:
        channels = silence.resolve_scope(conn, fault.channel_scope())
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
    failed: list[str] = []
    for fault in fault_list.schedulable():
        # One fault must not take the others down with it: a catalog change
        # that leaves a device undeclared fails its own fault loudly, while
        # the rest of the list — the volume watchdog last of all — still
        # runs. The job still exits non-zero, so the CronJob shows it.
        try:
            _run_fault(settings, fault, dry_run=args.dry_run)
        except Exception:
            log.exception("fault_run_failed", fault=fault.name, kind=str(fault.kind))
            failed.append(fault.name)
    if failed:
        log.error("detect_faults_incomplete", failed=failed)
        return 1
    return 0


def _subject_kind(fault: Fault) -> SubjectKind[Any, Any] | None:
    """The per-subject shape this fault runs in, if it has one. Drift picks
    it by the signal the file declares — the loader rejects one without, so
    a fault that got here signalless is a new kind of drift nobody wired up,
    and it fails rather than reporting nothing.
    """
    if fault.kind is MeasurementKind.DRIFT:
        if fault.signal is None:
            raise ValueError(f"fault {fault.name}: a drift fault declares which series it walks")
        return _DRIFT_SIGNALS[fault.signal]
    return _SUBJECT_KINDS.get(fault.kind)


def _run_fault(settings: Settings, fault: Fault, *, dry_run: bool) -> None:
    kind = _subject_kind(fault)
    if kind is not None:
        _run_subjects(settings, fault, kind, dry_run=dry_run)
    elif fault.kind is MeasurementKind.SILENCE:
        _run_silence(settings, fault, dry_run=dry_run)
    elif fault.kind is MeasurementKind.VOLUME:
        _run_volume(settings, fault, dry_run=dry_run)
    elif fault.kind is MeasurementKind.EXTERNAL:
        _run_external(settings, fault, dry_run=dry_run)
    else:
        # Arrives with its own ticket; a declared fault must not fail
        # the ones already running.
        log.warning("fault_kind_not_implemented", fault=fault.name, kind=str(fault.kind))
