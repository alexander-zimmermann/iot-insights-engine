"""Detect-faults job: run the declared fault list end to end.

For each schedulable fault the runner resolves the scope where the catalog
lives (`ga_catalog` in TSDB — never a hand-written address list), measures,
folds the observations into episodes behind the pure pipeline seam, and
reconciles the result with the episodes the database already holds. What
leaves the engine is a severity 0–3 per main group on
`anomaly.<fault>.<main_group>`; the knx-nats-bridge writer rules carry it
to the group's Zentral diagnosis address, where Basalte owns the text.

Every kind that reports per subject runs in one shape: a `runner.Kind`
declares how its series is measured and how its payload is shaped, and
the runner module owns the rest — the window, the fold, the
reconciliation, the log record, the dry run and the publish-then-write
tail, behind its injected store and publisher ends. This module is the
job: it loads the fault list, wires each kind's declaration, and keeps
the loops the runner does not cover yet.

Channel silence measures per channel but reports per main group, so its
declaration carries its own plan; the lifecycle around it is the same
runner as everything else's.

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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from .episodes import EpisodePolicy, fold_observations
from .faults import DriftSignal, Fault, FaultList, MeasurementKind
from .logging_setup import get_logger
from .runner import (
    LOOKBACK,
    DbStore,
    Kind,
    NatsPublisher,
    run_subjects,
)
from .severity import severity_name

log = get_logger(__name__)


_SUBJECT_KINDS: Mapping[MeasurementKind, Kind[Any, Any]] = {
    MeasurementKind.DURATION: Kind(
        event="appliance_runtime_run",
        delivery="per_device",
        frontier=duration.frontier,
        measure=duration.measure,
        publish_for=duration.publish_for,
        payload=duration.payload,
    ),
    MeasurementKind.DEVIATION: Kind(
        event="room_deviation_run",
        delivery="per_room",
        frontier=silence.frontier,
        measure=deviation.measure,
        publish_for=deviation.publish_for,
        payload=deviation.payload,
    ),
    MeasurementKind.SILENCE: Kind(
        event="channel_silence_run",
        delivery="per_main_group",
        frontier=silence.frontier,
        measure=silence.measure,
        # Measured per channel, delivered per main group — the one kind
        # whose plan is its own.
        plan=silence.plan_run,
        payload=silence.group_payload,
        # The dataless set is every never-sent symmetry address in the
        # catalog — a thousand of them, normal, and already counted by the
        # measurement's scope_drops record. The ones actually held open are
        # `stale_opens` in the run record.
        warn_dataless=False,
    ),
}


# The drift kind runs one shape per signal: same CUSUM, different series,
# so the run record and the payload's units differ with the signal the
# fault declares.
_DRIFT_SIGNALS: Mapping[DriftSignal, Kind[Any, Any]] = {
    DriftSignal.STANDBY: Kind(
        event="appliance_standby_run",
        delivery="per_device",
        frontier=duration.frontier,
        measure=drift.measure_standby,
        publish_for=drift.publish_for,
        payload=drift.payload_standby,
    ),
    DriftSignal.DUTY_CYCLE: Kind(
        event="duty_cycle_drift_run",
        delivery="per_device",
        frontier=duration.frontier,
        measure=drift.measure_duty_cycle,
        publish_for=drift.publish_for,
        payload=drift.payload_duty_cycle,
    ),
    DriftSignal.RECOVERY: Kind(
        event="heat_recovery_run",
        # One exchanger, one declared address — the volume watchdog's form,
        # not the appliances' per-device fan-out.
        delivery="ga",
        frontier=silence.frontier,
        measure=drift.measure_recovery,
        publish_for=drift.publish_for_exchanger,
        payload=drift.payload_recovery,
    ),
}


def _publish_volume(
    settings: Settings, fault_name: str, publish: volume.VolumePublish
) -> None:
    firing = publish.severity > 0
    nats_publisher.publish_anomaly(
        settings,
        fault_name,
        severity_name(publish.severity) if firing else None,
        volume.payload(publish),
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


def _subject_kind(fault: Fault) -> Kind[Any, Any] | None:
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
        run_subjects(DbStore(settings), NatsPublisher(settings), fault, kind, dry_run=dry_run)
    elif fault.kind is MeasurementKind.VOLUME:
        _run_volume(settings, fault, dry_run=dry_run)
    elif fault.kind is MeasurementKind.EXTERNAL:
        _run_external(settings, fault, dry_run=dry_run)
    else:
        # Arrives with its own ticket; a declared fault must not fail
        # the ones already running.
        log.warning("fault_kind_not_implemented", fault=fault.name, kind=str(fault.kind))
