"""Detect-faults job: run the declared fault list end to end.

For each schedulable fault the runner resolves the scope where the catalog
lives (`ga_catalog` in TSDB — never a hand-written address list), measures,
folds the observations into episodes behind the pure pipeline seam, and
reconciles the result with the episodes the database already holds. What
leaves the engine is a severity 0–3 per main group on
`anomaly.<fault>.<main_group>`; the knx-nats-bridge writer rules carry it
to the group's Zentral diagnosis address, where Basalte owns the text.

Every kind runs in one shape: a `runner.Kind` declares how its series is
measured and how its payload is shaped, and the runner module owns the
rest — the window, the fold, the reconciliation, the log record, the dry
run and the publish-then-write tail, behind its injected store and
publisher ends. This module is the job: it loads the fault list, wires
each kind's declaration, and runs the list fault by fault.

Channel silence measures per channel but reports per main group, so its
declaration carries its own plan; the lifecycle around it is the same
runner as everything else's.

A deviation fault that names an expectation runs the loop in days rather
than hours: the plant's whole-day yield against the kWh its named model
expected of it, on the plant's own address.

The volume watchdog runs the loop over the engine's own output: it counts
the incidents of the last seven days out of the episode stream and puts a
severity on one house-wide address, so drift back into noise arrives on the
same bus as everything else.

External faults run the same loop the other way round: Basalte detects,
writes the severity to the fault address and delivers itself; the engine
reads those writes back from the bus archive, records episodes marked
externally delivered, and publishes nothing. Their declared frontier is
the wall clock: the bus archive has no materialization lag, so an
orphaned row closes at wall-clock time.

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
from pathlib import Path
from typing import Any

from . import (
    deviation,
    drift,
    duration,
    external,
    silence,
    volume,
)
from .config import Settings
from .faults import DriftSignal, Fault, FaultList, MeasurementKind
from .logging_setup import get_logger
from .runner import (
    DbStore,
    Kind,
    NatsPublisher,
    run_subjects,
)

log = get_logger(__name__)


_KINDS: Mapping[MeasurementKind, Kind[Any, Any]] = {
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
    MeasurementKind.VOLUME: Kind(
        event="notification_volume_run",
        # One house-wide address; declared last in the fault list, so the
        # count already includes what this run's other faults just wrote.
        delivery="ga",
        frontier=silence.frontier,
        measure=volume.measure,
        publish_for=volume.publish_for,
        payload=volume.payload,
    ),
    MeasurementKind.EXTERNAL: Kind(
        event="external_severities_run",
        # Basalte detects and delivers itself: the fault declares no
        # target, the fold walks severity writes instead of observations,
        # the plan keeps rows open until their explicit 0, and nothing
        # is published — the engine only records.
        delivery=None,
        frontier=external.frontier,
        measure=external.measure,
        fold=external.fold,
        plan=external.plan,
        externally_delivered=True,
    ),
}


# The deviation kind's other shape: a fault that names an expectation
# measures the plant's whole-day yield against it, in daily buckets, on the
# plant's one declared address.
_DAILY_YIELD: Kind[Any, Any] = Kind(
    event="daily_yield_run",
    delivery="ga",
    frontier=deviation.yield_frontier,
    measure=deviation.measure_yield,
    publish_for=deviation.publish_for_plant,
    payload=deviation.payload_yield,
    policy=deviation.YIELD_POLICY,
)


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


def _kind_for(fault: Fault) -> Kind[Any, Any] | None:
    """The shape this fault runs in, if it has one. Drift picks it by the
    signal the file declares — the loader rejects one without, so a fault
    that got here signalless is a new kind of drift nobody wired up, and it
    fails rather than reporting nothing. Deviation picks it by whether the
    fault names an expectation: with one it measures a daily yield against
    that model, without one a room against its setpoint.
    """
    if fault.kind is MeasurementKind.DRIFT:
        if fault.signal is None:
            raise ValueError(f"fault {fault.name}: a drift fault declares which series it walks")
        return _DRIFT_SIGNALS[fault.signal]
    if fault.kind is MeasurementKind.DEVIATION and fault.expectation is not None:
        return _DAILY_YIELD
    return _KINDS.get(fault.kind)


def _run_fault(settings: Settings, fault: Fault, *, dry_run: bool) -> None:
    kind = _kind_for(fault)
    if kind is None:
        # Arrives with its own ticket; a declared fault must not fail
        # the ones already running.
        log.warning("fault_kind_not_implemented", fault=fault.name, kind=str(fault.kind))
        return
    run_subjects(DbStore(settings), NatsPublisher(settings), fault, kind, dry_run=dry_run)
