"""Appliance-runtime measurement: the `duration` kind of the fault list.

A device draws current for longer than its declared limit — "it was left
on, or it is stuck". Activity comes from the hourly appliance aggregate
(`knx_appliance_1h`, scoped to the Stromwert channels with the on-threshold
baked in); an hour counts as active when at least the declared fraction of
its samples sat above the standby valley. Consecutive active hours form a
run via the shared runs preparation; every bucket past the device's limit
becomes an observation whose score is the run length in units of that
limit — the fault's declared unit.

The limits are declared per device in the fault file and resolved against
the catalog scope strictly both ways: a limit that names no channel and a
channel no limit names are both config errors that fail the run loudly —
never a silently unmeasured device.

Like silence, time is the aggregate's own frontier, so the materialization
lag cancels instead of ending every run an hour early.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from .episodes import Observation
from .nats_publisher import slugify
from .reconcile import Measured, Plan, Window, subject_plan
from .runs import split_runs
from .silence import BUCKET, pair_by_match, resolve_scope

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow
    from .episodes import Episode
    from .faults import DeviceLimit, Fault
    from .silence import Channel


@dataclass(frozen=True, slots=True)
class Device:
    """One monitored appliance: its Stromwert channel and declared limit."""

    ga: str
    name: str
    label: str
    max_run: timedelta


@dataclass(frozen=True, slots=True)
class DeviceState:
    """Whether the device is running at the frontier, and for how long."""

    device: Device
    running_since: datetime | None = None
    run_hours: float | None = None


@dataclass(frozen=True, slots=True)
class Activity:
    """The window's activity per channel: the active buckets, and the newest
    bucket each channel delivered a row for at all — running or not, that is
    how far the device was measured.
    """

    active: Mapping[str, list[datetime]]
    last_bucket: Mapping[str, datetime]


def resolve_devices(channels: Sequence[Channel], limits: Sequence[DeviceLimit]) -> list[Device]:
    """The declared limits married to the scoped channels — never a silently
    unmeasured device."""
    return [
        Device(
            ga=channel.ga,
            name=channel.name,
            label=limit.match,
            max_run=timedelta(hours=limit.max_run_hours),
        )
        for channel, limit in pair_by_match(channels, limits, noun="limit")
    ]


def duration_observations(
    ga: str, active: Sequence[datetime], max_run: timedelta
) -> list[Observation]:
    """One observation per bucket a run stands past the device's limit, for
    the episode pipeline. The score is the run length so far in units of the
    limit; the value is that length in hours. A bucket covers its full hour,
    so a run's first bucket already counts as one.
    """
    observations: list[Observation] = []
    for run in split_runs(active, BUCKET):
        t = run.start
        while t <= run.end:
            elapsed = t - run.start + BUCKET
            if elapsed > max_run:
                observations.append(
                    Observation(subject=ga, time=t, score=elapsed / max_run, value=elapsed / BUCKET)
                )
            t += BUCKET
    return observations


def classify(device: Device, active: Sequence[datetime], frontier: datetime) -> DeviceState:
    """The device's current run, if one reaches the frontier — what the
    publish payload names alongside the severity."""
    runs = split_runs(active, BUCKET)
    if runs and runs[-1].end == frontier:
        current = runs[-1]
        return DeviceState(
            device, running_since=current.start, run_hours=current.duration / BUCKET
        )
    return DeviceState(device)


@dataclass(frozen=True, slots=True)
class DevicePublish:
    """One device whose severity or openness moved — the payload names the
    device, its current run and its limit; the writer rule carries only the
    severity to the device's Dauerbetrieb address.
    """

    ga: str
    severity: int
    device: str
    name: str
    running_since: datetime | None
    run_hours: float | None
    limit_hours: float | None

    @property
    def subject(self) -> str:
        return self.ga

    @property
    def entity(self) -> str:
        return slugify(self.ga)


DurationPlan = Plan[DevicePublish]


def plan_run(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    states_by_ga: Mapping[str, DeviceState],
    dataless: frozenset[str],
    frontier: datetime,
) -> DurationPlan:
    """The shared reconciliation, delivered per device."""

    def payload(subject: str, severity: int) -> DevicePublish:
        return publish_for(subject, severity, states_by_ga.get(subject))

    return subject_plan(
        episodes=episodes,
        open_rows=open_rows,
        dataless=dataless,
        frontier=frontier,
        publish_for=payload,
    )


def publish_for(subject: str, severity: int, state: DeviceState | None) -> DevicePublish:
    # A device that left the scope while its row was open still gets its
    # clear; the payload then only names the address.
    if state is None:
        return DevicePublish(
            ga=subject,
            severity=severity,
            device=subject,
            name=subject,
            running_since=None,
            run_hours=None,
            limit_hours=None,
        )
    return DevicePublish(
        ga=subject,
        severity=severity,
        device=state.device.label,
        name=state.device.name,
        running_since=state.running_since,
        run_hours=state.run_hours,
        limit_hours=state.device.max_run / BUCKET,
    )


def frontier(conn: psycopg.Connection[DictRow]) -> datetime | None:
    """The appliance aggregate's newest bucket anywhere — the 'now' every
    run is measured against."""
    row = conn.execute("SELECT max(bucket) AS frontier FROM knx_appliance_1h").fetchone()
    return row["frontier"] if row else None


def activity(
    conn: psycopg.Connection[DictRow],
    gas: Sequence[str],
    window_start: datetime,
    active_fraction: float,
) -> Activity:
    """The devices' active buckets over the window, one query for the whole
    scope — 21 appliances, not 2500 channels."""
    rows = conn.execute(
        """
        SELECT ga, bucket, on_samples, total_samples FROM knx_appliance_1h
        WHERE ga = ANY(%(gas)s) AND bucket >= %(start)s
        ORDER BY ga, bucket
        """,
        {"gas": list(gas), "start": window_start},
    ).fetchall()
    active: dict[str, list[datetime]] = defaultdict(list)
    last_bucket: dict[str, datetime] = {}
    for row in rows:
        last_bucket[row["ga"]] = row["bucket"]
        total = row["total_samples"]
        if total and row["on_samples"] / total >= active_fraction:
            active[row["ga"]].append(row["bucket"])
    return Activity(active=dict(active), last_bucket=last_bucket)


def measure(
    conn: psycopg.Connection[DictRow], fault: Fault, window: Window
) -> Measured[DeviceState]:
    """The kind's whole measurement: the declared limits married to the
    scope, the window's activity, and one run reconstruction per device.
    """
    active_fraction = float(fault.parameters["active_hour_fraction"])
    channels = resolve_scope(conn, fault.channel_scope())
    # A limit without a channel or a channel without a limit fails the run
    # loudly — never a silently unmeasured device.
    devices = resolve_devices(channels, fault.devices)
    seen = activity(conn, [d.ga for d in devices], window.start, active_fraction)

    states: dict[str, DeviceState] = {}
    observations: list[Observation] = []
    dataless: set[str] = set()
    for device in devices:
        if not window.reaches(seen.last_bucket.get(device.ga)):
            dataless.add(device.ga)
        active = seen.active.get(device.ga, [])
        states[device.ga] = classify(device, active, window.frontier)
        observations.extend(duration_observations(device.ga, active, device.max_run))

    return Measured(
        states=states,
        observations=tuple(observations),
        dataless=frozenset(dataless),
        counts={
            "devices": len(devices),
            "running": sum(1 for s in states.values() if s.running_since is not None),
            "over_limit": sum(
                1
                for s in states.values()
                if s.run_hours is not None and s.run_hours * BUCKET > s.device.max_run
            ),
        },
        labels={d.ga: d.label for d in devices},
    )
