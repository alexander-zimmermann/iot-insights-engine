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

from .episodes import Episode, Observation
from .runs import split_runs
from .silence import BUCKET

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow
    from .faults import DeviceLimit
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
    """The window's activity per channel: the active buckets, and which
    channels delivered any row at all — a channel absent here is silent,
    which is not this fault's call to clear.
    """

    active: Mapping[str, list[datetime]]
    present: frozenset[str]


def resolve_devices(channels: Sequence[Channel], limits: Sequence[DeviceLimit]) -> list[Device]:
    """Marry the declared limits to the scoped channels, strictly both ways.
    Every problem is reported at once — a config error should name the whole
    repair, not one field per run.
    """
    problems: list[str] = []
    matched: dict[str, Device] = {}
    for limit in limits:
        hits = [c for c in channels if limit.match in c.name]
        if len(hits) != 1:
            gas = ", ".join(c.ga for c in hits)
            problems.append(
                f"device {limit.match!r} matches no channel in scope"
                if not hits
                else f"device {limit.match!r} matches {len(hits)} channels: {gas}"
            )
            continue
        channel = hits[0]
        if channel.ga in matched:
            problems.append(
                f"channel {channel.ga} matched by {matched[channel.ga].label!r} and {limit.match!r}"
            )
            continue
        matched[channel.ga] = Device(
            ga=channel.ga,
            name=channel.name,
            label=limit.match,
            max_run=timedelta(hours=limit.max_run_hours),
        )
    problems.extend(
        f"channel {c.ga} ({c.name}) has no declared limit" for c in channels if c.ga not in matched
    )
    if problems:
        raise ValueError("device limits do not fit the scope: " + "; ".join(problems))
    return sorted(matched.values(), key=lambda d: d.ga)


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


@dataclass(frozen=True, slots=True)
class DurationPlan:
    """What one run changes: new episodes, reconciled open rows, orphaned
    rows to close, rows kept open for want of data, and the devices whose
    severity moved.
    """

    inserts: tuple[Episode, ...]
    updates: tuple[tuple[int, Episode], ...]
    orphan_closes: tuple[tuple[int, datetime], ...]
    stale_opens: tuple[str, ...]
    publishes: tuple[DevicePublish, ...]


def plan_run(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    states_by_ga: Mapping[str, DeviceState],
    dataless: frozenset[str],
    frontier: datetime,
) -> DurationPlan:
    """Pure reconciliation: computed episodes against the stored open rows,
    with the same guarantees the silence plan gives — a stored severity is
    never lowered, an open row whose device produced no episode closes at
    the frontier unless the device is `dataless` (no bucket in the whole
    window: with nothing to decide a recovery, it must not self-clear).

    Delivery is per device: a publish goes out when a device's severity
    moved, including the 0 when its episode ends.
    """
    open_by_subject = {row.subject: row for row in open_rows}

    # Several episodes per device can fall in the window; the stored open
    # row can only correspond to the latest one.
    latest_by_subject: dict[str, Episode] = {}
    for episode in episodes:
        current = latest_by_subject.get(episode.subject)
        if current is None or episode.started_at > current.started_at:
            latest_by_subject[episode.subject] = episode

    inserts: list[Episode] = []
    updates: list[tuple[int, Episode]] = []
    after: dict[str, int] = {}

    for subject, episode in sorted(latest_by_subject.items()):
        row = open_by_subject.get(subject)
        if episode.ended_at is None:
            after[subject] = max(episode.severity, row.severity if row else 0)
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
            # Still over its limit as far as anyone can tell — it keeps counting.
            after[row.subject] = row.severity
        else:
            orphan_closes.append((row.id, frontier))

    before = {row.subject: row.severity for row in open_rows}
    publishes = tuple(
        _publish_for(subject, after.get(subject, 0), states_by_ga.get(subject))
        for subject in sorted(set(before) | set(after))
        if before.get(subject) != after.get(subject)
    )

    return DurationPlan(
        inserts=tuple(inserts),
        updates=tuple(updates),
        orphan_closes=tuple(orphan_closes),
        stale_opens=tuple(stale_opens),
        publishes=publishes,
    )


def _publish_for(subject: str, severity: int, state: DeviceState | None) -> DevicePublish:
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
    present: set[str] = set()
    for row in rows:
        present.add(row["ga"])
        total = row["total_samples"]
        if total and row["on_samples"] / total >= active_fraction:
            active[row["ga"]].append(row["bucket"])
    return Activity(active=dict(active), present=frozenset(present))
