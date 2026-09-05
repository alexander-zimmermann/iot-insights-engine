"""Standby-drift measurement: the `drift` kind of the fault list.

A device's standby draw sits persistently above the healthy level written
for it in the fault file — a relay that no longer opens, not a savings
topic. The measurement is a tabular CUSUM (Page 1954) with a **pinned**
reference: the healthy value is declared, never derived from history, so a
fault that has stood for months cannot quietly become the new normal. A
trailing reference would be worse than useless here — against a rolling
window a linear ramp scores the same whatever its slope (#1593), which is
why the z-score this replaces could never have seen a creeping standby.

Three steps, each its own function:

1. **The standby valley** — an appliance's hourly floor is its idle draw
   only in the hours it does not run, so the sample is the lowest hourly
   floor of the trailing day (`standby_floors`). A day the device ran
   through, or one too thinly covered to hold a valley, yields no sample.
2. **The accumulation** — every hour the valley sits more than the declared
   rise above healthy adds its excess to a budget in mA·h; an hour back
   inside the band starts the count over (`accumulate`). The rise is a
   floor, not a noise band: a permanent step smaller than it never
   accumulates, however long it stands, which is what keeps the sentence's
   "more than 40 mA" honest — the Vorratsraum dehumidifier's real +10 mA
   step would otherwise fire after a few months of standing still.
3. **The observations** — once the budget is spent, every further hour is
   an observation for the episode pipeline (`drift_observations`). The
   score is the excess in units of the declared rise, so magnitude and
   persistence stay separate: the budget decides *whether* it fires, the
   score says *how far* past the line it sits.

State is two floats and is recomputed from the aggregate on every run,
never persisted — a redeploy cannot corrupt or lose it, and replaying a
window is the same run twice. Time is the aggregate's own frontier, as
everywhere else, so a stalled materialization freezes the picture instead
of clearing episodes nobody fixed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from .episodes import Observation
from .reconcile import reconcile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow
    from .episodes import Episode
    from .faults import DeviceReference
    from .silence import Channel


@dataclass(frozen=True, slots=True)
class Device:
    """One monitored appliance: its Stromwert channel and the healthy
    standby draw declared for it, in mA."""

    ga: str
    name: str
    label: str
    healthy: float


@dataclass(frozen=True, slots=True)
class Sample:
    """One bucket of the CUSUM walk: the standby valley there, how far it
    sits above healthy, the budget spent so far, and when the current
    accumulation began.
    """

    time: datetime
    standby: float
    excess: float
    budget_used: float
    since: datetime | None = None


@dataclass(frozen=True, slots=True)
class DeviceState:
    """The device at the frontier: what it idles at now, how far that is
    above healthy, and since when it has been climbing."""

    device: Device
    standby: float | None = None
    excess: float | None = None
    rising_since: datetime | None = None


def resolve_devices(
    channels: Sequence[Channel], references: Sequence[DeviceReference]
) -> list[Device]:
    """Marry the declared references to the scoped channels, strictly both
    ways. Every problem is reported at once — a config error should name
    the whole repair, not one field per run.
    """
    problems: list[str] = []
    matched: dict[str, Device] = {}
    for reference in references:
        hits = [c for c in channels if reference.match in c.name]
        if len(hits) != 1:
            gas = ", ".join(c.ga for c in hits)
            problems.append(
                f"device {reference.match!r} matches no channel in scope"
                if not hits
                else f"device {reference.match!r} matches {len(hits)} channels: {gas}"
            )
            continue
        channel = hits[0]
        if channel.ga in matched:
            problems.append(
                f"channel {channel.ga} matched by "
                f"{matched[channel.ga].label!r} and {reference.match!r}"
            )
            continue
        matched[channel.ga] = Device(
            ga=channel.ga,
            name=channel.name,
            label=reference.match,
            healthy=reference.healthy_ma,
        )
    problems.extend(
        f"channel {c.ga} ({c.name}) has no declared reference"
        for c in channels
        if c.ga not in matched
    )
    if problems:
        raise ValueError("device references do not fit the scope: " + "; ".join(problems))
    return sorted(matched.values(), key=lambda d: d.ga)


def standby_floors(
    buckets: Sequence[tuple[datetime, float]],
    *,
    window: timedelta,
    min_samples: int,
) -> list[tuple[datetime, float]]:
    """The device's standby valley at each bucket: the lowest hourly floor
    in the trailing `window`, including the bucket itself.

    A window carrying fewer than `min_samples` buckets yields no sample at
    all — the device may simply have run through the day, and a valley read
    off three hours is not one. That also swallows the first hours of any
    window, which have no day behind them yet.
    """
    floors: list[tuple[datetime, float]] = []
    start = 0
    for end, (time, _) in enumerate(buckets):
        while buckets[start][0] <= time - window:
            start += 1
        span = buckets[start : end + 1]
        if len(span) >= min_samples:
            floors.append((time, min(v for _, v in span)))
    return floors


def accumulate(
    floors: Sequence[tuple[datetime, float]], *, healthy: float, rise: float
) -> list[Sample]:
    """The CUSUM walk over the standby valleys: `S = S + (excess - rise)`
    while the valley sits more than `rise` above `healthy`, back to zero the
    moment it returns into the band.

    Zeroing on return is what keeps an episode honest about the present: a
    plain CUSUM decays only at `rise` per hour, so a repaired device would
    go on reporting for days on nothing but accumulated history. The cost is
    that a fault flickering in and out of the band never fills its budget —
    which is the reading the sentence asks for, since such a device is not
    *persistently* high.

    An hour without a valley contributes nothing: the budget counts hours
    the device was measurably high, never hours nobody looked.
    """
    trace: list[Sample] = []
    budget_used = 0.0
    since: datetime | None = None
    for time, standby in floors:
        excess = standby - healthy
        if excess <= rise:
            budget_used = 0.0
            since = None
        else:
            budget_used += excess - rise
            since = since or time
        trace.append(
            Sample(
                time=time,
                standby=standby,
                excess=excess,
                budget_used=budget_used,
                since=since,
            )
        )
    return trace


def drift_observations(
    ga: str, trace: Sequence[Sample], *, rise: float, budget: float
) -> list[Observation]:
    """One observation per hour the spent budget stands past the declared
    one, for the episode pipeline. The score is the excess in units of the
    declared rise — the fault's own unit — and the value is that excess in
    mA, the number a human acts on.
    """
    return [
        Observation(subject=ga, time=s.time, score=s.excess / rise, value=s.excess)
        for s in trace
        if s.budget_used > budget
    ]


def classify(device: Device, trace: Sequence[Sample], *, frontier: datetime) -> DeviceState:
    """What the device idles at now — the payload's side of the severity.
    A trace that does not reach the frontier says nothing about now.
    """
    if not trace or trace[-1].time != frontier:
        return DeviceState(device)
    last = trace[-1]
    return DeviceState(
        device=device,
        standby=last.standby,
        excess=last.excess,
        rising_since=last.since,
    )


@dataclass(frozen=True, slots=True)
class DevicePublish:
    """One device whose severity moved — the payload names the device, what
    it idles at, what it should idle at, and since when it has been high;
    the writer rule carries only the severity to the device's
    Stromwert-Standby-Anomalie address.
    """

    ga: str
    severity: int
    device: str
    name: str
    standby: float | None
    healthy: float | None
    excess: float | None
    rising_since: datetime | None


@dataclass(frozen=True, slots=True)
class DriftPlan:
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
) -> DriftPlan:
    """The shared reconciliation, delivered per device: a publish goes out
    when a device's severity moved, including the 0 when its episode ends.
    """
    result = reconcile(
        episodes=episodes, open_rows=open_rows, dataless=dataless, frontier=frontier
    )
    return DriftPlan(
        inserts=result.inserts,
        updates=result.updates,
        orphan_closes=result.orphan_closes,
        stale_opens=result.stale_opens,
        publishes=tuple(
            _publish_for(subject, severity, states_by_ga.get(subject))
            for subject, severity in result.moved
        ),
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
            standby=None,
            healthy=None,
            excess=None,
            rising_since=None,
        )
    return DevicePublish(
        ga=subject,
        severity=severity,
        device=state.device.label,
        name=state.device.name,
        standby=state.standby,
        healthy=state.device.healthy,
        excess=state.excess,
        rising_since=state.rising_since,
    )


def frontier(conn: psycopg.Connection[DictRow]) -> datetime | None:
    """The appliance aggregate's newest bucket anywhere — the 'now' every
    run is measured against."""
    row = conn.execute("SELECT max(bucket) AS frontier FROM knx_appliance_1h").fetchone()
    return row["frontier"] if row else None


def hourly_floors(
    conn: psycopg.Connection[DictRow], gas: Sequence[str], window_start: datetime
) -> dict[str, list[tuple[datetime, float]]]:
    """The devices' hourly idle floors over the window, one query for the
    whole scope — 21 appliances, not 2500 channels."""
    rows = conn.execute(
        """
        SELECT ga, bucket, idle_floor FROM knx_appliance_1h
        WHERE ga = ANY(%(gas)s) AND bucket >= %(start)s AND idle_floor IS NOT NULL
        ORDER BY ga, bucket
        """,
        {"gas": list(gas), "start": window_start},
    ).fetchall()
    series: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for row in rows:
        series[row["ga"]].append((row["bucket"], float(row["idle_floor"])))
    return dict(series)
