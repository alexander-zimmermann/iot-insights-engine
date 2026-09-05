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
from math import ceil
from typing import TYPE_CHECKING

from .episodes import Observation
from .nats_publisher import slugify
from .reconcile import Measured, Plan, Window, measurement_reaches, subject_plan
from .silence import BUCKET, pair_by_match, resolve_scope

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow
    from .episodes import Episode
    from .faults import DeviceReference, Fault
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
    """The declared references married to the scoped channels — never a
    silently unmeasured device."""
    return [
        Device(
            ga=channel.ga,
            name=channel.name,
            label=reference.match,
            healthy=reference.healthy_ma,
        )
        for channel, reference in pair_by_match(channels, references, noun="reference")
    ]


def min_window_samples(window: timedelta, fraction: float) -> int:
    """How many hourly buckets a trailing `window` must carry before its
    valley counts: the fault declares the fraction, the aggregate's hourly
    grid turns it into a count.
    """
    return ceil(fraction * (window / BUCKET))


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


def reaches_frontier(
    floors: Sequence[tuple[datetime, float]], *, frontier: datetime, max_gap: timedelta
) -> bool:
    """The shared `dataless` test read off the valley series, which is where
    a drift measurement ends: a device that ran through the last day, or
    sent too thinly to hold a valley, is unmeasured rather than well.
    """
    return measurement_reaches(
        floors[-1][0] if floors else None, frontier=frontier, max_gap=max_gap
    )


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

    Because the valley is a trailing minimum, one in-band sample means the
    device really did return to its healthy level at some point in the last
    day, and the count is right to start over: the budget only ever runs
    while the device did not reach healthy once in a whole day. The flip
    side is that a single low reading shadows the next 24 h, so a stuck
    relay that briefly drops out is reported a day later, not never.

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

    `rising_since` is bounded by the replay window: a drift older than the
    lookback reports the window's own start, which advances with it. The
    episode's `started_at` in the database is the stable onset.
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

    @property
    def subject(self) -> str:
        return self.ga

    @property
    def entity(self) -> str:
        return slugify(self.ga)


DriftPlan = Plan[DevicePublish]


def plan_run(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    states_by_ga: Mapping[str, DeviceState],
    dataless: frozenset[str],
    frontier: datetime,
) -> DriftPlan:
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


def hourly_floors(
    conn: psycopg.Connection[DictRow], gas: Sequence[str], window_start: datetime
) -> dict[str, list[tuple[datetime, float]]]:
    """The devices' hourly idle floors over the window, one query for the
    whole scope — 21 appliances, not 2500 channels."""
    rows = conn.execute(
        """
        SELECT ga, bucket, min(idle_floor) AS idle_floor FROM knx_appliance_1h
        WHERE ga = ANY(%(gas)s) AND bucket >= %(start)s AND idle_floor IS NOT NULL
        -- One row per hour: the aggregate also groups by knx_name, so a
        -- channel renamed mid-hour would otherwise be counted twice.
        GROUP BY ga, bucket
        ORDER BY ga, bucket
        """,
        {"gas": list(gas), "start": window_start},
    ).fetchall()
    series: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for row in rows:
        series[row["ga"]].append((row["bucket"], float(row["idle_floor"])))
    return dict(series)


def measure(
    conn: psycopg.Connection[DictRow], fault: Fault, window: Window
) -> Measured[DeviceState]:
    """The kind's whole measurement: the declared references married to the
    scope, then valley, CUSUM walk and observations per device.
    """
    rise = float(fault.parameters["rise_ma"])
    budget = float(fault.parameters["budget_ma_h"])
    trailing = timedelta(hours=float(fault.parameters["window_hours"]))
    if trailing > window.lookback:
        # A window the lookback cannot cover measures nothing, forever, and
        # would pin every open episode open. Fail the way a bad reference does.
        raise ValueError(
            f"fault {fault.name}: window_hours exceeds the {window.lookback.days}-day lookback"
        )
    min_samples = min_window_samples(trailing, float(fault.parameters["min_window_fraction"]))

    channels = resolve_scope(conn, fault.channel_scope())
    # A reference without a channel or a channel without a reference fails
    # the run loudly — never a silently unmeasured device.
    devices = resolve_devices(channels, fault.references)
    series = hourly_floors(conn, [d.ga for d in devices], window.start)

    states: dict[str, DeviceState] = {}
    observations: list[Observation] = []
    dataless: set[str] = set()
    for device in devices:
        floors = standby_floors(
            series.get(device.ga, []), window=trailing, min_samples=min_samples
        )
        if not reaches_frontier(
            floors, frontier=window.frontier, max_gap=window.policy.max_gap
        ):
            dataless.add(device.ga)
        trace = accumulate(floors, healthy=device.healthy, rise=rise)
        states[device.ga] = classify(device, trace, frontier=window.frontier)
        observations.extend(drift_observations(device.ga, trace, rise=rise, budget=budget))

    return Measured(
        states=states,
        observations=tuple(observations),
        dataless=frozenset(dataless),
        counts={
            "devices": len(devices),
            "high": sum(1 for s in states.values() if s.excess is not None and s.excess > rise),
        },
        labels={d.ga: d.label for d in devices},
    )
