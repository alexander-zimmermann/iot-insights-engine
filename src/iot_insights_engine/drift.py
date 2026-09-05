"""Drift measurement: the `drift` kind of the fault list.

Something a device does sits persistently above the healthy level declared
for it. The measurement is a tabular CUSUM (Page 1954) with a **pinned**
reference: the healthy value is declared, never derived from history, so a
fault that has stood for months cannot quietly become the new normal. A
trailing reference would be worse than useless here — against a rolling
window a linear ramp scores the same whatever its slope (#1593), which is
why the z-score this replaces could never have seen a creeping standby.

Two signals walk that same CUSUM, and the fault file says which:

* **standby** — the device's idle draw in mA. A relay that no longer opens,
  not a savings topic.
* **duty_cycle** — the share of the day a compressor runs, in percent. An
  iced-up evaporator buys the same cold with far more running, months
  before anything in the freezer thaws.

Three steps, each its own function:

1. **The level** — the trailing window's reading of the signal. Standby is
   the *lowest* hourly floor of the trailing day (`standby_floors`): an
   appliance's hourly floor is its idle draw only in the hours it does not
   run. Duty cycle is the share of the trailing day the compressor actually
   drew above the declared on-threshold (`duty_cycles`), with door events
   cut out first. A window too thinly covered to hold a reading yields no
   sample.
2. **The accumulation** — every hour the level sits more than the declared
   rise above healthy adds its excess to a budget; an hour back inside the
   band starts the count over (`accumulate`). The rise is a floor, not a
   noise band: a permanent step smaller than it never accumulates, however
   long it stands, which is what keeps the sentence's "more than 40 mA"
   honest — the Vorratsraum dehumidifier's real +10 mA step would otherwise
   fire after a few months of standing still.
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
from .runs import split_runs
from .silence import BUCKET, pair_by_match, resolve_scope

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow
    from .episodes import Episode
    from .faults import DeviceReference, Fault
    from .silence import Channel


@dataclass(frozen=True, slots=True)
class Device:
    """One monitored appliance: its Stromwert channel and the healthy level
    declared for it, in the signal's own unit."""

    ga: str
    name: str
    label: str
    healthy: float


# A telegram's value stands until the next one, but no longer than this: a
# longer gap is the bridge missing, not a compressor running for hours, and
# silence is its own fault.
MAX_HOLD = BUCKET
# An hour telegrams account for less than half of is a delivery gap, not a
# reading.
MIN_COVERAGE = 0.5
# ... and one with less than a minute of idle in it counts as never idle:
# every appliance channel sends cyclically even standing still, so a truly
# running hour reads as a clean 1.0.
IDLE_TOLERANCE = timedelta(minutes=1)


@dataclass(frozen=True, slots=True)
class OnTime:
    """One hour of a device's on-time: how much of the hour it drew above the
    on threshold, and how much of the hour telegrams accounted for at all.

    Time, not telegram counts: KNX sends on change, so a running compressor
    emits hundreds of telegrams an hour and an idle one a handful. A share
    of counts would read a freezer running half the day as running almost
    all of it — the pinned reference would then be a number about telegram
    rates instead of about the day.
    """

    time: datetime
    on: timedelta
    total: timedelta

    @property
    def covered(self) -> bool:
        return self.total >= MIN_COVERAGE * BUCKET

    @property
    def saturated(self) -> bool:
        """The compressor never stopped inside this hour — which ordinary
        cycling, at 25 to 46 minutes a run, never does."""
        return self.covered and self.total - self.on <= IDLE_TOLERANCE


@dataclass(frozen=True, slots=True)
class Step:
    """One bucket of the CUSUM walk: the level there, how far it sits above
    healthy, the budget spent so far, and when the current accumulation
    began.
    """

    time: datetime
    level: float
    excess: float
    budget_used: float
    since: datetime | None = None


@dataclass(frozen=True, slots=True)
class DeviceState:
    """The device at the frontier: what its level reads now, how far that is
    above healthy, and since when it has been climbing."""

    device: Device
    level: float | None = None
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
            healthy=reference.healthy,
        )
        for channel, reference in pair_by_match(channels, references, noun="reference")
    ]


def min_window_samples(window: timedelta, fraction: float) -> int:
    """How many hourly buckets a trailing `window` must carry before its
    reading counts: the fault declares the fraction, the aggregate's hourly
    grid turns it into a count.
    """
    return ceil(fraction * (window / BUCKET))


def trailing_windows(
    times: Sequence[datetime], *, window: timedelta, min_samples: int
) -> Iterator[tuple[datetime, slice]]:
    """Each bucket's trailing `window` as a slice into the series, including
    the bucket itself — the shape both signals read their level off.

    A window carrying fewer than `min_samples` buckets yields nothing at
    all: the device may simply have run through the day, and a reading taken
    off three hours is not one. That also swallows the first hours of any
    series, which have no day behind them yet.
    """
    start = 0
    for end, time in enumerate(times):
        while times[start] <= time - window:
            start += 1
        if end - start + 1 >= min_samples:
            yield time, slice(start, end + 1)


def standby_floors(
    buckets: Sequence[tuple[datetime, float]],
    *,
    window: timedelta,
    min_samples: int,
) -> list[tuple[datetime, float]]:
    """The device's standby valley at each bucket: the lowest hourly floor
    in the trailing `window`.
    """
    times = [time for time, _ in buckets]
    return [
        (time, min(v for _, v in buckets[span]))
        for time, span in trailing_windows(times, window=window, min_samples=min_samples)
    ]


def door_hours(buckets: Sequence[OnTime], *, door_run: timedelta) -> frozenset[datetime]:
    """The hours a door-open event covers: at least `door_run` in a row in
    which the compressor never dropped below the on-threshold.

    A door left open is the one thing that looks like an iced evaporator and
    is not — both make the compressor run far more than it should — so it is
    cut out of the series rather than scored. They are told apart by shape,
    not by duration alone: ordinary cycling leaves an idle sample in every
    hour even when a warm load stretches its runs, while a door ajar has the
    compressor running through hours on end. That is also why the hourly
    grid suffices here, where the runtime fault could not use it at all.

    Those events are Basalte's fault to report, on the door addresses it
    owns; here they only have to stop counting as ice.
    """
    saturated = [b.time for b in buckets if b.saturated]
    hours: set[datetime] = set()
    for run in split_runs(saturated, BUCKET):
        if run.duration < door_run:
            continue
        time = run.start
        while time <= run.end:
            hours.add(time)
            time += BUCKET
    return frozenset(hours)


def duty_cycles(
    buckets: Sequence[OnTime],
    *,
    window: timedelta,
    min_samples: int,
    excluded: frozenset[datetime],
) -> list[tuple[datetime, float]]:
    """The compressor's duty cycle at each covered bucket, in percent: its
    on-time over the trailing `window`, counting only hours that are neither
    a delivery gap nor part of a door event.

    The series stays hourly even across a door event — the level is read
    *at* every covered hour, only never *from* an excluded one. A hole would
    be worse than a wrong number here: an observation gap longer than the
    episode policy's tolerance splits one incident into one episode a day,
    and a freezer whose door is used every afternoon would clear and re-fire
    its address daily instead of standing as the one situation it is.

    Leaving the door hours out of the coverage too is what keeps that
    honest: a day that spent most of itself with the door open drops out
    under `min_samples` instead of reporting a reading taken around it.
    """
    kept = [b for b in buckets if b.covered and b.time not in excluded]
    levels: list[tuple[datetime, float]] = []
    first, past = 0, 0
    for bucket in buckets:
        if not bucket.covered:
            continue
        while past < len(kept) and kept[past].time <= bucket.time:
            past += 1
        while first < past and kept[first].time <= bucket.time - window:
            first += 1
        span = kept[first:past]
        if len(span) < min_samples:
            continue
        on = sum((b.on for b in span), timedelta())
        total = sum((b.total for b in span), timedelta())
        levels.append((bucket.time, 100.0 * (on / total)))
    return levels


def reaches_frontier(
    levels: Sequence[tuple[datetime, float]], *, frontier: datetime, max_gap: timedelta
) -> bool:
    """The shared `dataless` test read off the level series, which is where
    a drift measurement ends: a device that ran through the last day, or
    sent too thinly to hold a reading, is unmeasured rather than well.
    """
    return measurement_reaches(
        levels[-1][0] if levels else None, frontier=frontier, max_gap=max_gap
    )


def accumulate(
    levels: Sequence[tuple[datetime, float]], *, healthy: float, rise: float
) -> list[Step]:
    """The CUSUM walk over the levels: `S = S + (excess - rise)` while the
    level sits more than `rise` above `healthy`, back to zero the moment it
    returns into the band.

    Zeroing on return is what keeps an episode honest about the present: a
    plain CUSUM decays only at `rise` per hour, so a repaired device would
    go on reporting for days on nothing but accumulated history. The cost is
    that a fault flickering in and out of the band never fills its budget —
    which is the reading the sentence asks for, since such a device is not
    *persistently* high.

    Because the level is a trailing window, one in-band sample means the
    device really did read healthy at some point in the last day, and the
    count is right to start over: the budget only ever runs while the device
    did not reach healthy once in a whole day. The flip side is that a
    single low reading shadows the next 24 h, so a stuck relay that briefly
    drops out is reported a day later, not never.

    An hour without a level contributes nothing: the budget counts hours the
    device was measurably high, never hours nobody looked.
    """
    trace: list[Step] = []
    budget_used = 0.0
    since: datetime | None = None
    for time, level in levels:
        excess = level - healthy
        if excess <= rise:
            budget_used = 0.0
            since = None
        else:
            budget_used += excess - rise
            since = since or time
        trace.append(
            Step(
                time=time,
                level=level,
                excess=excess,
                budget_used=budget_used,
                since=since,
            )
        )
    return trace


def drift_observations(
    ga: str, trace: Sequence[Step], *, rise: float, budget: float
) -> list[Observation]:
    """One observation per hour the spent budget stands past the declared
    one, for the episode pipeline. The score is the excess in units of the
    declared rise — the fault's own unit — and the value is that excess in
    the signal's unit, the number a human acts on.
    """
    return [
        Observation(subject=ga, time=s.time, score=s.excess / rise, value=s.excess)
        for s in trace
        if s.budget_used > budget
    ]


def classify(device: Device, trace: Sequence[Step], *, frontier: datetime) -> DeviceState:
    """What the device reads now — the payload's side of the severity. A
    trace that does not reach the frontier says nothing about now.

    `rising_since` is bounded by the replay window: a drift older than the
    lookback reports the window's own start, which advances with it. The
    episode's `started_at` in the database is the stable onset.
    """
    if not trace or trace[-1].time != frontier:
        return DeviceState(device)
    last = trace[-1]
    return DeviceState(
        device=device,
        level=last.level,
        excess=last.excess,
        rising_since=last.since,
    )


@dataclass(frozen=True, slots=True)
class DevicePublish:
    """One device whose severity moved — the payload names the device, what
    it reads, what it should read, and since when it has been high; the
    writer rule carries only the severity to the device's fault address.
    """

    ga: str
    severity: int
    device: str
    name: str
    level: float | None
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
            level=None,
            healthy=None,
            excess=None,
            rising_since=None,
        )
    return DevicePublish(
        ga=subject,
        severity=severity,
        device=state.device.label,
        name=state.device.name,
        level=state.level,
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


def hourly_on_time(
    conn: psycopg.Connection[DictRow], gas: Sequence[str], window: Window, threshold: float
) -> dict[str, list[OnTime]]:
    """The devices' hourly on-time over the window, from the bus archive.

    The appliance aggregate cannot answer this: it counts telegrams, and a
    running compressor sends two orders of magnitude more of them than an
    idle one. So the archive is read the way KNX means it — each telegram's
    value stands until the next — and the holds are summed per hour. The
    query is indexed on `ga` (the compression segment) and bounded on both
    sides, and it aggregates in the database: what comes back is one row per
    hour per device, not the telegrams themselves.

    Buckets past the aggregate's frontier are left out, so this fault ends
    its window where its siblings end theirs.
    """
    rows = conn.execute(
        """
        WITH held AS (
            SELECT ga, time, value,
                   least(
                       lead(time) OVER (PARTITION BY ga ORDER BY time) - time,
                       %(max_hold)s
                   ) AS hold
            FROM knx
            WHERE ga = ANY(%(gas)s) AND time >= %(start)s AND time < %(end)s
        )
        SELECT ga, time_bucket('1 hour', time) AS bucket,
               coalesce(sum(hold) FILTER (WHERE value > %(threshold)s), '0'::interval)
                   AS on_time,
               sum(hold) AS total_time
        FROM held WHERE hold IS NOT NULL
        GROUP BY ga, bucket
        ORDER BY ga, bucket
        """,
        {
            "gas": list(gas),
            "start": window.start,
            "end": window.frontier + BUCKET,
            "max_hold": MAX_HOLD,
            "threshold": threshold,
        },
    ).fetchall()
    series: dict[str, list[OnTime]] = defaultdict(list)
    for row in rows:
        series[row["ga"]].append(
            OnTime(time=row["bucket"], on=row["on_time"], total=row["total_time"])
        )
    return dict(series)


def _trailing_window(fault: Fault, window: Window) -> tuple[timedelta, int]:
    """The declared trailing window as an interval and a bucket count."""
    trailing = timedelta(hours=float(fault.parameters["window_hours"]))
    if trailing > window.lookback:
        # A window the lookback cannot cover measures nothing, forever, and
        # would pin every open episode open. Fail the way a bad reference does.
        raise ValueError(
            f"fault {fault.name}: window_hours exceeds the {window.lookback.days}-day lookback"
        )
    return trailing, min_window_samples(
        trailing, float(fault.parameters["min_window_fraction"])
    )


def _devices(conn: psycopg.Connection[DictRow], fault: Fault) -> list[Device]:
    # A reference without a channel or a channel without a reference fails
    # the run loudly — never a silently unmeasured device.
    return resolve_devices(resolve_scope(conn, fault.channel_scope()), fault.references)


def _walk(
    devices: Sequence[Device],
    levels: Mapping[str, Sequence[tuple[datetime, float]]],
    *,
    window: Window,
    rise: float,
    budget: float,
    counts: Mapping[str, int],
) -> Measured[DeviceState]:
    """The part both signals share: CUSUM walk, state and observations per
    device, once the levels are read."""
    states: dict[str, DeviceState] = {}
    observations: list[Observation] = []
    dataless: set[str] = set()
    for device in devices:
        series = levels.get(device.ga, ())
        if not reaches_frontier(
            series, frontier=window.frontier, max_gap=window.policy.max_gap
        ):
            dataless.add(device.ga)
        trace = accumulate(series, healthy=device.healthy, rise=rise)
        states[device.ga] = classify(device, trace, frontier=window.frontier)
        observations.extend(drift_observations(device.ga, trace, rise=rise, budget=budget))

    return Measured(
        states=states,
        observations=tuple(observations),
        dataless=frozenset(dataless),
        counts={
            "devices": len(devices),
            "high": sum(1 for s in states.values() if s.excess is not None and s.excess > rise),
            **counts,
        },
        labels={d.ga: d.label for d in devices},
    )


def measure_standby(
    conn: psycopg.Connection[DictRow], fault: Fault, window: Window
) -> Measured[DeviceState]:
    """The standby signal: the declared references married to the scope,
    then valley, CUSUM walk and observations per device.
    """
    trailing, min_samples = _trailing_window(fault, window)
    devices = _devices(conn, fault)
    series = hourly_floors(conn, [d.ga for d in devices], window.start)
    levels = {
        device.ga: standby_floors(
            series.get(device.ga, []), window=trailing, min_samples=min_samples
        )
        for device in devices
    }
    return _walk(
        devices,
        levels,
        window=window,
        rise=float(fault.parameters["rise_ma"]),
        budget=float(fault.parameters["budget_ma_h"]),
        counts={},
    )


def measure_duty_cycle(
    conn: psycopg.Connection[DictRow], fault: Fault, window: Window
) -> Measured[DeviceState]:
    """The duty-cycle signal: the same walk over the share of the day the
    compressor runs, with door events cut out of the series first.
    """
    trailing, min_samples = _trailing_window(fault, window)
    door_run = timedelta(hours=float(fault.parameters["door_run_hours"]))
    devices = _devices(conn, fault)
    series = hourly_on_time(
        conn, [d.ga for d in devices], window, float(fault.parameters["on_ma"])
    )
    doors = {
        device.ga: door_hours(series.get(device.ga, []), door_run=door_run)
        for device in devices
    }
    levels = {
        device.ga: duty_cycles(
            series.get(device.ga, []),
            window=trailing,
            min_samples=min_samples,
            excluded=doors[device.ga],
        )
        for device in devices
    }
    return _walk(
        devices,
        levels,
        window=window,
        rise=float(fault.parameters["rise_pct"]),
        budget=float(fault.parameters["budget_pct_h"]),
        # The hours cut out as door events — the run record says how much of
        # the window the fault deliberately did not look at. A device that
        # loses its whole window to them lands in `dataless`, where a
        # compressor running through days on end belongs: at this
        # resolution it is indistinguishable from a door standing open,
        # and Basalte owns that one.
        counts={"door_hours": sum(len(hours) for hours in doors.values())},
    )
