"""Room-deviation measurement: the `deviation` kind of the fault list.

A value sits too far under its declared reference while a gate condition
holds — for the FBH fault: a room at least its declared gap under the
setpoint while the valve is open past the gate threshold, for at least the
declared hours. The reference is swappable by declaration: the fault file
names a channel pattern for the reference and the optional gate, which
follow a uniform naming rule across rooms, while each room names the
channel it is measured on. The two marry into triples — strictly both
ways, like the duration kind's device limits.

Values come from the hourly aggregate as a dense series: KNX channels are
state, not samples — a setpoint that sent once holds until changed — so
each role's last seen hourly average is carried forward across silent
buckets, deliberately without a staleness bound (a role dying mid-window
freezes the room's picture; the silence fault owns the dead channel). A
room measures only from the bucket where every role has appeared inside
the window; a room where one never does is `dataless`, must not
self-clear, and is warned about. Several value channels per room (the
sensor-less halls measure via their BWM heads) are averaged; a value
channel constant at zero for a day or more is a dead register, not a cold
room, and drops out before measurement.

Consecutive cold buckets form a run — a warm hour restarts the clock — and
every bucket from `min_hours` onward becomes an observation whose score is
the gap in units of the room's declared threshold, the fault's declared
unit. Like silence and duration, time is the aggregate's own frontier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from .episodes import Episode, Observation
from .nats_publisher import slugify
from .runs import split_runs
from .silence import BUCKET, DEAD_MIN_BUCKETS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow
    from .faults import Roles, RoomRule
    from .silence import Channel


@dataclass(frozen=True, slots=True)
class Room:
    """One monitored room: its channel triple and declared threshold. The
    slug doubles as episode subject and NATS entity — the writer rules pin
    `anomaly.<fault>.<slug>` to the room's anomaly address."""

    label: str
    slug: str
    value_gas: tuple[str, ...]
    reference_ga: str
    gate_ga: str | None
    min_gap: float

    @property
    def gas(self) -> tuple[str, ...]:
        """Every channel the room measures with — the fetch list and the
        carry keys must enumerate the roles identically."""
        gate = (self.gate_ga,) if self.gate_ga is not None else ()
        return (*self.value_gas, self.reference_ga, *gate)


@dataclass(frozen=True, slots=True)
class RoomBucket:
    """One measurable hour of a room: every role has a (carried) value."""

    time: datetime
    value: float
    reference: float
    gate: float | None

    @property
    def gap(self) -> float:
        return self.reference - self.value


@dataclass(frozen=True, slots=True)
class RoomState:
    """Whether the room is cold at the frontier, since when, and the
    numbers behind it — what the publish payload names beside the severity."""

    room: Room
    cold_since: datetime | None = None
    gap: float | None = None
    value: float | None = None
    reference: float | None = None
    gate: float | None = None


def _like(pattern: str, name: str) -> bool:
    """SQL LIKE against a full name (`%` any run, `_` any character) — the
    same dialect the scope's catalog query speaks, so a role pattern reads
    like a scope line."""
    regex = ".*".join(re.escape(part).replace("_", ".") for part in pattern.split("%"))
    return re.fullmatch(regex, name) is not None


def resolve_rooms(
    channels: Sequence[Channel], rules: Sequence[RoomRule], roles: Roles
) -> list[Room]:
    """Marry the declared rooms to the scoped channels, strictly both ways:
    every room resolves one reference (and gate, where declared) and at
    least one value channel; every scoped channel belongs to exactly one
    room and role. Every problem is reported at once.
    """
    problems: list[str] = []
    rooms: list[Room] = []
    claimed: dict[str, str] = {}
    slugs: dict[str, str] = {}
    for rule in rules:
        slug = slugify(rule.match)
        if slug in slugs:
            problems.append(
                f"rooms {slugs[slug]!r} and {rule.match!r} share the slug {slug!r}"
            )
            continue
        slugs[slug] = rule.match
        mine = [c for c in channels if f".{rule.match}." in c.name]
        by_role: dict[str, list[Channel]] = {"value": [], "reference": [], "gate": []}
        for channel in mine:
            if channel.ga in claimed:
                problems.append(
                    f"channel {channel.ga} matched by {claimed[channel.ga]!r} "
                    f"and {rule.match!r}"
                )
                continue
            claimed[channel.ga] = rule.match
            hits = [
                role
                for role, pattern in (
                    ("value", rule.value),
                    ("reference", roles.reference),
                    ("gate", roles.gate),
                )
                if pattern is not None and _like(pattern, channel.name)
            ]
            if len(hits) != 1:
                problems.append(
                    f"room {rule.match!r}: channel {channel.ga} ({channel.name}) "
                    + ("matches no role" if not hits else f"matches roles {', '.join(hits)}")
                )
                continue
            by_role[hits[0]].append(channel)

        broken = False
        for role, pattern in (("reference", roles.reference), ("gate", roles.gate)):
            if pattern is None:
                continue
            if len(by_role[role]) != 1:
                gas = ", ".join(c.ga for c in by_role[role])
                problems.append(
                    f"room {rule.match!r}: role {role} matches no channel"
                    if not by_role[role]
                    else f"room {rule.match!r}: role {role} matches "
                    f"{len(by_role[role])} channels: {gas}"
                )
                broken = True
        if not by_role["value"]:
            problems.append(f"room {rule.match!r}: role value matches no channel")
            broken = True
        if broken:
            continue

        rooms.append(
            Room(
                label=rule.match,
                slug=slug,
                value_gas=tuple(sorted(c.ga for c in by_role["value"])),
                reference_ga=by_role["reference"][0].ga,
                gate_ga=by_role["gate"][0].ga if roles.gate is not None else None,
                min_gap=rule.min_gap_k,
            )
        )
    problems.extend(
        f"channel {c.ga} ({c.name}) belongs to no declared room"
        for c in channels
        if c.ga not in claimed
    )
    if problems:
        raise ValueError("room rules do not fit the scope: " + "; ".join(problems))
    return sorted(rooms, key=lambda r: r.label)


def dead_value_gas(
    rooms: Sequence[Room], by_ga: Mapping[str, Mapping[datetime, float]]
) -> list[str]:
    """Value channels constant at zero over a day or more of buckets: a dead
    register reading 0.0 would score as a huge gap, so it drops out before
    measurement — the silence fault owns reporting it. Only the value role
    is checked: a closed valve legitimately sits at 0 % for weeks."""
    value_gas = {ga for room in rooms for ga in room.value_gas}
    return [
        ga
        for ga in sorted(value_gas)
        if (buckets := by_ga.get(ga))
        and len(buckets) >= DEAD_MIN_BUCKETS
        and all(v == 0.0 for v in buckets.values())
    ]


def room_series(
    room: Room,
    by_ga: Mapping[str, Mapping[datetime, float]],
    window_start: datetime,
    frontier: datetime,
) -> list[RoomBucket]:
    """The room's dense hourly series: each role's last seen hourly average,
    carried forward across silent buckets. Buckets before every role has
    appeared are unmeasurable and skipped; a room with none is dataless."""
    carry: dict[str, float] = {}
    buckets: list[RoomBucket] = []
    gas = room.gas
    t = window_start
    while t <= frontier:
        for ga in gas:
            value = by_ga.get(ga, {}).get(t)
            if value is not None:
                carry[ga] = value
        values = [carry[ga] for ga in room.value_gas if ga in carry]
        reference = carry.get(room.reference_ga)
        gate = carry.get(room.gate_ga) if room.gate_ga is not None else None
        if values and reference is not None and (room.gate_ga is None or gate is not None):
            buckets.append(
                RoomBucket(
                    time=t, value=sum(values) / len(values), reference=reference, gate=gate
                )
            )
        t += BUCKET
    return buckets


def cold_buckets(
    room: Room, buckets: Sequence[RoomBucket], gate_min: float | None
) -> list[RoomBucket]:
    """The buckets where the fault condition stands: the gap at or past the
    room's threshold while the gate (where declared) sits strictly above
    its minimum."""
    return [
        b
        for b in buckets
        if b.gap >= room.min_gap
        and (gate_min is None or (b.gate is not None and b.gate > gate_min))
    ]


def deviation_observations(
    room: Room, cold: Sequence[RoomBucket], min_hours: float
) -> list[Observation]:
    """One observation per bucket once a cold run has stood `min_hours`, for
    the episode pipeline. A run is consecutive cold buckets — a warm hour
    restarts the clock. The score is the gap in units of the room's own
    threshold — the fault's declared unit; the value is the gap itself. A
    bucket covers its full hour, so a run's first bucket already counts as
    one.
    """
    observations: list[Observation] = []
    run_start: datetime | None = None
    previous: datetime | None = None
    for bucket in cold:
        if run_start is None or previous is None or bucket.time - previous > BUCKET:
            run_start = bucket.time
        previous = bucket.time
        if bucket.time - run_start + BUCKET >= min_hours * BUCKET:
            observations.append(
                Observation(
                    subject=room.slug,
                    time=bucket.time,
                    score=bucket.gap / room.min_gap,
                    value=bucket.gap,
                )
            )
    return observations


def classify(room: Room, cold: Sequence[RoomBucket], frontier: datetime) -> RoomState:
    """The room's current cold stretch, if one reaches the frontier — what
    the publish payload names alongside the severity."""
    runs = split_runs((b.time for b in cold), BUCKET)
    if runs and runs[-1].end == frontier:
        latest = max(cold, key=lambda b: b.time)
        return RoomState(
            room=room,
            cold_since=runs[-1].start,
            gap=latest.gap,
            value=latest.value,
            reference=latest.reference,
            gate=latest.gate,
        )
    return RoomState(room=room)


@dataclass(frozen=True, slots=True)
class RoomPublish:
    """One room whose severity or openness moved — the payload names the
    room and its numbers; the writer rule carries only the severity to the
    room's anomaly address."""

    slug: str
    severity: int
    room: str
    cold_since: datetime | None = None
    gap: float | None = None
    value: float | None = None
    reference: float | None = None
    gate: float | None = None
    min_gap: float | None = None


@dataclass(frozen=True, slots=True)
class DeviationPlan:
    """What one run changes: new episodes, reconciled open rows, orphaned
    rows to close, rows kept open for want of data, and the rooms whose
    severity moved.
    """

    inserts: tuple[Episode, ...]
    updates: tuple[tuple[int, Episode], ...]
    orphan_closes: tuple[tuple[int, datetime], ...]
    stale_opens: tuple[str, ...]
    publishes: tuple[RoomPublish, ...]


def plan_run(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    states_by_slug: Mapping[str, RoomState],
    dataless: frozenset[str],
    frontier: datetime,
) -> DeviationPlan:
    """Pure reconciliation: computed episodes against the stored open rows,
    with the same guarantees the silence and duration plans give — a stored
    severity is never lowered, an open row whose room produced no episode
    closes at the frontier unless the room is `dataless` (no measurable
    bucket in the whole window: with nothing to decide a recovery, it must
    not self-clear).

    Delivery is per room: a publish goes out when a room's severity moved,
    including the 0 when its episode ends.
    """
    open_by_subject = {row.subject: row for row in open_rows}

    # Several episodes per room can fall in the window; the stored open
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
            # Still cold as far as anyone can tell — it keeps counting.
            after[row.subject] = row.severity
        else:
            orphan_closes.append((row.id, frontier))

    before = {row.subject: row.severity for row in open_rows}
    publishes = tuple(
        _publish_for(subject, after.get(subject, 0), states_by_slug.get(subject))
        for subject in sorted(set(before) | set(after))
        if before.get(subject) != after.get(subject)
    )

    return DeviationPlan(
        inserts=tuple(inserts),
        updates=tuple(updates),
        orphan_closes=tuple(orphan_closes),
        stale_opens=tuple(stale_opens),
        publishes=publishes,
    )


def _publish_for(subject: str, severity: int, state: RoomState | None) -> RoomPublish:
    # A room that left the scope while its row was open still gets its
    # clear; the payload then only names the slug.
    if state is None:
        return RoomPublish(slug=subject, severity=severity, room=subject)
    return RoomPublish(
        slug=subject,
        severity=severity,
        room=state.room.label,
        cold_since=state.cold_since,
        gap=state.gap,
        value=state.value,
        reference=state.reference,
        gate=state.gate,
        min_gap=state.room.min_gap,
    )


def values(
    conn: psycopg.Connection[DictRow], gas: Sequence[str], window_start: datetime
) -> dict[str, dict[datetime, float]]:
    """The rooms' hourly averages over the window, one query for the whole
    scope — 14 room triples, not 2500 channels."""
    rows = conn.execute(
        """
        SELECT ga, bucket, avg_value FROM knx_1h
        WHERE ga = ANY(%(gas)s) AND bucket >= %(start)s
        ORDER BY ga, bucket
        """,
        {"gas": list(gas), "start": window_start},
    ).fetchall()
    series: dict[str, dict[datetime, float]] = {}
    for row in rows:
        series.setdefault(row["ga"], {})[row["bucket"]] = float(row["avg_value"])
    return series
