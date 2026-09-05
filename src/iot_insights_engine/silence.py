"""Channel-silence measurement: the `silence` kind of the fault list.

A channel that used to send has gone silent — longer than `gap_factor`
times its own normal pause, measured from the latest bucket per channel
over the hourly aggregate. That pause is the `gap_quantile` of the
channel's own gaps rather than their median — see `normal_pause` for why
the median misreads every channel a human switches. Channel state
(alive / silent / never sent) is computed on demand from `knx_1h`, never
stored: a staleness detector working off stale data is the joke that
tells itself.

Gaps are measured against the aggregate's *frontier* (its newest bucket
anywhere), not the wall clock: the continuous aggregate materializes with
an end offset, so the newest visible bucket lags real time by an hour or
two for every channel at once. The frontier cancels that lag; a house-wide
outage is deliberately not this fault's problem (the bridge watchdog owns
it).

Channels that never sent are excluded without a report — symmetry
addresses are normal, not findings. Dead registers (a constant zero over
the whole window, like the L2 voltage that read 0.0 for 717 buckets) drop
out where the scope resolves; both drops are logged by the caller.

This module also holds what every kind needs before it can measure
anything: the `Channel` a catalog query resolves to, the query itself
(`resolve_scope`), and the strict pairing of declared per-device rules to
those channels (`pair_by_match`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from math import ceil
from typing import Any, Protocol

import psycopg
from psycopg.rows import DictRow

from .episodes import Observation
from .faults import Scope

BUCKET = timedelta(hours=1)


class DeviceRule(Protocol):
    """What `pair_by_match` needs of a declared per-device rule: the unique
    fragment of the catalog name it claims. The rest — a runtime limit, a
    healthy reference — belongs to the kind that declared it.
    """

    @property
    def match(self) -> str: ...


# Constant-zero evidence shorter than a day is thin — an idle binary
# channel, not a dead register.
DEAD_MIN_BUCKETS = 24

# Two buckets is the least a gap can be read off — a channel that sent
# fewer has no pause, and nothing measured it.
MIN_PAUSE_BUCKETS = 2


def main_group(ga: str) -> int:
    """The KNX main group of a group address — the granularity silence
    reports at (one Zentral diagnosis address per main group).
    """
    return int(ga.split("/", 1)[0])


class ChannelState(StrEnum):
    ALIVE = "alive"
    SILENT = "silent"
    NEVER_SENT = "never_sent"
    DEAD = "dead"


@dataclass(frozen=True, slots=True)
class Channel:
    """One catalog channel the fault's scope resolved to."""

    ga: str
    name: str
    dpt: str

    @property
    def main_group(self) -> int:
        return main_group(self.ga)


@dataclass(frozen=True, slots=True)
class ChannelStats:
    """Per-channel aggregate over the measurement window, one cheap query
    for the whole scope: enough to rule most channels alive without
    touching their bucket series.
    """

    ga: str
    buckets: int
    last_bucket: datetime
    floor_value: float
    ceil_value: float


@dataclass(frozen=True, slots=True)
class SilenceState:
    """The measured state of one channel, computed on demand."""

    channel: Channel
    state: ChannelState
    silent_since: datetime | None = None
    pause: timedelta | None = None


def normal_pause(buckets: list[datetime], quantile: float) -> timedelta | None:
    """The channel's own normal pause: the `quantile` of the gaps between
    its hourly buckets, floored at one bucket (hourly resolution sees
    nothing shorter). None below two buckets — a single send has no pause.

    The quantile is the estimator, not the median, because gaps are bimodal
    wherever a human drives the channel: an hour inside an evening's
    switching, a day until the next one. The median lands inside the
    cluster and calls every ordinary night a fault; a high quantile lands
    on the gap that actually ends the quiet phase. For the cyclic channels
    the fault exists for — power, temperature, status — every gap is the
    cycle, so the quantile reads the same value the median did and costs
    no sensitivity.

    Read by nearest rank: the pause is a gap the channel really showed, not
    an interpolation between two unlike ones.
    """
    if len(buckets) < MIN_PAUSE_BUCKETS:
        return None
    gaps = sorted(b - a for a, b in zip(buckets, buckets[1:], strict=False))
    rank = max(1, ceil(quantile * len(gaps)))
    return max(gaps[rank - 1], BUCKET)


def classify(
    channel: Channel,
    buckets: list[datetime],
    *,
    frontier: datetime,
    gap_factor: float,
    gap_quantile: float,
) -> SilenceState:
    """State of one channel from its time-ordered bucket series. Silent
    means the frontier sits strictly more than `gap_factor` of the
    channel's own pauses past its last bucket; silent-since names the last
    bucket, not the detection time.
    """
    if not buckets:
        return SilenceState(channel, ChannelState.NEVER_SENT)
    pause = normal_pause(buckets, gap_quantile)
    if pause is None:
        return SilenceState(channel, ChannelState.ALIVE)
    if frontier - buckets[-1] > gap_factor * pause:
        return SilenceState(channel, ChannelState.SILENT, silent_since=buckets[-1], pause=pause)
    return SilenceState(channel, ChannelState.ALIVE, pause=pause)


def silence_observations(
    ga: str,
    buckets: list[datetime],
    pause: timedelta,
    gap_factor: float,
    frontier: datetime,
) -> list[Observation]:
    """One observation per silent bucket, for the episode pipeline: every
    hourly bucket sitting strictly more than `gap_factor × pause` past the
    channel's preceding bucket, inside historical gaps and in the tail up
    to the frontier. The score is the gap in units of the channel's own
    pause — the fault's declared unit; the value is the gap in hours.
    """
    threshold = gap_factor * pause
    observations: list[Observation] = []
    for i, prev in enumerate(buckets):
        # Gap end: the next bucket (exclusive — it is the recovery), or the
        # frontier (inclusive — the silence is still running).
        end = buckets[i + 1] if i + 1 < len(buckets) else frontier + BUCKET
        if end - prev <= threshold:
            continue
        t = prev + BUCKET
        while t < end:
            gap = t - prev
            if gap > threshold:
                observations.append(
                    Observation(subject=ga, time=t, score=gap / pause, value=gap / BUCKET)
                )
            t += BUCKET
    return observations


def frontier(conn: psycopg.Connection[DictRow]) -> datetime | None:
    """The aggregate's newest bucket anywhere — the 'now' all gaps are
    measured against."""
    row = conn.execute("SELECT max(bucket) AS frontier FROM knx_1h").fetchone()
    return row["frontier"] if row else None


def resolve_scope(conn: psycopg.Connection[DictRow], scope: Scope) -> list[Channel]:
    """The fault's channels, resolved where the catalog lives — never a
    hand-written address list."""
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if scope.dpt:
        clauses.append("dpt = ANY(%(dpt)s)")
        params["dpt"] = list(scope.dpt)
    if scope.name_like:
        clauses.append("name LIKE ANY(%(name_like)s)")
        params["name_like"] = list(scope.name_like)
    if scope.exclude_name_like:
        clauses.append("NOT (name LIKE ANY(%(exclude_name_like)s))")
        params["exclude_name_like"] = list(scope.exclude_name_like)
    sql = "SELECT ga, name, dpt FROM ga_catalog"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    rows = conn.execute(sql + " ORDER BY ga", params).fetchall()
    return [Channel(ga=r["ga"], name=r["name"], dpt=r["dpt"]) for r in rows]


def pair_by_match[R: DeviceRule](
    channels: Sequence[Channel], rules: Sequence[R], *, noun: str
) -> list[tuple[Channel, R]]:
    """Marry declared per-device rules to the scoped channels, strictly both
    ways: every rule names exactly one channel, every channel is named by
    exactly one rule. Ordered by group address.

    Every problem is reported at once — a config error should name the whole
    repair, not one field per run — and `noun` names what the file declares
    ("limit", "reference"), so the message points at the block to edit.
    """
    problems: list[str] = []
    matched: dict[str, tuple[Channel, R]] = {}
    for rule in rules:
        hits = [c for c in channels if rule.match in c.name]
        if len(hits) != 1:
            gas = ", ".join(c.ga for c in hits)
            problems.append(
                f"device {rule.match!r} matches no channel in scope"
                if not hits
                else f"device {rule.match!r} matches {len(hits)} channels: {gas}"
            )
            continue
        channel = hits[0]
        if channel.ga in matched:
            problems.append(
                f"channel {channel.ga} matched by "
                f"{matched[channel.ga][1].match!r} and {rule.match!r}"
            )
            continue
        matched[channel.ga] = (channel, rule)
    problems.extend(
        f"channel {c.ga} ({c.name}) has no declared {noun}"
        for c in channels
        if c.ga not in matched
    )
    if problems:
        raise ValueError(f"device {noun}s do not fit the scope: " + "; ".join(problems))
    return [matched[ga] for ga in sorted(matched)]


def channel_stats(
    conn: psycopg.Connection[DictRow], window_start: datetime
) -> dict[str, ChannelStats]:
    rows = conn.execute(
        """
        SELECT ga, count(*) AS buckets, max(bucket) AS last_bucket,
               min(min_value) AS floor_value, max(max_value) AS ceil_value
        FROM knx_1h WHERE bucket >= %(start)s GROUP BY ga
        """,
        {"start": window_start},
    ).fetchall()
    return {
        r["ga"]: ChannelStats(
            ga=r["ga"],
            buckets=r["buckets"],
            last_bucket=r["last_bucket"],
            floor_value=r["floor_value"],
            ceil_value=r["ceil_value"],
        )
        for r in rows
    }


def bucket_series(
    conn: psycopg.Connection[DictRow], gas: list[str], window_start: datetime
) -> dict[str, list[datetime]]:
    """Bucket series for the channels that need one, fetched in main-group
    chunks so a single result set stays bounded on the small database.
    """
    by_group: dict[int, list[str]] = defaultdict(list)
    for ga in gas:
        by_group[main_group(ga)].append(ga)
    series: dict[str, list[datetime]] = defaultdict(list)
    for group_gas in by_group.values():
        rows = conn.execute(
            """
            SELECT ga, bucket FROM knx_1h
            WHERE ga = ANY(%(gas)s) AND bucket >= %(start)s
            ORDER BY ga, bucket
            """,
            {"gas": group_gas, "start": window_start},
        ).fetchall()
        for r in rows:
            series[r["ga"]].append(r["bucket"])
    return dict(series)


def drop_unmeasurable(
    channels: list[Channel], stats_by_ga: dict[str, ChannelStats]
) -> tuple[list[Channel], dict[ChannelState, list[Channel]]]:
    """The filter that sits where the scope resolves: never-sent channels
    (no data in the window) and dead registers (constant zero throughout)
    drop out before any measurement sees them. The caller logs the drops.
    """
    kept: list[Channel] = []
    drops: dict[ChannelState, list[Channel]] = {
        ChannelState.NEVER_SENT: [],
        ChannelState.DEAD: [],
    }
    for channel in channels:
        stats = stats_by_ga.get(channel.ga)
        if stats is None:
            drops[ChannelState.NEVER_SENT].append(channel)
        elif (
            stats.floor_value == stats.ceil_value == 0.0
            and stats.buckets >= DEAD_MIN_BUCKETS
        ):
            drops[ChannelState.DEAD].append(channel)
        else:
            kept.append(channel)
    return kept, drops
