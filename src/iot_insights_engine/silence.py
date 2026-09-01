"""Channel-silence measurement: the `silence` kind of the fault list.

A channel that used to send has gone silent — longer than `gap_factor`
times its own normal pause, measured from the latest bucket per channel
over the hourly aggregate. Channel state (alive / silent / never sent) is
computed on demand from `knx_1h`, never stored: a staleness detector
working off stale data is the joke that tells itself.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from statistics import median

from .episodes import Observation

BUCKET = timedelta(hours=1)

# Constant-zero evidence shorter than a day is thin — an idle binary
# channel, not a dead register.
DEAD_MIN_BUCKETS = 24


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
        return int(self.ga.split("/", 1)[0])


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


def median_pause(buckets: list[datetime]) -> timedelta | None:
    """The channel's own normal pause: the median gap between its hourly
    buckets, floored at one bucket (hourly resolution sees nothing
    shorter). None below two buckets — a single send has no pause.
    """
    if len(buckets) < 2:
        return None
    gaps = [(b - a).total_seconds() for a, b in zip(buckets, buckets[1:], strict=False)]
    return max(timedelta(seconds=median(gaps)), BUCKET)


def classify(
    channel: Channel, buckets: list[datetime], frontier: datetime, gap_factor: float
) -> SilenceState:
    """State of one channel from its time-ordered bucket series. Silent
    means the frontier sits strictly more than `gap_factor` of the
    channel's own pauses past its last bucket; silent-since names the last
    bucket, not the detection time.
    """
    if not buckets:
        return SilenceState(channel, ChannelState.NEVER_SENT)
    pause = median_pause(buckets)
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
