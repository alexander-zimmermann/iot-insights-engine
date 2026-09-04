"""Notification volume: the `volume` kind of the fault list.

"More than N incidents in a week" is itself a declared fault. The whole
rebuild exists to get the notification count down to about five a week, and
the dashboard that shows the count is pull — so drift back into noise has to
push, over the same machinery every other fault uses.

The measured stream is not a channel but the episodes themselves: at every
hourly bucket the watchdog counts the incidents that began in the seven days
ending there, and every bucket past the declared limit becomes an observation
whose score is the count in units of that limit — the fault's declared unit.
A rolling window, not calendar weeks: "this week" means the last seven days,
and a burst stays visible for seven days after it ends rather than being
forgiven at midnight on Sunday.

Two things are deliberately not filtered out of the count. Externally
delivered episodes count, because Basalte's notifications reach the same
human as the engine's. And the watchdog's own episodes count, because a
watchdog exempt from its own measure would report a quieter week than the
one that happened. It contributes at most one incident per week to its own
tally, so it cannot run away with itself; what that one costs is a tier of
hysteresis — over the limit at N+1 incidents from other faults, back under
it at N-1 — which is what stops it flapping around the threshold.

Folded episodes are excluded: they are the imported detector era, a
different system's noise, and counting them would leave the watchdog firing
on history for as long as the fold-in stays in the window.

Like the channel kinds, time is the hourly aggregate's frontier — the
episode stream inherits that lag from the aggregates its faults measure, so
a stalled refresh freezes the picture instead of letting the count decay to
a recovery nobody earned.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from .episodes import Episode, Observation
from .runs import split_runs
from .silence import BUCKET

if TYPE_CHECKING:
    from collections.abc import Sequence

    import psycopg
    from psycopg.rows import DictRow

    from .episode_store import OpenEpisodeRow

# "This week": the rolling window the incidents are counted over.
WINDOW = timedelta(days=7)

# The one subject — the fault measures the house's whole episode stream, so
# there is nothing to fan out over, and one house-wide address carries it.
SUBJECT = "house"


@dataclass(frozen=True, slots=True)
class EpisodeStart:
    """One counted incident: when it began, and which fault produced it."""

    time: datetime
    fault: str


@dataclass(frozen=True, slots=True)
class VolumeBucket:
    """How many incidents began in the week ending at this bucket."""

    time: datetime
    episodes: int


@dataclass(frozen=True, slots=True)
class FaultCount:
    """One fault's share of the current week — what the payload names so a
    human sees which fault is making the noise."""

    fault: str
    episodes: int


@dataclass(frozen=True, slots=True)
class VolumeState:
    """The volume at the frontier: how many incidents the last week holds,
    against which limit, since when it has been over it, and who produced
    them."""

    episodes: int
    limit: float
    over_since: datetime | None = None
    by_fault: tuple[FaultCount, ...] = ()


@dataclass(frozen=True, slots=True)
class VolumePublish:
    """The house-wide severity and the week that earned it; the writer rule
    carries only the severity to the central Diagnose address, the payload
    carries the state for Basalte's e-mail."""

    severity: int
    state: VolumeState


@dataclass(frozen=True, slots=True)
class VolumePlan:
    """What one run changes: the new episode, the reconciled open row, the
    orphaned row to close, and the house-wide publish if the severity moved.
    """

    inserts: tuple[Episode, ...]
    updates: tuple[tuple[int, Episode], ...]
    orphan_closes: tuple[tuple[int, datetime], ...]
    publish: VolumePublish | None


def count_series(
    starts: Sequence[EpisodeStart], window_start: datetime, frontier: datetime
) -> list[VolumeBucket]:
    """One count per hourly bucket from `window_start` to the frontier: the
    incidents that began in the week ending at that bucket. Counting needs a
    week of history before the first bucket, so `starts` must reach back
    that far.
    """
    times = sorted(s.time for s in starts)
    buckets: list[VolumeBucket] = []
    t = window_start
    while t <= frontier:
        # (t - WINDOW, t]: an incident belongs to the week it began in, and
        # to exactly one bucket boundary.
        opened = bisect_right(times, t) - bisect_right(times, t - WINDOW)
        buckets.append(VolumeBucket(time=t, episodes=opened))
        t += BUCKET
    return buckets


def volume_observations(buckets: Sequence[VolumeBucket], limit: float) -> list[Observation]:
    """One observation per bucket whose week stands past the limit, for the
    episode pipeline. The score is the count in units of the limit — the
    fault's declared unit; the value is the count itself.
    """
    return [
        Observation(
            subject=SUBJECT, time=b.time, score=b.episodes / limit, value=float(b.episodes)
        )
        for b in buckets
        if b.episodes > limit
    ]


def classify(
    starts: Sequence[EpisodeStart],
    buckets: Sequence[VolumeBucket],
    limit: float,
    frontier: datetime,
) -> VolumeState:
    """The volume at the frontier — what the publish payload names alongside
    the severity. `over_since` is the start of the over-limit stretch the
    frontier sits in, and None once the count is back under the limit.
    """
    runs = split_runs((b.time for b in buckets if b.episodes > limit), BUCKET)
    over_since = runs[-1].start if runs and runs[-1].end == frontier else None
    counts = Counter(s.fault for s in starts if frontier - WINDOW < s.time <= frontier)
    return VolumeState(
        episodes=buckets[-1].episodes,
        limit=limit,
        over_since=over_since,
        by_fault=tuple(
            FaultCount(fault=fault, episodes=count)
            for fault, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    )


def plan_run(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    state: VolumeState,
    frontier: datetime,
) -> VolumePlan:
    """Pure reconciliation: the computed episodes against the stored open
    row. One subject, so this is the other kinds' reconcile without the
    fan-out — and without their `dataless` case: the episode stream is
    always countable, so no computed episode really does mean the volume
    fell back, never "no data to tell".

    A stored severity is never lowered — the recompute window may have slid
    past the loudest stretch — and only a change publishes, including the 0
    when the week goes quiet again.
    """
    # One subject, so one row can be open for this fault at a time.
    open_row = next((row for row in open_rows if row.subject == SUBJECT), None)
    # Flicker beyond the quiet window can leave several episodes in the
    # window; that row can only correspond to the latest one.
    latest = max(episodes, key=lambda e: e.started_at, default=None)

    inserts: list[Episode] = []
    updates: list[tuple[int, Episode]] = []
    orphan_closes: list[tuple[int, datetime]] = []
    after = 0

    if latest is not None and latest.ended_at is None:
        after = max(latest.severity, open_row.severity if open_row else 0)
        if open_row is not None:
            updates.append((open_row.id, latest))
        else:
            inserts.append(latest)
    elif open_row is not None:
        # The row's incident is over: the computed episode closes it, or —
        # with nothing computed at all — the frontier does.
        if latest is not None:
            updates.append((open_row.id, latest))
        else:
            orphan_closes.append((open_row.id, frontier))

    before = open_row.severity if open_row else 0
    publish = VolumePublish(severity=after, state=state) if after != before else None
    return VolumePlan(tuple(inserts), tuple(updates), tuple(orphan_closes), publish)


def episode_starts(
    conn: psycopg.Connection[DictRow], window_start: datetime
) -> list[EpisodeStart]:
    """Every incident since `window_start`, from the episodes table itself —
    the stream this fault measures. Folded episodes are the imported
    detector era and are not this system's noise; everything else counts,
    externally delivered episodes and the watchdog's own included.
    """
    rows = conn.execute(
        """
        SELECT started_at, fault FROM episodes
        WHERE NOT folded AND started_at >= %(start)s
        ORDER BY started_at
        """,
        {"start": window_start},
    ).fetchall()
    return [EpisodeStart(time=r["started_at"], fault=r["fault"]) for r in rows]
