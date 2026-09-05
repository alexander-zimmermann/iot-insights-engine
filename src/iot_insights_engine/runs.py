"""Run preparation: activity samples -> "it ran from here to here".

The shared preparation step of the fault list. Two faults read runs —
appliance runtime reads the length of the current run, freezer icing reads
the stretches the compressor never left to cut door events out of its duty
cycle — so the awkward parts (gaps, dropouts, where one run ends and the
next begins) are built once and can be wrong in at most one place.

Cadence-agnostic: a run is a stretch of samples each one cadence step after
the previous; a missing step ends it. A sample covers its full step, so a
single sample is already a run of one step's length.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class Run:
    """One continuous stretch of activity: first and last active sample,
    and the covered duration (`end - start` plus one step)."""

    start: datetime
    end: datetime
    duration: timedelta


def split_runs(times: Iterable[datetime], step: timedelta) -> tuple[Run, ...]:
    """The active samples' runs, in time order. Consecutive means exactly
    one step apart; anything larger is a stop and starts a new run.
    """
    ordered = sorted(times)
    runs: list[Run] = []
    start: datetime | None = None
    previous: datetime | None = None
    for time in ordered:
        if start is None or previous is None or time - previous > step:
            if start is not None and previous is not None:
                runs.append(Run(start=start, end=previous, duration=previous - start + step))
            start = time
        previous = time
    if start is not None and previous is not None:
        runs.append(Run(start=start, end=previous, duration=previous - start + step))
    return tuple(runs)
