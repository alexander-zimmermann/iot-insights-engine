"""Episode reconciliation: computed episodes against the stored open rows.

Every kind that reports per subject — a device, a room, an appliance's
standby — reconciles the same way, and the guarantees are the interesting
part rather than the bookkeeping:

* only open computed episodes materialize as new rows; a computed episode
  that already ended matters only to close the open row it reconciles;
* a stored severity is never lowered — the recompute window may have slid
  past the peak;
* an open row whose subject produced no episode closes at the frontier,
  unless the subject is `dataless` (in scope, but with nothing measurable
  in the whole window): with no data to decide a recovery, the episode
  stays open instead of self-clearing.

What leaves here is `moved`: the subjects whose severity changed, with the
severity they carry now — including the 0 that ends an episode. Each kind
turns those into its own publish payload; only channel silence reconciles
differently, because it reports per main group rather than per subject.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from .episodes import Episode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .episode_store import OpenEpisodeRow


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """What one run changes: new episodes, reconciled open rows, orphaned
    rows to close, rows kept open for want of data, and the subjects whose
    severity moved.
    """

    inserts: tuple[Episode, ...]
    updates: tuple[tuple[int, Episode], ...]
    orphan_closes: tuple[tuple[int, datetime], ...]
    stale_opens: tuple[str, ...]
    moved: tuple[tuple[str, int], ...]


def reconcile(
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    dataless: frozenset[str],
    frontier: datetime,
) -> Reconciliation:
    """Pure reconciliation — the guarantees are in the module docstring."""
    open_by_subject = {row.subject: row for row in open_rows}

    # Several episodes per subject can fall in the window; the stored open
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
            # Still firing as far as anyone can tell — it keeps counting.
            after[row.subject] = row.severity
        else:
            orphan_closes.append((row.id, frontier))

    before = {row.subject: row.severity for row in open_rows}
    moved = tuple(
        (subject, after.get(subject, 0))
        for subject in sorted(set(before) | set(after))
        if before.get(subject) != after.get(subject)
    )

    return Reconciliation(
        inserts=tuple(inserts),
        updates=tuple(updates),
        orphan_closes=tuple(orphan_closes),
        stale_opens=tuple(stale_opens),
        moved=moved,
    )
