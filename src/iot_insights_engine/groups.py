"""Delivery per KNX main group: the shape a channel-scoped fault reports in.

Two kinds measure per channel and report per main group — silence and
constancy. Per-channel addresses are impossible (roughly 500 channels carry
data), so the address plan gives every main group one Diagnose address
instead: the address says roughly where, the payload names the channels
exactly.

Beyond the shared reconciliation both need one rule, and it lives here
once: **a group publishes when its severity moved or its set of open
channels changed**. Without the second half, a channel going wrong at a
tier another channel already holds would ride a stale payload — the group
would still say "one channel", naming the wrong one.

What differs between the kinds is only what a channel report says — how
long silent, how long stuck, at which value — so each hands in the factory
that builds one of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .reconcile import plan_from, reconcile

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import datetime

    from .episode_store import OpenEpisodeRow
    from .episodes import Episode
    from .reconcile import Measured, Plan


def main_group(ga: str) -> int:
    """The KNX main group of a group address — the granularity these kinds
    report at (one Zentral diagnosis address per main group).
    """
    return int(ga.split("/", 1)[0])


class GroupChannel(Protocol):
    """What a group publish needs of one channel's report: which address it
    speaks for, the severity it carries, and how bad it is in the kind's own
    unit — the last of these orders the group's list, and is None where this
    run could not measure it.
    """

    @property
    def ga(self) -> str: ...

    @property
    def severity(self) -> int: ...

    @property
    def magnitude(self) -> float | None: ...


@dataclass(frozen=True, slots=True)
class GroupPublish[R: GroupChannel]:
    """One main group's severity, with the reports of the channels that
    earned it — worst first."""

    main_group: int
    severity: int
    channels: tuple[R, ...]

    @property
    def subject(self) -> str:
        return str(self.main_group)

    @property
    def entity(self) -> str:
        return str(self.main_group)


class ReportFor[S, R](Protocol):
    """A kind's channel-report factory. The state is None where the subject
    was not measured this run, the episode where it is held open for want of
    data — a report then names the address and the tier it still carries,
    and nothing it cannot know.
    """

    def __call__(
        self, subject: str, severity: int, state: S | None, episode: Episode | None
    ) -> R: ...


def group_plan[S, R: GroupChannel](
    *,
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    measured: Measured[S],
    frontier: datetime,
    report_for: ReportFor[S, R],
) -> Plan[GroupPublish[R]]:
    """The shared reconciliation, delivered per main group."""
    result = reconcile(
        episodes=episodes,
        open_rows=open_rows,
        dataless=measured.dataless,
        frontier=frontier,
    )
    # The fold leaves at most one open episode per subject, and it is the
    # one the reconciliation carried into `after`.
    open_episodes = {e.subject: e for e in episodes if e.ended_at is None}
    reports = {
        subject: report_for(
            subject, severity, measured.states.get(subject), open_episodes.get(subject)
        )
        for subject, severity in result.after.items()
    }

    before = _group_state((row.subject, row.severity) for row in open_rows)
    after = _group_state((subject, report.severity) for subject, report in reports.items())

    publishes: list[GroupPublish[R]] = []
    for group in sorted(set(before) | set(after)):
        if after.get(group) == before.get(group):
            continue
        severity, _ = after.get(group, (0, frozenset()))
        channels = sorted(
            (r for r in reports.values() if main_group(r.ga) == group),
            key=lambda r: (-r.severity, -(r.magnitude or 0.0), r.ga),
        )
        publishes.append(GroupPublish(group, severity, tuple(channels)))

    return plan_from(result, publishes)


def _group_state(
    subject_severities: Iterable[tuple[str, int]],
) -> dict[int, tuple[int, frozenset[str]]]:
    """Per main group: the maximum severity and the set of open subjects —
    the two things whose change warrants a publish."""
    severities: dict[int, int] = {}
    subjects: dict[int, set[str]] = {}
    for subject, severity in subject_severities:
        group = main_group(subject)
        severities[group] = max(severities.get(group, 0), severity)
        subjects.setdefault(group, set()).add(subject)
    return {g: (severities[g], frozenset(subjects[g])) for g in severities}
