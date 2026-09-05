"""Tests for the shape every per-subject kind reconciles and delivers with.

The kinds' own test files cover their measurements; these cover what they
share — the effective severity map, the one meaning of `dataless`, and the
plan a moved subject turns into. No cluster, no live database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from iot_insights_engine.episode_store import OpenEpisodeRow
from iot_insights_engine.episodes import Episode
from iot_insights_engine.reconcile import (
    measurement_reaches,
    reconcile,
    subject_plan,
)

_T0 = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)
_FRONTIER = _T0 + 8 * _HOUR


def _episode(subject: str, severity: int, *, ended: bool = False) -> Episode:
    return Episode(
        fault="appliance_standby",
        subject=subject,
        started_at=_T0,
        last_seen_at=_T0 + 5 * _HOUR,
        ended_at=_T0 + 6 * _HOUR if ended else None,
        severity=severity,
        peak_score=7.0,
        evidence=(),
        events=(),
    )


class TestAfter:
    def test_an_open_episode_carries_its_severity(self) -> None:
        result = reconcile(
            episodes=[_episode("2/2/227", 2)],
            open_rows=[],
            dataless=frozenset(),
            frontier=_FRONTIER,
        )
        assert dict(result.after) == {"2/2/227": 2}

    def test_the_stored_severity_wins_where_it_is_higher(self) -> None:
        # The window slid past the peak; `after` reports the tier the bus
        # still carries, not the recomputed one.
        result = reconcile(
            episodes=[_episode("2/2/227", 1)],
            open_rows=[OpenEpisodeRow(id=7, subject="2/2/227", severity=3)],
            dataless=frozenset(),
            frontier=_FRONTIER,
        )
        assert dict(result.after) == {"2/2/227": 3}
        assert result.moved == ()

    def test_a_subject_kept_open_for_want_of_data_keeps_counting(self) -> None:
        result = reconcile(
            episodes=[],
            open_rows=[OpenEpisodeRow(id=7, subject="2/2/227", severity=2)],
            dataless=frozenset({"2/2/227"}),
            frontier=_FRONTIER,
        )
        assert dict(result.after) == {"2/2/227": 2}
        assert result.stale_opens == ("2/2/227",)
        assert result.moved == ()

    def test_an_ended_episode_leaves_after_and_moves_to_zero(self) -> None:
        result = reconcile(
            episodes=[_episode("2/2/227", 1, ended=True)],
            open_rows=[OpenEpisodeRow(id=7, subject="2/2/227", severity=1)],
            dataless=frozenset(),
            frontier=_FRONTIER,
        )
        assert dict(result.after) == {}
        assert result.moved == (("2/2/227", 0),)

    def test_moved_is_the_part_of_after_that_changed(self) -> None:
        result = reconcile(
            episodes=[_episode("2/2/227", 3), _episode("2/2/224", 1)],
            open_rows=[OpenEpisodeRow(id=7, subject="2/2/224", severity=1)],
            dataless=frozenset(),
            frontier=_FRONTIER,
        )
        assert dict(result.after) == {"2/2/224": 1, "2/2/227": 3}
        assert result.moved == (("2/2/227", 3),)


class TestMeasurementReaches:
    _MAX_GAP = 4 * _HOUR

    def test_nothing_measured_reaches_nothing(self) -> None:
        assert not measurement_reaches(None, frontier=_FRONTIER, max_gap=self._MAX_GAP)

    def test_a_measurement_at_the_frontier_reaches_it(self) -> None:
        assert measurement_reaches(_FRONTIER, frontier=_FRONTIER, max_gap=self._MAX_GAP)

    def test_a_lag_within_the_pipeline_gap_still_reaches(self) -> None:
        assert measurement_reaches(
            _FRONTIER - self._MAX_GAP, frontier=_FRONTIER, max_gap=self._MAX_GAP
        )

    def test_a_lag_past_the_pipeline_gap_does_not(self) -> None:
        assert not measurement_reaches(
            _FRONTIER - self._MAX_GAP - _HOUR, frontier=_FRONTIER, max_gap=self._MAX_GAP
        )


@dataclass(frozen=True, slots=True)
class _Publish:
    subject: str
    severity: int

    @property
    def entity(self) -> str:
        return self.subject


class TestSubjectPlan:
    def test_every_moved_subject_becomes_one_publish(self) -> None:
        episode = _episode("2/2/227", 2)
        plan = subject_plan(
            episodes=[episode],
            open_rows=[OpenEpisodeRow(id=9, subject="2/2/224", severity=1)],
            dataless=frozenset(),
            frontier=_FRONTIER,
            publish_for=_Publish,
        )
        assert plan.inserts == (episode,)
        assert plan.orphan_closes == ((9, _FRONTIER),)
        assert plan.stale_opens == ()
        assert plan.publishes == (_Publish("2/2/224", 0), _Publish("2/2/227", 2))

    def test_a_run_that_changes_nothing_publishes_nothing(self) -> None:
        plan = subject_plan(
            episodes=[],
            open_rows=[],
            dataless=frozenset(),
            frontier=_FRONTIER,
            publish_for=_Publish,
        )
        assert plan.publishes == ()
        assert plan.inserts == ()
