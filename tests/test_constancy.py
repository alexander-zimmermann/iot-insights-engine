"""Constancy-measurement tests — the `constancy` kind against invented series.

Each test feeds invented hourly readings (the shape `knx_1h` hands the
measurement) into the real computation and asserts only what comes out: the
constant runs, the per-bucket observations for the episode pipeline, the
state the payload is shaped from, or which channels are worth fetching a
series for. The spec fixture is the dead L2 voltage register — 0.0 for 717
buckets — which must fire. The shared reconciliation is `test_reconcile`'s;
no cluster, no live database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from iot_insights_engine.constancy import (
    ConstancyState,
    ConstantRun,
    Reading,
    WindowStats,
    candidates,
    classify,
    constancy_observations,
    constant_runs,
    report_for,
)
from iot_insights_engine.episodes import EpisodePolicy, fold_observations
from iot_insights_engine.reconcile import Window
from iot_insights_engine.silence import Channel

_T0 = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)

_VOLTAGE_L2 = Channel(
    ga="15/1/22", name="Versorgungstechnik.Energiezähler.Strom.Spannung-L2", dpt="14.027"
)
_VOLTAGE_L3 = Channel(
    ga="15/1/23", name="Versorgungstechnik.Energiezähler.Strom.Spannung-L3", dpt="14.027"
)


def _at(hours: float) -> datetime:
    return _T0 + hours * _HOUR


def _flat(value: float, hours: int, *, start: int = 0) -> list[Reading]:
    """`hours` contiguous buckets, all holding exactly `value`."""
    return [Reading(bucket=_at(start + h), low=value, high=value) for h in range(hours)]


def _window(frontier: datetime) -> Window:
    return Window(start=frontier - timedelta(days=30), frontier=frontier, policy=EpisodePolicy())


class TestConstantRuns:
    def test_one_held_value_is_one_run(self) -> None:
        runs = constant_runs(_flat(230.1, 5), same_within=0.0)
        assert runs == (ConstantRun(start=_at(0), end=_at(4), low=230.1, high=230.1),)
        assert runs[0].duration == 5 * _HOUR
        assert runs[0].value == 230.1

    def test_a_new_value_starts_a_new_run(self) -> None:
        readings = _flat(230.1, 3) + _flat(231.4, 2, start=3)
        runs = constant_runs(readings, same_within=0.0)
        assert [(r.start, r.duration, r.value) for r in runs] == [
            (_at(0), 3 * _HOUR, 230.1),
            (_at(3), 2 * _HOUR, 231.4),
        ]

    def test_a_missing_bucket_ends_the_run(self) -> None:
        # The channel stopped sending for an hour: silence's business, and
        # constancy cannot claim the hours it saw nothing in.
        readings = _flat(0.0, 3) + _flat(0.0, 3, start=4)
        assert [(r.start, r.duration) for r in constant_runs(readings, same_within=0.0)] == [
            (_at(0), 3 * _HOUR),
            (_at(4), 3 * _HOUR),
        ]

    def test_a_swing_inside_one_bucket_breaks_constancy(self) -> None:
        # Same hourly average, and the register is alive: the extremes are
        # what the measurement reads, never the average.
        readings = [
            Reading(bucket=_at(0), low=230.0, high=230.0),
            Reading(bucket=_at(1), low=229.0, high=231.0),
            Reading(bucket=_at(2), low=230.0, high=230.0),
        ]
        # The hour it moved in belongs to no run at all, and it separates
        # the two that do.
        assert [r.start for r in constant_runs(readings, same_within=0.0)] == [_at(0), _at(2)]

    def test_wobble_inside_the_declared_band_stays_one_run(self) -> None:
        readings = [
            Reading(bucket=_at(h), low=v, high=v)
            for h, v in enumerate((230.0, 230.2, 230.1, 230.3))
        ]
        (run,) = constant_runs(readings, same_within=0.5)
        assert run.duration == 4 * _HOUR

    def test_the_band_is_measured_over_the_whole_run_not_step_by_step(self) -> None:
        # Half a degree per hour, four hours: every step fits the band, the
        # run does not — a slow walk is drift, not a stuck register.
        readings = [
            Reading(bucket=_at(h), low=v, high=v)
            for h, v in enumerate((20.0, 20.4, 20.8, 21.2))
        ]
        assert len(constant_runs(readings, same_within=0.5)) > 1

    def test_no_readings_yield_no_runs(self) -> None:
        assert constant_runs([], same_within=0.0) == ()


class TestConstancyObservations:
    def test_a_run_within_the_limit_yields_nothing(self) -> None:
        runs = constant_runs(_flat(230.0, 4), same_within=0.0)
        assert constancy_observations("15/1/22", runs, timedelta(hours=4)) == []

    def test_every_bucket_past_the_limit_is_an_observation(self) -> None:
        runs = constant_runs(_flat(0.0, 6), same_within=0.0)
        observations = constancy_observations("15/1/22", runs, timedelta(hours=4))
        assert [(o.time, o.score, o.value) for o in observations] == [
            (_at(4), 5 / 4, 5.0),
            (_at(5), 6 / 4, 6.0),
        ]

    def test_a_value_change_restarts_the_clock(self) -> None:
        readings = _flat(230.0, 4) + _flat(231.0, 4, start=4)
        runs = constant_runs(readings, same_within=0.0)
        assert constancy_observations("15/1/22", runs, timedelta(hours=4)) == []

    def test_the_dead_voltage_register_fires(self) -> None:
        # The spec fixture: 15/1/22 read exactly 0.0 for 717 hourly buckets
        # while its sibling phases carried real voltages.
        runs = constant_runs(_flat(0.0, 717), same_within=0.0)
        observations = constancy_observations("15/1/22", runs, timedelta(hours=48))
        assert len(observations) == 717 - 48
        assert observations[0].time == _at(48)
        assert observations[-1].score == 717 / 48

        episodes = fold_observations(
            "channel_constancy", observations, [], EpisodePolicy(), _at(716)
        )
        (episode,) = episodes
        assert episode.subject == "15/1/22"
        assert episode.started_at == _at(48)
        assert episode.ended_at is None

    def test_a_live_register_never_fires(self) -> None:
        # The L3 phase over the same month: never the same value two hours
        # running.
        readings = [
            Reading(bucket=_at(h), low=230.0 + (h % 7) * 0.1, high=230.0 + (h % 7) * 0.1)
            for h in range(717)
        ]
        runs = constant_runs(readings, same_within=0.0)
        assert constancy_observations("15/1/23", runs, timedelta(hours=48)) == []


class TestState:
    def test_the_run_standing_at_the_frontier_is_the_current_one(self) -> None:
        runs = constant_runs(_flat(0.0, 6), same_within=0.0)
        state = classify(_VOLTAGE_L2, runs, _window(_at(5)))
        assert state.run is not None
        assert state.run.start == _at(0)
        assert state.stuck_hours == 6.0
        assert state.value == 0.0

    def test_a_run_the_channel_already_left_is_not_current(self) -> None:
        runs = constant_runs(_flat(0.0, 6), same_within=0.0)
        state = classify(_VOLTAGE_L2, runs, _window(_at(40)))
        assert state.run is None
        assert state.stuck_hours is None

    def test_a_lagging_channel_still_counts_as_standing(self) -> None:
        # Two hours behind the frontier is inside what the fold tolerates:
        # the channel is stuck, not unmeasured.
        runs = constant_runs(_flat(0.0, 6), same_within=0.0)
        state = classify(_VOLTAGE_L2, runs, _window(_at(7)))
        assert state.run is not None

    def test_a_channel_without_runs_has_no_current_one(self) -> None:
        assert classify(_VOLTAGE_L2, (), _window(_at(5))).run is None


class TestCandidates:
    def _stats(self, ga: str, *, low: float, high: float, buckets: int = 49) -> WindowStats:
        return WindowStats(
            ga=ga, last_bucket=_at(48), tail_buckets=buckets, tail_low=low, tail_high=high
        )

    def test_a_channel_that_never_moved_over_the_tail_is_a_candidate(self) -> None:
        stats = {"15/1/22": self._stats("15/1/22", low=0.0, high=0.0)}
        assert candidates([_VOLTAGE_L2], stats, frozenset(), same_within=0.0) == [_VOLTAGE_L2]

    def test_a_channel_that_moved_is_ruled_out_without_its_series(self) -> None:
        stats = {"15/1/23": self._stats("15/1/23", low=229.0, high=231.0)}
        assert candidates([_VOLTAGE_L3], stats, frozenset(), same_within=0.0) == []

    def test_a_channel_with_an_open_episode_is_always_a_candidate(self) -> None:
        # Its recovery is only visible in the series, so it is fetched even
        # though the tail says it moved.
        stats = {"15/1/23": self._stats("15/1/23", low=229.0, high=231.0)}
        assert candidates(
            [_VOLTAGE_L3], stats, frozenset({"15/1/23"}), same_within=0.0
        ) == [_VOLTAGE_L3]

    def test_a_single_reading_is_not_constancy(self) -> None:
        stats = {"15/1/22": self._stats("15/1/22", low=0.0, high=0.0, buckets=1)}
        assert candidates([_VOLTAGE_L2], stats, frozenset(), same_within=0.0) == []

    def test_a_channel_without_data_in_the_window_is_ruled_out(self) -> None:
        assert candidates([_VOLTAGE_L2], {}, frozenset(), same_within=0.0) == []


class TestPublish:
    def test_the_report_names_the_channel_and_what_it_is_stuck_at(self) -> None:
        runs = constant_runs(_flat(0.0, 60), same_within=0.0)
        state = classify(_VOLTAGE_L2, runs, _window(_at(59)))
        report = report_for("15/1/22", 2, state, None)
        assert report.ga == "15/1/22"
        assert report.name == _VOLTAGE_L2.name
        assert report.stuck_since == _T0
        assert report.stuck_hours == 60.0
        assert report.value == 0.0
        assert report.magnitude == 60.0

    def test_a_subject_held_open_for_want_of_data_reports_only_its_tier(self) -> None:
        report = report_for("15/1/22", 2, None, None)
        assert report.ga == "15/1/22"
        assert report.name == "15/1/22"
        assert report.stuck_since is None
        assert report.stuck_hours is None
        assert report.value is None

    def test_a_state_without_a_current_run_reports_the_channel_only(self) -> None:
        report = report_for("15/1/22", 1, ConstancyState(_VOLTAGE_L2), None)
        assert report.name == _VOLTAGE_L2.name
        assert report.stuck_since is None
