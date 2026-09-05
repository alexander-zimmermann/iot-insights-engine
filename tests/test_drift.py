"""Drift-measurement tests — the `drift` kind against invented series.

Every test feeds an invented series and asserts only what comes out: the
level the signal reads (a standby valley, a duty cycle), the accumulation,
observations with their scores, the current state, a reconciliation plan,
or a resolution error naming the device. No cluster, no live database.

The load-bearing cases are the two the z-score provably could not tell
apart: a slow ramp must fire, and a small permanent step must never fire,
however long it stands. For the duty-cycle signal a third joins them: a
door left open must not read as an iced evaporator.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from iot_insights_engine.drift import (
    Device,
    DevicePublish,
    DeviceState,
    DriftPlan,
    OnTime,
    accumulate,
    classify,
    door_hours,
    drift_observations,
    duty_cycles,
    min_window_samples,
    plan_run,
    reaches_frontier,
    resolve_devices,
    standby_floors,
)
from iot_insights_engine.episode_store import OpenEpisodeRow
from iot_insights_engine.episodes import (
    Episode,
    EpisodePolicy,
    Observation,
    fold_observations,
)
from iot_insights_engine.faults import DeviceReference
from iot_insights_engine.silence import Channel

_T0 = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)
_MINUTE = timedelta(minutes=1)
_DAY = timedelta(hours=24)

# The declared numbers of the appliance_standby fault.
_RISE = 40.0
_BUDGET = 480.0

# The declared numbers of the freezer_icing fault: a healthy freezer runs
# its compressor half the day, 15 points more is the line, and a stretch
# the compressor never leaves for three hours is a door, not ice.
_HEALTHY_DUTY = 50.0
_DUTY_RISE = 15.0
_DUTY_BUDGET = 600.0
_DOOR_RUN = timedelta(hours=3)
# The icing fault's declared coverage, through the engine's own reading of
# it: the hours a door event covers are cut out of the day, so it asks for
# less of the day than the standby fault does.
_DUTY_COVERAGE = min_window_samples(_DAY, 0.6)

_FREEZER = Channel(
    ga="2/2/227", name="Schalten.EG.Küche.K15-L1.Gefrierschrank.Stromwert", dpt="7.012"
)
_WASHER = Channel(
    ga="2/1/197", name="Schalten.KG.Hauswirtschaftsraum.K4-L1.Waschmaschine.Stromwert", dpt="7.012"
)

_REFERENCES = (
    DeviceReference(match="Hauswirtschaftsraum.K4-L1.Waschmaschine", healthy=0),
    DeviceReference(match="Küche.K15-L1.Gefrierschrank", healthy=48),
)


def _device(channel: Channel, reference: DeviceReference) -> Device:
    return Device(
        ga=channel.ga,
        name=channel.name,
        label=reference.match,
        healthy=reference.healthy,
    )


def _series(values: Sequence[float], start: datetime = _T0) -> list[tuple[datetime, float]]:
    return [(start + n * _HOUR, v) for n, v in enumerate(values)]


def _floors(values: Sequence[float], start: datetime = _T0) -> list[tuple[datetime, float]]:
    """A standby-valley series straight from values, skipping the rolling
    window — the accumulation tests are about the CUSUM, not the valley."""
    return _series(values, start)


class TestStandbyFloors:
    def test_valley_is_the_lowest_hour_of_the_trailing_day(self) -> None:
        # A freezer idling at 48 mA whose compressor lifts whole hours.
        hourly = _series([48, 270, 48, 390, 55, 48] * 4)
        floors = standby_floors(hourly, window=_DAY, min_samples=20)
        # The first 19 hours carry no day behind them; the rest read the
        # valley through the compressor's hours.
        assert [v for _, v in floors] == [48.0] * 5

    def test_no_valley_before_the_window_is_covered(self) -> None:
        assert standby_floors(_series([48] * 19), window=_DAY, min_samples=20) == []

    def test_a_thin_window_yields_no_sample(self) -> None:
        # Every third hour missing: 16 buckets span the day, under the gate.
        thin = [(_T0 + n * 3 * _HOUR, 48.0) for n in range(16)]
        assert standby_floors(thin, window=_DAY, min_samples=20) == []

    def test_the_valley_follows_the_window(self) -> None:
        # A day at 48, then a day at 100: the valley crosses over once the
        # last 48 has left the trailing day.
        hourly = _series([48.0] * 24 + [100.0] * 24)
        floors = dict(standby_floors(hourly, window=_DAY, min_samples=20))
        assert floors[_T0 + 23 * _HOUR] == 48.0
        assert floors[_T0 + 40 * _HOUR] == 48.0
        assert floors[_T0 + 47 * _HOUR] == 100.0


def _samples(duty: Sequence[float], start: datetime = _T0) -> list[OnTime]:
    """An hourly on-time series from duty cycles in percent — 100 means the
    compressor never stopped in that hour."""
    return [
        OnTime(time=start + n * _HOUR, on=d / 100 * _HOUR, total=_HOUR)
        for n, d in enumerate(duty)
    ]


def _duty(buckets: Sequence[OnTime], min_samples: int = 0) -> list[tuple[datetime, float]]:
    """The levels the icing fault would read off this series."""
    return duty_cycles(
        buckets,
        window=_DAY,
        min_samples=min_samples or _DUTY_COVERAGE,
        excluded=door_hours(buckets, door_run=_DOOR_RUN),
    )


class TestDoorHours:
    """A door left open is the one thing that looks like icing and is not:
    the compressor runs through instead of cycling. It is excluded by its
    signature — whole hours without a single idle sample, several in a row."""

    def test_a_stretch_the_compressor_never_leaves_is_a_door(self) -> None:
        # Four hours flat out in the middle of ordinary cycling.
        hours = door_hours(_samples([60] * 4 + [100] * 4 + [60] * 4), door_run=_DOOR_RUN)
        assert sorted(hours) == [_T0 + n * _HOUR for n in range(4, 8)]

    def test_a_short_saturated_stretch_is_ordinary_running(self) -> None:
        # Two hours flat out is a long cycle after a warm load, not a door.
        short = _samples([60] * 4 + [100] * 2 + [60] * 4)
        assert door_hours(short, door_run=_DOOR_RUN) == frozenset()

    def test_a_gap_ends_the_stretch(self) -> None:
        # Two saturated hours, a normal hour, two more: no door anywhere.
        broken = _samples([100] * 2 + [70] + [100] * 2)
        assert door_hours(broken, door_run=_DOOR_RUN) == frozenset()


class TestDutyCycles:
    def test_the_level_is_the_share_of_the_trailing_window(self) -> None:
        levels = _duty(_samples([40] * 24))
        # Every bucket with a covered day behind it, from the 15th on.
        assert [t for t, _ in levels] == [_T0 + n * _HOUR for n in range(14, 24)]
        assert [v for _, v in levels] == pytest.approx([40.0] * 10)

    def test_no_level_before_the_window_is_covered(self) -> None:
        assert _duty(_samples([40] * 14)) == []

    def test_hours_weigh_by_their_time_not_by_their_ratio(self) -> None:
        # A quarter-hour of delivery says less about the day than a full
        # one — a mean of hourly ratios would call this 62.5 %.
        uneven = [
            OnTime(time=_T0, on=_HOUR / 4, total=_HOUR / 4),
            OnTime(time=_T0 + _HOUR, on=_HOUR / 4, total=_HOUR),
            OnTime(time=_T0 + 2 * _HOUR, on=_HOUR / 4, total=_HOUR),
        ]
        [(_, level)] = _duty(uneven, min_samples=2)
        # The thin hour is a delivery gap and is left out altogether.
        assert level == pytest.approx(25.0)

    def test_a_door_event_is_not_counted_as_duty(self) -> None:
        # A day of ordinary cycling around a five-hour door event: the level
        # reads the cycling, not the door, so the day stays healthy.
        levels = _duty(_samples([50] * 10 + [100] * 5 + [50] * 9))
        assert levels
        assert [v for _, v in levels] == pytest.approx([50.0] * len(levels))

    def test_the_series_stays_hourly_across_a_door_event(self) -> None:
        # The load-bearing one: a hole here would be an observation gap, and
        # the episode pipeline would split one iced freezer into one episode
        # a day. The level is read *at* the door hours, only never *from*
        # them.
        levels = _duty(_samples([80] * 24 + [100] * 5 + [80] * 24))
        times = [t for t, _ in levels]
        assert max(b - a for a, b in zip(times, times[1:], strict=False)) == _HOUR
        assert times[-1] == _T0 + 52 * _HOUR

    def test_a_day_mostly_spent_with_the_door_open_is_unmeasurable(self) -> None:
        # Ten hours of cycling left of 24 is under the declared coverage:
        # better no reading than one taken around an open door.
        assert _duty(_samples([50] * 6 + [100] * 14 + [50] * 4)) == []

    def test_a_delivery_gap_is_not_a_reading(self) -> None:
        # Two minutes of telegrams in an hour says nothing about that hour.
        thin = [OnTime(time=_T0 + n * _HOUR, on=timedelta(), total=2 * _MINUTE) for n in range(24)]
        assert _duty(thin) == []


class TestIcing:
    """The freezer_icing fault end to end over its own signal: duty cycles
    in, observations out, with the declared numbers of the fault file."""

    def _levels(self, duty: Sequence[float]) -> list[tuple[datetime, float]]:
        return _duty(_samples(duty))

    def _observations(self, duty: Sequence[float]) -> list[Observation]:
        trace = accumulate(self._levels(duty), healthy=_HEALTHY_DUTY, rise=_DUTY_RISE)
        return drift_observations(
            _FREEZER.ga, trace, rise=_DUTY_RISE, budget=_DUTY_BUDGET
        )

    def test_an_iced_evaporator_fires(self) -> None:
        # The freezer as measured in June: the same cold bought with 80 % of
        # the day instead of half of it.
        duty = [80] * 24 * 5
        observations = self._observations(duty)
        assert observations
        # 30 points high, 15 of them past the declared rise: the 600
        # point-hour budget fills 40 h after the first reading, not before.
        first_level, _ = self._levels(duty)[0]
        assert observations[0].time == first_level + 40 * _HOUR
        assert observations[0].value == pytest.approx(30.0)
        assert observations[0].score == pytest.approx(30.0 / _DUTY_RISE)

    def test_a_healthy_freezer_with_a_daily_door_event_never_fires(self) -> None:
        # Five hours of door a day for a month, ordinary cycling around it:
        # the acceptance criterion that door events must not fire icing.
        month = ([50] * 10 + [100] * 5 + [50] * 9) * 30
        assert self._observations(month) == []

    def test_a_warm_load_for_a_day_does_not_fire(self) -> None:
        # A full shopping load run in at 90 % duty for a day, then back to
        # normal: high, but not standing — the budget is what tells them
        # apart, and it must not fill here.
        assert self._observations([50] * 24 + [90] * 24 + [50] * 48) == []

    def test_an_iced_freezer_whose_door_is_used_is_one_episode(self) -> None:
        # Icing and a five-hour door event every afternoon, ten days on end.
        # The door hours are cut out of the reading but not out of the
        # series, so this stays the one situation it is instead of clearing
        # and re-firing the address daily.
        duty = ([80] * 10 + [100] * 5 + [80] * 9) * 10
        observations = self._observations(duty)
        assert observations
        episodes = fold_observations(
            "freezer_icing", observations, [], EpisodePolicy(), self._levels(duty)[-1][0]
        )
        assert len(episodes) == 1
        assert episodes[0].ended_at is None

    def test_a_compressor_that_never_stops_is_unmeasured_not_healthy(self) -> None:
        # Days on end without a single idle minute: at this resolution that
        # is a door standing open, and it is Basalte's fault to report. The
        # engine says nothing rather than guessing — no level at all, which
        # makes the device dataless and holds any open episode open.
        never_idle = _samples([100] * 24 * 3)
        assert _duty(never_idle) == []
        assert self._observations([100] * 24 * 3) == []

    def test_the_episode_ends_after_a_defrost(self) -> None:
        # Iced for a week, defrosted, back to half the day: the episode must
        # close so the address clears.
        duty = [80] * 24 * 7 + [50] * 24 * 4
        observations = self._observations(duty)
        levels = self._levels(duty)
        episodes = fold_observations(
            "freezer_icing", observations, [], EpisodePolicy(), levels[-1][0]
        )
        assert len(episodes) == 1
        assert episodes[0].ended_at is not None


class TestMinWindowSamples:
    def test_the_declared_fraction_becomes_a_bucket_count(self) -> None:
        # The number that decides measurability at all: 0.8 of a day is
        # 19.2 hourly buckets, and a fifth of a day may not go missing.
        assert min_window_samples(_DAY, 0.8) == 20

    def test_a_full_window_demands_every_bucket(self) -> None:
        assert min_window_samples(_DAY, 1.0) == 24

    def test_it_rounds_up_so_a_thin_window_cannot_pass(self) -> None:
        assert min_window_samples(timedelta(hours=6), 0.5) == 3
        assert min_window_samples(timedelta(hours=7), 0.5) == 4


class TestReachesFrontier:
    """The difference between "recovered" and "unmeasured" — without it an
    episode ends on missing data and clears a still-stuck relay."""

    _MAX_GAP = EpisodePolicy().max_gap

    def test_valleys_up_to_the_frontier_are_measurable(self) -> None:
        floors = _floors([48] * 6)
        assert reaches_frontier(floors, frontier=_T0 + 5 * _HOUR, max_gap=self._MAX_GAP)

    def test_a_short_lag_still_counts(self) -> None:
        # One late bucket is materialization lag, not a blind device.
        floors = _floors([48] * 6)
        assert reaches_frontier(floors, frontier=_T0 + 7 * _HOUR, max_gap=self._MAX_GAP)

    def test_a_gap_past_the_episode_gap_is_unmeasurable(self) -> None:
        # Five missing hours stop the valley for ~29 h; the episode would
        # otherwise end and publish a clear the device never earned.
        floors = _floors([48] * 6)
        assert not reaches_frontier(floors, frontier=_T0 + 30 * _HOUR, max_gap=self._MAX_GAP)

    def test_no_valley_at_all_is_unmeasurable(self) -> None:
        assert not reaches_frontier([], frontier=_T0, max_gap=self._MAX_GAP)


class TestAccumulate:
    def test_a_healthy_device_never_accumulates(self) -> None:
        trace = accumulate(_floors([48, 49, 47, 48]), healthy=48, rise=_RISE)
        assert [a.budget_used for a in trace] == [0.0, 0.0, 0.0, 0.0]
        assert all(a.since is None for a in trace)

    def test_only_the_excess_past_the_declared_rise_accumulates(self) -> None:
        # 148 mA on a 48 mA device is 100 mA high; 60 of them count.
        trace = accumulate(_floors([148, 148, 148]), healthy=48, rise=_RISE)
        assert [a.budget_used for a in trace] == [60.0, 120.0, 180.0]
        assert [a.excess for a in trace] == [100.0, 100.0, 100.0]
        assert all(a.since == _T0 for a in trace)

    def test_a_rise_within_the_band_never_accumulates(self) -> None:
        # The Vorratsraum case: a real, permanent 10 mA step that is not
        # this fault. A budget without this floor would fire eventually.
        trace = accumulate(_floors([58] * 500), healthy=48, rise=_RISE)
        assert max(a.budget_used for a in trace) == 0.0

    def test_returning_into_the_band_starts_the_count_over(self) -> None:
        trace = accumulate(_floors([148, 148, 48, 148]), healthy=48, rise=_RISE)
        assert [a.budget_used for a in trace] == [60.0, 120.0, 0.0, 60.0]
        assert trace[-1].since == _T0 + 3 * _HOUR


class TestDriftObservations:
    def test_a_slow_ramp_fires(self) -> None:
        # 43 mA climbing 6 mA/day for six weeks — the case the 30-day
        # z-score scored at 1.83 whatever the slope (#1593).
        ramp = [43.0 + 6.0 * (n / 24) for n in range(42 * 24)]
        trace = accumulate(_floors(ramp), healthy=43, rise=_RISE)
        observations = drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET)
        assert observations
        first = observations[0]
        # Nothing fires before the ramp has climbed past the declared rise.
        assert first.value is not None
        assert first.value > _RISE
        assert first.score == pytest.approx(first.value / _RISE)
        # The excess passes the declared 40 mA on day 6.7; the budget then
        # fills against a 0.25 mA/h growth in another 2.6 days.
        assert (first.time - _T0) / _DAY == pytest.approx(9.25, abs=0.05)

    def test_a_stuck_relay_fires_within_hours(self) -> None:
        # 500 mA where 48 belong: 460 mA of excess fills a 480 mA·h budget
        # in the second hour.
        trace = accumulate(_floors([500] * 6), healthy=48, rise=_RISE)
        observations = drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET)
        assert [o.time for o in observations] == [_T0 + n * _HOUR for n in range(1, 6)]
        assert observations[0].value == 452.0
        assert observations[0].score == pytest.approx(452.0 / _RISE)

    def test_a_plateau_below_the_rise_never_fires(self) -> None:
        trace = accumulate(_floors([78] * 24 * 90), healthy=48, rise=_RISE)
        assert drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET) == []

    def test_recovery_ends_the_observations(self) -> None:
        trace = accumulate(_floors([500] * 6 + [48] * 6), healthy=48, rise=_RISE)
        observations = drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET)
        assert max(o.time for o in observations) == _T0 + 5 * _HOUR

    def test_the_same_history_yields_the_same_observations(self) -> None:
        # Recomputed from history on every run, never persisted: a rerun
        # over the same window is the same run.
        floors = _floors([500] * 6 + [48] * 3 + [300] * 8)
        again = accumulate(floors, healthy=48, rise=_RISE)
        once = accumulate(floors, healthy=48, rise=_RISE)
        assert drift_observations(
            "2/2/227", once, rise=_RISE, budget=_BUDGET
        ) == drift_observations("2/2/227", again, rise=_RISE, budget=_BUDGET)


class TestEpisodes:
    def test_one_stuck_relay_is_one_episode(self) -> None:
        trace = accumulate(_floors([500] * 12), healthy=48, rise=_RISE)
        observations = drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET)
        episodes = fold_observations(
            "appliance_standby", observations, [], EpisodePolicy(), _T0 + 11 * _HOUR
        )
        assert len(episodes) == 1
        assert episodes[0].subject == _FREEZER.ga
        assert episodes[0].ended_at is None

    def test_the_episode_ends_after_the_device_recovers(self) -> None:
        trace = accumulate(_floors([500] * 6 + [48] * 12), healthy=48, rise=_RISE)
        observations = drift_observations(_FREEZER.ga, trace, rise=_RISE, budget=_BUDGET)
        episodes = fold_observations(
            "appliance_standby", observations, [], EpisodePolicy(), _T0 + 17 * _HOUR
        )
        assert len(episodes) == 1
        assert episodes[0].ended_at is not None


class TestClassify:
    def test_a_drifting_device_names_its_excess_and_since(self) -> None:
        device = _device(_FREEZER, _REFERENCES[1])
        trace = accumulate(_floors([500] * 6), healthy=device.healthy, rise=_RISE)
        state = classify(device, trace, frontier=_T0 + 5 * _HOUR)
        assert state == DeviceState(
            device=device, level=500.0, excess=452.0, rising_since=_T0
        )

    def test_a_healthy_device_names_only_its_standby(self) -> None:
        device = _device(_FREEZER, _REFERENCES[1])
        trace = accumulate(_floors([48] * 6), healthy=device.healthy, rise=_RISE)
        state = classify(device, trace, frontier=_T0 + 5 * _HOUR)
        assert state == DeviceState(device=device, level=48.0, excess=0.0)

    def test_a_series_short_of_the_frontier_leaves_the_state_empty(self) -> None:
        device = _device(_FREEZER, _REFERENCES[1])
        trace = accumulate(_floors([500] * 6), healthy=device.healthy, rise=_RISE)
        assert classify(device, trace, frontier=_T0 + 20 * _HOUR) == DeviceState(device)


class TestResolveDevices:
    def test_maps_each_reference_to_its_channel(self) -> None:
        devices = resolve_devices([_FREEZER, _WASHER], _REFERENCES)
        assert devices == [
            _device(_WASHER, _REFERENCES[0]),
            _device(_FREEZER, _REFERENCES[1]),
        ]

    def test_reference_matching_no_channel_is_an_error(self) -> None:
        with pytest.raises(ValueError, match=r"Gefrierschrank.*no channel"):
            resolve_devices([_WASHER], _REFERENCES)

    def test_channel_without_reference_is_an_error(self) -> None:
        with pytest.raises(ValueError, match=r"2/2/227.*no declared reference"):
            resolve_devices([_FREEZER, _WASHER], _REFERENCES[:1])


class TestPlanRun:
    def _episode(self, subject: str, *, ended: bool, severity: int = 1) -> Episode:
        return Episode(
            fault="appliance_standby",
            subject=subject,
            started_at=_T0,
            last_seen_at=_T0 + 5 * _HOUR,
            ended_at=_T0 + 9 * _HOUR if ended else None,
            severity=severity,
            peak_score=11.3,
            evidence=(),
            events=(),
        )

    def test_an_open_episode_without_a_row_is_inserted_and_published(self) -> None:
        device = _device(_FREEZER, _REFERENCES[1])
        state = DeviceState(device, level=500.0, excess=452.0, rising_since=_T0)
        plan = plan_run(
            episodes=[self._episode(_FREEZER.ga, ended=False)],
            open_rows=[],
            states_by_ga={_FREEZER.ga: state},
            dataless=frozenset(),
            frontier=_T0 + 9 * _HOUR,
        )
        assert plan == DriftPlan(
            inserts=(self._episode(_FREEZER.ga, ended=False),),
            updates=(),
            orphan_closes=(),
            stale_opens=(),
            publishes=(
                DevicePublish(
                    ga=_FREEZER.ga,
                    severity=1,
                    device=device.label,
                    name=device.name,
                    level=500.0,
                    healthy=48.0,
                    excess=452.0,
                    rising_since=_T0,
                ),
            ),
        )

    def test_a_recovered_device_closes_its_row_and_publishes_the_clear(self) -> None:
        device = _device(_FREEZER, _REFERENCES[1])
        plan = plan_run(
            episodes=[],
            open_rows=[OpenEpisodeRow(id=7, subject=_FREEZER.ga, severity=2)],
            states_by_ga={_FREEZER.ga: DeviceState(device, level=48.0, excess=0.0)},
            dataless=frozenset(),
            frontier=_T0 + 9 * _HOUR,
        )
        assert plan.orphan_closes == ((7, _T0 + 9 * _HOUR),)
        assert [(p.ga, p.severity) for p in plan.publishes] == [(_FREEZER.ga, 0)]

    def test_a_device_without_data_keeps_its_open_row(self) -> None:
        plan = plan_run(
            episodes=[],
            open_rows=[OpenEpisodeRow(id=7, subject=_FREEZER.ga, severity=2)],
            states_by_ga={},
            dataless=frozenset({_FREEZER.ga}),
            frontier=_T0 + 9 * _HOUR,
        )
        assert plan.orphan_closes == ()
        assert plan.stale_opens == (_FREEZER.ga,)
        assert plan.publishes == ()

    def test_a_stored_severity_is_never_lowered(self) -> None:
        device = _device(_FREEZER, _REFERENCES[1])
        state = DeviceState(device, level=500.0, excess=452.0, rising_since=_T0)
        plan = plan_run(
            episodes=[self._episode(_FREEZER.ga, ended=False, severity=1)],
            open_rows=[OpenEpisodeRow(id=7, subject=_FREEZER.ga, severity=3)],
            states_by_ga={_FREEZER.ga: state},
            dataless=frozenset(),
            frontier=_T0 + 9 * _HOUR,
        )
        assert plan.updates == ((7, self._episode(_FREEZER.ga, ended=False, severity=1)),)
        assert plan.publishes == ()
