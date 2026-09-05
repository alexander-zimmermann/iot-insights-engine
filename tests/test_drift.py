"""Drift-measurement tests — the `drift` kind against invented series.

Every test feeds invented hourly floors and asserts only what comes out:
the standby valley, the accumulation, observations with their scores, the
current state, a reconciliation plan, or a resolution error naming the
device. No cluster, no live database.

The load-bearing cases are the two the z-score provably could not tell
apart (#1593): a slow ramp must fire, and a small permanent step must
never fire, however long it stands.
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
    accumulate,
    classify,
    drift_observations,
    plan_run,
    resolve_devices,
    standby_floors,
)
from iot_insights_engine.episode_store import OpenEpisodeRow
from iot_insights_engine.episodes import Episode, EpisodePolicy, fold_observations
from iot_insights_engine.faults import DeviceReference
from iot_insights_engine.silence import Channel

_T0 = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)
_DAY = timedelta(hours=24)

# The declared numbers of the appliance_standby fault.
_RISE = 40.0
_BUDGET = 480.0

_FREEZER = Channel(
    ga="2/2/227", name="Schalten.EG.Küche.K15-L1.Gefrierschrank.Stromwert", dpt="7.012"
)
_WASHER = Channel(
    ga="2/1/197", name="Schalten.KG.Hauswirtschaftsraum.K4-L1.Waschmaschine.Stromwert", dpt="7.012"
)

_REFERENCES = (
    DeviceReference(match="Hauswirtschaftsraum.K4-L1.Waschmaschine", healthy_ma=0),
    DeviceReference(match="Küche.K15-L1.Gefrierschrank", healthy_ma=48),
)


def _device(channel: Channel, reference: DeviceReference) -> Device:
    return Device(
        ga=channel.ga,
        name=channel.name,
        label=reference.match,
        healthy=reference.healthy_ma,
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
            device=device, standby=500.0, excess=452.0, rising_since=_T0
        )

    def test_a_healthy_device_names_only_its_standby(self) -> None:
        device = _device(_FREEZER, _REFERENCES[1])
        trace = accumulate(_floors([48] * 6), healthy=device.healthy, rise=_RISE)
        state = classify(device, trace, frontier=_T0 + 5 * _HOUR)
        assert state == DeviceState(device=device, standby=48.0, excess=0.0)

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
        state = DeviceState(device, standby=500.0, excess=452.0, rising_since=_T0)
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
                    standby=500.0,
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
            states_by_ga={_FREEZER.ga: DeviceState(device, standby=48.0, excess=0.0)},
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
        state = DeviceState(device, standby=500.0, excess=452.0, rising_since=_T0)
        plan = plan_run(
            episodes=[self._episode(_FREEZER.ga, ended=False, severity=1)],
            open_rows=[OpenEpisodeRow(id=7, subject=_FREEZER.ga, severity=3)],
            states_by_ga={_FREEZER.ga: state},
            dataless=frozenset(),
            frontier=_T0 + 9 * _HOUR,
        )
        assert plan.updates == ((7, self._episode(_FREEZER.ga, ended=False, severity=1)),)
        assert plan.publishes == ()
