"""Duration-measurement tests — the `duration` kind against invented series.

Every test feeds invented active buckets and asserts only what comes out:
observations with their scores, the current-run state, a reconciliation
plan, or a resolution error naming the device. No cluster, no live
database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from iot_insights_engine.duration import (
    Device,
    DevicePublish,
    DeviceState,
    DurationPlan,
    classify,
    duration_observations,
    plan_run,
    resolve_devices,
)
from iot_insights_engine.episode_store import OpenEpisodeRow
from iot_insights_engine.episodes import (
    Episode,
    EpisodePolicy,
    EventKind,
    EvidenceRow,
    NotificationEvent,
    fold_observations,
)
from iot_insights_engine.faults import DeviceLimit
from iot_insights_engine.silence import Channel

_T0 = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)

_WASHER = Channel(
    ga="2/1/197", name="Schalten.KG.Hauswirtschaftsraum.K4-L1.Waschmaschine.Stromwert", dpt="7.012"
)
_FREEZER = Channel(
    ga="2/2/227", name="Schalten.EG.Küche.K15-L1.Gefrierschrank.Stromwert", dpt="7.012"
)

_LIMITS = (
    DeviceLimit(match="Hauswirtschaftsraum.K4-L1.Waschmaschine", max_run_hours=4),
    DeviceLimit(match="Küche.K15-L1.Gefrierschrank", max_run_hours=6),
)


def _hours(*offsets: int) -> list[datetime]:
    return [_T0 + n * _HOUR for n in offsets]


def _device(channel: Channel, limit: DeviceLimit) -> Device:
    return Device(
        ga=channel.ga,
        name=channel.name,
        label=limit.match,
        max_run=timedelta(hours=limit.max_run_hours),
    )


class TestResolveDevices:
    def test_maps_each_limit_to_its_channel(self) -> None:
        devices = resolve_devices([_FREEZER, _WASHER], _LIMITS)
        assert devices == [_device(_WASHER, _LIMITS[0]), _device(_FREEZER, _LIMITS[1])]

    def test_limit_matching_no_channel_is_an_error(self) -> None:
        with pytest.raises(ValueError, match=r"Gefrierschrank.*no channel"):
            resolve_devices([_WASHER], _LIMITS)

    def test_channel_without_limit_is_an_error(self) -> None:
        with pytest.raises(ValueError, match=r"2/2/227.*no declared limit"):
            resolve_devices([_FREEZER, _WASHER], _LIMITS[:1])

    def test_ambiguous_limit_is_an_error(self) -> None:
        twin = Channel(
            ga="2/1/199",
            name="Schalten.KG.Hauswirtschaftsraum.K4-L1.Waschmaschine.Stromwert-Anomalie",
            dpt="5.010",
        )
        with pytest.raises(ValueError, match=r"Waschmaschine.*2 channels"):
            resolve_devices([_WASHER, twin, _FREEZER], _LIMITS)

    def test_all_problems_reported_at_once(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            resolve_devices([_FREEZER], _LIMITS[:1])
        assert "Waschmaschine" in str(excinfo.value)
        assert "2/2/227" in str(excinfo.value)


class TestRuntimeObservations:
    def test_run_within_limit_yields_nothing(self) -> None:
        assert duration_observations("2/1/197", _hours(0, 1, 2), timedelta(hours=4)) == []

    def test_overlong_run_yields_one_observation_per_excess_bucket(self) -> None:
        # 6 active hours against a 4 h limit: the 5th and 6th hour are over.
        observations = duration_observations(
            "2/1/197", _hours(0, 1, 2, 3, 4, 5), timedelta(hours=4)
        )
        assert [(o.time, o.score, o.value) for o in observations] == [
            (_T0 + 4 * _HOUR, 5 / 4, 5.0),
            (_T0 + 5 * _HOUR, 6 / 4, 6.0),
        ]

    def test_idle_hour_restarts_the_clock(self) -> None:
        # Two 3 h runs separated by an idle hour never exceed a 4 h limit.
        active = _hours(0, 1, 2, 4, 5, 6)
        assert duration_observations("2/1/197", active, timedelta(hours=4)) == []

    def test_fractional_limit(self) -> None:
        observations = duration_observations("2/1/197", _hours(0, 1), timedelta(hours=1.5))
        assert [(o.time, o.score) for o in observations] == [(_T0 + _HOUR, 2 / 1.5)]


class TestState:
    _dev = Device(ga="2/1/197", name="x", label="Waschmaschine", max_run=timedelta(hours=4))

    def test_running_at_the_frontier(self) -> None:
        s = classify(self._dev, _hours(0, 1, 2), frontier=_T0 + 2 * _HOUR)
        assert s.running_since == _T0
        assert s.run_hours == 3.0

    def test_stopped_before_the_frontier(self) -> None:
        s = classify(self._dev, _hours(0, 1, 2), frontier=_T0 + 5 * _HOUR)
        assert s.running_since is None
        assert s.run_hours is None

    def test_never_active(self) -> None:
        s = classify(self._dev, [], frontier=_T0)
        assert s.running_since is None


_FRONTIER = _T0 + 8 * _HOUR

_WASHER_DEVICE = Device(
    ga="2/1/197",
    name=_WASHER.name,
    label="Hauswirtschaftsraum.K4-L1.Waschmaschine",
    max_run=timedelta(hours=4),
)


def _episode(subject: str, severity: int, *, ended: bool = False) -> Episode:
    start = _T0 + 6 * _HOUR
    evidence = (
        EvidenceRow(time=start, score=1.25, severity=severity, value=5.0),
        EvidenceRow(time=start + _HOUR, score=1.5, severity=severity, value=6.0),
    )
    events = [NotificationEvent(EventKind.APPEARED, start, severity)]
    ended_at = None
    if ended:
        ended_at = start + 5 * _HOUR
        events.append(NotificationEvent(EventKind.ENDED, ended_at, 0))
    return Episode(
        fault="appliance_runtime",
        subject=subject,
        started_at=start,
        last_seen_at=start + _HOUR,
        ended_at=ended_at,
        severity=severity,
        peak_score=1.5,
        evidence=evidence,
        events=tuple(events),
    )


def _running(device: Device, hours: float) -> DeviceState:
    return DeviceState(device, running_since=_T0, run_hours=hours)


class TestPlanRun:
    def _plan(
        self,
        episodes: list[Episode],
        open_rows: list[OpenEpisodeRow],
        states: dict[str, DeviceState] | None = None,
        dataless: frozenset[str] = frozenset(),
    ) -> DurationPlan:
        return plan_run(
            episodes=episodes,
            open_rows=open_rows,
            states_by_ga=states
            if states is not None
            else {"2/1/197": _running(_WASHER_DEVICE, 6.0)},
            dataless=dataless,
            frontier=_FRONTIER,
        )

    def test_new_overlong_run_is_inserted_and_published(self) -> None:
        episode = _episode("2/1/197", severity=1)
        plan = self._plan([episode], open_rows=[])
        assert plan.inserts == (episode,)
        assert plan.updates == ()
        (publish,) = plan.publishes
        assert publish == DevicePublish(
            ga="2/1/197",
            severity=1,
            device="Hauswirtschaftsraum.K4-L1.Waschmaschine",
            name=_WASHER.name,
            running_since=_T0,
            run_hours=6.0,
            limit_hours=4.0,
        )

    def test_ongoing_episode_with_unchanged_severity_publishes_nothing(self) -> None:
        episode = _episode("2/1/197", severity=1)
        row = OpenEpisodeRow(id=7, subject="2/1/197", severity=1)
        plan = self._plan([episode], open_rows=[row])
        assert plan.inserts == ()
        assert plan.updates == ((7, episode),)
        assert plan.publishes == ()

    def test_escalation_publishes_the_new_severity(self) -> None:
        episode = _episode("2/1/197", severity=2)
        row = OpenEpisodeRow(id=7, subject="2/1/197", severity=1)
        plan = self._plan([episode], open_rows=[row])
        (publish,) = plan.publishes
        assert publish.severity == 2

    def test_stored_severity_is_never_lowered(self) -> None:
        episode = _episode("2/1/197", severity=1)
        row = OpenEpisodeRow(id=7, subject="2/1/197", severity=2)
        plan = self._plan([episode], open_rows=[row])
        assert plan.publishes == ()

    def test_recovery_publishes_zero_and_reconciles_the_ended_episode(self) -> None:
        episode = _episode("2/1/197", severity=1, ended=True)
        row = OpenEpisodeRow(id=7, subject="2/1/197", severity=1)
        plan = self._plan([episode], open_rows=[row])
        assert plan.updates == ((7, episode),)
        (publish,) = plan.publishes
        assert publish.severity == 0
        assert publish.ga == "2/1/197"

    def test_open_row_without_computed_counterpart_is_closed_at_frontier(self) -> None:
        row = OpenEpisodeRow(id=9, subject="2/1/197", severity=1)
        plan = self._plan([], open_rows=[row])
        assert plan.orphan_closes == ((9, _FRONTIER),)
        (publish,) = plan.publishes
        assert publish.severity == 0

    def test_device_outliving_the_window_stays_open(self) -> None:
        row = OpenEpisodeRow(id=9, subject="2/1/197", severity=2)
        plan = self._plan([], open_rows=[row], dataless=frozenset({"2/1/197"}))
        assert plan.orphan_closes == ()
        assert plan.stale_opens == ("2/1/197",)
        assert plan.publishes == ()

    def test_historical_ended_episode_without_open_row_is_ignored(self) -> None:
        plan = self._plan([_episode("2/1/197", severity=1, ended=True)], open_rows=[])
        assert plan.inserts == ()
        assert plan.updates == ()
        assert plan.publishes == ()


def test_four_hour_run_becomes_one_episode_with_few_events() -> None:
    # The acceptance case: a 4 h run of a device that normally runs 90 min
    # is one episode with at most three notification events.
    observations = duration_observations("2/1/197", _hours(0, 1, 2, 3), timedelta(hours=1.5))
    episodes = fold_observations(
        "appliance_runtime", observations, [], EpisodePolicy(), _T0 + 3 * _HOUR
    )
    assert len(episodes) == 1
    [episode] = episodes
    assert episode.subject == "2/1/197"
    assert episode.started_at == _T0 + _HOUR
    assert episode.last_seen_at == _T0 + 3 * _HOUR
    assert len(episode.events) <= 3
