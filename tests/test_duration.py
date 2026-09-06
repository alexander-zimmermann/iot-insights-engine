"""Duration-measurement tests — the `duration` kind against invented series.

Every test feeds invented active buckets and asserts only what comes out:
observations with their scores, the current-run state, the published
payload, or a resolution error naming the device. The shared
reconciliation is `test_reconcile`'s; no cluster, no live database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from iot_insights_engine.duration import (
    Device,
    DevicePublish,
    DeviceState,
    classify,
    duration_observations,
    publish_for,
    resolve_devices,
)
from iot_insights_engine.episodes import EpisodePolicy, fold_observations
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


_WASHER_DEVICE = Device(
    ga="2/1/197",
    name=_WASHER.name,
    label="Hauswirtschaftsraum.K4-L1.Waschmaschine",
    max_run=timedelta(hours=4),
)


def _running(device: Device, hours: float) -> DeviceState:
    return DeviceState(device, running_since=_T0, run_hours=hours)


class TestPublishFor:
    def test_the_payload_names_the_device_its_run_and_its_limit(self) -> None:
        publish = publish_for("2/1/197", 1, _running(_WASHER_DEVICE, 6.0))
        assert publish == DevicePublish(
            ga="2/1/197",
            severity=1,
            device="Hauswirtschaftsraum.K4-L1.Waschmaschine",
            name=_WASHER.name,
            running_since=_T0,
            run_hours=6.0,
            limit_hours=4.0,
        )
        assert publish.subject == "2/1/197"
        assert publish.entity == "2-1-197"

    def test_a_device_that_left_the_scope_still_gets_its_clear(self) -> None:
        publish = publish_for("2/1/197", 0, None)
        assert publish == DevicePublish(
            ga="2/1/197",
            severity=0,
            device="2/1/197",
            name="2/1/197",
            running_since=None,
            run_hours=None,
            limit_hours=None,
        )


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
