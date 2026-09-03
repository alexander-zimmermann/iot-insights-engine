"""Deviation-measurement tests — the `deviation` kind against invented series.

Every test feeds invented hourly values for a room's channel triple (value,
reference, gate) and asserts only what comes out: the dense room series,
which buckets count as cold, observations with their scores, the current
state, a reconciliation plan, or a resolution error naming the room. No
cluster, no live database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from iot_insights_engine.deviation import (
    DeviationPlan,
    Room,
    RoomBucket,
    RoomPublish,
    RoomState,
    classify,
    cold_buckets,
    dead_value_gas,
    deviation_observations,
    plan_run,
    resolve_rooms,
    room_series,
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
from iot_insights_engine.faults import Roles, RoomRule
from iot_insights_engine.silence import Channel

_T0 = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)

_BUERO_SENSOR = Channel(ga="8/2/61", name="Sensorik.EG.Büro.Sensor.Temperatur", dpt="9.001")
_BUERO_SOLL = Channel(
    ga="6/2/42", name="Raumklima.EG.Büro.FBH.Soll-Temperatur-Status", dpt="9.001"
)
_BUERO_VALVE = Channel(ga="6/2/40", name="Raumklima.EG.Büro.FBH.Stellwert-Status", dpt="5.001")

_FLUR_BWM_DIELE = Channel(ga="8/1/11", name="Sensorik.EG.Flur.BWM.Diele.Temperatur", dpt="9.001")
_FLUR_BWM_EINGANG = Channel(
    ga="8/1/21", name="Sensorik.EG.Flur.BWM.Eingang.Temperatur", dpt="9.001"
)
_FLUR_SOLL = Channel(ga="6/1/2", name="Raumklima.EG.Flur.FBH.Soll-Temperatur-Status", dpt="9.001")
_FLUR_VALVE = Channel(ga="6/1/1", name="Raumklima.EG.Flur.FBH.Stellwert-Status", dpt="5.001")

_ROLES = Roles(
    value="%.Sensor.Temperatur",
    reference="%.FBH.Soll-Temperatur-Status",
    gate="%.FBH.Stellwert-Status",
)
_RULES = (
    RoomRule(match="EG.Büro", min_gap_k=1.0),
    RoomRule(match="EG.Flur", min_gap_k=1.0, value="Sensorik.EG.Flur.BWM.%.Temperatur"),
)
_ALL_CHANNELS = [
    _BUERO_SENSOR,
    _BUERO_SOLL,
    _BUERO_VALVE,
    _FLUR_BWM_DIELE,
    _FLUR_BWM_EINGANG,
    _FLUR_SOLL,
    _FLUR_VALVE,
]

_BUERO = Room(
    label="EG.Büro",
    slug="eg-buero",
    value_gas=("8/2/61",),
    reference_ga="6/2/42",
    gate_ga="6/2/40",
    min_gap=1.0,
)
_FLUR = Room(
    label="EG.Flur",
    slug="eg-flur",
    value_gas=("8/1/11", "8/1/21"),
    reference_ga="6/1/2",
    gate_ga="6/1/1",
    min_gap=1.0,
)


def _hours(*offsets: int) -> list[datetime]:
    return [_T0 + n * _HOUR for n in offsets]


def _series(
    values: dict[int, float],
    references: dict[int, float],
    gates: dict[int, float],
    room: Room = _BUERO,
) -> dict[str, dict[datetime, float]]:
    return {
        room.value_gas[0]: {_T0 + n * _HOUR: v for n, v in values.items()},
        room.reference_ga: {_T0 + n * _HOUR: v for n, v in references.items()},
        str(room.gate_ga): {_T0 + n * _HOUR: v for n, v in gates.items()},
    }


def _bucket(
    n: int, value: float, reference: float, gate: float | None = None
) -> RoomBucket:
    return RoomBucket(
        time=_T0 + n * _HOUR, value=value, reference=reference, gate=gate
    )


class TestResolveRooms:
    def test_maps_each_room_to_its_channel_triple(self) -> None:
        rooms = resolve_rooms(_ALL_CHANNELS, _RULES, _ROLES)
        assert rooms == [_BUERO, _FLUR]

    def test_slug_transliterates_umlauts(self) -> None:
        wc_channels = [
            Channel(ga="8/2/70", name="Sensorik.EG.Gäste-WC.Sensor.Temperatur", dpt="9.001"),
            Channel(
                ga="6/2/22",
                name="Raumklima.EG.Gäste-WC.FBH.Soll-Temperatur-Status",
                dpt="9.001",
            ),
            Channel(
                ga="6/2/20", name="Raumklima.EG.Gäste-WC.FBH.Stellwert-Status", dpt="5.001"
            ),
        ]
        [room] = resolve_rooms(wc_channels, (RoomRule(match="EG.Gäste-WC", min_gap_k=1.0),), _ROLES)
        assert room.slug == "eg-gaeste-wc"

    def test_room_without_reference_channel_is_an_error(self) -> None:
        channels = [_BUERO_SENSOR, _BUERO_VALVE]
        with pytest.raises(ValueError, match=r"EG\.Büro.*reference"):
            resolve_rooms(channels, _RULES[:1], _ROLES)

    def test_channel_without_room_is_an_error(self) -> None:
        with pytest.raises(ValueError, match=r"8/1/11.*no declared room"):
            resolve_rooms(_ALL_CHANNELS, _RULES[:1], _ROLES)

    def test_room_channel_matching_no_role_is_an_error(self) -> None:
        stray = Channel(ga="6/2/49", name="Raumklima.EG.Büro.FBH.Diagnose", dpt="20.102")
        with pytest.raises(ValueError, match=r"6/2/49.*no role"):
            resolve_rooms([*_ALL_CHANNELS, stray], _RULES, _ROLES)

    def test_ambiguous_reference_is_an_error(self) -> None:
        twin = Channel(
            ga="6/2/43", name="Raumklima.EG.Büro.Zweit.FBH.Soll-Temperatur-Status", dpt="9.001"
        )
        with pytest.raises(ValueError, match=r"EG\.Büro.*reference.*2 channels"):
            resolve_rooms([*_ALL_CHANNELS, twin], _RULES, _ROLES)

    def test_missing_gate_role_leaves_gate_unset(self) -> None:
        roles = Roles(value="%.Sensor.Temperatur", reference="%.FBH.Soll-Temperatur-Status")
        rooms = resolve_rooms(
            [_BUERO_SENSOR, _BUERO_SOLL], (RoomRule(match="EG.Büro", min_gap_k=1.0),), roles
        )
        assert rooms[0].gate_ga is None

    def test_channel_claimed_by_two_rooms_is_an_error(self) -> None:
        # "EG.Flur" and a leftover "Flur" rule both contain the hall's
        # channels — a merge artifact that must fail, not double-measure.
        rules = (*_RULES, RoomRule(match="Flur", min_gap_k=2.0))
        with pytest.raises(ValueError, match=r"8/1/11.*'EG\.Flur'.*'Flur'"):
            resolve_rooms(_ALL_CHANNELS, rules, _ROLES)

    def test_colliding_slugs_are_an_error(self) -> None:
        # The slug is episode subject and NATS entity: two labels that
        # transliterate to one slug would silently merge two rooms.
        rules = (
            RoomRule(match="EG.Büro", min_gap_k=1.0),
            RoomRule(match="EG.Buero", min_gap_k=1.0),
        )
        with pytest.raises(ValueError, match=r"'EG\.Büro'.*'EG\.Buero'.*eg-buero"):
            resolve_rooms([_BUERO_SENSOR, _BUERO_SOLL, _BUERO_VALVE], rules, _ROLES)

    def test_all_problems_reported_at_once(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            resolve_rooms([_BUERO_SENSOR, _BUERO_VALVE, _FLUR_BWM_DIELE], _RULES[:1], _ROLES)
        assert "EG.Büro" in str(excinfo.value)
        assert "8/1/11" in str(excinfo.value)


class TestRoomSeries:
    def test_reference_and_gate_are_carried_forward(self) -> None:
        # Setpoint and valve send on change only; the sensor sends hourly.
        by_ga = _series(
            values={0: 20.5, 1: 20.4, 2: 20.6},
            references={0: 22.0},
            gates={0: 80.0},
        )
        buckets = room_series(_BUERO, by_ga, _T0, _T0 + 2 * _HOUR)
        assert buckets == [
            _bucket(0, 20.5, 22.0, 80.0),
            _bucket(1, 20.4, 22.0, 80.0),
            _bucket(2, 20.6, 22.0, 80.0),
        ]

    def test_value_is_carried_forward_over_a_silent_hour(self) -> None:
        by_ga = _series(values={0: 20.5, 2: 20.7}, references={0: 22.0}, gates={0: 80.0})
        buckets = room_series(_BUERO, by_ga, _T0, _T0 + 2 * _HOUR)
        assert [b.value for b in buckets] == [20.5, 20.5, 20.7]

    def test_value_averages_the_room_sensors_seen_so_far(self) -> None:
        by_ga = {
            "8/1/11": {_T0: 19.0, _T0 + _HOUR: 19.2},
            "8/1/21": {_T0 + _HOUR: 20.2},
            "6/1/2": {_T0: 22.0},
            "6/1/1": {_T0: 80.0},
        }
        buckets = room_series(_FLUR, by_ga, _T0, _T0 + _HOUR)
        assert [b.value for b in buckets] == [19.0, pytest.approx(19.7)]

    def test_buckets_before_all_roles_appeared_are_skipped(self) -> None:
        by_ga = _series(values={0: 20.5, 1: 20.4}, references={1: 22.0}, gates={0: 80.0})
        buckets = room_series(_BUERO, by_ga, _T0, _T0 + _HOUR)
        assert [b.time for b in buckets] == [_T0 + _HOUR]

    def test_room_without_any_data_yields_nothing(self) -> None:
        assert room_series(_BUERO, {}, _T0, _T0 + 5 * _HOUR) == []


class TestDeadValueGas:
    def test_value_channel_constant_at_zero_for_a_day_is_dead(self) -> None:
        by_ga = {"8/2/61": dict.fromkeys(_hours(*range(24)), 0.0)}
        assert dead_value_gas([_BUERO], by_ga) == ["8/2/61"]

    def test_short_zero_stretch_is_not_dead(self) -> None:
        by_ga = {"8/2/61": dict.fromkeys(_hours(*range(23)), 0.0)}
        assert dead_value_gas([_BUERO], by_ga) == []

    def test_live_sensor_is_not_dead(self) -> None:
        by_ga = {"8/2/61": {_T0 + n * _HOUR: 0.0 if n else 20.5 for n in range(24)}}
        assert dead_value_gas([_BUERO], by_ga) == []

    def test_gate_at_zero_is_not_dead(self) -> None:
        # A closed valve legitimately sits at 0 % for weeks — only the
        # value role is checked.
        by_ga = {"6/2/40": dict.fromkeys(_hours(*range(24)), 0.0)}
        assert dead_value_gas([_BUERO], by_ga) == []


class TestColdBuckets:
    def test_gap_at_open_valve_is_cold(self) -> None:
        buckets = [_bucket(0, 20.5, 22.0, 80.0)]
        assert cold_buckets(_BUERO, buckets, gate_min=50.0) == buckets

    def test_closed_valve_is_not_cold(self) -> None:
        # The acceptance case's quiet half: the same gap at 20 % stays quiet.
        buckets = [_bucket(0, 20.5, 22.0, 20.0)]
        assert cold_buckets(_BUERO, buckets, gate_min=50.0) == []

    def test_gap_under_the_room_threshold_is_not_cold(self) -> None:
        buckets = [_bucket(0, 21.5, 22.0, 80.0)]
        assert cold_buckets(_BUERO, buckets, gate_min=50.0) == []

    def test_gap_of_exactly_the_threshold_is_cold(self) -> None:
        buckets = [_bucket(0, 21.0, 22.0, 80.0)]
        assert cold_buckets(_BUERO, buckets, gate_min=50.0) == buckets

    def test_without_a_gate_role_the_gap_alone_decides(self) -> None:
        room = Room(
            label="EG.Büro",
            slug="eg-buero",
            value_gas=("8/2/61",),
            reference_ga="6/2/42",
            gate_ga=None,
            min_gap=1.0,
        )
        buckets = [_bucket(0, 20.5, 22.0, None)]
        assert cold_buckets(room, buckets, gate_min=None) == buckets


class TestDeviationObservations:
    def test_three_cold_hours_fire_from_the_second(self) -> None:
        # The acceptance case: 1.5 K under setpoint at 80 % valve for 3 h.
        cold = [_bucket(n, 20.5, 22.0, 80.0) for n in (0, 1, 2)]
        observations = deviation_observations(_BUERO, cold, min_hours=2.0)
        assert [(o.subject, o.time, o.score, o.value) for o in observations] == [
            ("eg-buero", _T0 + _HOUR, 1.5, 1.5),
            ("eg-buero", _T0 + 2 * _HOUR, 1.5, 1.5),
        ]

    def test_a_single_cold_hour_stays_quiet(self) -> None:
        assert deviation_observations(_BUERO, [_bucket(0, 20.5, 22.0, 80.0)], min_hours=2.0) == []

    def test_a_warm_hour_restarts_the_clock(self) -> None:
        cold = [_bucket(n, 20.5, 22.0, 80.0) for n in (0, 1, 3)]
        observations = deviation_observations(_BUERO, cold, min_hours=2.0)
        assert [o.time for o in observations] == [_T0 + _HOUR]

    def test_score_is_the_gap_in_units_of_the_room_threshold(self) -> None:
        room = Room(
            label="EG.Büro",
            slug="eg-buero",
            value_gas=("8/2/61",),
            reference_ga="6/2/42",
            gate_ga="6/2/40",
            min_gap=2.0,
        )
        cold = [_bucket(n, 19.0, 22.0, 80.0) for n in (0, 1)]
        observations = deviation_observations(room, cold, min_hours=2.0)
        assert [(o.score, o.value) for o in observations] == [(1.5, 3.0)]


class TestState:
    def test_cold_at_the_frontier(self) -> None:
        cold = [_bucket(n, 20.5, 22.0, 80.0) for n in (1, 2)]
        state = classify(_BUERO, cold, frontier=_T0 + 2 * _HOUR)
        assert state == RoomState(
            room=_BUERO,
            cold_since=_T0 + _HOUR,
            gap=1.5,
            value=20.5,
            reference=22.0,
            gate=80.0,
        )

    def test_unordered_cold_buckets_still_name_the_frontier_hour(self) -> None:
        cold = [_bucket(2, 20.0, 22.0, 80.0), _bucket(1, 20.5, 22.0, 80.0)]
        state = classify(_BUERO, cold, frontier=_T0 + 2 * _HOUR)
        assert state.gap == 2.0
        assert state.cold_since == _T0 + _HOUR

    def test_recovered_before_the_frontier(self) -> None:
        cold = [_bucket(n, 20.5, 22.0, 80.0) for n in (1, 2)]
        state = classify(_BUERO, cold, frontier=_T0 + 5 * _HOUR)
        assert state == RoomState(room=_BUERO)

    def test_never_cold(self) -> None:
        assert classify(_BUERO, [], frontier=_T0) == RoomState(room=_BUERO)


_FRONTIER = _T0 + 8 * _HOUR


def _episode(subject: str, severity: int, *, ended: bool = False) -> Episode:
    start = _T0 + 6 * _HOUR
    evidence = (
        EvidenceRow(time=start, score=1.5, severity=severity, value=1.5),
        EvidenceRow(time=start + _HOUR, score=1.8, severity=severity, value=1.8),
    )
    events = [NotificationEvent(EventKind.APPEARED, start, severity)]
    ended_at = None
    if ended:
        ended_at = start + 5 * _HOUR
        events.append(NotificationEvent(EventKind.ENDED, ended_at, 0))
    return Episode(
        fault="fbh_cold",
        subject=subject,
        started_at=start,
        last_seen_at=start + _HOUR,
        ended_at=ended_at,
        severity=severity,
        peak_score=1.8,
        evidence=evidence,
        events=tuple(events),
    )


def _cold_state(room: Room) -> RoomState:
    return RoomState(
        room=room, cold_since=_T0, gap=1.8, value=20.2, reference=22.0, gate=85.0
    )


class TestPlanRun:
    def _plan(
        self,
        episodes: list[Episode],
        open_rows: list[OpenEpisodeRow],
        states: dict[str, RoomState] | None = None,
        dataless: frozenset[str] = frozenset(),
    ) -> DeviationPlan:
        return plan_run(
            episodes=episodes,
            open_rows=open_rows,
            states_by_slug=states if states is not None else {"eg-buero": _cold_state(_BUERO)},
            dataless=dataless,
            frontier=_FRONTIER,
        )

    def test_new_cold_room_is_inserted_and_published(self) -> None:
        episode = _episode("eg-buero", severity=1)
        plan = self._plan([episode], open_rows=[])
        assert plan.inserts == (episode,)
        assert plan.updates == ()
        (publish,) = plan.publishes
        assert publish == RoomPublish(
            slug="eg-buero",
            severity=1,
            room="EG.Büro",
            cold_since=_T0,
            gap=1.8,
            value=20.2,
            reference=22.0,
            gate=85.0,
            min_gap=1.0,
        )

    def test_ongoing_episode_with_unchanged_severity_publishes_nothing(self) -> None:
        episode = _episode("eg-buero", severity=1)
        row = OpenEpisodeRow(id=7, subject="eg-buero", severity=1)
        plan = self._plan([episode], open_rows=[row])
        assert plan.inserts == ()
        assert plan.updates == ((7, episode),)
        assert plan.publishes == ()

    def test_escalation_publishes_the_new_severity(self) -> None:
        episode = _episode("eg-buero", severity=2)
        row = OpenEpisodeRow(id=7, subject="eg-buero", severity=1)
        plan = self._plan([episode], open_rows=[row])
        (publish,) = plan.publishes
        assert publish.severity == 2

    def test_stored_severity_is_never_lowered(self) -> None:
        episode = _episode("eg-buero", severity=1)
        row = OpenEpisodeRow(id=7, subject="eg-buero", severity=2)
        plan = self._plan([episode], open_rows=[row])
        assert plan.publishes == ()

    def test_recovery_publishes_zero_and_reconciles_the_ended_episode(self) -> None:
        episode = _episode("eg-buero", severity=1, ended=True)
        row = OpenEpisodeRow(id=7, subject="eg-buero", severity=1)
        plan = self._plan([episode], open_rows=[row])
        assert plan.updates == ((7, episode),)
        (publish,) = plan.publishes
        assert publish.severity == 0
        assert publish.slug == "eg-buero"

    def test_open_row_without_computed_counterpart_is_closed_at_frontier(self) -> None:
        row = OpenEpisodeRow(id=9, subject="eg-buero", severity=1)
        plan = self._plan([], open_rows=[row])
        assert plan.orphan_closes == ((9, _FRONTIER),)
        (publish,) = plan.publishes
        assert publish.severity == 0

    def test_room_outliving_the_window_stays_open(self) -> None:
        row = OpenEpisodeRow(id=9, subject="eg-buero", severity=2)
        plan = self._plan([], open_rows=[row], dataless=frozenset({"eg-buero"}))
        assert plan.orphan_closes == ()
        assert plan.stale_opens == ("eg-buero",)
        assert plan.publishes == ()

    def test_historical_ended_episode_without_open_row_is_ignored(self) -> None:
        plan = self._plan([_episode("eg-buero", severity=1, ended=True)], open_rows=[])
        assert plan.inserts == ()
        assert plan.updates == ()
        assert plan.publishes == ()


def test_three_cold_hours_become_one_episode_with_few_events() -> None:
    # The acceptance case end to end: 1.5 K under setpoint at 80 % valve for
    # 3 h is one episode with at most three notification events; the same
    # gap at 20 % valve yields nothing at all.
    by_ga = _series(
        values=dict.fromkeys(range(4), 20.5),
        references={0: 22.0},
        gates={0: 80.0},
    )
    buckets = room_series(_BUERO, by_ga, _T0, _T0 + 3 * _HOUR)
    cold = cold_buckets(_BUERO, buckets, gate_min=50.0)
    observations = deviation_observations(_BUERO, cold, min_hours=2.0)
    episodes = fold_observations("fbh_cold", observations, [], EpisodePolicy(), _T0 + 3 * _HOUR)
    assert len(episodes) == 1
    [episode] = episodes
    assert episode.subject == "eg-buero"
    assert episode.started_at == _T0 + _HOUR
    assert episode.last_seen_at == _T0 + 3 * _HOUR
    assert len(episode.events) <= 3

    closed_valve = _series(
        values=dict.fromkeys(range(4), 20.5),
        references={0: 22.0},
        gates={0: 20.0},
    )
    quiet = cold_buckets(
        _BUERO, room_series(_BUERO, closed_valve, _T0, _T0 + 3 * _HOUR), gate_min=50.0
    )
    assert deviation_observations(_BUERO, quiet, min_hours=2.0) == []
