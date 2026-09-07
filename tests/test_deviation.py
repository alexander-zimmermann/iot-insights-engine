"""Deviation-measurement tests — the `deviation` kind against invented series.

Both shapes of the kind. For rooms, every test feeds invented hourly values
for a room's channel triple (value, reference, gate); for the daily yield,
invented days of kWh and an invented forecast curve. Each asserts only what
comes out: the series, which buckets or days count, observations with their
scores, the current state, the published payload, or a resolution error
naming the room. The shared reconciliation is `test_reconcile`'s; no
cluster, no live database.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from iot_insights_engine.deviation import (
    DAY,
    PLANT,
    YIELD_POLICY,
    Room,
    RoomBucket,
    RoomPublish,
    RoomState,
    YieldDay,
    YieldState,
    classify,
    classify_yield,
    cold_buckets,
    daily_energy,
    daily_yield,
    dead_value_gas,
    deviation_observations,
    judged_days,
    payload_yield,
    publish_for,
    publish_for_plant,
    resolve_rooms,
    room_series,
    yield_observations,
)
from iot_insights_engine.episodes import EpisodePolicy, fold_observations
from iot_insights_engine.faults import DeviationExpectation, Roles, RoomRule
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
    reference="%.FBH.Soll-Temperatur-Status",
    gate="%.FBH.Stellwert-Status",
)
_RULES = (
    RoomRule(match="EG.Büro", min_gap_k=1.0, value="Sensorik.EG.Büro.Sensor.Temperatur"),
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
        rule = RoomRule(
            match="EG.Gäste-WC", min_gap_k=1.0, value="Sensorik.EG.Gäste-WC.Sensor.Temperatur"
        )
        [room] = resolve_rooms(wc_channels, (rule,), _ROLES)
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
        roles = Roles(reference="%.FBH.Soll-Temperatur-Status")
        rooms = resolve_rooms([_BUERO_SENSOR, _BUERO_SOLL], _RULES[:1], roles)
        assert rooms[0].gate_ga is None

    def test_channel_claimed_by_two_rooms_is_an_error(self) -> None:
        # "EG.Flur" and a leftover "Flur" rule both contain the hall's
        # channels — a merge artifact that must fail, not double-measure.
        stray = RoomRule(match="Flur", min_gap_k=2.0, value="Sensorik.EG.Flur.BWM.%.Temperatur")
        with pytest.raises(ValueError, match=r"8/1/11.*'EG\.Flur'.*'Flur'"):
            resolve_rooms(_ALL_CHANNELS, (*_RULES, stray), _ROLES)

    def test_colliding_slugs_are_an_error(self) -> None:
        # The slug is episode subject and NATS entity: two labels that
        # transliterate to one slug would silently merge two rooms.
        rules = (
            _RULES[0],
            RoomRule(
                match="EG.Buero", min_gap_k=1.0, value="Sensorik.EG.Büro.Sensor.Temperatur"
            ),
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


def _cold_state(room: Room) -> RoomState:
    return RoomState(
        room=room, cold_since=_T0, gap=1.8, value=20.2, reference=22.0, gate=85.0
    )


class TestPublishFor:
    def test_the_payload_names_the_room_and_its_numbers(self) -> None:
        publish = publish_for("eg-buero", 1, _cold_state(_BUERO))
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
        assert publish.subject == "eg-buero"
        assert publish.entity == "eg-buero"

    def test_a_room_that_left_the_scope_still_gets_its_clear(self) -> None:
        publish = publish_for("eg-buero", 0, None)
        assert publish == RoomPublish(slug="eg-buero", severity=0, room="eg-buero")


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


# --- the daily-yield shape -------------------------------------------------
#
# Days of invented kWh and an invented forecast curve, never a database: a
# known bad day must fire, and the cloudy days around it must not.

_DAY0 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
_MIN_SHORTFALL = 35.0
_FORECAST = DeviationExpectation.FORECAST_SOLAR


def _day(n: int) -> datetime:
    return _DAY0 + n * DAY


def _days(pairs: dict[int, tuple[float, float]]) -> list[YieldDay]:
    """One `YieldDay` per (actual, expected) pair, keyed by day offset."""
    return [
        YieldDay(day=_day(n), actual_kwh=actual, expected_kwh=expected)
        for n, (actual, expected) in sorted(pairs.items())
    ]


def _curve(*points: tuple[datetime, float]) -> list[tuple[datetime, float]]:
    return list(points)


def _hourly_day(n: int, watts: dict[int, float]) -> list[tuple[datetime, float]]:
    """A day's forecast curve as hourly samples, hour → watts."""
    return [(_day(n) + hour * _HOUR, value) for hour, value in sorted(watts.items())]


class TestDailyEnergy:
    def test_an_hourly_curve_integrates_to_kwh_per_day(self) -> None:
        # 0 → 2000 → 2000 → 0 W over four hourly samples: two trapezoids of
        # 1 kWh each plus the flat 2 kWh hour between them.
        energy = daily_energy(_hourly_day(0, {6: 0.0, 7: 2000.0, 8: 2000.0, 9: 0.0}))
        assert energy == {_day(0): 4.0}

    def test_the_night_between_two_days_is_not_integrated_across(self) -> None:
        # Sunset on one day and sunrise on the next are not one interval:
        # bridging them would invent a whole night of production.
        curve = _hourly_day(0, {8: 0.0, 9: 2000.0}) + _hourly_day(1, {8: 0.0, 9: 2000.0})
        assert daily_energy(curve) == {_day(0): 1.0, _day(1): 1.0}

    def test_a_hole_in_the_curve_lowers_the_expectation(self) -> None:
        # A missing midday sample drops its intervals rather than drawing a
        # straight line through the peak — expectations only fall.
        whole = _hourly_day(0, {6: 0.0, 7: 2000.0, 8: 2000.0, 9: 0.0})
        holed = [point for point in whole if point[0] != _day(0) + 7 * _HOUR]
        assert daily_energy(holed)[_day(0)] < daily_energy(whole)[_day(0)]

    def test_a_session_in_another_timezone_still_buckets_utc_days(self) -> None:
        # psycopg renders timestamptz in the session timezone; the day a
        # sample belongs to must not move with it.
        berlin = timezone(timedelta(hours=2))
        curve = [
            (point.astimezone(berlin), watts)
            for point, watts in _hourly_day(0, {6: 0.0, 7: 2000.0})
        ]
        assert daily_energy(curve) == {_day(0): 1.0}

    def test_an_empty_curve_expects_nothing(self) -> None:
        assert daily_energy(_curve()) == {}


class TestDailyYield:
    def test_a_day_is_measured_from_the_previous_days_close(self) -> None:
        # An inverter powers down at night, so its first reading of the day
        # already includes the first daylight hour; only the previous day's
        # closing counter puts that hour back in the right day.
        counters = {1: {_day(0): 1_000.0, _day(1): 31_000.0, _day(2): 55_000.0}}
        assert daily_yield(counters) == {_day(1): 30.0, _day(2): 24.0}

    def test_the_inverters_sum(self) -> None:
        counters = {
            1: {_day(0): 0.0, _day(1): 20_000.0},
            2: {_day(0): 5_000.0, _day(1): 17_000.0},
        }
        assert daily_yield(counters) == {_day(1): 32.0}

    def test_a_day_whose_predecessor_is_missing_is_left_out(self) -> None:
        # Crediting the gap to the day that followed it would invent a
        # record day and hide the outage.
        counters = {1: {_day(0): 1_000.0, _day(2): 61_000.0, _day(3): 81_000.0}}
        assert daily_yield(counters) == {_day(3): 20.0}

    def test_a_counter_reset_floors_at_zero(self) -> None:
        counters = {1: {_day(0): 900_000.0, _day(1): 2_000.0, _day(2): 22_000.0}}
        assert daily_yield(counters) == {_day(1): 0.0, _day(2): 20.0}

    def test_one_inverter_missing_a_day_does_not_drop_the_other(self) -> None:
        counters = {
            1: {_day(0): 0.0, _day(1): 20_000.0},
            2: {_day(1): 5_000.0},
        }
        assert daily_yield(counters) == {_day(1): 20.0}


class TestJudgedDays:
    def test_only_days_both_sides_know_and_that_expected_enough(self) -> None:
        actual = {_day(0): 30.0, _day(1): 5.0, _day(2): 40.0}
        expected = {_day(0): 40.0, _day(1): 1.0, _day(3): 40.0}
        judged = judged_days(
            actual, expected, min_expected_kwh=3.0, frontier=_day(3)
        )
        # Day 1 expected too little to score, day 2 has no forecast and day 3
        # no production.
        assert [d.day for d in judged] == [_day(0)]

    def test_a_day_past_the_frontier_is_not_judged_yet(self) -> None:
        actual = {_day(0): 30.0, _day(1): 10.0}
        expected = {_day(0): 40.0, _day(1): 40.0}
        judged = judged_days(actual, expected, min_expected_kwh=3.0, frontier=_day(0))
        assert [d.day for d in judged] == [_day(0)]

    def test_the_shortfall_is_a_percentage_of_the_expectation(self) -> None:
        [day] = judged_days(
            {_day(0): 26.0}, {_day(0): 40.0}, min_expected_kwh=3.0, frontier=_day(0)
        )
        assert day.shortfall_pct == 35.0


class TestYieldObservations:
    def test_a_known_bad_day_fires_and_cloudy_days_stay_quiet(self) -> None:
        # The acceptance case: the forecast is already weather-adjusted, so
        # a cloudy day that met its lowered expectation is not a fault, and
        # a day that made a third of it is.
        days = _days({0: (7.0, 8.0), 1: (12.0, 40.0), 2: (30.0, 34.0)})
        [observation] = yield_observations(days, _MIN_SHORTFALL)
        assert observation.subject == PLANT
        assert observation.time == _day(1)
        assert observation.value == pytest.approx(70.0)
        # The score is the shortfall in units of the declared minimum.
        assert observation.score == pytest.approx(2.0)

    def test_exactly_the_declared_shortfall_already_counts(self) -> None:
        [observation] = yield_observations(_days({0: (26.0, 40.0)}), _MIN_SHORTFALL)
        assert observation.score == pytest.approx(1.0)

    def test_a_day_over_its_expectation_is_never_a_fault(self) -> None:
        assert yield_observations(_days({0: (48.0, 40.0)}), _MIN_SHORTFALL) == []


class TestClassifyYield:
    def test_a_short_stretch_reaching_the_frontier(self) -> None:
        days = _days({0: (38.0, 40.0), 1: (10.0, 40.0), 2: (12.0, 40.0)})
        state = classify_yield(days, _MIN_SHORTFALL, _FORECAST, frontier=_day(2))
        assert state.short_since == _day(1)
        assert state.day == _day(2)
        assert state.actual_kwh == 12.0
        assert state.expected_kwh == 40.0
        assert state.shortfall_pct == pytest.approx(70.0)
        assert state.expectation is _FORECAST
        assert state.min_shortfall_pct == _MIN_SHORTFALL

    def test_a_plant_that_recovered_still_reports_its_last_day(self) -> None:
        # The episode stays open for a few quiet days; a severity published
        # in that window must still say what the last measured day did.
        days = _days({0: (10.0, 40.0), 1: (39.0, 40.0)})
        state = classify_yield(days, _MIN_SHORTFALL, _FORECAST, frontier=_day(1))
        assert state.short_since is None
        assert state.day == _day(1)
        assert state.actual_kwh == 39.0
        assert state.shortfall_pct == pytest.approx(2.5)

    def test_a_window_with_nothing_judged_reports_no_day(self) -> None:
        state = classify_yield([], _MIN_SHORTFALL, _FORECAST, frontier=_day(1))
        assert state == YieldState(
            expectation=_FORECAST, min_shortfall_pct=_MIN_SHORTFALL
        )


class TestPublishForPlant:
    def test_the_payload_names_the_day_and_what_it_was_measured_against(self) -> None:
        state = classify_yield(
            _days({0: (12.0, 40.0)}), _MIN_SHORTFALL, _FORECAST, frontier=_day(0)
        )
        publish = publish_for_plant(PLANT, 2, state)
        assert publish.subject == PLANT
        # One plant, one declared address: no entity token on the bus.
        assert publish.entity is None
        assert payload_yield(publish) == {
            "expectation": "forecast_solar",
            "day": _day(0),
            "actual_kwh": 12.0,
            "expected_kwh": 40.0,
            "shortfall_pct": pytest.approx(70.0),
            "min_shortfall_pct": _MIN_SHORTFALL,
            "short_since": _day(0),
        }

    def test_a_publish_without_a_measured_state_is_a_wiring_error(self) -> None:
        # The plant is the whole scope and is always measured, so this can
        # only mean the kind was wired to the wrong measurement.
        with pytest.raises(ValueError, match="without a measured state"):
            publish_for_plant(PLANT, 0, None)


def test_three_short_days_become_one_episode_with_few_events() -> None:
    # The acceptance case end to end, in the cadence the fault folds in: a
    # dead string over three days is one incident, not three.
    days = _days({0: (12.0, 40.0), 1: (11.0, 40.0), 2: (13.0, 40.0)})
    observations = yield_observations(days, _MIN_SHORTFALL)
    [episode] = fold_observations(
        "pv_underperformance", observations, [], YIELD_POLICY, _day(2)
    )
    assert episode.subject == PLANT
    assert episode.started_at == _day(0)
    assert episode.last_seen_at == _day(2)
    assert episode.ended_at is None
    assert len(episode.events) <= 3


def test_one_good_day_between_two_bad_ones_stays_one_episode() -> None:
    # Weather flickers; a fault does not become two incidents because of it.
    days = _days({0: (12.0, 40.0), 1: (39.0, 40.0), 2: (13.0, 40.0)})
    observations = yield_observations(days, _MIN_SHORTFALL)
    [episode] = fold_observations(
        "pv_underperformance", observations, [], YIELD_POLICY, _day(2)
    )
    assert episode.started_at == _day(0)
    assert episode.last_seen_at == _day(2)


def test_a_recovered_plant_ends_its_episode_after_the_quiet_days() -> None:
    days = _days({0: (12.0, 40.0), 1: (39.0, 40.0), 2: (40.0, 40.0), 3: (39.0, 40.0)})
    observations = yield_observations(days, _MIN_SHORTFALL)
    [episode] = fold_observations(
        "pv_underperformance", observations, [], YIELD_POLICY, _day(3)
    )
    assert episode.ended_at == _day(2)
