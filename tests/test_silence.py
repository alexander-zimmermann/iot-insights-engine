"""Silence-measurement tests — seam 2 of the detection rebuild.

Each test feeds an invented bucket series (the shape `knx_1h` hands the
measurement) into the real computation and asserts only what comes out:
a channel state, and per-bucket observations for the episode pipeline.
The spec fixture: "sent hourly until 10:00, then nothing" must yield
"silent since 10:00"; a never-sent channel must yield nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from iot_insights_engine.silence import (
    BUCKET,
    Channel,
    ChannelState,
    ChannelStats,
    classify,
    drop_unmeasurable,
    median_pause,
    silence_observations,
)

_T0 = datetime(2026, 8, 30, 0, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)
_FREEZER = Channel(ga="2/2/227", name="Schalten.Gefrierschrank.Stromwert", dpt="9.021")


def _at(hours: float) -> datetime:
    return _T0 + hours * _HOUR

def _hourly_until(last_hour: int) -> list[datetime]:
    return [_at(h) for h in range(last_hour + 1)]


# ---------------------------------------------------------------- median_pause

def test_median_pause_of_an_hourly_sender_is_one_bucket() -> None:
    assert median_pause(_hourly_until(10)) == _HOUR


def test_median_pause_ignores_one_outage_in_a_regular_series() -> None:
    # Hourly sends with one 8-hour hole: the median gap stays 1 hour.
    buckets = [_at(h) for h in [0, 1, 2, 3, 4, 12, 13, 14, 15]]
    assert median_pause(buckets) == _HOUR


def test_median_pause_of_a_daily_sender_is_a_day() -> None:
    buckets = [_at(24 * d) for d in range(10)]
    assert median_pause(buckets) == timedelta(days=1)


def test_median_pause_needs_two_buckets() -> None:
    assert median_pause([_at(0)]) is None
    assert median_pause([]) is None


# -------------------------------------------------------------------- classify

def test_hourly_sender_gone_quiet_is_silent_since_its_last_bucket() -> None:
    # The spec fixture: sent hourly until 10:00, then nothing.
    state = classify(_FREEZER, _hourly_until(10), frontier=_at(16), gap_factor=5.0)
    assert state.state is ChannelState.SILENT
    assert state.silent_since == _at(10)
    assert state.pause == _HOUR


def test_hourly_sender_within_five_pauses_is_alive() -> None:
    state = classify(_FREEZER, _hourly_until(10), frontier=_at(15), gap_factor=5.0)
    assert state.state is ChannelState.ALIVE
    assert state.silent_since is None


def test_never_sent_channel_is_excluded_not_silent() -> None:
    state = classify(_FREEZER, [], frontier=_at(16), gap_factor=5.0)
    assert state.state is ChannelState.NEVER_SENT


def test_daily_sender_mid_pause_is_alive() -> None:
    buckets = [_at(24 * d) for d in range(10)]
    state = classify(_FREEZER, buckets, frontier=buckets[-1] + 30 * _HOUR, gap_factor=5.0)
    assert state.state is ChannelState.ALIVE


def test_daily_sender_past_five_of_its_own_pauses_is_silent() -> None:
    buckets = [_at(24 * d) for d in range(10)]
    state = classify(_FREEZER, buckets, frontier=buckets[-1] + 121 * _HOUR, gap_factor=5.0)
    assert state.state is ChannelState.SILENT
    assert state.silent_since == buckets[-1]


def test_single_bucket_channel_has_no_measurable_pause() -> None:
    # One send ever: "its own normal pause" does not exist, so silence is
    # not decidable — the channel stays alive and produces nothing.
    state = classify(_FREEZER, [_at(0)], frontier=_at(500), gap_factor=5.0)
    assert state.state is ChannelState.ALIVE
    assert state.pause is None


# ------------------------------------------------------- silence_observations

def test_silent_channel_yields_one_observation_per_silent_bucket() -> None:
    obs = silence_observations(
        "2/2/227", _hourly_until(10), pause=_HOUR, gap_factor=5.0, frontier=_at(18)
    )
    assert [o.time for o in obs] == [_at(16), _at(17), _at(18)]
    # Score is the gap in units of the channel's own pause.
    assert [o.score for o in obs] == [6.0, 7.0, 8.0]
    assert all(o.subject == "2/2/227" for o in obs)


def test_alive_channel_yields_no_observations() -> None:
    assert (
        silence_observations(
            "2/2/227", _hourly_until(10), pause=_HOUR, gap_factor=5.0, frontier=_at(15)
        )
        == []
    )


def test_recovered_channel_stops_producing_observations() -> None:
    # Quiet from 10:00, back at 22:00: observations exist only inside the gap.
    buckets = [*_hourly_until(10), _at(22)]
    obs = silence_observations(
        "2/2/227", buckets, pause=_HOUR, gap_factor=5.0, frontier=_at(23)
    )
    assert [o.time for o in obs] == [_at(h) for h in range(16, 22)]
    assert obs[-1].score == 11.0


def test_observation_value_carries_the_gap_in_hours() -> None:
    obs = silence_observations(
        "2/2/227", _hourly_until(10), pause=_HOUR, gap_factor=5.0, frontier=_at(16)
    )
    assert [o.value for o in obs] == [6.0]


def test_daily_sender_observations_score_against_its_own_pause() -> None:
    day = timedelta(days=1)
    buckets = [_at(24 * d) for d in range(3)]
    obs = silence_observations(
        "2/2/227", buckets, pause=day, gap_factor=5.0, frontier=buckets[-1] + 122 * _HOUR
    )
    # Silent buckets start strictly past 5 × 24 h after the last send.
    assert obs[0].time == buckets[-1] + 121 * _HOUR
    assert obs[0].score == 121 / 24
    assert obs[-1].time == buckets[-1] + 122 * _HOUR


# ------------------------------------------------------------ scope filtering

def _stats(ga: str, buckets: int, floor: float, ceil: float) -> ChannelStats:
    return ChannelStats(
        ga=ga, buckets=buckets, last_bucket=_at(10), floor_value=floor, ceil_value=ceil
    )


def test_constant_zero_channel_is_dropped_as_dead() -> None:
    # 15/1/22 read exactly 0.0 for 717 hourly buckets and looks healthy to
    # every measurement that does not ask.
    dead_stats = _stats("15/1/22", buckets=717, floor=0.0, ceil=0.0)
    voltage = Channel(ga="15/1/22", name="Versorgung.Netz.Spannung-L2", dpt="14.027")
    kept, drops = drop_unmeasurable([voltage], {"15/1/22": dead_stats})
    assert kept == []
    assert drops[ChannelState.DEAD] == [voltage]


def test_constant_nonzero_channel_is_kept() -> None:
    stats = _stats(_FREEZER.ga, buckets=717, floor=1.0, ceil=1.0)
    kept, drops = drop_unmeasurable([_FREEZER], {_FREEZER.ga: stats})
    assert kept == [_FREEZER]
    assert drops[ChannelState.DEAD] == []


def test_briefly_constant_zero_channel_is_kept() -> None:
    # A handful of zero samples is thin evidence, not a dead register.
    stats = _stats(_FREEZER.ga, buckets=5, floor=0.0, ceil=0.0)
    kept, _ = drop_unmeasurable([_FREEZER], {_FREEZER.ga: stats})
    assert kept == [_FREEZER]


def test_channel_without_data_is_dropped_as_never_sent() -> None:
    kept, drops = drop_unmeasurable([_FREEZER], {})
    assert kept == []
    assert drops[ChannelState.NEVER_SENT] == [_FREEZER]


def test_channel_main_group() -> None:
    assert _FREEZER.main_group == 2
    assert BUCKET == _HOUR
