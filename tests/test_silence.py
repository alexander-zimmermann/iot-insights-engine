"""Silence-measurement tests — seam 2 of the detection rebuild.

Each test feeds an invented bucket series (the shape `knx_1h` hands the
measurement) into the real computation and asserts only what comes out:
a channel state, per-bucket observations for the episode pipeline, and —
for the gap walk — what the run would hold open.
The spec fixture: "sent hourly until 10:00, then nothing" must yield
"silent since 10:00"; a never-sent channel must yield nothing.

Series are written at the scale the fault runs at — 30 days of hourly
buckets — wherever the assertion is about the *pause estimator*: a
quantile over nine gaps says nothing about one over seven hundred.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

from iot_insights_engine.episode_store import OpenEpisodeRow
from iot_insights_engine.episodes import EpisodePolicy
from iot_insights_engine.faults import Fault, MeasurementKind, Target
from iot_insights_engine.groups import GroupPublish
from iot_insights_engine.reconcile import Measured, Plan, Window
from iot_insights_engine.silence import (
    BUCKET,
    RULE_REVISION,
    Channel,
    ChannelReport,
    ChannelState,
    ChannelStats,
    SilenceState,
    classify,
    drop_unmeasurable,
    fingerprint,
    gap_walk,
    normal_pause,
    plan_run,
    silence_observations,
)

_T0 = datetime(2026, 8, 30, 0, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)
_FREEZER = Channel(ga="2/2/227", name="Schalten.Gefrierschrank.Stromwert", dpt="9.021")
_HALLWAY = Channel(ga="9/1/7", name="Bewegungsmelder.KG.Flur.KNX.Decke.Präsenz", dpt="1.001")


def _at(hours: float) -> datetime:
    return _T0 + hours * _HOUR

def _hourly_until(last_hour: int) -> list[datetime]:
    return [_at(h) for h in range(last_hour + 1)]


def _cyclic(days: int = 30) -> list[datetime]:
    """A sensor that sends every hour, for the whole window."""
    return [_at(h) for h in range(24 * days)]


def _bursty(days: int = 30, *, events: int = 5, first_hour: int = 18) -> list[datetime]:
    """An event-driven channel's day: a handful of telegrams in the evening
    (a light switched a few times), then nothing until the next evening —
    the shape whose median gap is one hour and whose real return interval
    is a day.
    """
    return [_at(24 * d + first_hour + e) for d in range(days) for e in range(events)]


# ---------------------------------------------------------------- normal_pause

def test_pause_of_an_hourly_sender_is_one_bucket_at_every_quantile() -> None:
    # The channels the fault exists for are cyclic: for them the quantile
    # is a no-op, so raising it costs no sensitivity.
    assert normal_pause(_cyclic(), 0.5) == _HOUR
    assert normal_pause(_cyclic(), 0.9) == _HOUR


def test_pause_of_an_hourly_sender_survives_a_single_outage() -> None:
    # One 8-hour hole in 30 days is 0.1 % of the gaps — far below the tail
    # the quantile reads, so the estimator does not widen its own threshold
    # every time the channel hiccups.
    buckets = [b for b in _cyclic() if not _at(100) < b < _at(108)]
    assert normal_pause(buckets, 0.9) == _HOUR


def test_pause_of_a_burst_sender_is_the_gap_between_bursts_not_inside_them() -> None:
    # The measurement error #1678 exists for: five telegrams in an evening
    # give a median gap of one hour, so every quiet day trips the threshold.
    # p90 reads the gap that actually ends the quiet phase.
    assert normal_pause(_bursty(), 0.5) == _HOUR
    assert normal_pause(_bursty(), 0.9) == timedelta(hours=20)


def test_pause_of_a_daily_sender_is_a_day_at_every_quantile() -> None:
    buckets = [_at(24 * d) for d in range(30)]
    assert normal_pause(buckets, 0.5) == timedelta(days=1)
    assert normal_pause(buckets, 0.9) == timedelta(days=1)


def test_pause_is_always_a_gap_the_channel_actually_showed() -> None:
    # Nearest rank, no interpolation: a threshold has to be defensible as
    # "it went this long once", not as an average of two unlike gaps.
    buckets = [_at(0), _at(1), _at(2), _at(3), _at(13)]
    assert normal_pause(buckets, 0.9) == timedelta(hours=10)


def test_pause_needs_two_buckets() -> None:
    assert normal_pause([_at(0)], 0.9) is None
    assert normal_pause([], 0.9) is None


# -------------------------------------------------------------------- classify

def test_hourly_sender_gone_quiet_is_silent_since_its_last_bucket() -> None:
    # The spec fixture: sent hourly until 10:00, then nothing.
    state = classify(
        _FREEZER,
        _hourly_until(10),
        start=_at(0),
        frontier=_at(16),
        gap_factor=5.0,
        gap_quantile=0.9,
    )
    assert state.state is ChannelState.SILENT
    assert state.silent_since == _at(10)
    assert state.pause == _HOUR


def test_hourly_sender_within_five_pauses_is_alive() -> None:
    state = classify(
        _FREEZER,
        _hourly_until(10),
        start=_at(0),
        frontier=_at(15),
        gap_factor=5.0,
        gap_quantile=0.9,
    )
    assert state.state is ChannelState.ALIVE
    assert state.silent_since is None


def test_never_sent_channel_is_excluded_not_silent() -> None:
    state = classify(_FREEZER, [], start=_at(0), frontier=_at(16), gap_factor=5.0, gap_quantile=0.9)
    assert state.state is ChannelState.NEVER_SENT


def test_daily_sender_mid_pause_is_alive() -> None:
    buckets = [_at(24 * d) for d in range(10)]
    state = classify(
        _FREEZER,
        buckets,
        start=_at(0),
        frontier=buckets[-1] + 30 * _HOUR,
        gap_factor=5.0,
        gap_quantile=0.9,
    )
    assert state.state is ChannelState.ALIVE


def test_daily_sender_past_five_of_its_own_pauses_is_silent() -> None:
    buckets = [_at(24 * d) for d in range(10)]
    state = classify(
        _FREEZER,
        buckets,
        start=_at(0),
        frontier=buckets[-1] + 121 * _HOUR,
        gap_factor=5.0,
        gap_quantile=0.9,
    )
    assert state.state is ChannelState.SILENT
    assert state.silent_since == buckets[-1]


def test_burst_sender_idle_for_a_day_is_alive_where_the_median_called_it_silent() -> None:
    # One skipped evening: at the median pause (1 h) the threshold is 5 h and
    # the channel is silent every single night; at p90 it is what it is —
    # a light nobody touched today.
    buckets = _bursty()
    frontier = buckets[-1] + 30 * _HOUR
    assert (
        classify(
            _FREEZER, buckets, start=_at(0), frontier=frontier, gap_factor=5.0, gap_quantile=0.5
        ).state
        is ChannelState.SILENT
    )
    assert (
        classify(
            _FREEZER, buckets, start=_at(0), frontier=frontier, gap_factor=5.0, gap_quantile=0.9
        ).state
        is ChannelState.ALIVE
    )


def test_burst_sender_quiet_for_five_of_its_own_days_is_still_silent() -> None:
    # The finds the fault exists for survive the quantile: a switch actuator
    # nobody has reached in a working week is a fault, not a quiet evening.
    buckets = _bursty()
    state = classify(
        _FREEZER,
        buckets,
        start=_at(0),
        frontier=buckets[-1] + 101 * _HOUR,
        gap_factor=5.0,
        gap_quantile=0.9,
    )
    assert state.state is ChannelState.SILENT
    assert state.pause == timedelta(hours=20)


def test_single_bucket_channel_has_no_measurable_pause() -> None:
    # One send ever: "its own normal pause" does not exist, so silence is
    # not decidable — the channel stays alive and produces nothing.
    state = classify(
        _FREEZER, [_at(0)], start=_at(0), frontier=_at(500), gap_factor=5.0, gap_quantile=0.9
    )
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


# ---------------------------------------------------------------- unproven

# The window opens at _at(0); this is the hour its last day begins.
_LAST_DAY = 24 * 29


def _newcomer(hours: tuple[int, ...] = (15, 16, 17, 18)) -> list[datetime]:
    """A motion detector the coupler let through on the last afternoon of
    the window: a few consecutive buckets, nothing before them."""
    return [_at(_LAST_DAY + h) for h in hours]


def test_channel_first_seen_in_the_window_is_unproven_in_its_first_night() -> None:
    # Four consecutive buckets: every gap it ever showed is an hour, so the
    # first quiet night would read as eight of its own pauses.
    state = classify(
        _HALLWAY,
        _newcomer(),
        start=_at(0),
        frontier=_at(_LAST_DAY + 26),
        gap_factor=5.0,
        gap_quantile=0.95,
    )
    assert state.state is ChannelState.UNPROVEN
    assert state.pause == _HOUR
    assert state.silent_since is None


def test_newcomer_is_proven_once_the_quantile_is_more_than_its_largest_gap() -> None:
    # Two days on: twenty gaps, two nights among them — the p95 is now the
    # shorter night rather than the longest gap, and the next one is well
    # within five of it.
    buckets = _newcomer() + [_at(_LAST_DAY + 24 + h) for h in range(5, 21)] + [_at(_LAST_DAY + 53)]
    assert len(buckets) - 1 == 20
    state = classify(
        _HALLWAY,
        buckets,
        start=_at(0),
        frontier=_at(_LAST_DAY + 61),
        gap_factor=5.0,
        gap_quantile=0.95,
    )
    assert state.state is ChannelState.ALIVE
    assert state.pause == 9 * _HOUR


def test_nineteen_gaps_leave_a_newcomer_unproven_at_p95() -> None:
    # One short of twenty, and "19 von 20 Fällen" is still its largest gap.
    buckets = _newcomer() + [_at(_LAST_DAY + 24 + h) for h in range(5, 21)]
    assert len(buckets) - 1 == 19
    state = classify(
        _HALLWAY,
        buckets,
        start=_at(0),
        frontier=_at(_LAST_DAY + 24 + 20 + 60),
        gap_factor=5.0,
        gap_quantile=0.95,
    )
    assert state.state is ChannelState.UNPROVEN


def test_sparse_channel_present_since_the_window_start_is_judged_as_before() -> None:
    # Bordbar: five sends in its first week, then nothing — it was there
    # when the window opened, so its silence counts however thin it is.
    buckets = [_at(h) for h in (5, 59, 84, 118, 121)]
    state = classify(
        _FREEZER,
        buckets,
        start=_at(0),
        frontier=_at(24 * 30),
        gap_factor=5.0,
        gap_quantile=0.95,
    )
    assert state.state is ChannelState.SILENT
    assert state.silent_since == _at(121)


def test_lead_shorter_than_its_own_threshold_does_not_make_a_channel_new() -> None:
    # First seen 18 h into the window, sending daily: the empty lead is far
    # shorter than the five days it would have to be silent — an
    # established channel with a late first bucket, not a newcomer.
    buckets = [_at(18 + 24 * d) for d in range(4)]
    state = classify(
        _FREEZER,
        buckets,
        start=_at(0),
        frontier=buckets[-1] + 121 * _HOUR,
        gap_factor=5.0,
        gap_quantile=0.95,
    )
    assert state.state is ChannelState.SILENT


def test_hourly_newcomer_is_proven_after_a_day() -> None:
    # A cyclic sender that appeared yesterday: twenty hourly gaps prove its
    # pause, and it is silent after five of them like any other.
    buckets = [_at(_LAST_DAY - 20 + h) for h in range(21)]
    state = classify(
        _FREEZER,
        buckets,
        start=_at(0),
        frontier=buckets[-1] + 6 * _HOUR,
        gap_factor=5.0,
        gap_quantile=0.95,
    )
    assert state.state is ChannelState.SILENT


# ---------------------------------------------------------------- gap_walk


# The fingerprint this run measures by, and the one the median estimator
# ran on before the quantile replaced it.
_FINGERPRINT = "silence(gap_factor=5.0, gap_quantile=0.95)"
_OLD_FINGERPRINT = "silence(gap_factor=5.0, gap_quantile=0.5)"
# A second channel in the same scope and main group, never heard from.
_STAIRS = Channel(ga="9/1/9", name="Bewegungsmelder.KG.Treppe.KNX.Decke.Präsenz", dpt="1.001")


def _walk(
    buckets: list[datetime],
    frontier: datetime,
    *,
    open_rows: Sequence[OpenEpisodeRow] = (),
    series: Mapping[str, list[datetime]] | None = None,
) -> tuple[Measured[SilenceState], Window]:
    """The hallway detector with the given series — and the stairs detector
    in scope beside it, without a single bucket."""
    series = {_HALLWAY.ga: buckets} if series is None else dict(series)
    stats_by_ga = {
        ga: ChannelStats(
            ga=ga, buckets=len(b), last_bucket=b[-1], floor_value=0.0, ceil_value=1.0
        )
        for ga, b in series.items()
        if b
    }
    kept = [c for c in (_HALLWAY, _STAIRS) if c.ga in stats_by_ga]
    window = Window(start=_at(0), frontier=frontier, policy=EpisodePolicy())
    measured = gap_walk(
        channels=[_HALLWAY, _STAIRS],
        kept=kept,
        candidates=kept,
        series=series,
        stats_by_ga=stats_by_ga,
        open_rows=open_rows,
        window=window,
        gap_factor=5.0,
        gap_quantile=0.95,
        fingerprint=_FINGERPRINT,
    )
    return measured, window


def _plan(
    measured: Measured[SilenceState], window: Window, *rows: OpenEpisodeRow
) -> Plan[GroupPublish[ChannelReport]]:
    return plan_run(episodes=(), open_rows=list(rows), measured=measured, frontier=window.frontier)


def test_unproven_channel_is_unmeasured_and_yields_no_observations() -> None:
    measured, _ = _walk(_newcomer(), frontier=_at(_LAST_DAY + 26))
    assert measured.states[_HALLWAY.ga].state is ChannelState.UNPROVEN
    assert measured.observations == ()
    assert _HALLWAY.ga in measured.dataless
    assert measured.record["unproven"] == 1


def test_open_episode_this_rule_made_on_an_unproven_subject_is_held_not_closed() -> None:
    # A device that died with a real history slides into `new ∧ thin` once
    # enough of its buckets have left the window: the accusation was this
    # rule's own, and stands until data says otherwise.
    row = OpenEpisodeRow(id=7, subject=_HALLWAY.ga, severity=1, fingerprint=_FINGERPRINT)
    measured, window = _walk(_newcomer(), frontier=_at(_LAST_DAY + 26), open_rows=[row])
    assert measured.stranded == frozenset()
    plan = _plan(measured, window, row)
    assert plan.stale_opens == (_HALLWAY.ga,)
    assert plan.orphan_closes == ()
    assert plan.publishes == ()


def test_open_episode_another_rule_left_on_an_unproven_subject_is_closed() -> None:
    # The median estimator accused a hand-driven socket the quantile never
    # would: a stranded accusation, closed at the frontier — and the group
    # it pinned is published clear, because the row was the group's last.
    row = OpenEpisodeRow(id=7, subject=_HALLWAY.ga, severity=2, fingerprint=_OLD_FINGERPRINT)
    measured, window = _walk(_newcomer(), frontier=_at(_LAST_DAY + 26), open_rows=[row])
    # Still dataless — the channel is unmeasured either way; stranded is
    # what the row is, not what the channel is.
    assert _HALLWAY.ga in measured.dataless
    assert measured.stranded == frozenset({_HALLWAY.ga})
    plan = _plan(measured, window, row)
    assert plan.orphan_closes == ((7, window.frontier),)
    assert plan.stale_opens == ()
    assert plan.publishes == (GroupPublish(main_group=9, severity=0, channels=()),)


def test_open_episode_without_a_fingerprint_on_an_unproven_subject_is_held() -> None:
    # A row older than the stamp: nobody can say which rule made it, so it
    # is held like any other — the bench mark for a hand backfill, never a
    # close by default.
    row = OpenEpisodeRow(id=7, subject=_HALLWAY.ga, severity=1)
    measured, window = _walk(_newcomer(), frontier=_at(_LAST_DAY + 26), open_rows=[row])
    assert measured.stranded == frozenset()
    plan = _plan(measured, window, row)
    assert plan.stale_opens == (_HALLWAY.ga,)
    assert plan.orphan_closes == ()


def test_never_sent_channel_holds_its_row_whatever_rule_left_it() -> None:
    # The window slid past every bucket of a dead device: there is no data
    # to decide a recovery with, and the rule that opened the row is beside
    # the point.
    row = OpenEpisodeRow(id=8, subject=_STAIRS.ga, severity=2, fingerprint=_OLD_FINGERPRINT)
    measured, window = _walk(_hourly_until(10), frontier=_at(18), open_rows=[row])
    assert _STAIRS.ga in measured.dataless
    assert measured.stranded == frozenset()
    plan = _plan(measured, window, row)
    assert plan.stale_opens == (_STAIRS.ga,)
    assert plan.orphan_closes == ()


def test_single_bucket_channel_holds_its_row_whatever_rule_left_it() -> None:
    # One bucket left in the window: no pause to read a gap off, so the
    # channel was not measured, and its row is held like a never-sent one.
    row = OpenEpisodeRow(id=8, subject=_STAIRS.ga, severity=2, fingerprint=_OLD_FINGERPRINT)
    measured, window = _walk(
        _hourly_until(10),
        frontier=_at(18),
        open_rows=[row],
        series={_HALLWAY.ga: _hourly_until(10), _STAIRS.ga: [_at(2)]},
    )
    assert _STAIRS.ga in measured.dataless
    assert measured.stranded == frozenset()
    plan = _plan(measured, window, row)
    assert plan.stale_opens == (_STAIRS.ga,)


def test_run_record_splits_the_stale_opens_by_channel_state_and_names_the_stranded() -> None:
    # What the log shows instead of the bus: which rows this run holds, and
    # why, and which it lets go as another rule's leftovers.
    hallway = OpenEpisodeRow(id=7, subject=_HALLWAY.ga, severity=2, fingerprint=_OLD_FINGERPRINT)
    stairs = OpenEpisodeRow(id=8, subject=_STAIRS.ga, severity=2, fingerprint=_OLD_FINGERPRINT)
    measured, _ = _walk(
        _newcomer(), frontier=_at(_LAST_DAY + 26), open_rows=[hallway, stairs]
    )
    assert measured.record["stale_opens_by_state"] == {
        "unproven": [],
        "never_sent": [_STAIRS.ga],
        "single_bucket": [],
    }
    assert measured.record["stranded"] == [_HALLWAY.ga]


def test_the_kinds_fingerprint_folds_the_code_revision_in() -> None:
    # The unproven guard changed which channels the rule accuses without a
    # line of the fault file moving; the stamp has to move with the code.
    fault = Fault(
        name="channel_silence",
        sentence="ein Kanal schweigt",
        unit="× der üblichen Sendepause",
        kind=MeasurementKind.SILENCE,
        parameters={"gap_factor": 5, "gap_quantile": 0.95},
        target=Target(per_main_group=True),
    )
    assert fingerprint(fault) == f"{fault.fingerprint}@{RULE_REVISION}"
    assert fingerprint(fault).startswith("silence(gap_factor=5.0, gap_quantile=0.95)@")


def test_run_record_lists_a_held_unproven_row_under_unproven() -> None:
    row = OpenEpisodeRow(id=7, subject=_HALLWAY.ga, severity=1, fingerprint=_FINGERPRINT)
    measured, _ = _walk(_newcomer(), frontier=_at(_LAST_DAY + 26), open_rows=[row])
    assert measured.record["stale_opens_by_state"]["unproven"] == [_HALLWAY.ga]
    assert measured.record["stranded"] == []


def test_proven_channel_walks_its_gaps_as_before() -> None:
    measured, _ = _walk(_hourly_until(10), frontier=_at(18))
    assert measured.states[_HALLWAY.ga].state is ChannelState.SILENT
    assert len(measured.observations) == 3
    assert _HALLWAY.ga not in measured.dataless
    assert measured.record["unproven"] == 0
