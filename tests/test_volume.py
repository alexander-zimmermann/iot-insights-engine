"""Notification-volume tests — seam 2 (the measurement over an invented
episode stream) plus the runner's single-subject reconcile.

The fixtures are weeks of incidents, not database rows: a week with ten
episodes must fire and a week with three must stay quiet, and the watchdog's
own episode must count like any other.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from iot_insights_engine.episode_store import OpenEpisodeRow
from iot_insights_engine.episodes import (
    Episode,
    EpisodePolicy,
    EventKind,
    EvidenceRow,
    NotificationEvent,
    fold_observations,
)
from iot_insights_engine.volume import (
    SUBJECT,
    WINDOW,
    EpisodeStart,
    FaultCount,
    VolumeBucket,
    VolumePlan,
    VolumePublish,
    VolumeState,
    classify,
    count_series,
    plan_run,
    volume_observations,
)

_HOUR = timedelta(hours=1)
_FRONTIER = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
_WINDOW_START = _FRONTIER - timedelta(days=14)
_LIMIT = 5.0


def _week(
    count: int, *, ending: datetime = _FRONTIER, fault: str = "channel_silence"
) -> list[EpisodeStart]:
    """`count` incidents spread evenly across the seven days ending at
    `ending` — the synthetic week the acceptance criteria talk about."""
    step = WINDOW / (count + 1)
    return [
        EpisodeStart(time=ending - WINDOW + (i + 1) * step, fault=fault)
        for i in range(count)
    ]


def _series(starts: list[EpisodeStart]) -> list[VolumeBucket]:
    return count_series(starts, _WINDOW_START, _FRONTIER)


def test_a_week_with_ten_episodes_fires() -> None:
    observations = volume_observations(_series(_week(10)), _LIMIT)
    assert observations
    latest = observations[-1]
    assert latest.time == _FRONTIER
    assert latest.value == 10.0
    # The score is the count in units of the declared limit.
    assert latest.score == 2.0


def test_a_week_with_three_episodes_stays_quiet() -> None:
    assert volume_observations(_series(_week(3)), _LIMIT) == []


def test_exactly_the_limit_is_not_yet_a_fault() -> None:
    # "More than N": five a week is the target, not the fault.
    assert volume_observations(_series(_week(5)), _LIMIT) == []


def test_the_window_rolls_so_old_incidents_stop_counting() -> None:
    # Ten incidents, all of them more than a week before the frontier: the
    # watchdog fired back then and is quiet now.
    buckets = _series(_week(10, ending=_FRONTIER - WINDOW))
    observations = volume_observations(buckets, _LIMIT)
    assert max(b.episodes for b in buckets) == 10
    assert buckets[-1].episodes == 0
    assert observations
    assert observations[-1].time < _FRONTIER


def test_the_watchdogs_own_episodes_count_normally() -> None:
    # Its own incident is an incident: no fault is exempt from the tally,
    # which is what keeps the count honest about what reaches a human.
    starts = [*_week(5), EpisodeStart(time=_FRONTIER - _HOUR, fault="notification_volume")]
    buckets = _series(starts)
    assert buckets[-1].episodes == 6
    assert volume_observations(buckets, _LIMIT)


def test_a_noisy_week_is_one_incident_with_at_most_three_events() -> None:
    observations = volume_observations(_series(_week(10)), _LIMIT)
    (episode,) = fold_observations(
        "notification_volume", observations, [], EpisodePolicy(), _FRONTIER
    )
    assert episode.subject == SUBJECT
    assert episode.ended_at is None
    assert len(episode.events) <= 3
    assert episode.events[0].kind is EventKind.APPEARED


def test_state_names_the_count_the_faults_and_since_when() -> None:
    starts = [*_week(6, fault="channel_silence"), *_week(2, fault="fbh_cold")]
    state = classify(starts, _series(starts), _LIMIT, _FRONTIER)
    assert state.episodes == 8
    assert state.limit == _LIMIT
    assert state.over_since is not None
    # Loudest fault first — the payload says where the noise comes from.
    assert state.by_fault == (
        FaultCount(fault="channel_silence", episodes=6),
        FaultCount(fault="fbh_cold", episodes=2),
    )


def test_state_below_the_limit_has_no_start() -> None:
    starts = _week(3)
    state = classify(starts, _series(starts), _LIMIT, _FRONTIER)
    assert state.episodes == 3
    assert state.over_since is None


def test_state_after_recovery_has_no_start() -> None:
    # Over the limit earlier in the window, back under it at the frontier.
    starts = _week(10, ending=_FRONTIER - WINDOW)
    state = classify(starts, _series(starts), _LIMIT, _FRONTIER)
    assert state.episodes == 0
    assert state.over_since is None


def _state(
    episodes: int = 10, over_since: datetime | None = _FRONTIER - 6 * _HOUR
) -> VolumeState:
    return VolumeState(
        episodes=episodes,
        limit=_LIMIT,
        over_since=over_since,
        by_fault=(FaultCount(fault="channel_silence", episodes=episodes),),
    )


def _episode(severity: int, *, ended: bool = False, age_hours: int = 6) -> Episode:
    start = _FRONTIER - age_hours * _HOUR
    evidence = (
        EvidenceRow(time=start, score=1.2, severity=severity, value=6.0),
        EvidenceRow(time=start + _HOUR, score=2.0, severity=severity, value=10.0),
    )
    events = [NotificationEvent(EventKind.APPEARED, start, severity)]
    ended_at = None
    if ended:
        ended_at = start + 5 * _HOUR
        events.append(NotificationEvent(EventKind.ENDED, ended_at, 0))
    return Episode(
        fault="notification_volume",
        subject=SUBJECT,
        started_at=start,
        last_seen_at=start + _HOUR,
        ended_at=ended_at,
        severity=severity,
        peak_score=2.0,
        evidence=evidence,
        events=tuple(events),
    )


def _plan(
    episodes: list[Episode],
    open_row: OpenEpisodeRow | None,
    state: VolumeState | None = None,
) -> VolumePlan:
    return plan_run(
        episodes=episodes,
        open_rows=[open_row] if open_row is not None else [],
        state=state if state is not None else _state(),
        frontier=_FRONTIER,
    )


def test_a_fresh_over_volume_is_inserted_and_published() -> None:
    episode = _episode(severity=1)
    plan = _plan([episode], open_row=None)
    assert plan.inserts == (episode,)
    assert plan.updates == ()
    assert plan.publish == VolumePublish(severity=1, state=_state())


def test_an_unchanged_severity_publishes_nothing() -> None:
    episode = _episode(severity=1)
    row = OpenEpisodeRow(id=4, subject=SUBJECT, severity=1)
    plan = _plan([episode], open_row=row)
    assert plan.inserts == ()
    assert plan.updates == ((4, episode),)
    assert plan.publish is None


def test_escalation_publishes_the_new_severity() -> None:
    row = OpenEpisodeRow(id=4, subject=SUBJECT, severity=1)
    plan = _plan([_episode(severity=2)], open_row=row)
    assert plan.publish is not None
    assert plan.publish.severity == 2


def test_stored_severity_is_never_lowered() -> None:
    # The window slid past the loudest stretch: the recomputed severity is
    # lower, but the bus keeps the stored tier and nothing is re-published.
    row = OpenEpisodeRow(id=4, subject=SUBJECT, severity=2)
    plan = _plan([_episode(severity=1)], open_row=row)
    assert plan.publish is None


def test_recovery_reconciles_the_ended_episode_and_clears() -> None:
    episode = _episode(severity=2, ended=True)
    row = OpenEpisodeRow(id=4, subject=SUBJECT, severity=2)
    plan = _plan([episode], open_row=row, state=_state(episodes=3, over_since=None))
    assert plan.inserts == ()
    assert plan.updates == ((4, episode),)
    assert plan.publish is not None
    assert plan.publish.severity == 0
    assert plan.publish.state.episodes == 3


def test_an_open_row_without_a_computed_episode_closes_at_the_frontier() -> None:
    # Unlike the channel kinds there is no dataless case: the episode stream
    # is always countable, so no episode really means the volume fell back.
    row = OpenEpisodeRow(id=9, subject=SUBJECT, severity=2)
    plan = _plan([], open_row=row, state=_state(episodes=2, over_since=None))
    assert plan.orphan_closes == ((9, _FRONTIER),)
    assert plan.publish is not None
    assert plan.publish.severity == 0


def test_a_historical_episode_without_an_open_row_is_ignored() -> None:
    plan = _plan([_episode(severity=1, ended=True)], open_row=None)
    assert plan == VolumePlan((), (), (), None)


def test_only_the_latest_of_several_episodes_reconciles_the_open_row() -> None:
    # Flicker beyond the quiet window splits the fortnight into two
    # incidents; the stored open row can only correspond to the newer one.
    older = _episode(severity=1, ended=True, age_hours=200)
    newer = _episode(severity=2)
    row = OpenEpisodeRow(id=4, subject=SUBJECT, severity=1)
    plan = _plan([older, newer], open_row=row)
    assert plan.updates == ((4, newer),)
    assert plan.inserts == ()


def _run(starts: list[EpisodeStart], open_row: OpenEpisodeRow | None = None) -> VolumePlan:
    """The whole chain the runner drives, minus its SQL and NATS edges."""
    buckets = _series(starts)
    observations = volume_observations(buckets, _LIMIT)
    return plan_run(
        episodes=fold_observations(
            "notification_volume", observations, [], EpisodePolicy(), _FRONTIER
        ),
        open_rows=[open_row] if open_row is not None else [],
        state=classify(starts, buckets, _LIMIT, _FRONTIER),
        frontier=_FRONTIER,
    )


def test_ten_episodes_put_a_severity_on_the_house_wide_address() -> None:
    plan = _run(_week(10))
    (inserted,) = plan.inserts
    assert inserted.ended_at is None
    assert plan.publish is not None
    assert plan.publish.severity > 0
    assert plan.publish.state.episodes == 10
    assert plan.publish.state.over_since is not None


def test_three_episodes_reach_the_bus_as_nothing_at_all() -> None:
    plan = _run(_week(3))
    assert plan == VolumePlan((), (), (), None)


def test_its_own_episode_keeps_counting_while_the_week_drains() -> None:
    # The hysteresis the own-count buys: four incidents from other faults
    # plus its own is five — at the limit, not over it — so the watchdog
    # clears one incident earlier than it fired instead of flapping.
    starts = [*_week(4), EpisodeStart(time=_FRONTIER - 3 * _HOUR, fault="notification_volume")]
    assert _series(starts)[-1].episodes == 5
    plan = _run(starts, open_row=OpenEpisodeRow(id=4, subject=SUBJECT, severity=2))
    assert plan.orphan_closes == ((4, _FRONTIER),)
    assert plan.publish is not None
    assert plan.publish.severity == 0
