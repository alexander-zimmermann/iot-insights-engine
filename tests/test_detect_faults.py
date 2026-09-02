"""Runner-planning tests for the detect-faults job.

The plan is a pure function: computed episodes plus the open rows the
database holds in, inserts/updates/closes plus per-main-group publishes
out. The SQL and NATS edges stay thin; the cluster smoke test covers them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from iot_insights_engine import detect_faults, duration, nats_publisher
from iot_insights_engine.config import Settings
from iot_insights_engine.detect_faults import GroupPublish, plan_run
from iot_insights_engine.episode_store import OpenEpisodeRow
from iot_insights_engine.episodes import (
    Episode,
    EventKind,
    EvidenceRow,
    NotificationEvent,
)
from iot_insights_engine.silence import Channel, ChannelState, SilenceState

_T0 = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)
_FRONTIER = _T0 + 8 * _HOUR

_FREEZER = Channel(ga="2/2/227", name="Schalten.Gefrierschrank.Stromwert", dpt="9.021")
_BOILER = Channel(ga="2/2/224", name="Schalten.Geschirrspueler.Stromwert", dpt="9.021")


def _episode(subject: str, severity: int, *, ended: bool = False) -> Episode:
    start = _T0 + 6 * _HOUR
    evidence = (
        EvidenceRow(time=start, score=6.0, severity=severity, value=6.0),
        EvidenceRow(time=start + _HOUR, score=7.0, severity=severity, value=7.0),
    )
    events = [NotificationEvent(EventKind.APPEARED, start, severity)]
    ended_at = None
    if ended:
        ended_at = start + 5 * _HOUR
        events.append(NotificationEvent(EventKind.ENDED, ended_at, 0))
    return Episode(
        fault="channel_silence",
        subject=subject,
        started_at=start,
        last_seen_at=start + _HOUR,
        ended_at=ended_at,
        severity=severity,
        peak_score=7.0,
        evidence=evidence,
        events=tuple(events),
    )


def _silent(channel: Channel) -> SilenceState:
    return SilenceState(
        channel, ChannelState.SILENT, silent_since=_T0, pause=_HOUR
    )


def _states(*channels: Channel) -> dict[str, SilenceState]:
    return {c.ga: _silent(c) for c in channels}


def _plan(
    episodes: list[Episode],
    open_rows: list[OpenEpisodeRow],
    states: dict[str, SilenceState] | None = None,
    dataless: frozenset[str] = frozenset(),
) -> detect_faults.RunPlan:
    return plan_run(
        episodes=episodes,
        open_rows=open_rows,
        states_by_ga=states if states is not None else _states(_FREEZER, _BOILER),
        dataless=dataless,
        frontier=_FRONTIER,
    )


def test_new_silent_channel_is_inserted_and_published() -> None:
    episode = _episode("2/2/227", severity=1)
    plan = _plan([episode], open_rows=[])
    assert plan.inserts == (episode,)
    assert plan.updates == ()
    assert plan.publishes == (
        GroupPublish(main_group=2, severity=1, channels=plan.publishes[0].channels),
    )
    # The notification names the exact channel.
    (report,) = plan.publishes[0].channels
    assert report.ga == "2/2/227"
    assert report.name == "Schalten.Gefrierschrank.Stromwert"
    assert report.silent_since == _T0
    assert report.gap_hours == 7.0


def test_ongoing_episode_with_unchanged_severity_publishes_nothing() -> None:
    episode = _episode("2/2/227", severity=1)
    row = OpenEpisodeRow(id=7, subject="2/2/227", severity=1)
    plan = _plan([episode], open_rows=[row])
    assert plan.inserts == ()
    assert plan.updates == ((7, episode),)
    assert plan.publishes == ()


def test_escalation_publishes_the_new_group_severity() -> None:
    episode = _episode("2/2/227", severity=2)
    row = OpenEpisodeRow(id=7, subject="2/2/227", severity=1)
    plan = _plan([episode], open_rows=[row])
    assert plan.publishes[0].severity == 2


def test_recovery_publishes_zero_and_reconciles_the_ended_episode() -> None:
    episode = _episode("2/2/227", severity=1, ended=True)
    row = OpenEpisodeRow(id=7, subject="2/2/227", severity=1)
    plan = _plan([episode], open_rows=[row])
    assert plan.updates == ((7, episode),)
    assert plan.publishes == (GroupPublish(main_group=2, severity=0, channels=()),)


def test_open_row_without_computed_counterpart_is_closed_at_frontier() -> None:
    row = OpenEpisodeRow(id=9, subject="2/2/227", severity=1)
    plan = _plan([], open_rows=[row])
    assert plan.orphan_closes == ((9, _FRONTIER),)
    assert plan.publishes == (GroupPublish(main_group=2, severity=0, channels=()),)


def test_stored_severity_is_never_lowered() -> None:
    # The window slid past the old peak: the recomputed severity is lower,
    # but the bus keeps the stored tier and nothing is re-published.
    episode = _episode("2/2/227", severity=1)
    row = OpenEpisodeRow(id=7, subject="2/2/227", severity=2)
    plan = _plan([episode], open_rows=[row])
    assert plan.publishes == ()


def test_group_severity_is_the_maximum_over_its_channels() -> None:
    freezer = _episode("2/2/227", severity=3)
    boiler = _episode("2/2/224", severity=1)
    row = OpenEpisodeRow(id=7, subject="2/2/224", severity=1)
    plan = _plan([boiler, freezer], open_rows=[row])
    (publish,) = plan.publishes
    assert publish.main_group == 2
    assert publish.severity == 3
    # Worst channel first.
    assert [c.ga for c in publish.channels] == ["2/2/227", "2/2/224"]


def test_historical_ended_episode_without_open_row_is_ignored() -> None:
    plan = _plan([_episode("2/2/227", severity=1, ended=True)], open_rows=[])
    assert plan == detect_faults.RunPlan((), (), (), (), ())


def test_second_channel_at_the_same_tier_is_still_published() -> None:
    # The group severity does not move, but the set of silent channels does
    # — the publish names the newcomer instead of going stale.
    freezer = _episode("2/2/227", severity=1)
    boiler = _episode("2/2/224", severity=1)
    row = OpenEpisodeRow(id=7, subject="2/2/224", severity=1)
    plan = _plan([boiler, freezer], open_rows=[row])
    (publish,) = plan.publishes
    assert publish.severity == 1
    assert {c.ga for c in publish.channels} == {"2/2/224", "2/2/227"}


def test_silence_outliving_the_window_stays_open() -> None:
    # In scope but without a single bucket in the window: there is no data
    # to decide a recovery with, so the episode must not self-clear.
    row = OpenEpisodeRow(id=9, subject="2/2/227", severity=2)
    plan = _plan([], open_rows=[row], dataless=frozenset({"2/2/227"}))
    assert plan.orphan_closes == ()
    assert plan.stale_opens == ("2/2/227",)
    assert plan.publishes == ()


def _settings() -> Settings:
    return Settings(
        db_host="localhost",
        db_name="x",
        db_username="x",
        db_password="x",  # noqa: S106 — test stub
        nats_servers="nats://localhost:4222",
    )


def test_publish_group_carries_severity_level_and_channels() -> None:
    settings = _settings()
    episode = _episode("2/2/227", severity=2)
    plan = _plan([episode], open_rows=[])
    with patch.object(nats_publisher, "publish") as pub:
        detect_faults._publish_groups(settings, "channel_silence", plan.publishes)
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.channel_silence.2"
    payload = call.args[2]
    assert payload["severity_level"] == 2
    assert payload["firing"] is True
    assert payload["channels"][0]["ga"] == "2/2/227"


def test_publish_clear_forces_level_zero() -> None:
    settings = _settings()
    with patch.object(nats_publisher, "publish") as pub:
        detect_faults._publish_groups(
            settings,
            "channel_silence",
            (GroupPublish(main_group=15, severity=0, channels=()),),
        )
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.channel_silence.15"
    assert call.args[2]["severity_level"] == 0
    assert call.args[2]["severity"] is None
    assert call.args[2]["firing"] is False


def test_publish_device_carries_run_details_on_the_slug_subject() -> None:
    settings = _settings()
    publish = duration.DevicePublish(
        ga="2/1/197",
        severity=2,
        device="Hauswirtschaftsraum.K4-L1.Waschmaschine",
        name="Schalten.KG.Hauswirtschaftsraum.K4-L1.Waschmaschine.Stromwert",
        running_since=_T0,
        run_hours=6.0,
        limit_hours=4.0,
    )
    with patch.object(nats_publisher, "publish") as pub:
        detect_faults._publish_devices(settings, "appliance_runtime", (publish,))
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.appliance_runtime.2-1-197"
    payload = call.args[2]
    assert payload["severity_level"] == 2
    assert payload["firing"] is True
    assert payload["device"] == "Hauswirtschaftsraum.K4-L1.Waschmaschine"
    assert payload["run_hours"] == 6.0
    assert payload["limit_hours"] == 4.0


def test_publish_device_clear_forces_level_zero() -> None:
    settings = _settings()
    publish = duration.DevicePublish(
        ga="2/1/197",
        severity=0,
        device="Hauswirtschaftsraum.K4-L1.Waschmaschine",
        name="Schalten.KG.Hauswirtschaftsraum.K4-L1.Waschmaschine.Stromwert",
        running_since=None,
        run_hours=None,
        limit_hours=4.0,
    )
    with patch.object(nats_publisher, "publish") as pub:
        detect_faults._publish_devices(settings, "appliance_runtime", (publish,))
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.appliance_runtime.2-1-197"
    assert call.args[2]["severity_level"] == 0
    assert call.args[2]["severity"] is None
    assert call.args[2]["firing"] is False
