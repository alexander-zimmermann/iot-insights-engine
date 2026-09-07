"""Plan and wire tests for the detect-faults kinds.

The silence plan is a pure function: computed episodes plus the open rows
the database holds in, inserts/updates/closes plus per-main-group
publishes out. The wire tests pin each kind's subject and payload bytes.
The SQL and NATS edges stay thin; the cluster smoke test covers them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from iot_insights_engine import (
    deviation,
    drift,
    duration,
    nats_publisher,
    silence,
    volume,
)
from iot_insights_engine.config import Settings
from iot_insights_engine.detect_faults import _kind_for
from iot_insights_engine.episode_store import OpenEpisodeRow
from iot_insights_engine.episodes import (
    Episode,
    EventKind,
    EvidenceRow,
    NotificationEvent,
)
from iot_insights_engine.faults import (
    DeviationExpectation,
    Fault,
    MeasurementKind,
    Target,
)
from iot_insights_engine.reconcile import Measured, Plan
from iot_insights_engine.runner import NatsPublisher, publish_subjects
from iot_insights_engine.silence import (
    Channel,
    ChannelState,
    GroupPublish,
    SilenceState,
    plan_run,
)

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
) -> Plan[GroupPublish]:
    measured = Measured(
        states=states if states is not None else _states(_FREEZER, _BOILER),
        observations=(),
        dataless=dataless,
        record={},
    )
    return plan_run(
        episodes=episodes,
        open_rows=open_rows,
        measured=measured,
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
    assert plan == Plan((), (), (), (), ())


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
        publish_subjects(
            NatsPublisher(settings), "channel_silence", plan.publishes, silence.group_payload
        )
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.channel_silence.2"
    payload = call.args[2]
    assert payload["severity_level"] == 2
    assert payload["firing"] is True
    assert payload["channels"][0]["ga"] == "2/2/227"


def test_publish_clear_forces_level_zero() -> None:
    settings = _settings()
    with patch.object(nats_publisher, "publish") as pub:
        publish_subjects(
            NatsPublisher(settings),
            "channel_silence",
            (GroupPublish(main_group=15, severity=0, channels=()),),
            silence.group_payload,
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
        publish_subjects(
            NatsPublisher(settings), "appliance_runtime", (publish,), duration.payload
        )
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
        publish_subjects(
            NatsPublisher(settings), "appliance_runtime", (publish,), duration.payload
        )
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.appliance_runtime.2-1-197"
    assert call.args[2]["severity_level"] == 0
    assert call.args[2]["severity"] is None
    assert call.args[2]["firing"] is False


def test_publish_standby_carries_drift_details_on_the_slug_subject() -> None:
    settings = _settings()
    publish = drift.DevicePublish(
        ga="2/2/227",
        severity=2,
        device="Küche.K15-L1.Gefrierschrank",
        name="Schalten.EG.Küche.K15-L1.Gefrierschrank.Stromwert",
        level=500.0,
        healthy=48.0,
        excess=452.0,
        rising_since=_T0,
    )
    with patch.object(nats_publisher, "publish") as pub:
        publish_subjects(
            NatsPublisher(settings), "appliance_standby", (publish,), drift.payload_standby
        )
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.appliance_standby.2-2-227"
    payload = call.args[2]
    assert payload["severity_level"] == 2
    assert payload["firing"] is True
    assert payload["standby_ma"] == 500.0
    assert payload["healthy_ma"] == 48.0
    assert payload["excess_ma"] == 452.0
    assert payload["rising_since"] == _T0


def test_publish_duty_cycle_carries_drift_details_on_the_slug_subject() -> None:
    settings = _settings()
    publish = drift.DevicePublish(
        ga="2/2/227",
        severity=1,
        device="Küche.K15-L1.Gefrierschrank",
        name="Schalten.EG.Küche.K15-L1.Gefrierschrank.Stromwert",
        level=71.0,
        healthy=50.0,
        excess=21.0,
        rising_since=_T0,
    )
    with patch.object(nats_publisher, "publish") as pub:
        publish_subjects(
            NatsPublisher(settings), "freezer_icing", (publish,), drift.payload_duty_cycle
        )
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.freezer_icing.2-2-227"
    payload = call.args[2]
    assert payload["severity_level"] == 1
    assert payload["firing"] is True
    assert payload["duty_pct"] == 71.0
    assert payload["healthy_pct"] == 50.0
    assert payload["excess_pct"] == 21.0
    assert payload["rising_since"] == _T0


def test_publish_room_carries_cold_details_on_the_slug_subject() -> None:
    settings = _settings()
    publish = deviation.RoomPublish(
        slug="eg-buero",
        severity=2,
        room="EG.Büro",
        cold_since=_T0,
        gap=1.5,
        value=20.5,
        reference=22.0,
        gate=80.0,
        min_gap=1.0,
    )
    with patch.object(nats_publisher, "publish") as pub:
        publish_subjects(NatsPublisher(settings), "fbh_cold", (publish,), deviation.payload)
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.fbh_cold.eg-buero"
    payload = call.args[2]
    assert payload["severity_level"] == 2
    assert payload["firing"] is True
    assert payload["room"] == "EG.Büro"
    assert payload["gap"] == 1.5
    assert payload["reference"] == 22.0


def test_publish_room_clear_forces_level_zero() -> None:
    settings = _settings()
    publish = deviation.RoomPublish(
        slug="eg-buero",
        severity=0,
        room="EG.Büro",
        cold_since=None,
        gap=None,
        value=None,
        reference=None,
        gate=None,
        min_gap=1.0,
    )
    with patch.object(nats_publisher, "publish") as pub:
        publish_subjects(NatsPublisher(settings), "fbh_cold", (publish,), deviation.payload)
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.fbh_cold.eg-buero"
    assert call.args[2]["severity_level"] == 0
    assert call.args[2]["severity"] is None
    assert call.args[2]["firing"] is False


def test_publish_exchanger_carries_recovery_details_on_the_slug_subject() -> None:
    settings = _settings()
    publish = drift.ExchangerPublish(
        slug="kwl",
        severity=2,
        exchanger="KWL",
        efficiency=68.0,
        healthy=88.0,
        deficit=20.0,
        falling_since=_T0,
    )
    with patch.object(nats_publisher, "publish") as pub:
        publish_subjects(
            NatsPublisher(settings), "heat_recovery_decay", (publish,), drift.payload_recovery
        )
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.heat_recovery_decay.kwl"
    payload = call.args[2]
    assert payload["severity_level"] == 2
    assert payload["firing"] is True
    assert payload["exchanger"] == "KWL"
    assert payload["efficiency_pct"] == 68.0
    assert payload["healthy_pct"] == 88.0
    assert payload["deficit_pct"] == 20.0
    assert payload["falling_since"] == _T0


def test_publish_exchanger_clear_forces_level_zero() -> None:
    settings = _settings()
    publish = drift.ExchangerPublish(
        slug="kwl",
        severity=0,
        exchanger="KWL",
        efficiency=86.0,
        healthy=88.0,
        deficit=2.0,
        falling_since=None,
    )
    with patch.object(nats_publisher, "publish") as pub:
        publish_subjects(
            NatsPublisher(settings), "heat_recovery_decay", (publish,), drift.payload_recovery
        )
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.heat_recovery_decay.kwl"
    assert call.args[2]["severity_level"] == 0
    assert call.args[2]["severity"] is None
    assert call.args[2]["firing"] is False


def _volume_publish(severity: int) -> volume.VolumePublish:
    """Over the limit while firing, back under it on the clear."""
    return volume.VolumePublish(
        severity=severity,
        state=volume.VolumeState(
            episodes=11 if severity else 3,
            limit=5.0,
            over_since=_T0 if severity else None,
            by_fault=(
                volume.FaultCount(fault="channel_silence", episodes=8),
                volume.FaultCount(fault="fbh_cold", episodes=3),
            ),
        ),
    )


def test_publish_volume_carries_the_week_on_the_house_wide_subject() -> None:
    settings = _settings()
    with patch.object(nats_publisher, "publish") as pub:
        publish_subjects(
            NatsPublisher(settings),
            "notification_volume",
            (_volume_publish(2),),
            volume.payload,
        )
    (call,) = pub.call_args_list
    # One house-wide address, so a 1:1 subject with no entity token.
    assert call.args[1] == "anomaly.notification_volume"
    payload = call.args[2]
    assert payload["severity_level"] == 2
    assert payload["firing"] is True
    assert payload["episodes"] == 11
    assert payload["limit"] == 5.0
    assert payload["window_days"] == 7
    assert payload["by_fault"][0] == {"fault": "channel_silence", "episodes": 8}


def test_publish_volume_clear_forces_level_zero() -> None:
    settings = _settings()
    with patch.object(nats_publisher, "publish") as pub:
        publish_subjects(
            NatsPublisher(settings),
            "notification_volume",
            (_volume_publish(0),),
            volume.payload,
        )
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.notification_volume"
    assert call.args[2]["severity_level"] == 0
    assert call.args[2]["severity"] is None
    assert call.args[2]["firing"] is False


def _plant_publish(severity: int) -> deviation.PlantPublish:
    """Short against the forecast while firing, back on target on the clear."""
    return deviation.PlantPublish(
        severity=severity,
        state=deviation.YieldState(
            expectation=DeviationExpectation.FORECAST_SOLAR,
            min_shortfall_pct=35.0,
            day=_T0,
            actual_kwh=12.0 if severity else 39.0,
            expected_kwh=40.0,
            shortfall_pct=70.0 if severity else 2.5,
            short_since=_T0 if severity else None,
        ),
    )


def test_publish_plant_carries_the_day_against_its_expectation() -> None:
    settings = _settings()
    with patch.object(nats_publisher, "publish") as pub:
        publish_subjects(
            NatsPublisher(settings),
            "pv_underperformance",
            (_plant_publish(2),),
            deviation.payload_yield,
        )
    (call,) = pub.call_args_list
    # One plant-wide address, so a 1:1 subject with no entity token — the
    # writer rule pins exactly this string to 15/4/11.
    assert call.args[1] == "anomaly.pv_underperformance"
    payload = call.args[2]
    assert payload["severity_level"] == 2
    assert payload["firing"] is True
    assert payload["expectation"] == "forecast_solar"
    assert payload["day"] == _T0
    assert payload["actual_kwh"] == 12.0
    assert payload["expected_kwh"] == 40.0
    assert payload["shortfall_pct"] == 70.0
    assert payload["short_since"] == _T0


def test_publish_plant_clear_forces_level_zero() -> None:
    settings = _settings()
    with patch.object(nats_publisher, "publish") as pub:
        publish_subjects(
            NatsPublisher(settings),
            "pv_underperformance",
            (_plant_publish(0),),
            deviation.payload_yield,
        )
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.pv_underperformance"
    assert call.args[2]["severity_level"] == 0
    assert call.args[2]["severity"] is None
    assert call.args[2]["firing"] is False


def _deviation_fault(expectation: DeviationExpectation | None) -> Fault:
    return Fault(
        name="pv_underperformance" if expectation else "fbh_cold",
        sentence="ein Wert liegt unter seiner Erwartung",
        unit="× der erlaubten Abweichung",
        kind=MeasurementKind.DEVIATION,
        parameters={},
        target=Target(ga="15/4/11") if expectation else Target(per_room=True),
        expectation=expectation,
    )


def test_a_named_expectation_picks_the_daily_yield_shape() -> None:
    # The one thing that decides which of the kind's two shapes runs.
    kind = _kind_for(_deviation_fault(DeviationExpectation.FORECAST_SOLAR))
    assert kind is not None
    assert kind.measure is deviation.measure_yield
    assert kind.frontier is deviation.yield_frontier
    assert kind.policy.bucket == deviation.DAY


def test_a_deviation_without_an_expectation_stays_the_room_shape() -> None:
    kind = _kind_for(_deviation_fault(None))
    assert kind is not None
    assert kind.measure is deviation.measure
    assert kind.policy.bucket == timedelta(hours=1)
