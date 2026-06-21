"""Unit tests for the energy-balance job — pure balance math + publish gating.
The SQL counter-delta path is covered by the cluster smoke test after deploy."""

from __future__ import annotations

from unittest.mock import patch

from iot_insights_engine import energy_balance
from iot_insights_engine.config import Settings


def _settings() -> Settings:
    return Settings(
        db_host="localhost",
        db_name="x",
        db_username="x",
        db_password="x",  # noqa: S106 — test stub
    )


def test_compute_normal_closes() -> None:
    # The live-verified case: 21.9 generated, 7 imported, 15 exported.
    v = energy_balance._compute(21.9, 7.0, 15.0)
    assert v == {
        "generation_kwh": 21.9,
        "consumption_kwh": 13.9,  # 21.9 + 7 - 15
        "self_consumption_kwh": 6.9,  # 21.9 - 15
        "self_sufficiency_kwh": 6.9,
        "self_consumption_rate": 31.5,  # 6.9 / 21.9
        "self_sufficiency_rate": 49.6,  # 6.9 / 13.9
    }


def test_compute_night_zero_division_guard() -> None:
    # No generation, only grid draw → rates 0, not a division error.
    v = energy_balance._compute(0.0, 5.0, 0.0)
    assert v["generation_kwh"] == 0.0
    assert v["consumption_kwh"] == 5.0
    assert v["self_consumption_kwh"] == 0.0
    assert v["self_consumption_rate"] == 0.0
    assert v["self_sufficiency_rate"] == 0.0


def test_compute_clamps_direct_use_nonnegative() -> None:
    # Export momentarily exceeds generation (sampling/timing) → floored at 0.
    v = energy_balance._compute(5.0, 0.0, 6.0)
    assert v["self_consumption_kwh"] == 0.0
    assert 0.0 <= v["self_consumption_rate"] <= 100.0


def test_publish_skips_without_nats() -> None:
    s = _settings()  # nats_servers is None
    with patch.object(energy_balance.nats_publisher, "publish") as pub:
        energy_balance._publish(s, {"generation_kwh": 21.9})
    pub.assert_not_called()


def test_publish_emits_each_subject() -> None:
    s = _settings()
    s.nats_servers = "nats://localhost:4222"
    with patch.object(energy_balance.nats_publisher, "publish") as pub:
        energy_balance._publish(s, {"generation_kwh": 21.9, "self_consumption_rate": 31.5})
    subjects = {call.args[1] for call in pub.call_args_list}
    assert subjects == {"energy.pv.generation_kwh", "energy.pv.self_consumption_rate"}
    assert {"value": 21.9} in [call.args[2] for call in pub.call_args_list]
