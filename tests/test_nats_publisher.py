"""The NATS adapter: one publish per anomaly, on the subject the
knx-nats-bridge writer rules pin (`tests/test_slug` locks the entity token).
"""

from __future__ import annotations

from unittest.mock import patch

from iot_insights_engine import nats_publisher
from iot_insights_engine.config import Settings
from iot_insights_engine.nats_publisher import publish_anomaly


def test_publish_anomaly_slugs_the_raw_entity_once() -> None:
    # Kinds hand the entity over raw (a GA, a room label); this adapter is
    # the one place that speaks NATS dialect — subject token and payload
    # `entity` carry the same slug.
    settings = Settings(
        db_host="localhost",
        db_name="x",
        db_username="x",
        db_password="x",  # noqa: S106 — test stub
        nats_servers="nats://localhost:4222",
    )
    with patch.object(nats_publisher, "publish") as pub:
        publish_anomaly(settings, "appliance_runtime", "warning", {}, entity="2/1/197")
    (call,) = pub.call_args_list
    assert call.args[1] == "anomaly.appliance_runtime.2-1-197"
    assert call.args[2]["entity"] == "2-1-197"
