"""Unit tests for the anomaly subject/entity slug.

The slug is the last token of the `anomaly.<uc>.<entity>` NATS subject and is
pinned by the knx-nats-bridge writer-rules — these tests lock the mapping so a
refactor can't silently re-route anomalies to the wrong KNX-GA.
"""

from __future__ import annotations

from iot_insights_engine.nats_publisher import entity_slug, slugify


def test_slugify_deterministic() -> None:
    assert slugify("2/2/227") == "2-2-227"
    assert slugify("Schlafzimmer Eltern") == "schlafzimmer-eltern"
    assert slugify("Gäste WC") == "gaeste-wc"
    assert slugify("Küche") == "kueche"
    assert slugify("Begehbarer-Schrank") == "begehbarer-schrank"


def test_slugify_subject_safe() -> None:
    # No dots (NATS token separators), no leading/trailing/double dashes.
    assert "." not in slugify("a.b.c")
    assert slugify("  Foo  Bar  ") == "foo-bar"


def test_entity_slug_ungrouped_is_none() -> None:
    assert entity_slug({}) is None


def test_entity_slug_prefers_ga() -> None:
    # Appliance UCs group by (ga, knx_name) — the GA is the stable id.
    assert entity_slug({"ga": "2/2/227", "knx_name": "…Gefrierschrank.Stromwert"}) == "2-2-227"


def test_entity_slug_inverter_and_meter() -> None:
    assert entity_slug({"inverter_id": 1}) == "inv1"
    assert entity_slug({"inverter_id": 2}) == "inv2"
    assert entity_slug({"meter_id": 0}) == "meter0"


def test_entity_slug_falls_back_to_values() -> None:
    assert entity_slug({"room": "Küche"}) == "kueche"
