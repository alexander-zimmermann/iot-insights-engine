"""The entity slug the package exports.

The slug is the last token of the `anomaly.<fault>.<entity>` subject, pinned
by the knx-nats-bridge writer rules — and the lares generator imports it at
the deployed tag to write those rules, so both the mapping and the import
path are locked here.
"""

from __future__ import annotations

from iot_insights_engine import entity_slug


def test_entity_slug_renders_the_documented_examples() -> None:
    assert entity_slug("2/1/27") == "2-1-27"
    assert entity_slug("EG.Flur") == "eg-flur"


def test_entity_slug_is_deterministic() -> None:
    assert entity_slug("Schlafzimmer Eltern") == "schlafzimmer-eltern"
    assert entity_slug("Gäste WC") == "gaeste-wc"
    assert entity_slug("Küche") == "kueche"
    assert entity_slug("Begehbarer-Schrank") == "begehbarer-schrank"


def test_entity_slug_is_subject_safe() -> None:
    # No dots (NATS token separators), no leading/trailing/double dashes.
    assert "." not in entity_slug("a.b.c")
    assert entity_slug("  Foo  Bar  ") == "foo-bar"
