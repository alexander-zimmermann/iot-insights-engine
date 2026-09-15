"""The entity slug: the last token of an `anomaly.<fault>.<entity>` subject.

A device is named by its group address (`2/1/27` -> `2-1-27`), a room by
its entry in the fault's room map (`EG.Flur` -> `eg-flur`), an exchanger by
its label. The knx-nats-bridge writer rules pin the subject, and the lares
generator writes those rules with this very function, imported at the
deployed tag — so the mapping MUST stay deterministic; `tests/test_slug`
locks it.
"""

from __future__ import annotations

import re

_UMLAUTS = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


def entity_slug(value: str) -> str:
    """Stable, NATS-subject-safe token from an entity name: lowercase,
    German umlauts transliterated, every run of non-`[a-z0-9]` collapsed
    to one `-`.
    """
    value = value.lower()
    for umlaut, repl in _UMLAUTS.items():
        value = value.replace(umlaut, repl)
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")
