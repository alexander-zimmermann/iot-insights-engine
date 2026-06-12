"""Shared severity ordering for all detector families.

`severity_floor` on a registry entry suppresses findings below the
floor at classification time — warmup demotion (where a family has
one) runs *after* the floor check, matching the original
score_seasonal semantics.
"""

from __future__ import annotations

SEVERITY_ORDER: tuple[str, ...] = ("info", "warning", "critical")


def meets_floor(severity: str, floor: str) -> bool:
    return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(floor)


def escalated(old_severity: str | None, new_severity: str) -> bool:
    """True when an existing anomaly row was bumped to a higher tier.

    Detectors re-score the same (still maturing) bucket multiple times;
    a finding that first fired as info must still reach the bus when a
    CAGG refresh turns it critical. Downgrades and unchanged severity
    return False — no re-publish, no notification spam.
    """
    if old_severity is None:
        return False
    return SEVERITY_ORDER.index(new_severity) > SEVERITY_ORDER.index(old_severity)
