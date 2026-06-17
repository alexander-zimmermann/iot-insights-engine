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


def severity_level(severity: str | None) -> int:
    """Numeric tier for the KNX payload: clear/None=0, info=1, warning=2,
    critical=3. The writer-rules write this 1-byte value (DPT 5.010) onto the
    anomaly GA, and a Basalte LUT maps it to a notification."""
    if severity is None:
        return 0
    return SEVERITY_ORDER.index(severity) + 1


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
