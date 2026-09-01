"""Severity ordering shared by everything that publishes to the bus.

The numeric tier is the delivery contract: the knx-nats-bridge writer
rules carry it onto a group address, where Basalte turns it into a
notification.
"""

from __future__ import annotations

SEVERITY_ORDER: tuple[str, ...] = ("info", "warning", "critical")

# The 1-byte tiers written to the anomaly GAs; 0 clears.
CLEAR, INFO, WARNING, CRITICAL = 0, 1, 2, 3


def severity_name(level: int) -> str:
    """Inverse of severity_level for the firing tiers 1..3."""
    if not 1 <= level <= len(SEVERITY_ORDER):
        raise ValueError(f"no severity name for tier {level}")
    return SEVERITY_ORDER[level - 1]


def severity_level(severity: str | None) -> int:
    """Numeric tier for the KNX payload: clear/None=0, info=1, warning=2,
    critical=3. The writer-rules write this 1-byte value (DPT 5.010) onto the
    anomaly GA, and a Basalte LUT maps it to a notification."""
    if severity is None:
        return 0
    return SEVERITY_ORDER.index(severity) + 1
