"""Severity ordering shared by everything that publishes to the bus.

The numeric tier is the delivery contract: the knx-nats-bridge writer
rules carry it onto a group address, where Basalte turns it into a
notification.
"""

from __future__ import annotations

SEVERITY_ORDER: tuple[str, ...] = ("info", "warning", "critical")


def severity_level(severity: str | None) -> int:
    """Numeric tier for the KNX payload: clear/None=0, info=1, warning=2,
    critical=3. The writer-rules write this 1-byte value (DPT 5.010) onto the
    anomaly GA, and a Basalte LUT maps it to a notification."""
    if severity is None:
        return 0
    return SEVERITY_ORDER.index(severity) + 1
