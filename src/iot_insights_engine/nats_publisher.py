from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import nats

from .config import Settings
from .logging_setup import get_logger
from .severity import severity_level

log = get_logger(__name__)

_UMLAUTS = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


def slugify(value: str) -> str:
    """Stable, NATS-subject-safe token from an entity name (lowercase, German
    umlauts transliterated, every run of non-`[a-z0-9]` collapsed to one `-`).

    The `anomaly.<uc>.<entity>` subject is pinned by the knx-nats-bridge
    writer-rules, so this MUST stay deterministic — `tests/test_nats_publisher`
    locks the mapping.
    """
    value = value.lower()
    for umlaut, repl in _UMLAUTS.items():
        value = value.replace(umlaut, repl)
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def entity_slug(group: dict[str, Any]) -> str | None:
    """Deterministic entity slug for a grouped UC, or None for a 1:1 UC.

    Prefers stable unique identifiers: a KNX GA (`2/2/227` → `2-2-227`),
    inverter/meter id, else the slugified group values. The slug becomes the
    last NATS-subject token, so it must not contain dots.
    """
    if not group:
        return None
    if "ga" in group:
        return slugify(str(group["ga"]))
    if "inverter_id" in group:
        return f"inv{group['inverter_id']}"
    if "meter_id" in group:
        return f"meter{group['meter_id']}"
    return "-".join(slugify(str(v)) for v in group.values())


def _connect_opts(settings: Settings) -> dict[str, Any]:
    """Auth precedence: creds-file → NKey-seed-file → user/password →
    anonymous. Mirrors the knx-nats-bridge publisher so operators swap
    auth by changing the mounted secret, not the code."""
    if not settings.nats_servers:
        raise ValueError("MCP_NATS_SERVERS is required to publish to NATS")

    opts: dict[str, Any] = {
        "servers": [s.strip() for s in settings.nats_servers.split(",") if s.strip()],
        "name": "iot-insights-engine",
        "max_reconnect_attempts": 3,
        "connect_timeout": 5,
    }
    if settings.nats_creds_file:
        opts["user_credentials"] = settings.nats_creds_file
    elif settings.nats_nkey_seed_file:
        opts["nkeys_seed"] = settings.nats_nkey_seed_file
    elif settings.nats_user and settings.nats_password:
        opts["user"] = settings.nats_user
        opts["password"] = settings.nats_password
    return opts


async def _publish_async(settings: Settings, subject: str, payload: dict[str, Any]) -> None:
    nc = await nats.connect(**_connect_opts(settings))
    try:
        body = json.dumps(payload, default=str).encode("utf-8")
        await nc.publish(subject, body)
        await nc.flush(timeout=5)
        log.info("nats_publish", subject=subject, bytes=len(body))
    finally:
        await nc.close()


def publish(settings: Settings, subject: str, payload: dict[str, Any]) -> None:
    """Synchronous wrapper — each job invocation publishes a handful of
    events, so we open/close per call rather than wiring an event loop."""
    asyncio.run(_publish_async(settings, subject, payload))


def publish_anomaly(
    settings: Settings,
    uc: str,
    severity: str,
    payload: dict[str, Any],
    *,
    entity: str | None = None,
    firing: bool = True,
) -> None:
    """Publish one anomaly. Subject is **stable per routing target** —
    `anomaly.<uc>` for a 1:1 UC, `anomaly.<uc>.<entity>` for a grouped one —
    so the knx-nats-bridge writer-rules map exactly one rule per KNX-GA.

    The severity travels as a numeric `severity_level` in the payload (the
    writer-rule reads `$.severity_level`); `firing=False` forces level 0
    (auto-clear → GA falls back to 0).
    """
    subject = f"anomaly.{uc}.{entity}" if entity else f"anomaly.{uc}"
    body = {
        "firing": firing,
        "uc": uc,
        "entity": entity,
        "severity": severity,
        "severity_level": severity_level(severity) if firing else 0,
        **payload,
    }
    publish(settings, subject, body)
