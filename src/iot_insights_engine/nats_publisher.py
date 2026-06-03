from __future__ import annotations

import asyncio
import json
from typing import Any

import nats

from .config import Settings
from .logging_setup import get_logger

log = get_logger(__name__)


def _connect_opts(settings: Settings) -> dict[str, Any]:
    """Auth precedence: creds-file → NKey-seed-file → user/password →
    anonymous. Mirrors the knx-nats-bridge publisher so operators swap
    auth by changing the mounted secret, not the code."""
    if not settings.nats_servers:
        raise ValueError("MCP_NATS_SERVERS is required to publish to NATS")

    opts: dict[str, Any] = {
        "servers": [s.strip() for s in settings.nats_servers.split(",") if s.strip()],
        "name": "iot-mcp-bridge-jobs",
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
    firing: bool = True,
) -> None:
    """Subject: `anomaly.<uc>.<severity>`."""
    body = {"firing": firing, "uc": uc, "severity": severity, **payload}
    publish(settings, f"anomaly.{uc}.{severity}", body)
