"""Tests for the jobs-extension settings fields (db_write, NATS, S3, SMTP)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from iot_insights_engine.config import Settings


@pytest.fixture
def base_env(tmp_path: Path) -> Iterator[None]:
    """Minimum required env so Settings() validates; tests then override as needed."""
    keys = list(os.environ.keys())
    for k in keys:
        if k.startswith("MCP_"):
            del os.environ[k]
    os.environ.update(
        MCP_DB_HOST="localhost",
        MCP_DB_NAME="test",
        MCP_DB_USERNAME="ro",
        MCP_DB_PASSWORD="ro-pw",
        MCP_AUTH_ENABLED="false",
    )
    yield
    for k in list(os.environ.keys()):
        if k.startswith("MCP_"):
            del os.environ[k]


def test_db_write_dsn_raises_when_unset(base_env: None) -> None:
    s = Settings()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="MCP_DB_WRITE_USERNAME"):
        _ = s.db_write_dsn


def test_db_write_dsn_uses_inline_env(base_env: None) -> None:
    os.environ.update(MCP_DB_WRITE_USERNAME="rw", MCP_DB_WRITE_PASSWORD="rw-pw")
    s = Settings()  # type: ignore[call-arg]
    assert s.db_write_dsn == "postgresql://rw:rw-pw@localhost:5432/test"


def test_db_write_dsn_reads_file(base_env: None, tmp_path: Path) -> None:
    user_file = tmp_path / "user"
    pw_file = tmp_path / "pw"
    user_file.write_text("rw\n")
    pw_file.write_text("rw-pw\n")
    os.environ.update(
        MCP_DB_WRITE_USERNAME_FILE=str(user_file),
        MCP_DB_WRITE_PASSWORD_FILE=str(pw_file),
    )
    s = Settings()  # type: ignore[call-arg]
    assert s.db_write_dsn == "postgresql://rw:rw-pw@localhost:5432/test"


def test_db_dsns_url_encode_password(base_env: None) -> None:
    # openssl rand -base64 routinely produces passwords containing '/', '+',
    # '=', '@' — all of which break psycopg's URI parser when not encoded.
    os.environ.update(
        MCP_DB_USERNAME="ro",
        MCP_DB_PASSWORD="abc/def+gh=ij@kl",
        MCP_DB_WRITE_USERNAME="rw",
        MCP_DB_WRITE_PASSWORD="x/y+z=q@r",
    )
    s = Settings()  # type: ignore[call-arg]
    assert s.db_dsn == "postgresql://ro:abc%2Fdef%2Bgh%3Dij%40kl@localhost:5432/test"
    assert s.db_write_dsn == "postgresql://rw:x%2Fy%2Bz%3Dq%40r@localhost:5432/test"


def test_optional_fields_have_sane_defaults(base_env: None) -> None:
    s = Settings()  # type: ignore[call-arg]
    assert s.s3_bucket == "iot-mcp-bridge-models"
    assert s.smtp_host == "smtprelay.smtprelay.svc.cluster.local"
    assert s.smtp_port == 25
    assert s.nats_servers is None
    assert s.s3_endpoint is None


def test_nats_and_s3_secret_files_resolve(base_env: None, tmp_path: Path) -> None:
    pw = tmp_path / "nats-pw"
    pw.write_text("super-secret\n")
    ak = tmp_path / "s3-ak"
    sk = tmp_path / "s3-sk"
    ak.write_text("AKIA\n")
    sk.write_text("secretkey\n")
    os.environ.update(
        MCP_NATS_PASSWORD_FILE=str(pw),
        MCP_S3_ACCESS_KEY_FILE=str(ak),
        MCP_S3_SECRET_KEY_FILE=str(sk),
    )
    s = Settings()  # type: ignore[call-arg]
    assert s.nats_password == "super-secret"
    assert s.s3_access_key == "AKIA"
    assert s.s3_secret_key == "secretkey"
