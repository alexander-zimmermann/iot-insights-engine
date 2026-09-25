from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import DictRow, dict_row

from .config import Settings
from .logging_setup import get_logger

log = get_logger(__name__)


@contextmanager
def write_connection(settings: Settings) -> Iterator[psycopg.Connection[DictRow]]:
    """Open a short-lived synchronous write connection.

    Batch jobs are one-shot — open, work, close. No pool, because each
    CronJob run is a fresh process holding a single connection for its
    whole lifetime; there is nothing to multiplex.
    """
    conn = psycopg.connect(
        settings.db_write_dsn,
        autocommit=True,
        row_factory=dict_row,
    )
    try:
        log.info(
            "db_write_connect",
            host=settings.db_host,
            database=settings.db_name,
            user=settings.db_write_username,
        )
        yield conn
    finally:
        conn.close()


@contextmanager
def read_connection(settings: Settings) -> Iterator[psycopg.Connection[DictRow]]:
    """Open a short-lived read-only connection (the RO role).

    For jobs that only SELECT (e.g. energy_balance) so the CronJob does not
    need write credentials mounted. Same one-shot lifecycle as write_connection.
    """
    conn = psycopg.connect(
        settings.db_dsn,
        autocommit=True,
        row_factory=dict_row,
    )
    try:
        log.info(
            "db_read_connect",
            host=settings.db_host,
            database=settings.db_name,
            user=settings.db_username,
        )
        yield conn
    finally:
        conn.close()
