from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import structlog

from .config import PostgresConfig

logger = structlog.get_logger()


def _dsn_with_timeout(dsn: str, statement_timeout_ms: int | None) -> str:
    if not statement_timeout_ms:
        return dsn
    # For psycopg v3, can pass options="-c statement_timeout=..."; append cautiously
    if "options=" in dsn:
        return dsn
    return f"{dsn}?options=-c%20statement_timeout%3D{statement_timeout_ms}"


@contextlib.contextmanager
def get_connection(cfg: PostgresConfig) -> Iterator[Any]:
    """Yield a psycopg connection. Import is done lazily to avoid hard dependency.

    The return type is Any to keep mypy passing without the driver installed.
    """
    try:
        import psycopg
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError(
            "psycopg (v3) is required to use DB features. Install with `poetry add psycopg`"
        ) from e

    dsn = _dsn_with_timeout(cfg.dsn, cfg.statement_timeout_ms)
    logger.debug("pg_connect", dsn_masked=_mask_dsn(dsn))
    conn = psycopg.connect(dsn)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            logger.warning("pg_close_failed")


def _mask_dsn(dsn: str) -> str:
    # naive mask of password between : and @
    if "@" in dsn and ":" in dsn.split("@", 1)[0]:
        left, right = dsn.split("@", 1)
        if ":" in left:
            pre, _pwd = left.rsplit(":", 1)
            return pre + ":***@" + right
    return dsn
