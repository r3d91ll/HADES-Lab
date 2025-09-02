from __future__ import annotations

import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

_ARXIV_ID_RE = re.compile(r"^(?P<yymm>\d{2}(?:0[1-9]|1[0-2]))\.(?P<num>\d{5})(v\d+)?$")


def normalize_arxiv_id(raw_id: str) -> str:
    """Normalize arXiv ID by stripping version suffix (vN)."""
    rid = raw_id.strip()
    if "v" in rid and _ARXIV_ID_RE.match(rid):
        return rid.split("v", 1)[0]
    return rid


def parse_published_at(record: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    """Parse published_at and updated_at from snapshot/OAI-like record.

    Precedence:
      - versions[0].created (RFC-822 like) -> published_at
      - update_date or updated
    Returns (published_at, updated_at)
    """
    published_at: datetime | None = None
    updated_at: datetime | None = None

    # versions[0].created (RFC-822 or ISO-like)
    versions = record.get("versions") or []
    if isinstance(versions, list) and versions:
        created = versions[0].get("created") if isinstance(versions[0], dict) else None
        if created:
            published_at = _parse_any_ts(created)

    # Fallbacks
    if not published_at:
        for key in ("published", "publish_date", "created", "update_date"):
            val = record.get(key)
            if isinstance(val, str):
                published_at = _parse_any_ts(val)
                if published_at:
                    break

    # Updated
    for key in ("update_date", "updated"):
        val = record.get(key)
        if isinstance(val, str):
            updated_at = _parse_any_ts(val)
            if updated_at:
                break

    return published_at, updated_at


def derive_parts(dt: datetime | None) -> tuple[int | None, int | None, str | None]:
    if not dt:
        return None, None, None
    dtz = dt.astimezone(UTC)
    year = dtz.year
    month = dtz.month
    yymm = f"{str(year)[2:]}{month:02d}"
    return year, month, yymm


def _parse_any_ts(text: str) -> datetime | None:
    # Try RFC-822
    try:
        return parsedate_to_datetime(text)
    except Exception:
        pass
    # Try ISO-8601 variants
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(text, fmt)
            # If naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except Exception:
            continue
    return None
