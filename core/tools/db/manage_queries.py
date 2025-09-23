"""List and optionally kill long-running AQL queries directly via upstream socket.

Bypasses the RO/RW proxies to avoid proxy policy/timeouts.

Examples:
  poetry run python -m core.tools.db.manage_queries --database arxiv_repository --min-seconds 60
  poetry run python -m core.tools.db.manage_queries --database arxiv_repository --kill --min-seconds 300
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Any

from core.database.database_factory import DatabaseFactory


def list_queries(client) -> list[dict[str, Any]]:
    db = client._config.database  # type: ignore[attr-defined]
    path = f"/_db/{db}/_api/query/current"
    data = client._read_client.request("GET", path)  # type: ignore[attr-defined]
    return data if isinstance(data, list) else []


def kill_query(client, query_id: str) -> bool:
    db = client._config.database  # type: ignore[attr-defined]
    path = f"/_db/{db}/_api/query/{query_id}"
    client._write_client.request("DELETE", path)  # type: ignore[attr-defined]
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="List/kill long-running ArangoDB queries")
    p.add_argument("--database", default="arxiv_repository")
    p.add_argument("--min-seconds", type=int, default=60, help="Only show/kill queries running >= this many seconds")
    p.add_argument("--kill", action="store_true", help="Kill matching queries")
    args = p.parse_args(argv)

    client = DatabaseFactory.get_arango_memory_service(database=args.database, use_proxies=False)
    try:
        queries = list_queries(client)
        now = dt.datetime.utcnow().timestamp()
        matched = []
        for q in queries:
            secs = float(q.get("runTime", 0.0))
            if secs < args.min_seconds:
                continue
            matched.append(q)

        print(f"Found {len(queries)} queries; {len(matched)} >= {args.min_seconds}s")
        for q in matched:
            qid = q.get("id")
            secs = q.get("runTime")
            query_str = q.get("query", "").split("\n")[0][:120]
            print(f"  {qid}  {secs:>8.2f}s  {query_str}")
        if args.kill:
            for q in matched:
                qid = q.get("id")
                try:
                    kill_query(client, qid)
                    print(f"  KILLED {qid}")
                except Exception as exc:
                    print(f"  Failed to kill {qid}: {exc}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

