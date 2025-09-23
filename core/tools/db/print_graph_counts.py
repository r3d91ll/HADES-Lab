"""Print document counts for HiRAG graph collections and key inputs.

Usage:
  poetry run python -m core.tools.db.print_graph_counts --database arxiv_repository
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict

from core.database.database_factory import DatabaseFactory
from core.graph.schema import GraphSchemaManager, GraphCollections


def collection_count(memory_client, name: str) -> int | None:
    # Prefer the REST count endpoint to avoid scanning the collection.
    try:
        db_name = memory_client._config.database  # type: ignore[attr-defined]
        path = f"/_db/{db_name}/_api/collection/{name}/count"
        data = memory_client._read_client.request("GET", path)  # type: ignore[attr-defined]
        value = data.get("count")
        return int(value) if isinstance(value, int) else None
    except Exception:
        # Fallback to AQL count if REST is unavailable.
        try:
            rows = memory_client.execute_query(
                "FOR d IN @@c COLLECT WITH COUNT INTO c RETURN c",
                {"@@c": name},
            )
            return int(rows[0]) if rows else 0
        except Exception:
            return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Print counts for HiRAG graph collections")
    p.add_argument("--database", default="arxiv_repository")
    args = p.parse_args(argv)

    client = DatabaseFactory.get_arango_memory_service(database=args.database)
    try:
        mgr = GraphSchemaManager(client)
        cols = asdict(mgr.collections)

        # Include key input collections as well.
        extras = ["arxiv_metadata", "arxiv_abstract_embeddings", "arxiv_abstract_chunks"]

        names: list[str] = sorted(set(list(cols.values()) + extras))

        print(f"Database: {mgr.database}")
        for name in names:
            count = collection_count(client, name)
            status = f"{count:,}" if isinstance(count, int) else "<missing>"
            print(f"  {name:30} {status}")
    finally:
        client.close()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
