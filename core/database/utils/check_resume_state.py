"""Utility script to inspect ArangoDB ingest progress using the memory client."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import os
import sys

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from core.database.database_factory import DatabaseFactory
from core.database.arango import MemoryServiceError


def _count_documents(db, collection: str) -> int:
    """Return LENGTH(collection) or 0 if the collection is missing."""
    try:
        result = db.execute_query(f"RETURN LENGTH({collection})")
        return int(result[0]) if result else 0
    except MemoryServiceError as exc:
        # Collection may not exist in a given deployment; treat as zero
        print(f"[warn] Could not count collection '{collection}': {exc.details()}")
        return 0
    except Exception as exc:  # pragma: no cover - diagnostics
        print(f"[warn] Unexpected error counting '{collection}': {exc}")
        return 0


def _print_recent_papers(db) -> None:
    """Display the most recent processed papers."""
    print("\nRecent Papers")
    print("-" * 60)
    try:
        records = db.execute_query(
            """
            FOR doc IN arxiv_papers
                SORT doc.processing_timestamp DESC
                LIMIT 5
                RETURN doc
            """
        )
    except MemoryServiceError:
        print("No arxiv_papers collection available yet.")
        return

    if not records:
        print("No papers processed yet.")
        return

    for rec in records:
        arxiv_id = rec.get("arxiv_id", "<unknown>")
        processed = rec.get("processing_timestamp", "<unknown>")
        title = rec.get("title", "<untitled>")
        print(f"ID: {arxiv_id:20} | {processed} | {title[:80]}")

        emb = db.execute_query(
            """
            FOR doc IN arxiv_embeddings
                FILTER doc.arxiv_id == @id
                LIMIT 1
                RETURN doc
            """,
            bind_vars={"id": arxiv_id},
        )
        if emb:
            vector = emb[0].get("vector", [])
            dim = len(vector) if isinstance(vector, list) else emb[0].get("embedding_dim", "?")
            print(f"  ↳ embedding dimension: {dim}")
        else:
            print("  ↳ no embedding record found")


def _print_recent_activity(db) -> None:
    """Show records processed in the last minute."""
    print("\nRecent Activity (last minute)")
    print("-" * 60)
    since = (datetime.now() - timedelta(minutes=1)).isoformat()
    try:
        result = db.execute_query(
            """
            FOR doc IN arxiv_papers
                FILTER doc.processing_timestamp >= @ts
                COLLECT WITH COUNT INTO count
                RETURN count
            """,
            bind_vars={"ts": since},
        )
        recent = int(result[0]) if result else 0
        print(f"Papers processed: {recent}")
        print(f"Rate: ~{recent:.0f} papers/minute")
    except MemoryServiceError as exc:
        print(f"Unable to query recent activity: {exc.details()}")


def _print_unprocessed_metadata(db) -> None:
    """Show candidate metadata entries that still need embeddings (legacy workflow)."""
    legacy_embeddings = _count_documents(db, "arxiv_abstract_embeddings")
    legacy_metadata = _count_documents(db, "arxiv_metadata")
    if legacy_metadata == 0 or legacy_embeddings == 0:
        return

    print("\nLegacy Metadata Backlog")
    print("-" * 60)
    try:
        rows = db.execute_query(
            """
            LET processed = (
                FOR emb IN arxiv_abstract_embeddings
                    RETURN DISTINCT emb.arxiv_id
            )
            FOR doc IN arxiv_metadata
                FILTER doc.abstract != null
                FILTER doc.abstract_length >= 655
                FILTER doc.arxiv_id NOT IN processed
                SORT doc.abstract_length ASC
                LIMIT 10
                RETURN {id: doc.arxiv_id, length: doc.abstract_length}
            """
        )
        if not rows:
            print("All eligible metadata rows have embeddings.")
            return
        for idx, rec in enumerate(rows, start=1):
            print(f"{idx:2}. ID: {rec['id']:20} | Length: {rec['length']:4}")
    except MemoryServiceError as exc:
        print(f"Unable to compute backlog: {exc.details()}")


def main() -> None:
    password = os.environ.get("ARANGO_PASSWORD")
    if not password:
        print("ERROR: ARANGO_PASSWORD environment variable not set", file=sys.stderr)
        sys.exit(1)

    db = DatabaseFactory.get_arango_memory_service(database="arxiv_repository")

    print("=" * 60)
    print("CURRENT DATABASE STATE")
    print("=" * 60)

    papers_count = _count_documents(db, "arxiv_papers")
    embeddings_count = _count_documents(db, "arxiv_embeddings")
    structures_count = _count_documents(db, "arxiv_structures")
    legacy_metadata = _count_documents(db, "arxiv_metadata")

    print(f"arxiv_papers:      {papers_count:,}")
    print(f"arxiv_embeddings:  {embeddings_count:,}")
    print(f"arxiv_structures:  {structures_count:,}")
    if legacy_metadata:
        legacy_embeddings = _count_documents(db, "arxiv_abstract_embeddings")
        legacy_chunks = _count_documents(db, "arxiv_abstract_chunks")
        print(f"arxiv_metadata:    {legacy_metadata:,} (legacy)")
        print(f"legacy embeddings: {legacy_embeddings:,}")
        print(f"legacy chunks:     {legacy_chunks:,}")

    _print_recent_papers(db)
    _print_recent_activity(db)
    _print_unprocessed_metadata(db)


if __name__ == "__main__":
    main()
