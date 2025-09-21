"""CLI helpers for HiRAG × PathRAG graph maintenance."""

from __future__ import annotations

import argparse
import logging
from contextlib import closing
from typing import Sequence

from core.database.database_factory import DatabaseFactory

from .service import HiragPathragService

logger = logging.getLogger(__name__)


def _ensure_schema(args: argparse.Namespace) -> None:
    with closing(DatabaseFactory.get_arango_memory_service(database=args.database)) as client:
        service = HiragPathragService(client)
        service.ensure_schema()
        logger.info("Ensured schema for database '%s'", args.database)


def _ingest_entities(args: argparse.Namespace) -> None:
    with closing(DatabaseFactory.get_arango_memory_service(database=args.database)) as client:
        service = HiragPathragService(client)
        count = service.build_entities_from_arxiv(limit=args.limit)
        logger.info("Upserted %d entities", count)


def _build_relations(args: argparse.Namespace) -> None:
    with closing(DatabaseFactory.get_arango_memory_service(database=args.database)) as client:
        service = HiragPathragService(client)
        count = service.build_relations_from_embeddings(
            top_k=args.top_k,
            base_weight=args.base_weight,
            limit=args.limit,
        )
        logger.info("Upserted %d relations", count)


def _build_hierarchy_cmd(args: argparse.Namespace) -> None:
    with closing(DatabaseFactory.get_arango_memory_service(database=args.database)) as client:
        service = HiragPathragService(client)
        stats = service.build_hierarchy(limit=args.limit)
        logger.info(
            "Updated hierarchy: %d cluster links, %d membership edges",
            stats.get('cluster_links', 0),
            stats.get('membership_edges', 0),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HiRAG × PathRAG graph tools")
    parser.add_argument(
        "--database",
        default="arxiv_repository",
        help="Target ArangoDB database name",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest-entities", help="Upsert entities from arxiv_metadata")
    ingest_parser.add_argument("--limit", type=int, default=None, help="Maximum number of metadata records to ingest")
    ingest_parser.set_defaults(func=_ingest_entities)

    relations_parser = subparsers.add_parser("build-relations", help="Create bootstrap semantic relations")
    relations_parser.add_argument("--top-k", type=int, default=5, help="Neighbors per paper within the same category")
    relations_parser.add_argument("--base-weight", type=float, default=0.5, help="Default weight assigned to bootstrap edges")
    relations_parser.add_argument("--limit", type=int, default=None, help="Optional limit on source papers")
    relations_parser.set_defaults(func=_build_relations)

    hierarchy_parser = subparsers.add_parser("build-hierarchy", help="Construct category hierarchy and membership edges")
    hierarchy_parser.add_argument("--limit", type=int, default=None, help="Optional limit on metadata rows to consider")
    hierarchy_parser.set_defaults(func=_build_hierarchy_cmd)

    ensure_parser = subparsers.add_parser("ensure-schema", help="Create required collections and named graph")
    ensure_parser.set_defaults(func=_ensure_schema)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        args.func(args)
    except Exception as exc:  # pragma: no cover - CLI guardrail
        logger.error("Command failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
