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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HiRAG × PathRAG graph tools")
    parser.add_argument(
        "--database",
        default="arxiv_repository",
        help="Target ArangoDB database name",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

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
