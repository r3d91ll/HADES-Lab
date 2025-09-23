"""Utility to remove graph bootstrap collections from an Arango database."""

from __future__ import annotations

import argparse
import logging
from dataclasses import fields
from typing import Iterable

from core.database.arango.memory_client import MemoryServiceError
from core.database.database_factory import DatabaseFactory
from core.graph.schema import DEFAULT_DATABASE, DEFAULT_GRAPH_NAME, GraphCollections

logger = logging.getLogger(__name__)


def _collection_lookup(collections: GraphCollections) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for field in fields(GraphCollections):
        value = getattr(collections, field.name)
        lookup.setdefault(field.name, value)
        lookup.setdefault(value, value)
    return lookup


def _resolve_collections(
    collections: GraphCollections,
    include: Iterable[str] | None,
    skip: Iterable[str] | None,
) -> list[str]:
    lookup = _collection_lookup(collections)
    ordered_defaults = [getattr(collections, field.name) for field in fields(GraphCollections)]

    if include:
        resolved: list[str] = []
        for item in include:
            try:
                name = lookup[item]
            except KeyError as exc:  # pragma: no cover - defensive arg parsing
                raise ValueError(f"Unknown collection '{item}'") from exc
            if name not in resolved:
                resolved.append(name)
    else:
        resolved = list(dict.fromkeys(ordered_defaults))

    if skip:
        for item in skip:
            try:
                name = lookup[item]
            except KeyError as exc:  # pragma: no cover - defensive arg parsing
                raise ValueError(f"Unknown collection to skip '{item}'") from exc
            if name in resolved:
                resolved.remove(name)

    return resolved


def _confirm(database: str, graph_name: str | None, collections: list[str]) -> bool:
    print("About to drop data from ArangoDB:")
    print(f"  Database: {database}")
    if graph_name:
        print(f"  Named graph: {graph_name}")
    if collections:
        print("  Collections:")
        for name in collections:
            print(f"    - {name}")
    else:
        print("  Collections: (none)")
    print()
    prompt = f"Type the database name '{database}' to confirm: "
    try:
        response = input(prompt)
    except EOFError:
        return False
    return response.strip() == database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drop Phase 2 graph collections from Arango")
    parser.add_argument("--database", default=DEFAULT_DATABASE, help="Target Arango database name")
    parser.add_argument(
        "--graph-name",
        default=DEFAULT_GRAPH_NAME,
        help="Named graph to remove before collections are recreated",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip dropping the named graph",
    )
    parser.add_argument(
        "--graph-drop-collections",
        action="store_true",
        help="When dropping the named graph, also drop Arango-managed collections",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        help="Specific collections to drop (name or GraphCollections attribute)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        dest="skip_collections",
        help="Collections to skip when using the default set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without executing requests",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Bypass interactive confirmation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (INFO, DEBUG, ...)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    collections = GraphCollections()
    try:
        target_collections = _resolve_collections(collections, args.collections, args.skip_collections)
    except ValueError as exc:
        parser.error(str(exc))

    if args.skip_graph and not target_collections:
        logger.info("No graph or collections selected; nothing to do")
        return 0

    if not args.dry_run and not args.yes:
        confirmed = _confirm(
            args.database,
            None if args.skip_graph else args.graph_name,
            target_collections,
        )
        if not confirmed:
            logger.info("Aborted; confirmation failed")
            return 1

    client = DatabaseFactory.get_arango_memory_service(database=args.database)
    failures: list[str] = []
    try:
        if not args.skip_graph:
            if args.dry_run:
                logger.info(
                    "DRY RUN: would drop named graph %s (dropCollections=%s)",
                    args.graph_name,
                    "true" if args.graph_drop_collections else "false",
                )
            else:
                try:
                    dropped = client.drop_named_graph(
                        args.graph_name,
                        drop_collections=args.graph_drop_collections,
                        ignore_missing=True,
                    )
                except MemoryServiceError as exc:
                    logger.error("Failed to drop named graph %s: %s", args.graph_name, exc)
                    failures.append(args.graph_name)
                else:
                    if dropped:
                        logger.info("Dropped named graph %s", args.graph_name)
                    else:
                        logger.info("Named graph %s not present", args.graph_name)

        for name in target_collections:
            if args.dry_run:
                logger.info("DRY RUN: would drop collection %s", name)
                continue
            try:
                client.drop_collections([name], ignore_missing=True)
            except MemoryServiceError as exc:
                logger.error("Failed to drop collection %s: %s", name, exc)
                failures.append(name)
            else:
                logger.info("Removed collection %s (if it existed)", name)
    finally:
        client.close()

    if failures:
        logger.warning("Completed with errors; failed to drop: %s", ", ".join(failures))
        return 1

    logger.info("Completed drop of Phase 2 graph collections")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
