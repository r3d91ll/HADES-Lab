"""Command line entry point for the Arxiv graph build workflow."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence
import asyncio

from core.workflows.arxiv_repository.arxiv_graph_build.workflow_arxiv_graph_build import (
    ArxivGraphBuildConfig,
    ArxivGraphBuildWorkflow,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arxiv graph bootstrap")
    parser.add_argument("--total-buckets", type=int, default=64, help="Total hash buckets to process")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel workers")
    parser.add_argument(
        "--entity-chunk-size",
        type=int,
        default=5000,
        help="Number of entity keys to process per batch",
    )
    parser.add_argument("--rate-limit", type=int, default=0, help="Max concurrent jobs (0 disables throttling)")
    parser.add_argument(
        "--enable-semantic", action="store_true", default=False, help="Include semantic similarity edge phase"
    )
    parser.add_argument("--semantic-top-k", type=int, default=8, help="Top-K neighbors per paper for semantic edges")
    parser.add_argument(
        "--semantic-score-threshold", type=float, default=0.75, help="Minimum score for semantic edges"
    )
    parser.add_argument("--semantic-embed-source", default="jina_v4")
    parser.add_argument("--semantic-snapshot-id", default="bootstrap")
    parser.add_argument(
        "--buckets",
        type=str,
        help="Comma separated list of bucket ids to process (defaults to all)",
    )
    parser.add_argument(
        "--database", default="arxiv_repository", help="Target ArangoDB database name"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG)"
    )
    return parser


def parse_bucket_subset(bucket_arg: str | None, total: int) -> list[int] | None:
    if not bucket_arg:
        return None
    if bucket_arg.lower() in {"all", "*"}:
        return None
    subset: list[int] = []
    for token in bucket_arg.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            subset.extend(range(start, end + 1))
        else:
            subset.append(int(token))
    return [idx for idx in subset if 0 <= idx < total]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    rate_limit = args.rate_limit if args.rate_limit > 0 else None
    partition_subset = parse_bucket_subset(args.buckets, args.total_buckets)
    config = ArxivGraphBuildConfig(
        database=args.database,
        total_buckets=args.total_buckets,
        shard_count=args.total_buckets,
        num_workers=args.num_workers,
        rate_limit=rate_limit,
        enable_semantic=args.enable_semantic,
        semantic_top_k=args.semantic_top_k,
        semantic_score_threshold=args.semantic_score_threshold,
        semantic_embed_source=args.semantic_embed_source,
        semantic_snapshot_id=args.semantic_snapshot_id,
        partition_subset=partition_subset,
        entity_chunk_size=args.entity_chunk_size,
    )

    workflow = ArxivGraphBuildWorkflow(config)

    if not workflow.validate_inputs():
        logger.error("Validation failed; aborting")
        return 1

    try:
        result = workflow.execute()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning("Interrupted by user (Ctrl-C); cancelling outstanding work")
        return 130
    logger.info("Workflow success: %s", result.success)
    logger.info("Items processed: %s", result.items_processed)
    phase_stats = result.metadata.get("phase_stats", [])
    for s in phase_stats:
        try:
            logger.info(
                "Phase %s: partitions=%s, rows_written=%s, rows_read=%s, duration=%.2fs",
                s.get("phase"),
                s.get("partitions"),
                s.get("rows_written"),
                s.get("rows_read"),
                float(s.get("duration_seconds", 0.0)),
            )
        except Exception:
            pass
    if result.errors:
        logger.warning("Errors: %s", result.errors)
    return 0 if result.success else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
