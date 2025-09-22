"""Command-line entry for running sharded jobs."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
from typing import Any, Mapping

from . import CallableShardJob, FixedTokenBucket, InMemoryLeaseStore, NullTokenBucket, ShardRunner
from .lease import LeaseStore
from .spec import PartitionProvider
from .token_bucket import TokenBucket

logger = logging.getLogger(__name__)


def _load_object(path: str) -> Any:
    if ":" not in path:
        raise ValueError("component path must be module:object")
    module_name, attr = path.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr)
    return obj


def _instantiate(component: Any, config: Mapping[str, Any]) -> Any:
    if callable(component):
        if hasattr(component, "__call__") and hasattr(component, "__mro__"):
            return component(**config)
        return component
    raise TypeError("component must be callable or class")


def _parse_kv(payload: str) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON configuration") from exc


def _build_token_buckets(spec: str | None) -> dict[str, TokenBucket]:
    if not spec:
        return {}
    buckets: dict[str, TokenBucket] = {}
    pairs = spec.split(",")
    for pair in pairs:
        if not pair:
            continue
        name, value = pair.split("=", 1)
        capacity = int(value)
        buckets[name] = FixedTokenBucket(capacity)
    return buckets


def _lease_store_from_uri(uri: str | None) -> LeaseStore:
    if uri is None or uri == "memory://":
        return InMemoryLeaseStore()
    raise NotImplementedError("only memory:// lease store is implemented")


async def _run_async(args: argparse.Namespace) -> None:
    adapter_obj = _load_object(args.adapter)
    adapter = _instantiate(adapter_obj, _parse_kv(args.adapter_config or ""))
    if not isinstance(adapter, PartitionProvider):
        raise TypeError("adapter must implement PartitionProvider")

    job_obj = _load_object(args.job)
    job_instance = _instantiate(job_obj, _parse_kv(args.job_config or ""))
    if not hasattr(job_instance, "run"):
        job_instance = CallableShardJob(job_instance)

    lease_store = _lease_store_from_uri(args.lease_store)
    token_buckets = _build_token_buckets(args.rate_limit)

    def emit(name: str, spec, payload: Mapping[str, object]) -> None:
        logger.info("%s %s %s", name, spec.id, payload)

    runner = ShardRunner(
        adapter,
        job_instance,
        lease_store=lease_store,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        lease_ttl=args.lease_ttl,
        token_buckets=token_buckets,
        metrics=emit,
    )

    results = await runner.run()
    logger.info("completed %d partitions", len(results))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run sharded ingestion jobs")
    parser.add_argument("--adapter", required=True, help="Adapter path module:Class")
    parser.add_argument("--job", required=True, help="Job path module:Class or callable")
    parser.add_argument("--adapter-config", help="JSON configuration for adapter")
    parser.add_argument("--job-config", help="JSON configuration for job")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--lease-ttl", type=float, default=120.0)
    parser.add_argument("--lease-store", help="Lease store URI", default="memory://")
    parser.add_argument("--rate-limit", help="Comma-separated token buckets (name=capacity)")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        asyncio.run(_run_async(args))
    except Exception as exc:  # pragma: no cover - CLI guardrail
        logger.error("sharded run failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
