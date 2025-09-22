"""Async shard runner coordinating partitioned jobs."""

from __future__ import annotations

import asyncio
import logging
import socket
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Iterable, Mapping

from .jobs import JobOutput, ShardJob, enrich_result
from .lease import LeaseStore
from .spec import PartitionInvariantError, PartitionProvider, PartitionResult, PartitionSpec
from .token_bucket import NullTokenBucket, TokenBucket

logger = logging.getLogger(__name__)


MetricEmitter = Callable[[str, PartitionSpec, Mapping[str, Any]], None]


class ShardRunner:
    """Coordinates leases, throttles, and job execution across partitions."""

    def __init__(
        self,
        provider: PartitionProvider,
        job: ShardJob,
        *,
        lease_store: LeaseStore,
        concurrency: int = 8,
        max_retries: int = 3,
        lease_ttl: float = 120.0,
        token_buckets: Mapping[str, TokenBucket] | None = None,
        metrics: MetricEmitter | None = None,
        owner_id: str | None = None,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        self._provider = provider
        self._job = job
        self._lease_store = lease_store
        self._concurrency = concurrency
        self._max_retries = max_retries
        self._lease_ttl = lease_ttl
        self._token_buckets = dict(token_buckets or {})
        self._metrics = metrics
        self._owner_id = owner_id or f"{socket.gethostname()}-{uuid.uuid4()}"
        if "default" not in self._token_buckets:
            self._token_buckets["default"] = NullTokenBucket()
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run(self, *, partitions: Iterable[PartitionSpec] | None = None) -> list[PartitionResult]:
        specs = list(partitions) if partitions is not None else self._provider.describe_partitions()
        tasks = [
            asyncio.create_task(self._run_partition_with_guard(spec))
            for spec in specs
        ]
        results: list[PartitionResult] = []
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
        return results

    async def _run_partition_with_guard(self, spec: PartitionSpec) -> PartitionResult | None:
        async with self._semaphore:
            return await self._run_partition(spec)

    async def _run_partition(self, spec: PartitionSpec) -> PartitionResult | None:
        acquired = await self._lease_store.acquire(spec.id, self._owner_id, self._lease_ttl)
        if not acquired:
            self._emit("shard.skipped", spec, {"reason": "lease-unavailable"})
            return None
        async with self._lease_guard(spec):
            attempt = 0
            while attempt <= self._max_retries:
                attempt += 1
                try:
                    self._emit("shard.started", spec, {"attempt": attempt})
                    self._provider.pre_shard(spec)
                    params = self._provider.bind_params(spec)
                    output = await self._execute_with_tokens(spec, params)
                    result = enrich_result(spec, attempt, output)
                    self._provider.expected_invariants(spec, result)
                    self._provider.post_shard(spec, result)
                    await self._lease_store.mark_succeeded(spec.id, self._owner_id, result)
                    self._emit("shard.succeeded", spec, {"attempt": attempt, "rows": result.rows_written})
                    return result
                except PartitionInvariantError as exc:
                    await self._lease_store.mark_failed(spec.id, self._owner_id, f"invariant: {exc}")
                    self._emit("shard.failed", spec, {"attempt": attempt, "error": str(exc)})
                    raise
                except Exception as exc:
                    attempts = await self._lease_store.mark_failed(spec.id, self._owner_id, str(exc))
                    self._emit("shard.failed", spec, {"attempt": attempt, "error": str(exc)})
                    if attempts > self._max_retries:
                        raise
                    await asyncio.sleep(min(2**attempt, 30))
                    reacquired = await self._lease_store.acquire(spec.id, self._owner_id, self._lease_ttl)
                    if not reacquired:
                        self._emit("shard.skipped", spec, {"reason": "lease-lost"})
                        return None
            return None

    async def _execute_with_tokens(self, spec: PartitionSpec, params: Mapping[str, object]) -> JobOutput:
        async with self._token_buckets["default"].reserve():
            return await self._job.run(spec, params)

    @asynccontextmanager
    async def _lease_guard(self, spec: PartitionSpec):
        stop_event = asyncio.Event()
        heartbeat_task = asyncio.create_task(self._heartbeat(spec, stop_event))
        try:
            yield
        finally:
            stop_event.set()
            await heartbeat_task
            await self._lease_store.release(spec.id, self._owner_id)

    async def _heartbeat(self, spec: PartitionSpec, stop_event: asyncio.Event) -> None:
        interval = max(1.0, self._lease_ttl / 2)
        while not stop_event.is_set():
            await asyncio.sleep(interval)
            try:
                await self._lease_store.heartbeat(spec.id, self._owner_id, self._lease_ttl)
            except Exception:
                logger.exception(
                    "lease heartbeat failed for spec %s owner %s",
                    spec.id,
                    self._owner_id,
                    extra={"spec_id": spec.id, "owner_id": self._owner_id},
                )
                break

    def _emit(self, name: str, spec: PartitionSpec, payload: Mapping[str, Any]) -> None:
        if self._metrics is None:
            return
        self._metrics(name, spec, payload)
