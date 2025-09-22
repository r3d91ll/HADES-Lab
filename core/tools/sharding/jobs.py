"""Shard job abstractions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol

from .spec import PartitionResult, PartitionSpec


@dataclass
class JobOutput:
    """Raw output from a job executor before runner enrichment."""

    rows_read: int = 0
    rows_written: int = 0
    succeeded: bool = True
    metrics: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)


class ShardJob(Protocol):
    """Protocol implemented by all job executors."""

    async def run(self, spec: PartitionSpec, params: Mapping[str, Any]) -> JobOutput:
        ...


ShardExecutor = Callable[[PartitionSpec, Mapping[str, Any]], Awaitable[JobOutput]]


class PythonShardJob(ShardJob):
    """Wraps a Python callable into the shard job interface."""

    def __init__(
        self,
        func: Callable[[PartitionSpec, Mapping[str, Any]], JobOutput | Mapping[str, Any] | Awaitable[JobOutput | Mapping[str, Any]]],
    ) -> None:
        self._func = func

    async def run(self, spec: PartitionSpec, params: Mapping[str, Any]) -> JobOutput:
        result = self._func(spec, params)
        if asyncio.iscoroutine(result):
            result = await result  # type: ignore[assignment]
        if isinstance(result, JobOutput):
            return result
        if isinstance(result, Mapping):
            return JobOutput(
                rows_read=int(result.get("rows_read", 0)),
                rows_written=int(result.get("rows_written", 0)),
                succeeded=bool(result.get("succeeded", True)),
                metrics=dict(result.get("metrics", {})),
                details=dict(result.get("details", {})),
            )
        raise TypeError("job callable must return JobOutput or mapping")


class CallableShardJob(PythonShardJob):
    """Alias maintained for backwards compatibility."""


class AqlShardJob(ShardJob):
    """Executes AQL for a partition using the memory client."""

    def __init__(self, client, query: str, *, batch_size: int | None = None) -> None:
        self._client = client
        self._query = query
        self._batch_size = batch_size

    async def run(self, spec: PartitionSpec, params: Mapping[str, Any]) -> JobOutput:
        loop = asyncio.get_running_loop()

        def _execute() -> JobOutput:
            result = self._client.execute_query(self._query, dict(params), batch_size=self._batch_size)
            rows_written = len(result)
            return JobOutput(rows_read=rows_written, rows_written=rows_written, succeeded=True)

        return await loop.run_in_executor(None, _execute)


class ExternalShardJob(ShardJob):
    """Placeholder for out-of-process workers (e.g., Go, gRPC)."""

    def __init__(self, executor: ShardExecutor):
        self._executor = executor

    async def run(self, spec: PartitionSpec, params: Mapping[str, Any]) -> JobOutput:
        return await self._executor(spec, params)


def enrich_result(spec: PartitionSpec, attempt: int, output: JobOutput) -> PartitionResult:
    """Create a PartitionResult from the raw job output."""

    return PartitionResult(
        partition_id=spec.id,
        attempt=attempt,
        rows_read=output.rows_read,
        rows_written=output.rows_written,
        succeeded=output.succeeded,
        metrics=output.metrics,
        details=output.details,
    )
