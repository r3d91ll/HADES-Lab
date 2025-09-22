"""Shared partition specifications and provider contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class PartitionSpec:
    """Identifies a shardable span of work."""

    id: str
    bounds: Mapping[str, Any]
    shard_affinity: str | None = None
    est_rows: int = 0
    est_bytes: int = 0
    max_rows: int | None = None
    max_minutes: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PartitionResult:
    """Outcome details returned by shard jobs."""

    partition_id: str
    attempt: int
    rows_read: int
    rows_written: int
    succeeded: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)


class PartitionInvariantError(RuntimeError):
    """Raised when adapter invariants fail."""


@runtime_checkable
class PartitionProvider(Protocol):
    """Adapter interface describing dataset partitions."""

    def describe_partitions(self) -> list[PartitionSpec]:
        ...

    def bind_params(self, spec: PartitionSpec) -> MutableMapping[str, Any]:
        ...

    def expected_invariants(self, spec: PartitionSpec, result: PartitionResult) -> None:
        ...

    def pre_shard(self, spec: PartitionSpec) -> None:
        ...

    def post_shard(self, spec: PartitionSpec, result: PartitionResult) -> None:
        ...


class BasePartitionAdapter:
    """Convenience base class providing optional hooks."""

    def pre_shard(self, spec: PartitionSpec) -> None:  # noqa: D401 - trivial default
        pass

    def post_shard(self, spec: PartitionSpec, result: PartitionResult) -> None:  # noqa: D401 - trivial default
        pass

    def expected_invariants(self, spec: PartitionSpec, result: PartitionResult) -> None:
        pass
