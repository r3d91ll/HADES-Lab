"""ArXiv-specific partition adapter for sharded ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping

from ..planners import build_hash_partitions
from ..spec import BasePartitionAdapter, PartitionResult, PartitionSpec


@dataclass(slots=True)
class ArxivPartitionAdapter(BasePartitionAdapter):
    """Partitions the arXiv corpus by stable hash bucket."""

    total_buckets: int = 64
    shard_count: int | None = None
    metadata_collection: str = "arxiv_metadata"
    key_field: str = "_key"

    def describe_partitions(self) -> list[PartitionSpec]:
        return build_hash_partitions(
            total_buckets=self.total_buckets,
            shard_count=self.shard_count,
            prefix="arxiv",
        )

    def bind_params(self, spec: PartitionSpec) -> MutableMapping[str, object]:
        bounds = dict(spec.bounds)
        return {
            "hash_mod": bounds.get("hash_mod", self.total_buckets),
            "hash_low": bounds.get("hash_low", 0),
            "hash_high": bounds.get("hash_high", 0),
            "collection": self.metadata_collection,
            "key_field": self.key_field,
        }

    def expected_invariants(self, spec: PartitionSpec, result: PartitionResult) -> None:
        if result.rows_written < 0:
            raise ValueError("rows_written must be non-negative")
