"""Partition planning helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

from .spec import PartitionSpec


def build_hash_partitions(
    *, total_buckets: int, shard_count: int | None = None, prefix: str = "hash"
) -> list[PartitionSpec]:
    """Return shard-aligned hash partitions."""

    if total_buckets <= 0:
        raise ValueError("total_buckets must be positive")
    partitions: list[PartitionSpec] = []
    for bucket in range(total_buckets):
        shard_affinity = None
        if shard_count and shard_count > 0:
            shard_affinity = f"shard_{bucket % shard_count}"
        partitions.append(
            PartitionSpec(
                id=f"{prefix}-{bucket:02d}",
                bounds={"hash_mod": total_buckets, "hash_low": bucket, "hash_high": bucket},
                shard_affinity=shard_affinity,
            )
        )
    return partitions


def build_time_partitions(
    ranges: Sequence[tuple[datetime, datetime]],
    *,
    prefix: str = "window",
) -> list[PartitionSpec]:
    """Create time-window partitions from explicit ranges."""

    partitions: list[PartitionSpec] = []
    for idx, (start, end) in enumerate(ranges):
        if end <= start:
            raise ValueError("time range end must be after start")
        partitions.append(
            PartitionSpec(
                id=f"{prefix}-{idx:02d}",
                bounds={"start": start.isoformat(), "end": end.isoformat()},
            )
        )
    return partitions
