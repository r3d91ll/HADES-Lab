"""Durable leasing primitives used by the shard runner."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

from .spec import PartitionResult


@dataclass
class LeaseRecord:
    partition_id: str
    owner: str | None = None
    status: str = "PENDING"
    attempts: int = 0
    lease_until: float = 0.0
    last_error: str | None = None
    result: PartitionResult | None = None


class LeaseStore(Protocol):
    """Abstracts persistence of partition leases."""

    async def acquire(self, partition_id: str, owner: str, ttl: float) -> bool:
        ...

    async def heartbeat(self, partition_id: str, owner: str, ttl: float) -> None:
        ...

    async def release(self, partition_id: str, owner: str) -> None:
        ...

    async def mark_succeeded(self, partition_id: str, owner: str, result: PartitionResult) -> None:
        ...

    async def mark_failed(self, partition_id: str, owner: str, error: str) -> int:
        ...

    async def get_record(self, partition_id: str) -> LeaseRecord:
        ...


class InMemoryLeaseStore(LeaseStore):
    """Lease store suitable for tests and local runs."""

    def __init__(self) -> None:
        self._records: dict[str, LeaseRecord] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, partition_id: str, owner: str, ttl: float) -> bool:
        async with self._lock:
            record = self._records.get(partition_id)
            now = time.time()
            if record is None:
                record = LeaseRecord(partition_id=partition_id)
                self._records[partition_id] = record
            if record.status == "SUCCEEDED":
                return False
            if record.owner and record.owner != owner and record.lease_until > now:
                return False
            record.owner = owner
            record.lease_until = now + ttl
            if record.status == "PENDING":
                record.status = "LEASED"
            return True

    async def heartbeat(self, partition_id: str, owner: str, ttl: float) -> None:
        async with self._lock:
            record = self._records.get(partition_id)
            if not record or record.owner != owner:
                raise RuntimeError("lease heartbeat without ownership")
            record.lease_until = time.time() + ttl

    async def release(self, partition_id: str, owner: str) -> None:
        async with self._lock:
            record = self._records.get(partition_id)
            if not record or record.owner != owner:
                return
            if record.status != "SUCCEEDED":
                record.owner = None
                record.lease_until = 0.0
                if record.status != "FAILED":
                    record.status = "PENDING"

    async def mark_succeeded(self, partition_id: str, owner: str, result: PartitionResult) -> None:
        async with self._lock:
            record = self._records.get(partition_id)
            if not record or record.owner != owner:
                raise RuntimeError("cannot mark success without lease")
            record.status = "SUCCEEDED"
            record.owner = None
            record.result = result
            record.lease_until = 0.0

    async def mark_failed(self, partition_id: str, owner: str, error: str) -> int:
        async with self._lock:
            record = self._records.get(partition_id)
            if not record or record.owner != owner:
                raise RuntimeError("cannot mark failure without lease")
            record.status = "FAILED"
            record.attempts += 1
            record.last_error = error
            record.owner = None
            record.lease_until = 0.0
            return record.attempts

    async def get_record(self, partition_id: str) -> LeaseRecord:
        async with self._lock:
            record = self._records.get(partition_id)
            if record is None:
                record = LeaseRecord(partition_id=partition_id)
                self._records[partition_id] = record
            return record
