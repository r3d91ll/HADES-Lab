"""Redis-backed coordination primitives for the sharding toolkit."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Optional

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    redis = None  # type: ignore

from dataclasses import asdict

from .lease import LeaseRecord, LeaseStore
from .spec import PartitionResult
from .token_bucket import TokenBucket


class RedisLeaseStore(LeaseStore):
    """Durable lease store that uses Redis hashes per partition."""

    def __init__(
        self,
        *,
        socket_path: str,
        namespace: str = "hades:shard:lease",
        prefix: Optional[str] = None,
    ) -> None:
        if redis is None:
            raise RuntimeError("redis package is required for RedisLeaseStore")
        self._client = redis.Redis(unix_socket_path=socket_path, decode_responses=True)
        self._namespace = prefix or namespace
        self._acquire_script = self._client.register_script(
            """
            local key = KEYS[1]
            local owner = ARGV[1]
            local now = tonumber(ARGV[2])
            local ttl = tonumber(ARGV[3])
            local new_until = now + ttl

            if redis.call('EXISTS', key) == 0 then
              redis.call('HMSET', key,
                'owner', owner,
                'status', 'LEASED',
                'attempts', 0,
                'lease_until', new_until,
                'last_error', '',
                'result_json', ''
              )
              return 1
            end

            local status = redis.call('HGET', key, 'status')
            local current_owner = redis.call('HGET', key, 'owner')
            local lease_until = tonumber(redis.call('HGET', key, 'lease_until') or '0')
            if status == 'SUCCEEDED' then
              return 0
            end
            if current_owner == false or current_owner == nil or current_owner == owner or lease_until <= now then
              redis.call('HMSET', key,
                'owner', owner,
                'status', 'LEASED',
                'lease_until', new_until,
                'last_error', ''
              )
              return 1
            end
            return 0
            """
        )
        self._mark_failed_script = self._client.register_script(
            """
            local key = KEYS[1]
            local owner = ARGV[1]
            local error_msg = ARGV[2]
            local attempts = tonumber(redis.call('HGET', key, 'attempts') or '0')
            local current_owner = redis.call('HGET', key, 'owner')
            if current_owner ~= owner then
              return attempts
            end
            attempts = attempts + 1
            redis.call('HMSET', key,
              'status', 'FAILED',
              'owner', '',
              'lease_until', 0,
              'attempts', attempts,
              'last_error', error_msg
            )
            return attempts
            """
        )
        self._mark_succeeded_script = self._client.register_script(
            """
            local key = KEYS[1]
            local owner = ARGV[1]
            local payload = ARGV[2]
            local current_owner = redis.call('HGET', key, 'owner')
            if current_owner ~= owner then
              return 0
            end
            redis.call('HMSET', key,
              'status', 'SUCCEEDED',
              'owner', '',
              'lease_until', 0,
              'last_error', '',
              'result_json', payload
            )
            return 1
            """
        )
        self._release_script = self._client.register_script(
            """
            local key = KEYS[1]
            local owner = ARGV[1]
            local current_owner = redis.call('HGET', key, 'owner')
            if current_owner ~= owner then
              return 0
            end
            local status = redis.call('HGET', key, 'status')
            if status == 'SUCCEEDED' then
              redis.call('HMSET', key,
                'owner', '',
                'lease_until', 0
              )
            else
              redis.call('HMSET', key,
                'owner', '',
                'status', 'PENDING',
                'lease_until', 0
              )
            end
            return 1
            """
        )

    def _key(self, partition_id: str) -> str:
        return f"{self._namespace}:{partition_id}"

    async def acquire(self, partition_id: str, owner: str, ttl: float) -> bool:
        return await asyncio.to_thread(
            self._acquire_script,
            keys=[self._key(partition_id)],
            args=[owner, time.time(), ttl],
        )

    async def heartbeat(self, partition_id: str, owner: str, ttl: float) -> None:
        async def _heartbeat() -> None:
            key = self._key(partition_id)
            data = self._client.hgetall(key)
            if not data or data.get("owner") != owner:
                raise RuntimeError("lease heartbeat without ownership")
            self._client.hset(key, mapping={"lease_until": time.time() + ttl})

        await asyncio.to_thread(_heartbeat)

    async def release(self, partition_id: str, owner: str) -> None:
        await asyncio.to_thread(
            self._release_script,
            keys=[self._key(partition_id)],
            args=[owner],
        )

    async def mark_succeeded(self, partition_id: str, owner: str, result: PartitionResult) -> None:
        payload = json.dumps(asdict(result), default=str)
        await asyncio.to_thread(
            self._mark_succeeded_script,
            keys=[self._key(partition_id)],
            args=[owner, payload],
        )

    async def mark_failed(self, partition_id: str, owner: str, error: str) -> int:
        return await asyncio.to_thread(
            self._mark_failed_script,
            keys=[self._key(partition_id)],
            args=[owner, error],
        )

    async def get_record(self, partition_id: str) -> LeaseRecord:
        def _fetch() -> LeaseRecord:
            key = self._key(partition_id)
            data = self._client.hgetall(key)
            if not data:
                return LeaseRecord(partition_id=partition_id)
            result_payload = data.get("result_json")
            result_obj: Optional[PartitionResult] = None
            if result_payload:
                try:
                    raw = json.loads(result_payload)
                    result_obj = PartitionResult(
                        partition_id=raw.get("partition_id", partition_id),
                        attempt=raw.get("attempt", 0),
                        rows_read=raw.get("rows_read", 0),
                        rows_written=raw.get("rows_written", 0),
                        succeeded=raw.get("succeeded", False),
                        metrics=raw.get("metrics", {}),
                        details=raw.get("details", {}),
                    )
                except json.JSONDecodeError:
                    result_obj = None
            return LeaseRecord(
                partition_id=partition_id,
                owner=data.get("owner") or None,
                status=data.get("status") or "PENDING",
                attempts=int(data.get("attempts") or 0),
                lease_until=float(data.get("lease_until") or 0.0),
                last_error=data.get("last_error") or None,
                result=result_obj,
            )

        return await asyncio.to_thread(_fetch)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class RedisTokenBucket(TokenBucket):
    """Token bucket implemented with Redis counters."""

    def __init__(
        self,
        *,
        socket_path: str,
        bucket_id: str = "default",
        capacity: int = 32,
        namespace: str = "hades:shard:tokens",
        poll_interval: float = 0.1,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if redis is None:
            raise RuntimeError("redis package is required for RedisTokenBucket")
        self._client = redis.Redis(unix_socket_path=socket_path, decode_responses=True)
        self._bucket_key = f"{namespace}:{bucket_id}:available"
        self._capacity_key = f"{namespace}:{bucket_id}:capacity"
        self._capacity = capacity
        self._poll_interval = poll_interval
        # initialize capacity if not set
        self._client.setnx(self._capacity_key, capacity)
        self._client.setnx(self._bucket_key, capacity)
        self._reserve_script = self._client.register_script(
            """
            local available_key = KEYS[1]
            local capacity_key = KEYS[2]
            local cost = tonumber(ARGV[1])
            local capacity = tonumber(redis.call('GET', capacity_key) or '0')
            if capacity == 0 then
              return 0
            end
            local available = tonumber(redis.call('GET', available_key) or capacity)
            if available >= cost then
              redis.call('DECRBY', available_key, cost)
              return 1
            end
            return 0
            """
        )
        self._release_script = self._client.register_script(
            """
            local available_key = KEYS[1]
            local capacity_key = KEYS[2]
            local cost = tonumber(ARGV[1])
            local capacity = tonumber(redis.call('GET', capacity_key) or '0')
            local available = tonumber(redis.call('GET', available_key) or capacity)
            local new_value = available + cost
            if new_value > capacity then
              new_value = capacity
            end
            redis.call('SET', available_key, new_value)
            return new_value
            """
        )

    async def _wait_for_tokens(self, cost: int) -> None:
        while True:
            acquired = await asyncio.to_thread(
                self._reserve_script,
                keys=[self._bucket_key, self._capacity_key],
                args=[cost],
            )
            if acquired == 1:
                break
            await asyncio.sleep(self._poll_interval)

    @asynccontextmanager
    async def reserve(self, cost: int = 1):
        if cost <= 0:
            raise ValueError("cost must be positive")
        await self._wait_for_tokens(cost)
        try:
            yield
        finally:
            await asyncio.to_thread(
                self._release_script,
                keys=[self._bucket_key, self._capacity_key],
                args=[cost],
            )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
