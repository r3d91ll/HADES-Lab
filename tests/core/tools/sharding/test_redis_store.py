import asyncio
from pathlib import Path

import pytest

pytest.importorskip("redis")

from core.tools.sharding.redis_store import RedisLeaseStore, RedisTokenBucket
from core.tools.sharding.spec import PartitionResult


REDIS_SOCKET = Path("/run/redis/redis-server.sock")


pytestmark = pytest.mark.skipif(
    not REDIS_SOCKET.exists(),
    reason="Redis UNIX socket not available",
)


@pytest.mark.asyncio
async def test_redis_lease_store():
    store = RedisLeaseStore(socket_path=str(REDIS_SOCKET), namespace="test:lease")

    assert await store.acquire("part-1", "worker-a", ttl=0.5)
    record = await store.get_record("part-1")
    assert record.owner == "worker-a"

    await store.heartbeat("part-1", "worker-a", ttl=0.5)

    attempts = await store.mark_failed("part-1", "worker-a", "oops")
    assert attempts == 1

    assert await store.acquire("part-1", "worker-b", ttl=0.5)

    completion = PartitionResult(
        partition_id="part-1",
        attempt=1,
        rows_read=10,
        rows_written=10,
        succeeded=True,
        metrics={},
        details={},
    )
    await store.mark_succeeded("part-1", "worker-b", completion)
    final = await store.get_record("part-1")
    assert final.status == "SUCCEEDED"
    await store.release("part-1", "worker-b")

    store.close()


@pytest.mark.asyncio
async def test_redis_token_bucket():
    bucket = RedisTokenBucket(
        socket_path=str(REDIS_SOCKET),
        bucket_id="test-bucket",
        capacity=1,
        namespace="test:tokens",
        poll_interval=0.01,
    )

    acquired_once = False
    async with bucket.reserve():
        acquired_once = True
    assert acquired_once

    acquired_twice = False
    async with bucket.reserve():
        acquired_twice = True
    assert acquired_twice

    bucket.close()
