import pytest

from core.tools.sharding import NullTokenBucket, ShardRunner
from core.tools.sharding.adapters.arxiv import ArxivPartitionAdapter
from core.tools.sharding.jobs import PythonShardJob
from core.tools.sharding.lease import InMemoryLeaseStore
from core.tools.sharding.spec import BasePartitionAdapter, PartitionInvariantError, PartitionResult, PartitionSpec


@pytest.mark.asyncio
async def test_runner_executes_all_partitions():
    adapter = ArxivPartitionAdapter(total_buckets=3)
    seen: set[str] = set()

    def job_callable(spec: PartitionSpec, params: dict[str, object]):
        seen.add(spec.id)
        return {"rows_read": 1, "rows_written": 1}

    job = PythonShardJob(job_callable)
    lease_store = InMemoryLeaseStore()
    runner = ShardRunner(
        adapter,
        job,
        lease_store=lease_store,
        concurrency=2,
        token_buckets={"default": NullTokenBucket()},
        lease_ttl=2.0,
    )

    results = await runner.run()

    assert len(results) == 3
    assert seen == {"arxiv-00", "arxiv-01", "arxiv-02"}
    assert all(r.rows_written == 1 for r in results)


class FailingAdapter(BasePartitionAdapter):
    def __init__(self) -> None:
        self.spec = PartitionSpec(id="fail", bounds={})

    def describe_partitions(self):
        return [self.spec]

    def bind_params(self, spec):
        return {}

    def expected_invariants(self, spec, result: PartitionResult):
        raise PartitionInvariantError("forced failure")


@pytest.mark.asyncio
async def test_runner_raises_on_invariant_failure():
    adapter = FailingAdapter()
    job = PythonShardJob(lambda _spec, _params: {"rows_read": 0, "rows_written": 0})
    lease_store = InMemoryLeaseStore()
    runner = ShardRunner(
        adapter,
        job,
        lease_store=lease_store,
        concurrency=1,
        max_retries=0,
        token_buckets={"default": NullTokenBucket()},
        lease_ttl=2.0,
    )

    with pytest.raises(PartitionInvariantError):
        await runner.run()


@pytest.mark.asyncio
async def test_runner_skips_when_lease_held():
    adapter = ArxivPartitionAdapter(total_buckets=1)
    job = PythonShardJob(lambda _spec, _params: {"rows_read": 1, "rows_written": 1})
    lease_store = InMemoryLeaseStore()
    spec = adapter.describe_partitions()[0]

    acquired = await lease_store.acquire(spec.id, "external-owner", ttl=30.0)
    assert acquired

    runner = ShardRunner(
        adapter,
        job,
        lease_store=lease_store,
        concurrency=1,
        token_buckets={"default": NullTokenBucket()},
        lease_ttl=2.0,
    )

    results = await runner.run(partitions=[spec])
    assert results == []
