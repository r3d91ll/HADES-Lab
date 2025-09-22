import pytest

from core.tools.sharding.spec import PartitionResult, PartitionSpec
from core.tools.sharding import NullTokenBucket
from core.tools.sharding.lease import LeaseRecord
from core.workflows.arxiv_repository.arxiv_graph_build.workflow_arxiv_graph_build import (
    ArxivGraphBuildConfig,
    ArxivGraphBuildWorkflow,
)


def test_arxiv_graph_build_workflow(monkeypatch):
    module = __import__(
        "core.workflows.arxiv_repository.arxiv_graph_build.workflow_arxiv_graph_build",
        fromlist=["ArxivGraphBuildWorkflow"],
    )

    class DummyClient:
        def execute_query(self, query, *_, **__):
            if "arxiv_metadata" in query:
                return ["doc"]
            return []

        def close(self):
            pass

    class DummyFactory:
        @staticmethod
        def get_arango_memory_service(database):
            return DummyClient()

    class DummyService:
        def __init__(self, _client, *_args, **_kwargs):
            pass

        def ensure_schema(self):
            return None

    class DummySchema:
        def __init__(self, _client):
            pass

    class DummyAdapter:
        def __init__(self, total_buckets, shard_count=None):
            self.total_buckets = total_buckets

        def describe_partitions(self):
            return [
                PartitionSpec(id=f"part-{i}", bounds={"hash_mod": 2, "hash_low": i})
                for i in range(self.total_buckets)
            ]

    captured = {"phases": []}

    class DummyRunner:
        def __init__(self, adapter, job, **kwargs):
            captured["phases"].append(job.__class__.__name__)
            self._metrics = kwargs.get("metrics")

        async def run(self, *, partitions):
            if self._metrics:
                for spec in partitions:
                    self._metrics("shard.succeeded", spec, {"phase": "dummy"})
            return [
                PartitionResult(
                    partition_id=spec.id,
                    attempt=1,
                    rows_read=1,
                    rows_written=1,
                    succeeded=True,
                    metrics={},
                    details={},
                )
                for spec in partitions
            ]

    monkeypatch.setattr(module, "DatabaseFactory", DummyFactory)
    monkeypatch.setattr(module, "HiragService", DummyService)
    monkeypatch.setattr(module, "GraphSchemaManager", DummySchema)
    monkeypatch.setattr(module, "ArxivPartitionAdapter", DummyAdapter)
    monkeypatch.setattr(module, "ShardRunner", DummyRunner)

    class DummyLeaseStore:
        async def acquire(self, partition_id, owner, ttl):
            return True

        async def heartbeat(self, partition_id, owner, ttl):
            return None

        async def release(self, partition_id, owner):
            return None

        async def mark_succeeded(self, partition_id, owner, result):
            return None

        async def mark_failed(self, partition_id, owner, error):
            return 0

        async def get_record(self, partition_id):
            return LeaseRecord(partition_id=partition_id)

        def close(self):
            return None

    workflow = ArxivGraphBuildWorkflow(
        ArxivGraphBuildConfig(
            total_buckets=2,
            num_workers=2,
            rate_limit=None,
            semantic_top_k=5,
            semantic_score_threshold=0.8,
            semantic_embed_source="test_source",
            semantic_snapshot_id="snapshot-test",
            lease_store_factory=lambda: DummyLeaseStore(),
            token_bucket_factory=lambda: {"default": NullTokenBucket()},
        )
    )

    assert workflow.validate_inputs() is True
    result = workflow.execute()

    assert result.success is True
    assert result.items_processed == 8  # 2 partitions * 4 phases
    assert result.items_failed == 0
    assert captured["phases"] == [
        "HiragEntityIngestJob",
        "HiragRelationsJob",
        "HiragHierarchyJob",
        "HiragSemanticEdgesJob",
    ]
    assert "phase_results" in result.metadata
