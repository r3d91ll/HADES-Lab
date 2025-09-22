import asyncio

import pytest

from core.tools.sharding.hirag_jobs import (
    HiragEntityIngestJob,
    HiragHierarchyJob,
    HiragRelationsJob,
    HiragSemanticEdgesJob,
)
from core.tools.sharding.spec import PartitionSpec


@pytest.mark.asyncio
async def test_entity_job_invokes_service(monkeypatch):
    calls = {}

    class DummyService:
        def __init__(self, _client, *_args, **_kwargs):
            pass

        def build_entities_from_arxiv(self, **kwargs):
            calls.update(kwargs)
            return 5

    class DummyManager:
        def __init__(self, _client):
            pass

    class DummyClient:
        def close(self):
            pass

    monkeypatch.setattr(
        "core.tools.sharding.hirag_jobs.DatabaseFactory.get_arango_memory_service",
        lambda database: DummyClient(),
    )
    monkeypatch.setattr("core.tools.sharding.hirag_jobs.HiragService", DummyService)
    monkeypatch.setattr("core.tools.sharding.hirag_jobs.GraphSchemaManager", DummyManager)

    job = HiragEntityIngestJob(database="test_db")
    spec = PartitionSpec(id="p0", bounds={"hash_mod": 8, "hash_low": 3, "hash_high": 3})
    params = {"hash_mod": 8, "hash_low": 3, "hash_high": 3}

    result = await job.run(spec, params)

    assert result.rows_written == 5
    assert calls == {"hash_mod": 8, "hash_low": 3, "hash_high": 3}


@pytest.mark.asyncio
async def test_relations_job_uses_config(monkeypatch):
    received = {}

    class DummyService:
        def __init__(self, _client, *_args, **_kwargs):
            pass

        def build_relations_from_embeddings(self, **kwargs):
            received.update(kwargs)
            return 10

    class DummyManager:
        def __init__(self, _client):
            pass

    class DummyClient:
        def close(self):
            pass

    monkeypatch.setattr(
        "core.tools.sharding.hirag_jobs.DatabaseFactory.get_arango_memory_service",
        lambda database: DummyClient(),
    )
    monkeypatch.setattr("core.tools.sharding.hirag_jobs.HiragService", DummyService)
    monkeypatch.setattr("core.tools.sharding.hirag_jobs.GraphSchemaManager", DummyManager)

    job = HiragRelationsJob(top_k=7, base_weight=0.3, database="test_db")
    spec = PartitionSpec(id="p1", bounds={"hash_mod": 16, "hash_low": 5})
    params = {"hash_mod": 16, "hash_low": 5}

    result = await job.run(spec, params)

    assert result.metrics["relations"] == 10
    assert received["top_k"] == 7
    assert received["base_weight"] == 0.3
    assert received["hash_low"] == 5


@pytest.mark.asyncio
async def test_hierarchy_job_collects_stats(monkeypatch):
    class DummyService:
        def __init__(self, _client, *_args, **_kwargs):
            pass

        def build_hierarchy(self, **kwargs):
            assert kwargs["hash_low"] == 2
            return {"cluster_links": 2, "membership_edges": 3}

    class DummyManager:
        def __init__(self, _client):
            pass

    class DummyClient:
        def close(self):
            pass

    monkeypatch.setattr(
        "core.tools.sharding.hirag_jobs.DatabaseFactory.get_arango_memory_service",
        lambda database: DummyClient(),
    )
    monkeypatch.setattr("core.tools.sharding.hirag_jobs.HiragService", DummyService)
    monkeypatch.setattr("core.tools.sharding.hirag_jobs.GraphSchemaManager", DummyManager)

    job = HiragHierarchyJob(database="test_db")
    spec = PartitionSpec(id="p2", bounds={"hash_mod": 32, "hash_low": 2})
    params = {"hash_mod": 32, "hash_low": 2}

    result = await job.run(spec, params)

    assert result.rows_written == 5
    assert result.metrics == {"cluster_links": 2, "membership_edges": 3}


@pytest.mark.asyncio
async def test_semantic_edges_job(monkeypatch):
    captured = {}

    class DummyService:
        def __init__(self, _client, *_args, **_kwargs):
            pass

        def build_semantic_similarity_edges(self, **kwargs):
            captured.update(kwargs)
            return 7

    class DummyManager:
        def __init__(self, _client):
            pass

    class DummyClient:
        def close(self):
            pass

    monkeypatch.setattr(
        "core.tools.sharding.hirag_jobs.DatabaseFactory.get_arango_memory_service",
        lambda database: DummyClient(),
    )
    monkeypatch.setattr("core.tools.sharding.hirag_jobs.HiragService", DummyService)
    monkeypatch.setattr("core.tools.sharding.hirag_jobs.GraphSchemaManager", DummyManager)

    job = HiragSemanticEdgesJob(
        top_k=11,
        score_threshold=0.9,
        embed_source="unit-test",
        snapshot_id="snap-1",
        database="test_db",
    )
    spec = PartitionSpec(id="p3", bounds={"hash_mod": 8, "hash_low": 4})
    params = {"hash_mod": 8, "hash_low": 4}

    result = await job.run(spec, params)

    assert result.metrics["semantic_edges"] == 7
    assert captured["top_k"] == 11
    assert captured["score_threshold"] == 0.9
    assert captured["embed_source"] == "unit-test"
    assert captured["snapshot_id"] == "snap-1"
