"""Shard jobs that call HiRAG builders via the service layer."""

from __future__ import annotations

import asyncio
from typing import Mapping

from core.database.database_factory import DatabaseFactory
from core.graph.hirag.service import HiragService
from core.graph.schema import GraphSchemaManager

from .jobs import JobOutput, ShardJob
from .spec import PartitionSpec


class _BaseHiragJob(ShardJob):
    """Helper base class to run synchronous HiRAG service calls in an executor."""

    def __init__(self, database: str = "arxiv_repository") -> None:
        self._database = database

    async def _run_service(self, func) -> JobOutput:
        loop = asyncio.get_running_loop()

        def _execute() -> JobOutput:
            client = DatabaseFactory.get_arango_memory_service(database=self._database)
            try:
                service = HiragService(client, schema_manager=GraphSchemaManager(client))
                return func(service)
            finally:
                client.close()

        return await loop.run_in_executor(None, _execute)


class HiragEntityIngestJob(_BaseHiragJob):
    """Run entity ingestion for a single hash partition."""

    async def run(self, spec: PartitionSpec, params: Mapping[str, object]) -> JobOutput:
        hash_mod = int(params["hash_mod"])
        hash_low = int(params["hash_low"])
        hash_high = int(params.get("hash_high", hash_low))

        def _call(service: HiragService) -> JobOutput:
            count = service.build_entities_from_arxiv(
                hash_mod=hash_mod,
                hash_low=hash_low,
                hash_high=hash_high,
            )
            return JobOutput(
                rows_read=count,
                rows_written=count,
                metrics={"entities": count},
                details={"hash_mod": hash_mod, "hash_low": hash_low},
            )

        return await self._run_service(_call)


class HiragRelationsJob(_BaseHiragJob):
    """Build bootstrap relations for a partition."""

    def __init__(
        self,
        *,
        top_k: int = 10,
        base_weight: float = 0.5,
        database: str = "arxiv_repository",
    ) -> None:
        super().__init__(database=database)
        self._top_k = top_k
        self._base_weight = base_weight

    async def run(self, spec: PartitionSpec, params: Mapping[str, object]) -> JobOutput:
        hash_mod = int(params["hash_mod"])
        hash_low = int(params["hash_low"])
        hash_high = int(params.get("hash_high", hash_low))

        def _call(service: HiragService) -> JobOutput:
            count = service.build_relations_from_embeddings(
                top_k=self._top_k,
                base_weight=self._base_weight,
                hash_mod=hash_mod,
                hash_low=hash_low,
                hash_high=hash_high,
            )
            return JobOutput(
                rows_read=count,
                rows_written=count,
                metrics={"relations": count},
                details={"hash_mod": hash_mod, "hash_low": hash_low},
            )

        return await self._run_service(_call)


class HiragHierarchyJob(_BaseHiragJob):
    """Build hierarchy edges/documents for a partition."""

    async def run(self, spec: PartitionSpec, params: Mapping[str, object]) -> JobOutput:
        hash_mod = int(params["hash_mod"])
        hash_low = int(params["hash_low"])
        hash_high = int(params.get("hash_high", hash_low))

        def _call(service: HiragService) -> JobOutput:
            stats = service.build_hierarchy(
                hash_mod=hash_mod,
                hash_low=hash_low,
                hash_high=hash_high,
            )
            written = int(stats.get("cluster_links", 0) + stats.get("membership_edges", 0))
            return JobOutput(
                rows_read=written,
                rows_written=written,
                metrics=stats,
                details={"hash_mod": hash_mod, "hash_low": hash_low},
            )

        return await self._run_service(_call)


class HiragSemanticEdgesJob(_BaseHiragJob):
    """Build semantic similarity edges for a partition."""

    def __init__(
        self,
        *,
        top_k: int = 10,
        score_threshold: float = 0.7,
        embed_source: str = "jina_v4",
        snapshot_id: str = "snapshot-unknown",
        database: str = "arxiv_repository",
    ) -> None:
        super().__init__(database=database)
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._embed_source = embed_source
        self._snapshot_id = snapshot_id

    async def run(self, spec: PartitionSpec, params: Mapping[str, object]) -> JobOutput:
        hash_mod = int(params["hash_mod"])
        hash_low = int(params["hash_low"])
        hash_high = int(params.get("hash_high", hash_low))

        def _call(service: HiragService) -> JobOutput:
            count = service.build_semantic_similarity_edges(
                top_k=self._top_k,
                score_threshold=self._score_threshold,
                embed_source=self._embed_source,
                snapshot_id=self._snapshot_id,
                hash_mod=hash_mod,
                hash_low=hash_low,
                hash_high=hash_high,
            )
            return JobOutput(
                rows_read=count,
                rows_written=count,
                metrics={"semantic_edges": count},
                details={"hash_mod": hash_mod, "hash_low": hash_low},
            )

        return await self._run_service(_call)
