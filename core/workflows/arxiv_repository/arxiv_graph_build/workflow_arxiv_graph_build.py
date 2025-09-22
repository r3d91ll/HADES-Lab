"""Workflow orchestrating the ArXiv graph bootstrap using the sharding toolkit."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Mapping, Optional

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    redis = None  # type: ignore

from core.database.database_factory import DatabaseFactory
from core.graph.hirag.service import HiragService
from core.graph.schema import GraphSchemaManager
from core.tools.sharding import (
    ArxivPartitionAdapter,
    HiragEntityIngestJob,
    HiragHierarchyJob,
    HiragRelationsJob,
    HiragSemanticEdgesJob,
    NullTokenBucket,
    PartitionResult,
    RedisLeaseStore,
    RedisTokenBucket,
    ShardRunner,
)
from core.tools.sharding.lease import LeaseStore, InMemoryLeaseStore
from core.tools.sharding.token_bucket import TokenBucket
from core.tools.sharding.spec import PartitionSpec
from core.workflows.workflow_base import WorkflowBase, WorkflowConfig, WorkflowResult

logger = logging.getLogger(__name__)


LeaseStoreFactory = Callable[[], LeaseStore]
TokenBucketFactory = Callable[[], Dict[str, TokenBucket]]
@dataclass
class ArxivGraphBuildConfig(WorkflowConfig):
    """Configuration for the ArXiv graph build workflow."""

    name: str = "arxiv_graph_build"
    database: str = "arxiv_repository"
    total_buckets: int = 64
    shard_count: Optional[int] = None
    num_workers: int = 16  # inherited by WorkflowConfig, used as shard runner concurrency
    rate_limit: Optional[int] = None
    top_k: int = 10
    base_weight: float = 0.5
    semantic_top_k: int = 10
    semantic_score_threshold: float = 0.7
    semantic_embed_source: str = "jina_v4"
    semantic_snapshot_id: str = "snapshot-unknown"
    lease_store_factory: Optional[LeaseStoreFactory] = None
    token_bucket_factory: Optional[TokenBucketFactory] = None
    redis_socket_path: Optional[str] = "/run/redis/redis-server.sock"

    def build_lease_store(self) -> LeaseStore:
        if self.lease_store_factory is not None:
            return self.lease_store_factory()
        socket_path = self.redis_socket_path
        if socket_path and redis is not None:
            try:
                probe = redis.Redis(unix_socket_path=socket_path)
                probe.ping()
                probe.close()
                return RedisLeaseStore(
                    socket_path=socket_path,
                    namespace=f"hades:{self.name}:lease",
                )
            except Exception:
                pass
        # Fallback: in-memory store for local runs (non-durable)
        return InMemoryLeaseStore()

    def build_token_buckets(self) -> Dict[str, TokenBucket]:
        if self.token_bucket_factory is not None:
            return self.token_bucket_factory()
        if self.rate_limit is None or self.rate_limit <= 0:
            return {"default": NullTokenBucket()}
        socket_path = self.redis_socket_path
        if socket_path and redis is not None:
            try:
                probe = redis.Redis(unix_socket_path=socket_path)
                probe.ping()
                probe.close()
                bucket = RedisTokenBucket(
                    socket_path=socket_path,
                    bucket_id=f"{self.name}:default",
                    capacity=self.rate_limit,
                )
                return {"default": bucket}
            except Exception:
                pass
        return {"default": NullTokenBucket()}


class ArxivGraphBuildWorkflow(WorkflowBase):
    """Bootstrap the HiRAG graph for the arXiv repository database."""

    def __init__(self, config: Optional[ArxivGraphBuildConfig] = None) -> None:
        super().__init__(config or ArxivGraphBuildConfig())
        if not isinstance(self.config, ArxivGraphBuildConfig):
            raise TypeError("config must be ArxivGraphBuildConfig")

    # ------------------------------------------------------------------
    def validate_inputs(self, **kwargs) -> bool:
        logger.info("Validating prerequisites for arXiv graph build")
        client = DatabaseFactory.get_arango_memory_service(database=self.config.database)
        try:
            docs = client.execute_query("FOR doc IN arxiv_metadata LIMIT 1 RETURN doc._key")
            if not docs:
                logger.error("arxiv_metadata collection is empty")
                return False
            return True
        finally:
            client.close()

    # ------------------------------------------------------------------
    def execute(self, **kwargs) -> WorkflowResult:
        start_time = datetime.now(timezone.utc)
        total_partitions = self.config.total_buckets
        items_processed = 0
        errors: List[str] = []
        metadata: Dict[str, object] = {}

        try:
            self._ensure_graph_schema()
            adapter = ArxivPartitionAdapter(
                total_buckets=self.config.total_buckets,
                shard_count=self.config.shard_count,
            )
            partitions = adapter.describe_partitions()
            total_partitions = len(partitions)

            phase_results: Dict[str, List[PartitionResult]] = {}

            for phase_name, job in self._phases():
                logger.info("Starting phase %s", phase_name)
                lease_store = self.config.build_lease_store()
                token_buckets = self.config.build_token_buckets()
                runner = ShardRunner(
                    adapter,
                    job,
                    lease_store=lease_store,
                    concurrency=self.config.num_workers,
                    max_retries=2,
                    lease_ttl=180.0,
                    token_buckets=token_buckets,
                    metrics=self._emit_metric,
                )

                results = self._run_async(runner.run(partitions=partitions))
                phase_results[phase_name] = results
                items_processed += len(results)

                close_fn = getattr(lease_store, "close", None)
                if callable(close_fn):
                    close_fn()
                for bucket in token_buckets.values():
                    close_bucket = getattr(bucket, "close", None)
                    if callable(close_bucket):
                        close_bucket()

                if len(results) != total_partitions:
                    skipped = total_partitions - len(results)
                    errors.append(f"phase {phase_name} completed {len(results)}/{total_partitions} partitions; {skipped} skipped")
                    logger.warning(errors[-1])

            metadata["phase_results"] = {
                phase: {
                    "count": len(results),
                    "rows_written": sum(r.rows_written for r in results),
                }
                for phase, results in phase_results.items()
            }

            success = not errors
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("ArxivGraphBuildWorkflow failed")
            errors.append(str(exc))
            success = False
        end_time = datetime.now(timezone.utc)

        return WorkflowResult(
            workflow_name=self.name,
            success=success,
            items_processed=items_processed,
            items_failed=(total_partitions * len(self._phase_names()) - items_processed),
            start_time=start_time,
            end_time=end_time,
            metadata=metadata,
            errors=errors,
        )

    # ------------------------------------------------------------------
    def _ensure_graph_schema(self) -> None:
        logger.info("Ensuring HiRAG schema exists for database %s", self.config.database)
        client = DatabaseFactory.get_arango_memory_service(database=self.config.database)
        try:
            service = HiragService(client, schema_manager=GraphSchemaManager(client))
            service.ensure_schema()
        finally:
            client.close()

    def _phase_names(self) -> List[str]:
        return ["entities", "relations", "hierarchy", "semantic"]

    def _phases(self) -> Iterable[tuple[str, object]]:
        jobs = [
            ("entities", HiragEntityIngestJob(database=self.config.database)),
            (
                "relations",
                HiragRelationsJob(
                    top_k=self.config.top_k,
                    base_weight=self.config.base_weight,
                    database=self.config.database,
                ),
            ),
            ("hierarchy", HiragHierarchyJob(database=self.config.database)),
            (
                "semantic",
                HiragSemanticEdgesJob(
                    top_k=self.config.semantic_top_k,
                    score_threshold=self.config.semantic_score_threshold,
                    embed_source=self.config.semantic_embed_source,
                    snapshot_id=self.config.semantic_snapshot_id,
                    database=self.config.database,
                ),
            ),
        ]
        return jobs

    def _run_async(self, coro):
        return asyncio.run(coro)

    def _emit_metric(self, name: str, spec: PartitionSpec, payload: Mapping[str, object]) -> None:
        logger.debug("%s %s %s", name, spec.id, payload)
