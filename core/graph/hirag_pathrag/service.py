"""Service scaffolding for building and querying the HiRAG × PathRAG graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence

from core.database.arango.memory_client import ArangoMemoryClient

from . import aql
from .builders import build_category_hierarchy, build_semantic_relations, ingest_entities_from_arxiv
from .schema_manager import GraphSchemaManager, GraphCollections

logger = logging.getLogger(__name__)


def _relation_boost_defaults() -> dict[str, float]:
    return {
        "refers_to": 0.8,
        "implements": 0.7,
        "cites": 0.6,
        "derives_from": 0.6,
        "extends": 0.5,
        "coref": 0.4,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

@dataclass(slots=True)
class HiragPathragConfig:
    """Runtime parameters for hierarchy narrowing and path extraction."""

    topic_k: int = 8
    beam: Sequence[int] = (8, 4, 2)
    candidate_cap: int = 500
    deg_alpha: float = 0.005
    path_max_paths: int = 6
    path_max_hops: int = 5
    path_lambda: float = 0.3
    path_epsilon: float = 1e-6
    relation_boost: Mapping[str, float] = field(default_factory=_relation_boost_defaults)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "HiragPathragConfig":
        data = dict(payload)
        relation_boost = data.get("relation_boost")
        if relation_boost is not None:
            data["relation_boost"] = dict(relation_boost)  # shallow copy
        return cls(**data)


class HiragPathragService:
    """Coordinates schema management and graph workflows."""

    def __init__(
        self,
        client: ArangoMemoryClient,
        *,
        config: HiragPathragConfig | None = None,
        schema_manager: GraphSchemaManager | None = None,
    ) -> None:
        self._client = client
        self._config = config or HiragPathragConfig()
        self._schema_manager = schema_manager or GraphSchemaManager(client)

    # ------------------------------------------------------------------
    def ensure_schema(self) -> None:
        self._schema_manager.ensure_schema()

    # ------------------------------------------------------------------
    def build_entities_from_arxiv(self, *, limit: int | None = None) -> int:
        """Upsert paper entities from the arXiv corpus."""

        return ingest_entities_from_arxiv(self._client, self._schema_manager.collections, limit=limit)

    def build_relations_from_embeddings(
        self, *, top_k: int = 10, base_weight: float = 0.5, limit: int | None = None
    ) -> int:
        """Bootstrap semantic relations based on shared categories."""

        return build_semantic_relations(
            self._client,
            self._schema_manager.collections,
            top_k=top_k,
            base_weight=base_weight,
            limit=limit,
        )

    def build_hierarchy(self, *, limit: int | None = None) -> Mapping[str, int]:
        """Construct L1/L2 clusters from primary categories."""

        return build_category_hierarchy(
            self._client,
            self._schema_manager.collections,
            limit=limit,
        )

    # ------------------------------------------------------------------
    def get_candidate_subgraph(
        self,
        *,
        query_embedding: Sequence[float],
        allowed_types: Sequence[str] = ("paper", "concept", "method", "code_symbol"),
        topic_threshold: float = 0.5,
    ) -> list[dict[str, object]]:
        """Run the HiRAG narrowing query and return candidate vertices."""

        config = aql.HierarchyTraversalConfig(
            topic_k=self._config.topic_k,
            beam_widths=self._config.beam,
            candidate_cap=self._config.candidate_cap,
            deg_alpha=self._config.deg_alpha,
        )
        query = aql.build_hierarchy_candidate_query(config)
        bind_vars = {
            "query_embedding": list(query_embedding),
            "topic_threshold": topic_threshold,
            "graph_name": self._schema_manager.graph_name,
            "allowed_types": list(allowed_types),
        }
        logger.debug("Executing hierarchy candidate query")
        return self._client.execute_query(query, bind_vars)

    def get_weighted_paths(
        self,
        *,
        source_id: str,
        target_id: str,
        epsilon: float | None = None,
    ) -> list[dict[str, object]]:
        """Run the PathRAG weighted shortest-path query."""

        config = aql.PathExtractionConfig(
            max_paths=self._config.path_max_paths,
            max_hops=self._config.path_max_hops,
            lambda_term=self._config.path_lambda,
        )
        query = aql.build_weighted_path_query(config)
        bind_vars = {
            "source": source_id,
            "target": target_id,
            "graph_name": self._schema_manager.graph_name,
            "relation_boost": dict(self._config.relation_boost),
            "epsilon": epsilon if epsilon is not None else self._config.path_epsilon,
        }
        logger.debug("Executing weighted path query", extra={"source": source_id, "target": target_id})
        return self._client.execute_query(query, bind_vars)

    # ------------------------------------------------------------------
    def log_query_trace(self, payload: Mapping[str, object]) -> None:
        """Persist a query trace document in the query log collection."""

        doc = dict(payload)
        doc.setdefault('created_at', _utc_now_iso())
        query = f'INSERT @doc INTO {self._schema_manager.collections.query_logs} RETURN NEW._key'
        try:
            self._client.execute_query(query, {'doc': doc})
        except Exception:  # pragma: no cover - defensive logging
            logger.exception('Failed to record query trace')

    # ------------------------------------------------------------------
    @property
    def config(self) -> HiragPathragConfig:
        return self._config

    @property
    def collections(self) -> GraphCollections:
        return self._schema_manager.collections

    @property
    def graph_name(self) -> str:
        return self._schema_manager.graph_name

