"""Service scaffolding for HiRAG graph construction and narrowing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

from core.database.arango.memory_client import ArangoMemoryClient

from core.graph.schema import GraphCollections, GraphSchemaManager

from . import aql
from .builders import build_category_hierarchy, build_semantic_relations, ingest_entities_from_arxiv

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HiragConfig:
    """Runtime parameters for hierarchy narrowing and graph builders."""

    topic_k: int = 8
    beam: Sequence[int] = (8, 4, 2)
    candidate_cap: int = 500
    deg_alpha: float = 0.005

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "HiragConfig":
        data = dict(payload)
        return cls(**data)


class HiragService:
    """Coordinates schema management and HiRAG graph workflows."""

    def __init__(
        self,
        client: ArangoMemoryClient,
        *,
        config: HiragConfig | None = None,
        schema_manager: GraphSchemaManager | None = None,
    ) -> None:
        self._client = client
        self._config = config or HiragConfig()
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

    # ------------------------------------------------------------------
    @property
    def config(self) -> HiragConfig:
        return self._config

    @property
    def collections(self) -> GraphCollections:
        return self._schema_manager.collections

    @property
    def graph_name(self) -> str:
        return self._schema_manager.graph_name
