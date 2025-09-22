"""Service scaffolding for PathRAG graph retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

from core.database.arango.memory_client import ArangoMemoryClient

from core.graph.schema import GraphCollections, GraphSchemaManager

from . import aql

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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class PathragConfig:
    """Runtime parameters for path extraction."""

    path_max_paths: int = 6
    path_max_hops: int = 5
    path_lambda: float = 0.3
    path_epsilon: float = 1e-6
    relation_boost: Mapping[str, float] = field(default_factory=_relation_boost_defaults)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PathragConfig":
        data = dict(payload)
        relation_boost = data.get("relation_boost")
        if relation_boost is not None:
            data["relation_boost"] = dict(relation_boost)
        return cls(**data)


class PathragService:
    """Provides weighted path extraction and query logging."""

    def __init__(
        self,
        client: ArangoMemoryClient,
        *,
        config: PathragConfig | None = None,
        schema_manager: GraphSchemaManager | None = None,
    ) -> None:
        self._client = client
        self._config = config or PathragConfig()
        self._schema_manager = schema_manager or GraphSchemaManager(client)

    # ------------------------------------------------------------------
    def ensure_schema(self) -> None:
        self._schema_manager.ensure_schema()

    # ------------------------------------------------------------------
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

    def log_query_trace(self, payload: Mapping[str, object]) -> None:
        """Persist a query trace document in the query log collection."""

        doc = dict(payload)
        doc.setdefault("created_at", _utc_now_iso())
        query = f"INSERT @doc INTO {self._schema_manager.collections.query_logs} RETURN NEW._key"
        try:
            self._client.execute_query(query, {"doc": doc})
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to record query trace")

    # ------------------------------------------------------------------
    @property
    def config(self) -> PathragConfig:
        return self._config

    @property
    def collections(self) -> GraphCollections:
        return self._schema_manager.collections

    @property
    def graph_name(self) -> str:
        return self._schema_manager.graph_name
