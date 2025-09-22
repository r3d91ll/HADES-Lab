"""Shared schema management helpers for graph services."""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass

from core.database.arango.memory_client import (
    ArangoMemoryClient,
    CollectionDefinition,
    MemoryServiceError,
)

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_NAME = "HADES_KG"
DEFAULT_DATABASE = "arxiv_repository"


@dataclass(frozen=True)
class GraphCollections:
    """Canonical collection names backing the Arango knowledge graph."""

    entities: str = "entities"
    clusters: str = "clusters"
    relations: str = "relations"
    cluster_edges: str = "cluster_edges"
    bridge_cache: str = "bridge_cache"
    weight_config: str = "weight_config"
    query_logs: str = "query_logs"
    cluster_membership_journal: str = "cluster_membership_journal"
    papers_raw: str = "papers_raw"


class GraphSchemaManager:
    """Ensure required collections, indices, and named graph exist."""

    def __init__(
        self,
        client: ArangoMemoryClient,
        *,
        database: str | None = None,
        graph_name: str = DEFAULT_GRAPH_NAME,
        collections: GraphCollections | None = None,
    ) -> None:
        self._client = client
        self._database = database or getattr(client, "_config").database
        self._graph_name = graph_name
        self._collections = collections or GraphCollections()

    # ------------------------------------------------------------------
    def ensure_schema(self) -> None:
        """Ensure all collections, indices, and named graph exist."""

        self._ensure_collections()
        self._ensure_named_graph()

    # ------------------------------------------------------------------
    def _ensure_collections(self) -> None:
        definitions = [
            CollectionDefinition(
                name=self._collections.entities,
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["type"]},
                    {"type": "persistent", "fields": ["layer"]},
                    {"type": "persistent", "fields": ["cluster_id"]},
                    {
                        "type": "persistent",
                        "fields": ["name_lower", "type"],
                        "unique": True,
                        "sparse": True,
                    },
                    {"type": "persistent", "fields": ["deg_out"]},
                ],
            ),
            CollectionDefinition(
                name=self._collections.clusters,
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["level"]},
                    {"type": "persistent", "fields": ["centroid_version"]},
                ],
            ),
            CollectionDefinition(
                name=self._collections.bridge_cache,
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["query_hash"], "unique": True},
                    {"type": "persistent", "fields": ["expires_at"]},
                ],
            ),
            CollectionDefinition(
                name=self._collections.weight_config,
                type="document",
                indexes=[{"type": "persistent", "fields": ["snapshot_id"], "unique": True}],
            ),
            CollectionDefinition(
                name=self._collections.query_logs,
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["request_id"]},
                    {"type": "persistent", "fields": ["created_at"]},
                ],
            ),
            CollectionDefinition(
                name=self._collections.cluster_membership_journal,
                type="document",
                indexes=[
                    {"type": "persistent", "fields": ["entity_id"]},
                    {"type": "persistent", "fields": ["cluster_id"]},
                    {"type": "persistent", "fields": ["timestamp"]},
                ],
            ),
            CollectionDefinition(
                name=self._collections.papers_raw,
                type="document",
                indexes=[{"type": "persistent", "fields": ["arxiv_id"], "unique": True}],
            ),
            CollectionDefinition(
                name=self._collections.relations,
                type="edge",
                indexes=[
                    {"type": "persistent", "fields": ["type"]},
                    {"type": "persistent", "fields": ["layer_bridge"]},
                ],
            ),
            CollectionDefinition(
                name=self._collections.cluster_edges,
                type="edge",
                indexes=[
                    {"type": "persistent", "fields": ["edge_type"]},
                ],
            ),
        ]

        try:
            self._client.create_collections(definitions)
        except MemoryServiceError:  # pragma: no cover - defensive logging
            logger.exception("Failed to ensure graph collections")
            raise

    def _ensure_named_graph(self) -> None:
        edge_definitions = [
            {
                "collection": self._collections.relations,
                "from": [self._collections.entities],
                "to": [self._collections.entities],
            },
            {
                "collection": self._collections.cluster_edges,
                "from": [self._collections.entities, self._collections.clusters],
                "to": [self._collections.clusters],
            },
        ]

        orphan_collections: list[str] = [
            self._collections.bridge_cache,
            self._collections.weight_config,
            self._collections.query_logs,
            self._collections.cluster_membership_journal,
            self._collections.papers_raw,
        ]

        action = textwrap.dedent(
            """
            function (params) {
              const db = require('@arangodb').db;
              const name = params.graphName;
              const edgeDefinitions = params.edgeDefinitions;
              const orphans = params.orphanCollections;
              if (!db._graphs().some(g => g._key === name)) {
                db._createGraph(name, edgeDefinitions, orphans);
              }
              return true;
            }
            """
        )

        params = {
            "graphName": self._graph_name,
            "edgeDefinitions": edge_definitions,
            "orphanCollections": orphan_collections,
        }

        try:
            self._client.execute_transaction(
                write=["_graphs", self._collections.relations, self._collections.cluster_edges],
                action=action,
                params=params,
                wait_for_sync=True,
            )
        except MemoryServiceError as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to ensure named graph '%s'", self._graph_name)
            raise

    # ------------------------------------------------------------------
    @property
    def database(self) -> str:
        return self._database

    @property
    def graph_name(self) -> str:
        return self._graph_name

    @property
    def collections(self) -> GraphCollections:
        return self._collections
