"""Helpers that populate HiRAG × PathRAG graph collections."""

from __future__ import annotations

import logging
from typing import Mapping

from core.database.arango.memory_client import ArangoMemoryClient

from .schema_manager import GraphCollections

logger = logging.getLogger(__name__)


def _sum_result(result: list[int]) -> int:
    return int(sum(result)) if result else 0


def ingest_entities_from_arxiv(
    client: ArangoMemoryClient,
    collections: GraphCollections,
    *,
    limit: int | None = None,
) -> int:
    """Upsert paper entities from ``arxiv_metadata`` into ``entities``."""

    limit_clause = ""
    bind: dict[str, object] = {}
    if limit is not None:
        limit_clause = "LIMIT @limit"
        bind["limit"] = limit

    query = f"""
    LET now = DATE_ISO8601(DATE_NOW())
    FOR doc IN arxiv_metadata
      FILTER doc.title != null
      SORT doc.arxiv_id
      {limit_clause}
      LET key = CONCAT('paper_', doc.arxiv_id)
      LET lower_name = LOWER(doc.title)
      UPSERT {{ _key: key }}
        INSERT {{
          _key: key,
          type: 'paper',
          layer: 0,
          arxiv_id: doc.arxiv_id,
          name: doc.title,
          name_lower: lower_name,
          summary: doc.summary,
          categories: doc.categories,
          primary_category: doc.primary_category,
          deg_out: 0,
          cluster_id: null,
          created_at: now,
          updated_at: now
        }}
        UPDATE {{
          name: doc.title,
          name_lower: lower_name,
          summary: doc.summary,
          categories: doc.categories,
          primary_category: doc.primary_category,
          updated_at: now
        }}
      IN {collections.entities}
      OPTIONS {{ keepNull: false }}
      RETURN 1
    """

    logger.debug("Upserting entities from arxiv_metadata", extra={"limit": limit})
    result = client.execute_query(query, bind)
    return _sum_result(result)


def build_semantic_relations(
    client: ArangoMemoryClient,
    collections: GraphCollections,
    *,
    top_k: int,
    base_weight: float,
    limit: int | None = None,
) -> int:
    """Create simple co-category relations as a bootstrap semantic graph."""

    limit_clause = ""
    bind: dict[str, object] = {
        "top_k": top_k,
        "base_weight": base_weight,
        "entities_collection": collections.entities,
        "entities_prefix": f"{collections.entities}/",
    }
    if limit is not None:
        limit_clause = "LIMIT @limit"
        bind["limit"] = limit

    query = f"""
    LET now = DATE_ISO8601(DATE_NOW())
    FOR doc IN arxiv_metadata
      FILTER doc.primary_category != null
      SORT doc.arxiv_id
      {limit_clause}
      LET from_key = CONCAT('paper_', doc.arxiv_id)
      LET from_doc = DOCUMENT(@entities_collection, from_key)
      FILTER from_doc != null
      LET neighbors = (
        FOR other IN arxiv_metadata
          FILTER other.primary_category == doc.primary_category
          FILTER other.arxiv_id != doc.arxiv_id
          SORT other.updated DESC NULLS LAST, other.arxiv_id
          LIMIT @top_k
          RETURN other.arxiv_id
      )
      FOR neighbor_id IN neighbors
        LET neighbor_key = CONCAT('paper_', neighbor_id)
        LET neighbor_doc = DOCUMENT(@entities_collection, neighbor_key)
        FILTER neighbor_doc != null
        LET edge_key = CONCAT(from_key, '::', neighbor_key)
        UPSERT {{ _key: edge_key }}
          INSERT {{
            _key: edge_key,
            _from: CONCAT(@entities_prefix, from_key),
            _to: CONCAT(@entities_prefix, neighbor_key),
            type: 'refers_to',
            weight: @base_weight,
            weight_components: {{
              sim: 0.0,
              type_prior: @base_weight,
              evidence: 0.0,
              freshness: 0.0
            }},
            layer_bridge: false,
            created: now,
            updated: now
          }}
          UPDATE {{
            weight: @base_weight,
            updated: now,
            weight_components: MERGE(OLD.weight_components, {{ type_prior: @base_weight }})
          }}
        IN {collections.relations}
        OPTIONS {{ keepNull: false }}
        RETURN 1
    """

    logger.debug("Building bootstrap relations", extra={"top_k": top_k, "limit": limit})
    result = client.execute_query(query, bind)
    return _sum_result(result)


def build_category_hierarchy(
    client: ArangoMemoryClient,
    collections: GraphCollections,
    *,
    limit: int | None = None,
) -> Mapping[str, int]:
    """Create L1/L2 clusters based on primary categories and link entities."""

    limit_clause = ""
    bind: dict[str, object] = {
        "clusters_prefix": f"{collections.clusters}/",
        "entities_prefix": f"{collections.entities}/",
        "entities_collection": collections.entities,
    }
    if limit is not None:
        limit_clause = "LIMIT @limit"
        bind["limit"] = limit

    cluster_query = f"""
    LET now = DATE_ISO8601(DATE_NOW())
    FOR row IN (
      FOR doc IN arxiv_metadata
        FILTER doc.primary_category != null
        {limit_clause}
        COLLECT category = doc.primary_category WITH COUNT INTO count
        RETURN {{ category, count }}
    )
      LET sanitized = REGEX_REPLACE(row.category, '[^a-zA-Z0-9_]', '_', true)
      LET cluster_key = CONCAT('cat_', sanitized)
      LET super_raw = FIRST(SPLIT(row.category, '.'))
      LET super_sanitized = REGEX_REPLACE(super_raw, '[^a-zA-Z0-9_]', '_', true)
      LET super_key = CONCAT('super_', super_sanitized)
      LET super_name = super_raw
      LET cluster_id = CONCAT(@clusters_prefix, cluster_key)
      LET super_id = CONCAT(@clusters_prefix, super_key)
      UPSERT {{ _key: super_key }}
        INSERT {{
          _key: super_key,
          name: super_name,
          summary: super_name,
          level: 2,
          created_at: now,
          updated_at: now
        }}
        UPDATE {{
          name: super_name,
          summary: super_name,
          updated_at: now
        }}
      IN {collections.clusters}
      OPTIONS {{ keepNull: false }}
      UPSERT {{ _key: cluster_key }}
        INSERT {{
          _key: cluster_key,
          name: row.category,
          summary: row.category,
          level: 1,
          parent_super_key: super_key,
          size: row.count,
          created_at: now,
          updated_at: now
        }}
        UPDATE {{
          size: row.count,
          parent_super_key: super_key,
          updated_at: now
        }}
      IN {collections.clusters}
      OPTIONS {{ keepNull: false }}
      UPSERT {{ _key: CONCAT(cluster_key, '->', super_key) }}
        INSERT {{
          _key: CONCAT(cluster_key, '->', super_key),
          _from: cluster_id,
          _to: super_id,
          edge_type: 'SUBTOPIC_OF',
          created_at: now,
          updated_at: now
        }}
        UPDATE {{ updated_at: now }}
      IN {collections.cluster_edges}
      OPTIONS {{ keepNull: false }}
      RETURN 1
    """

    membership_query = f"""
    LET now = DATE_ISO8601(DATE_NOW())
    FOR doc IN arxiv_metadata
      FILTER doc.primary_category != null
      SORT doc.arxiv_id
      {limit_clause}
      LET entity_key = CONCAT('paper_', doc.arxiv_id)
      LET entity_doc = DOCUMENT(@entities_collection, entity_key)
      FILTER entity_doc != null
      LET sanitized = REGEX_REPLACE(doc.primary_category, '[^a-zA-Z0-9_]', '_', true)
      LET cluster_key = CONCAT('cat_', sanitized)
      LET cluster_id = CONCAT(@clusters_prefix, cluster_key)
      UPSERT {{ _key: CONCAT(entity_key, '->', cluster_key) }}
        INSERT {{
          _key: CONCAT(entity_key, '->', cluster_key),
          _from: CONCAT(@entities_prefix, entity_key),
          _to: cluster_id,
          edge_type: 'MEMBER_OF',
          created_at: now,
          updated_at: now
        }}
        UPDATE {{ updated_at: now }}
      IN {collections.cluster_edges}
      OPTIONS {{ keepNull: false }}
      UPDATE entity_doc WITH {{ cluster_id: cluster_id }} IN {collections.entities}
      OPTIONS {{ keepNull: false }}
      RETURN 1
    """

    logger.debug("Building category hierarchy", extra={"limit": limit})
    cluster_result = client.execute_query(cluster_query, bind)
    membership_result = client.execute_query(membership_query, bind)
    return {
        "cluster_links": _sum_result(cluster_result),
        "membership_edges": _sum_result(membership_result),
    }

