"""Helpers that populate HiRAG graph collections."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from core.database.arango.memory_client import ArangoMemoryClient

from core.graph.schema import GraphCollections

logger = logging.getLogger(__name__)

_SAFE_KEY_PATTERN = re.compile(r"[^a-zA-Z0-9_.:@-]")


def _sanitize_key(value: str | None) -> str:
    if not value:
        return ""
    return _SAFE_KEY_PATTERN.sub("_", value)


def _chunk_iterable(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _sum_result(result: list[Any]) -> int:
    return int(sum(result)) if result else 0


def ingest_entities_from_arxiv(
    client: ArangoMemoryClient,
    collections: GraphCollections,
    *,
    limit: int | None = None,
    hash_mod: int | None = None,
    hash_low: int | None = None,
    hash_high: int | None = None,
    keys: Sequence[str] | None = None,
) -> int:
    """Upsert paper entities from ``arxiv_metadata`` into ``entities``."""

    limit_clause = ""
    bind: dict[str, object] = {}
    if limit is not None:
        limit_clause = "LIMIT @limit"
        bind["limit"] = limit

    hash_clause = ""
    if hash_mod is not None and hash_low is not None:
        hash_clause = "FILTER HASH(doc._key) % @hash_mod == @hash_low"

    key_filter_clause = ""
    if keys:
        key_filter_clause = "FILTER doc._key IN @keys"

    query = f"""
    LET now = DATE_ISO8601(DATE_NOW())
    FOR doc IN arxiv_metadata
      FILTER doc.title != null
      {hash_clause}
      {key_filter_clause}
      SORT doc._key
      {limit_clause}
      LET base_key = doc._key != null ? doc._key : doc.arxiv_id
      LET safe_key = REGEX_REPLACE(base_key, '[^a-zA-Z0-9_.:@-]', '_', true)
      LET key = CONCAT('paper_', safe_key)
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
      COLLECT AGGREGATE num_written = LENGTH(1)
    RETURN {{ count: num_written }}
    """

    logger.debug("Upserting entities from arxiv_metadata", extra={"limit": limit})
    if hash_mod is not None and hash_low is not None:
        bind["hash_mod"] = hash_mod
        bind["hash_low"] = hash_low
    if keys:
        bind["keys"] = list(keys)

    result = client.execute_write_query(query, bind)
    return _sum_result([row.get("count", 0) for row in result])


def build_semantic_relations(
    client: ArangoMemoryClient,
    collections: GraphCollections,
    *,
    top_k: int,
    base_weight: float,
    limit: int | None = None,
    hash_mod: int | None = None,
    hash_low: int | None = None,
    hash_high: int | None = None,
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

    hash_clause = ""
    if hash_mod is not None and hash_low is not None:
        hash_clause = "FILTER HASH(doc._key) % @hash_mod == @hash_low"

    query = f"""
    LET now = DATE_ISO8601(DATE_NOW())
    LET total = SUM(
      FOR doc IN arxiv_metadata
        {hash_clause}
        SORT doc._key
        {limit_clause}
        LET cat = doc.primary_category != null ? doc.primary_category : (doc.categories != null && LENGTH(doc.categories) > 0 ? doc.categories[0] : null)
        FILTER cat != null
        LET from_base = doc._key != null ? doc._key : doc.arxiv_id
        LET from_safe = REGEX_REPLACE(from_base, '[^a-zA-Z0-9_.:@-]', '_', true)
        LET from_key = CONCAT('paper_', from_safe)
        LET from_doc = DOCUMENT(@entities_collection, from_key)
        FILTER from_doc != null
        LET neighbors = (
          FOR other IN arxiv_metadata
            LET other_cat = other.primary_category != null ? other.primary_category : (other.categories != null && LENGTH(other.categories) > 0 ? other.categories[0] : null)
            FILTER other_cat == cat
            FILTER other._key != doc._key
            SORT other.update_date DESC, other._key
            LIMIT @top_k
            RETURN other._key
        )
        FOR neighbor_id IN neighbors
          LET neighbor_safe = REGEX_REPLACE(neighbor_id, '[^a-zA-Z0-9_.:@-]', '_', true)
          LET neighbor_key = CONCAT('paper_', neighbor_safe)
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
    )
    RETURN {{ count: total }}
    """

    logger.debug("Building bootstrap relations", extra={"top_k": top_k, "limit": limit})
    if hash_mod is not None and hash_low is not None:
        bind["hash_mod"] = hash_mod
        bind["hash_low"] = hash_low

    result = client.execute_write_query(query, bind)
    return _sum_result([row.get("count", 0) for row in result])


def build_semantic_similarity_edges(
    client: ArangoMemoryClient,
    collections: GraphCollections,
    *,
    top_k: int,
    score_threshold: float,
    embed_source: str,
    snapshot_id: str,
    hash_mod: int | None = None,
    hash_low: int | None = None,
    hash_high: int | None = None,
) -> int:
    """Construct semantic edges using FAISS GPU cosine similarity."""

    hash_clause = ""
    bind: dict[str, object] = {}
    if hash_mod is not None and hash_low is not None:
        hash_clause = "FILTER HASH(doc.paper_key) % @hash_mod == @hash_low"
        bind["hash_mod"] = hash_mod
        bind["hash_low"] = hash_low

    logger.debug(
        "Loading embeddings for semantic similarity",
        extra={"top_k": top_k, "hash_mod": hash_mod, "hash_low": hash_low},
    )

    embeddings_query = f"""
    FOR doc IN arxiv_abstract_embeddings
      FILTER doc.embedding != null
      FILTER doc.paper_key != null
      {hash_clause}
      RETURN {{
        paper_key: doc.paper_key,
        chunk_id: doc.chunk_id,
        embedding: doc.embedding
      }}
    """

    rows = client.execute_query(embeddings_query, bind, batch_size=5000)
    if not rows:
        logger.debug("No embeddings found for semantic phase", extra={"hash_low": hash_low})
        return 0

    sanitized_keys = [_sanitize_key(row.get("paper_key")) for row in rows]
    chunk_ids = [row.get("chunk_id") for row in rows]
    embeddings = np.array([row.get("embedding", []) for row in rows], dtype=np.float32)

    neighbors = _compute_semantic_neighbors(embeddings, top_k, score_threshold)
    if not neighbors:
        logger.debug("No semantic neighbors above threshold", extra={"hash_low": hash_low})
        return 0

    edges = _prepare_semantic_edges(
        neighbors,
        sanitized_keys,
        chunk_ids,
        embeddings.shape[1],
        embed_source,
        snapshot_id,
        collections,
    )

    total_written = _upsert_semantic_edges(client, edges, collections)
    logger.debug(
        "Written semantic edges",
        extra={"count": total_written, "hash_low": hash_low},
    )
    return total_written


def build_category_hierarchy(
    client: ArangoMemoryClient,
    collections: GraphCollections,
    *,
    limit: int | None = None,
    hash_mod: int | None = None,
    hash_low: int | None = None,
    hash_high: int | None = None,
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

    hash_clause = ""
    category_bind: dict[str, object] = {}
    if hash_mod is not None and hash_low is not None:
        hash_clause = "FILTER HASH(doc._key) % @hash_mod == @hash_low"
        bind["hash_mod"] = hash_mod
        bind["hash_low"] = hash_low
        category_bind["hash_mod"] = hash_mod
        category_bind["hash_low"] = hash_low
    if limit is not None:
        category_bind["limit"] = limit

    category_query = f"""
    FOR doc IN arxiv_metadata
      {hash_clause}
      {limit_clause}
      LET cat = doc.primary_category != null ? doc.primary_category : (doc.categories != null && LENGTH(doc.categories) > 0 ? doc.categories[0] : null)
      FILTER cat != null
      COLLECT category = cat WITH COUNT INTO count
      RETURN {{ category, count }}
    """

    logger.debug("Building category hierarchy", extra={"limit": limit})
    categories = client.execute_query(category_query, category_bind)

    cluster_links = 0
    if categories:
        rows_bind = {"rows": categories}
        super_query = f"""
        FOR row IN @rows
          LET now = DATE_ISO8601(DATE_NOW())
          LET super_raw = FIRST(SPLIT(row.category, '.'))
          LET super_sanitized = REGEX_REPLACE(super_raw, '[^a-zA-Z0-9_]', '_', true)
          LET super_key = CONCAT('super_', super_sanitized)
          UPSERT {{ _key: super_key }}
            INSERT {{
              _key: super_key,
              name: super_raw,
              summary: super_raw,
              level: 2,
              created_at: now,
              updated_at: now
            }}
            UPDATE {{
              name: super_raw,
              summary: super_raw,
              updated_at: now
            }}
          IN {collections.clusters}
          OPTIONS {{ keepNull: false }}
        """

        cluster_doc_query = f"""
        FOR row IN @rows
          LET now = DATE_ISO8601(DATE_NOW())
          LET sanitized = REGEX_REPLACE(row.category, '[^a-zA-Z0-9_]', '_', true)
          LET cluster_key = CONCAT('cat_', sanitized)
          LET super_raw = FIRST(SPLIT(row.category, '.'))
          LET super_sanitized = REGEX_REPLACE(super_raw, '[^a-zA-Z0-9_]', '_', true)
          LET super_key = CONCAT('super_', super_sanitized)
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
        """

        edge_query = f"""
        FOR row IN @rows
          LET now = DATE_ISO8601(DATE_NOW())
          LET sanitized = REGEX_REPLACE(row.category, '[^a-zA-Z0-9_]', '_', true)
          LET cluster_key = CONCAT('cat_', sanitized)
          LET super_raw = FIRST(SPLIT(row.category, '.'))
          LET super_sanitized = REGEX_REPLACE(super_raw, '[^a-zA-Z0-9_]', '_', true)
          LET super_key = CONCAT('super_', super_sanitized)
          LET cluster_id = CONCAT(@clusters_prefix, cluster_key)
          LET super_id = CONCAT(@clusters_prefix, super_key)
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
          COLLECT AGGREGATE num_edges = LENGTH(1)
        RETURN {{ count: num_edges }}
        """

        client.execute_write_query(super_query, rows_bind)
        client.execute_write_query(cluster_doc_query, rows_bind)
        edge_result = client.execute_write_query(edge_query, {"rows": categories, "clusters_prefix": f"{collections.clusters}/"})
        cluster_links = _sum_result([row.get("count", 0) for row in edge_result])

    membership_query = f"""
    LET now = DATE_ISO8601(DATE_NOW())
    FOR doc IN arxiv_metadata
      {hash_clause}
      SORT doc._key
      {limit_clause}
      LET cat = doc.primary_category != null ? doc.primary_category : (doc.categories != null && LENGTH(doc.categories) > 0 ? doc.categories[0] : null)
      FILTER cat != null
      LET entity_base = doc._key != null ? doc._key : doc.arxiv_id
      LET entity_safe = REGEX_REPLACE(entity_base, '[^a-zA-Z0-9_.:@-]', '_', true)
      LET entity_key = CONCAT('paper_', entity_safe)
      LET entity_doc = DOCUMENT(@entities_collection, entity_key)
      FILTER entity_doc != null
      LET sanitized = REGEX_REPLACE(cat, '[^a-zA-Z0-9_]', '_', true)
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
      RETURN {{ count: 1 }}
    """

    membership_result = client.execute_write_query(membership_query, bind)
    return {
        "cluster_links": cluster_links,
        "membership_edges": _sum_result([row.get("count", 0) for row in membership_result]),
    }


def _compute_semantic_neighbors(
    embeddings: np.ndarray,
    top_k: int,
    threshold: float,
) -> list[tuple[int, int, float]]:
    n = embeddings.shape[0]
    if n <= 1:
        return []

    actual_k = min(top_k, n - 1)
    if actual_k <= 0:
        return []

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Semantic similarity requires PyTorch to run on the GPU; install torch before enabling the semantic phase."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch could not find a CUDA device. Ensure the GPU drivers/toolkit are installed and the process can access them."
        )

    device = torch.device("cuda")
    tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    tensor = torch.nn.functional.normalize(tensor, dim=1)

    neighbors: list[tuple[int, int, float]] = []
    chunk_size = 2048
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_tensor = tensor[start:end]
        sims = torch.matmul(chunk_tensor, tensor.T)
        row_indices = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), row_indices] = -1.0
        topk_scores, topk_indices = torch.topk(sims, k=actual_k, dim=1)
        topk_scores = topk_scores.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()
        for row_offset, (scores_row, idx_row) in enumerate(zip(topk_scores, topk_indices)):
            i = start + row_offset
            for score, j in zip(scores_row, idx_row):
                if score < threshold or j < 0:
                    continue
                neighbors.append((i, int(j), float(score)))

    return neighbors

def _prepare_semantic_edges(
    neighbors: list[tuple[int, int, float]],
    sanitized_keys: Sequence[str],
    chunk_ids: Sequence[Any],
    embedding_dim: int,
    embed_source: str,
    snapshot_id: str,
    collections: GraphCollections,
) -> list[dict[str, Any]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    entities_prefix = f"{collections.entities}/"
    edges: list[dict[str, Any]] = []
    for from_idx, to_idx, score in neighbors:
        from_safe = sanitized_keys[from_idx]
        to_safe = sanitized_keys[to_idx]
        if not from_safe or not to_safe:
            continue
        from_key = f"paper_{from_safe}"
        to_key = f"paper_{to_safe}"
        edge_key = f"{from_key}::semantic::{to_key}"
        edges.append(
            {
                "_key": edge_key,
                "_from": f"{entities_prefix}{from_key}",
                "_to": f"{entities_prefix}{to_key}",
                "type": "semantic_similar_to",
                "weight": score,
                "weight_components": {
                    "sim": score,
                    "type_prior": score,
                    "evidence": 0.0,
                    "freshness": 0.0,
                },
                "embed_source": embed_source,
                "embed_dim": embedding_dim,
                "snapshot_id": snapshot_id,
                "score": score,
                "quality": score,
                "recency": 0.0,
                "metadata": {
                    "from_chunk": chunk_ids[from_idx],
                    "to_chunk": chunk_ids[to_idx],
                },
                "ts": now_iso,
            }
        )
    return edges


def _upsert_semantic_edges(
    client: ArangoMemoryClient,
    edges: list[dict[str, Any]],
    collections: GraphCollections,
    *,
    chunk_size: int = 2000,
) -> int:
    if not edges:
        return 0

    query = f"""
    LET now = DATE_ISO8601(DATE_NOW())
    FOR edge IN @edges
      UPSERT {{ _key: edge._key }}
        INSERT MERGE(edge, {{ created: now, ts: now }})
        UPDATE {{
          weight: edge.weight,
          weight_components: MERGE(OLD.weight_components, edge.weight_components),
          score: edge.score,
          quality: edge.quality,
          embed_source: edge.embed_source,
          snapshot_id: edge.snapshot_id,
          ts: now,
          metadata: edge.metadata
        }}
      IN {collections.relations}
      OPTIONS {{ keepNull: false }}
    """

    total = 0
    for batch in _chunk_iterable(edges, chunk_size):
        client.execute_write_query(query, {"edges": batch})
        total += len(batch)
    return total
