# HiRAG — Hierarchical Graph Structure on ArangoDB (Phase 2)

## Summary

Define and operate the hierarchical structure (HiRAG) of the unified Arango graph for the `arxiv_repository` database. HiRAG covers collections, edge types, indices, graph creation, and hierarchy construction (topics → super‑topics), plus provenance and operational guardrails. Retrieval logic (PathRAG) is specified separately.

## Collections

```
papers_raw       // raw arXiv papers & metadata (late‑chunked refs; not traversed)
entities         // extracted entities (paper|concept|method|person|code_symbol)
clusters         // hierarchy summary nodes (L1 topics, L2 super‑topics)
cluster_membership_journal // membership changes for reproducibility/rollback
weight_config    // relation type/base priors and learning snapshots
query_logs       // retrieval traces for audits (shared)
```

## Edge Collections

```
relations       // entity↔entity (cites|implements|extends|refers_to|coref|derives_from)
cluster_edges   // member↔cluster and cluster↔cluster links (SUBTOPIC_OF, PART_OF)
```

## Indices

- `entities`: persistent on `type`, `layer`, `cluster_id`; optional unique `(name_lower, type)`; persistent on `deg_out` if used for beam demotion.
- `relations`: persistent on `type`, `layer_bridge`; edge index on `_from`, `_to`.
- `clusters`: persistent on `level` (1/2), `centroid_version`.

Directionality conventions

- `cites`: paper → paper (citer → cited)
- `implements`: paper/method → code_symbol
- `derives_from`: method_new → method_base
- `refers_to`: entity_a → entity_b (mention/reference)
- `extends`: method_new → method_base
- `coref`: mention → canonical entity (non‑bridge)

## Sharding

- Hash shard by `_key` or `community_id` (from clustering) for `entities` and `relations` to limit cross‑shard traversals.

## Hierarchy Construction (HiIndex)

1) Level 0 → Level 1 (Topic clusters)
- Input: `entities.layer=0.embedding` (Jina v4 per ingestion PRD).
- Method: HDBSCAN (cosine), `min_cluster_size≈5`, `min_samples≈3`.
- Output: create `clusters` (level=1) with centroid embeddings and abstractive summaries.

2) Structural refinement
- Build 1–2 hop neighborhoods via `relations`; run Louvain/Leiden; reconcile with embedding clusters via consensus labels; adjust memberships.

3) Level 1 → Level 2 (Super‑topics)
- Agglomerative over L1 centroids; enforce max fan‑out/depth limits for traversal.

4) Cluster edges
- `cluster_edges`: member→cluster (entities→L1), cluster→super (L1→L2), optional cross‑links between L1 siblings for known taxonomies.

5) Bridge labeling
- Mark `relations.layer_bridge=true` if edge crosses levels (0↔1, 1↔2) or joins distant communities with high semantic similarity/provenance.

## Graph Creation Snippet (arangosh)

```js
const graphName = 'HADES_KG';
if (!db._collection('entities')) db._createDocumentCollection('entities');
if (!db._collection('clusters')) db._createDocumentCollection('clusters');
if (!db._collection('relations')) db._createEdgeCollection('relations');
if (!db._collection('cluster_edges')) db._createEdgeCollection('cluster_edges');

const graphDef = [
  { collection: 'relations', from: ['entities'], to: ['entities'] },
  { collection: 'cluster_edges', from: ['entities','clusters'], to: ['clusters'] }
];
if (!db._graphs().some(g => g._key === graphName)) {
  db._createGraph(graphName, graphDef);
}
```

## Ops & Guardrails

- No secrets in source; sockets via env (`ARANGO_RO_UDS`, `ARANGO_RW_UDS`).
- Timeouts and circuit breakers match memory client defaults.
- Provenance: keep `cluster_membership_journal` and `weight_config` snapshots for rollback.

## QA & Monitoring

- Weekly Bridge QA job samples `relations` where `layer_bridge=true`, verifies supporting provenance (citations, semantic score ≥τ) and records findings in `cluster_membership_journal`. Failing samples demote the relation’s prior and emit alerts to protect Conveyance zero-propagation.
- Code edges (`CALLS/IMPORTS`) carry GitHub processor provenance: `{repo, commit_sha, language_parser}`; store linked snapshot IDs so time-travel audits can recreate call graphs.
- Surface bridge QA metrics (`bridge.qa_failure_rate`, `bridge.audit_sample_size`) in Conveyance dashboards alongside hierarchy counts.

## Acceptance (HiRAG)

- Collections/indices created; L1/L2 hierarchy built; schema conformance ≥99.5%; counts within ±2%; journaling enabled.
- Bridge QA job operational with failure rate ≤1 % and automated demotion of unverifiable edges.
