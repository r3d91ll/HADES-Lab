# Semantic Structure PRD (Similarity Layer on Arango)

## Summary

Define and operate the semantic similarity layer as first-class edges and ANN queries on the unified Arango graph. The goal is to expose high-quality semantic neighbors for nodes (papers, chunks, concepts) without altering the topological/temporal HiRAG structure, and to provide consistent inputs to GraphSAGE and PathRAG.

Companion PRDs:

- HiRAG: `docs/prd/phase2/hirag_arango_prd.md`
- PathRAG: `docs/prd/phase2/pathrag_arango_prd.md`
- GraphSAGE Integration: `docs/prd/phase2/graphsage_integration_prd.md`

## Goals

- Materialize `SEMANTIC_SIMILAR_TO` edges with configurable per-node top‑K (`k_sem ≤ 20`) and score threshold `τ_sem`.
- Provide ANN-backed AQL patterns for fast candidate retrieval with strict linting/guardrails.
- Version and provenance: edge fields capture embedding source/version, snapshot, timestamp, and score.
- Integrate with GraphSAGE (optionally produce/consume GNN embeddings as a semantic source).

## Non‑Goals

- Replacing HiRAG structural edges; semantic edges are additive.
- Building a general vector DB beyond Arango’s vector functions.

## Schema

- Edge type: `SEMANTIC_SIMILAR_TO` (stored in `relations` with `type='semantic_similar_to'`), or a dedicated collection if isolation is required.
- Edge fields:
  - `score` (float): raw similarity (cosine or dot-product) from the embedding space.
  - `embed_source` (enum): e.g., `jina_v4`, `sage_v1`.
  - `embed_dim` (int): embedding dimensionality.
  - `snapshot_id` (string): export/build snapshot identifier.
  - `ts` (ISO datetime): creation timestamp.
  - `quality` (float): optional rerank quality metric.
  - `recency` (float): decay-adjusted freshness when applicable.

## ANN Index & AQL Patterns

- Arango 3.12.4+ vector functions: `APPROX_NEAR_COSINE`, `APPROX_NEAR_L2` (experimental index must be enabled at startup; immutable dimension/metric per index).
- Lint rule: no `FILTER` between `FOR` and `LIMIT` inside ANN loop. Apply filters in an outer post-filter query.
- Example:

```aql
LET top = (
  FOR v IN entities
    SORT APPROX_NEAR_COSINE(v.text_emb, @q) DESC
    LIMIT @prefetch
    RETURN { _id: v._id, score: APPROX_NEAR_COSINE(v.text_emb, @q) }
)
  FOR t IN top
    LET v = DOCUMENT(t._id)
    FILTER v.type IN @allowed
    RETURN MERGE(t, { v })
```

## Index Lifecycle & Fallback

- Record `index_id`, `embed_source`, `embed_dim`, and `metric` for each vector index. Any change provisions a new index and triggers a parallel semantic edge rebuild.
- Canary criteria for index swaps: ≥95 % of nodes retain `k_sem/2` neighbors above `τ_sem`, ANN p99 latency ≤200 ms, and coverage verified for 48 h before cutover.
- Maintain external ANN (FAISS/HNSW) capability behind a feature flag; on Arango regression or dimensional change, rebuild against the external index and flip traffic when canary gates pass.

## Builders

1) Text-embedding builder
   - Aggregate paper/chunk embeddings (mean or length-weighted) to a paper-level vector.
   - For each node, compute top-K neighbors above `τ_sem`, apply degree-aware penalty `score_norm = score / (1 + α_deg · deg_out)` (default `α_deg=0.02`), and write `SEMANTIC_SIMILAR_TO` edges with metadata.

2) GNN-assisted builder (optional)
   - Use `sage_emb_vN` to compute neighbors; either update existing semantic edges with blended scores or maintain a parallel `embed_source='sage_vN'` view.

## Integration Points

- HiRAG: hierarchy remains topological/temporal; semantic edges do not alter cluster membership.
- GraphSAGE: consumes text/struct features; can output `sage_emb_vN` used for semantic neighbors or edge weights.
- PathRAG: may use semantic edges for candidate expansion and path scoring (semantic support term), respecting hop caps.

## Telemetry & Ops

- Metrics: `semantic_edge_coverage` (nodes with ≥K/2 neighbors), `ann.latency_p95/p99`, `semantic_build_time`, `semantic_rebuild_rows`.
- Alerts: ANN latency spikes, coverage drops, index offline, schema drift on vector fields.
- Rebuild cadence: nightly; on‑demand reflow when embedder changes (`embed_source`/`dim`).

## Acceptance Criteria

- Coverage: ≥95% of eligible nodes have ≥K/2 semantic neighbors above `τ_sem`.
- ANN p99 latency ≤200 ms for top‑200 prefetch, post‑filtering included.
- Edges carry complete provenance (snapshot_id, embed_source, dim, ts) ≥99.9%.

## Risks & Mitigations

- Hub bias: apply degree-aware down-weighting; cap `k_sem` and use recency decay.
- Index drift: enforce lifecycle rules with coverage/latency canaries before flipping indices; fall back to external ANN if Arango features regress.
- Embedder drift: maintain feature/embedding registry; block mismatched dimensions; full rebuild on version change.
- ANN anti-patterns: CI lint and canned query templates.
