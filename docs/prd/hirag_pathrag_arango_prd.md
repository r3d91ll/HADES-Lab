# HiRAG × PathRAG Graph Backend on ArangoDB

## Summary

Implement a production‑grade graph backend on ArangoDB for the `arxiv_repository` database that supports hierarchical retrieval (HiRAG) and relational path pruning (PathRAG). This PRD defines collections, edge types, indices, graph views, weighting/learning, and query patterns, plus integration points with our MemGPT‑style virtual memory and vLLM model manager. The same tooling and schemas will be reused to stand up a smaller, local graph DB for the MemGPT agent’s secondary memory (separate database/graph within Arango).

Companion PRDs:

- Memory: `docs/prd/memgpt_virtual_context_arango_prd.md`
- Model Manager: `docs/prd/model_manager_vllm_prd.md`

## Conveyance Scorecard (Efficiency stance)

| Dimension         | Score | Evidence |
| ----------------- | ----- | -------- |
| **W** (What)      | 0.88  | Hierarchical narrowing reduces off‑topic candidates; path pruning increases signal density in final context. |
| **R** (Where)     | 0.90  | Topology‑aware traversal across hierarchy and relational edges; explicit bridge edges with weights. |
| **H** (Who)       | 0.88  | Same LLM/tools as existing stack (vLLM) with function‑calling hooks for graph queries. |
| **T** (Time)      | 0.84  | Bounded beam search on hierarchy and flow‑pruned paths; hot bridge cache and indices reduce latency. |
| **Ctx** (L/I/A/G) | 0.89  | L/G from hierarchy summaries; A/L from path prompts; instrumentation aligns with existing observability. |

Using α = 1.7, Conveyance ≈ (W·R·H/T)·Ctx^α ≈ **0.68**. Fallback: if hierarchy is unavailable or late chunking is violated, degrade to semantic‑only retrieval (vector) plus PathRAG; reserve C = 0 for W/R/H = 0 or T → ∞ per policy.

## Problem Statement

Flat GraphRAG tends to over‑retrieve redundant facts, inflating tokens and latency. We need a graph backend that organizes knowledge hierarchically (HiRAG) and, at query time, extracts only a few high‑value relational paths (PathRAG) to feed the model. The same methods must power MemGPT’s secondary memory graph for long‑horizon agent tasks.

## Goals

- Define and deploy Arango collections, edge types, and a named Graph for `arxiv_repository` powering HiRAG × PathRAG.
- Implement hierarchy construction (clustering + summaries) and relational weighting with provenance.
- Provide query patterns/APIs for: local candidates, bridge paths, global summaries, and path‑prompt assembly.
- Integrate with MemGPT tools (read‑only graph queries; write hooks for entity/edge updates) and vLLM function calling.
- Instrument Conveyance metrics and caching for performance repeatability.

## Non‑Goals

- RL/agentic traversal learning, cross‑DB federation, or advanced UI dashboards (beyond basic metrics).
- Model training/finetuning.

## Stakeholders & Users

- Knowledge engineering and ingestion workflows (arXiv corpus).
- Agent runtime (MemGPT‑style memory service) and vLLM clients.
- Platform ops (Arango cluster management; proxies; observability).

## Architecture Overview

### Collections (Documents)

```text
papers_raw       // raw arXiv papers & metadata (late‑chunked refs; not traversed)
entities         // extracted entities (paper|concept|method|person|code_symbol)
clusters         // hierarchy summary nodes (L1 topics, L2 super‑topics)
bridge_cache     // precomputed hot bridge paths with weights and provenance
weight_config    // relation type/base priors and learning snapshots
query_logs       // retrieval traces for feedback/learning and audits
cluster_membership_journal // optional: membership changes for reproducibility/rollback
```

### Collections (Edges)

```text
relations       // entity↔entity (cites|implements|extends|refers_to|coref|derives_from)
cluster_edges   // member↔cluster and cluster↔cluster links (SUBTOPIC_OF, PART_OF)
```

### Representative Documents

```jsonc
// entities
{
  "_key": "e_123",
  "name": "multi-head attention",
  "type": "concept",     // paper|concept|method|person|code_symbol
  "layer": 0,             // 0=base, 1=topic(L1), 2=super-topic(L2)
  "cluster_id": "c_42",  // L1 parent
  "prev_cluster_id": null, // last cluster id, for drift audits
  "embedding": [ ... ],
  "summary": "...",
  "deg_out": 0            // precomputed outbound degree for hub demotion
}

// relations (edge)
{
  "_from": "entities/e_1",
  "_to":   "entities/e_2",
  "type":  "refers_to",   // cites|implements|extends|refers_to|coref|derives_from
  "weight": 0.0,           // learned/materialized weight (σ of components)
  "weight_components": {    // audit trail for reproducibility
    "sim": 0.0,
    "type_prior": 0.0,
    "evidence": 0.0,
    "freshness": 0.0
  },
  "layer_bridge": true,    // crosses hierarchy levels?
  "sim": 0.0,              // embedding similarity or text sim
  "created": "ISO_date",
  "provenance": { "doc": "papers_raw/0704_0001", "span": [s,e] }
}

// clusters
{
  "_key": "c_42",
  "level": 1,                 // 1=topic, 2=super-topic
  "centroid": [ ... ],
  "centroid_version": "v1",
  "summary": "...",
  "cluster_params": {         // reproducibility: HDBSCAN/Louvain params + seed
    "algo": "hdbscan",
    "min_cluster_size": 5,
    "min_samples": 3,
    "seed": 42
  }
}
```

### Graph Definition

- Named Graph: `HADES_KG` (database: `arxiv_repository`).
- Vertex collections: `entities`, `clusters`.
- Edge definitions:
  - `relations`: `entities -> entities`.
  - `cluster_edges`: `entities <-> clusters`, `clusters <-> clusters`.

Paper representation: we adopt the “paper as entity” model. Papers that participate in traversals are stored in `entities` with `type="paper"`; the `papers_raw` collection holds fulltext and metadata blobs and is not part of the graph.

### Indices

- `entities`: persistent on `type`, `layer`, `cluster_id`; sparse on `name` (lowercased), geo/none as applicable; optional unique constraint on `(name_lower, type)` to reduce synonym collisions; optional persistent index on `deg_out` if used for demotion.
- `relations`: persistent on `type`, `layer_bridge`; edge index on `_from`, `_to`.
- `clusters`: persistent on `level` (1/2), `centroid_version`.
- Optional: ArangoSearch view `entities_view` over `entities.name`, `entities.summary` (and `papers_raw.title` as needed) for lexical fallback/rerank; vector index (3.12.4+ experimental) for embeddings if used directly in Arango.

Directionality conventions

- `cites`: paper → paper (citer → cited)
- `implements`: paper/method → code_symbol
- `derives_from`: method_new → method_base
- `refers_to`: entity_a → entity_b (mention/reference)
- `extends`: method_new → method_base
- `coref`: mention → canonical entity (non‑bridge)

### Sharding

- Hash shard by `_key` or `community_id` (from clustering) for `entities` and `relations` to limit cross‑shard traversals.

## Hierarchy Construction (HiIndex)

(1) Level 0 → Level 1 (Topic clusters)

- Input: `entities.layer=0.embedding` (Jina v4 per ingestion PRD).
- Method: HDBSCAN (cosine), `min_cluster_size≈5`, `min_samples≈3`.
- Output: create `clusters` (level=1) with centroid embeddings and abstractive summaries.

(2) Structural refinement

- Build 1–2 hop neighborhoods via `relations`; run Louvain/Leiden; reconcile with embedding clusters via consensus labels; adjust memberships.

(3) Level 1 → Level 2 (Super‑topics)

- Agglomerative over L1 centroids; ensure max fan‑out and depth limits for traversal.

(4)) Cluster edges

- `cluster_edges`: member→cluster (entities→L1), cluster→super (L1→L2), optional cross‑links between L1 siblings for known taxonomies.

(5) Bridge labeling

- Mark `relations.layer_bridge=true` if edge crosses levels (0↔1, 1↔2) or joins distant communities with high semantic similarity/provenance.

## Retrieval (HiRetrieval × PathRAG)

Stage A — Hierarchical narrowing (HiRAG)

- Embed query; fetch top‑K L2/L1 topics by cosine; descend with bounded beam (e.g., 8→4→2) to candidate subgraph (hundreds of nodes, bounded by hops/degree).

Stage B — Flow‑pruned relational paths (PathRAG)

- From candidate subgraph, extract M best paths (length 2–5) emphasizing relation types (`refers_to`, `implements`, `cites`, etc.). Path score combines materialized edge weights and relation boosts.
- Export path prompts: `A --[rel; snippet]--> B --[...]--> C` with per‑path rationale.

Weighted shortest path (AQL sketch; single s→t). Note: K‑shortest simple paths are not available natively.

```aql
FOR v, e, p IN OUTBOUND SHORTEST_PATH @s TO @t
  GRAPH "HADES_KG"
  OPTIONS { weightAttribute: "weight", defaultWeight: 0.1 }
  LET eps = @eps
  LET raw = SUM(
    FOR ed IN p.edges
      LET boost = @rel_boost[ed.type] ? @rel_boost[ed.type] : 1.0
      RETURN LOG(GREATEST(ed.weight * boost, eps))
  )
  LET score = raw / (1 + @lambda * (LENGTH(p.edges) - 2))
  RETURN { nodes: p.vertices, edges: p.edges, score }
```

Candidate subgraph expansion (ranked beams; driver‑side per‑level loop)

```aql
LET topics = (
  FOR c IN clusters
    FILTER c.level IN [1,2]
    FILTER COSINE_DISTANCE(c.centroid, @qEmb) < @topicThresh
    SORT COSINE_DISTANCE(c.centroid, @qEmb) ASC
    LIMIT @kTopic
    RETURN c._id
)

// Example: one level of expansion; repeat per level in the driver
FOR v, e, p IN 1..1 OUTBOUND topics GRAPH "HADES_KG"
  FILTER v.type IN ["concept","method","paper","code_symbol"]
  LET hscore = (1 - COSINE_DISTANCE(v.embedding, @qEmb))
               * (1.0 / (1 + @deg_alpha * NVL(v.deg_out, 0)))
  SORT hscore DESC
  LIMIT @beamWidth
  RETURN v
```

## Edge Weighting & Learning

- Initialize `relations.weight` with logistic squashing: `weight = σ(α·sim + β·type_prior + γ·evidence + δ·freshness)`, where `freshness = exp(-(now - created)/tau)` applies time decay (per‑relation `tau` in `weight_config`, e.g., shorter for `implements`, longer for `cites`). Parameters come from `weight_config`; keep `weight_components` for audits.
- Online updates: prefer pairwise ranking (positive vs rejected paths) from `query_logs` with bounded learning rate; nightly batch refreshes `bridge_cache` for top‑moving clusters.
- Provenance recorded per update with snapshot ids for rollback.

## APIs & Integration

### Python API (core/tools/rag_utils)

- `get_hierarchy_candidates(q_emb, k_topic, thresholds) -> CandidateGraph`
- `get_best_paths(candidate_graph, M, L, relation_boosts) -> List[Path]`
- `format_path_prompt(paths, hierarchy_summary) -> str`

These functions use `core/database/arango/memory_client.py` (RO socket) and follow the same error/timeout handling patterns as existing utilities.

### MemGPT Hooks (read‑only, then RW)

- Tool: `graph_query_paths(query, k, L)` → returns path strings + citations; used during answer planning.
- Tool: `graph_entity_update(entity_id, fields)` (v2) for minor curation tasks; guarded by idempotency, quorum rules, and review thresholds.
- All calls go through AF_UNIX proxies (`ARANGO_RO_UDS`, `ARANGO_RW_UDS`).

### vLLM Prompting

- Path‑based prompting with a brief hierarchy ledger. Enforce “answer from provided paths only” and require citations `(path #, node ids)`.

### Prompt Budgets & API Contracts

- Stage‑A hierarchy summary ≤ 300 tokens.
- Paths: `M` paths × ≤ 120 tokens/path (cap total prompt tokens via config guardrail).
- Return both machine (IDs) and human (text) forms for each path to support citations.

### Driver‑level Query Safeguards

- Set driver timeouts and AQL `maxRuntime` per query; cap `batchSize` appropriately.
- For RW jobs (indexing/weights), enable retry on write‑write conflicts.
- Circuit breakers and backoff mirror memory client defaults.

## Configuration

YAML under `core/config/workflows/`:

```yaml
hirag_pathrag:
  topic_k: 8
  beam: [8,4,2]
  cand_cap: 500
  path:
    M: 6
    L: 5
    lambda: 0.3
    eps: 1e-6
    relation_boost:
      refers_to: 0.8
      implements: 0.7
      cites: 0.6
      derives_from: 0.6
      coref: 0.4
  beam_demote:
    deg_alpha: 0.005
```

## Deployment & Ops

- Graph creation done via `arangosh` or HTTP using RO/RW proxies; environment via `.env` (no hard‑coded credentials).
- Proxies: keep RO/RW socket paths (`/run/hades/readonly/arangod.sock`, `/run/hades/readwrite/arangod.sock`) in sync; systemd units as per deploy docs.
- Optional vector index requires `--experimental-vector-index` (3.12.4+); once enabled, it cannot be disabled. Ensure startup guards match Memory PRD.

## Non‑Functional Requirements

- P95 query latency ≤ 2.0 s on 10k‑paper subset; throughput ≥ 100 QPM.
- Bridge discovery rate ≥ 0.80; avg path length < 4 hops.
- Deterministic retries; timeouts and circuit breakers mirror memory client.
- Logging: emit W/R/H/T and context factors with per‑query ids; counters for candidate size, discarded nodes, kept paths, and final prompt tokens.

## Validation & Benchmarks

- Ground truth: ≥500 arXiv→GitHub pairs with expected bridge entities.
- Baselines: Flat RAG → GraphRAG → HiRAG → PathRAG → HiRAG×PathRAG.
- Metrics: Bridge Discovery (hit if any returned path contains the gold pair within ≤ L hops), Recall@M, path purity (fraction of edges in allowlist), Precision@k, Context Coherence, Latency, Token cost; estimate α from Δlog‑ratios holding H constant.

## Test Plan & Acceptance Criteria

- Unit: AQL builders, relation weighting math, prompt formatter.
- Integration: end‑to‑end candidate graph + path extraction on a 10k subset; RO vs RW paths exercised.
- Performance: measure P50/P95 end‑to‑end on warm caches; confirm `bridge_cache` hits.
- Security: UDS permissions, auth on proxies, no TCP listeners; env‑driven config only.

Exit Criteria (MVP)

- Graph and indices created; hierarchy build jobs complete (L1/L2); `get_best_paths` returns ≥1 valid path for ≥80% of GT pairs; P95 ≤ 2.0 s; Conveyance logger active.

## Risks & Mitigations

- Latency at high edge counts → shard by community; cap expansions; precompute hot bridges.
- Hierarchy drift → scheduled reclustering; cohesion tests; manual spot checks.
- Weight overfitting → bounded learning rate; A/B with held‑out set; rollback snapshots.

## Open Questions

- Do we park ArangoSearch view over `entities` for hybrid lexical/semantic rerank, or rely solely on embeddings + graph topology?
- How aggressively do we cache `bridge_cache` (TTL vs pin hot topics)?
- When to expose RW tools to agents for curation?

## References

- Memory PRD: `docs/prd/memgpt_virtual_context_arango_prd.md`
- Model Manager PRD: `docs/prd/model_manager_vllm_prd.md`
- Conveyance Framework: `docs/CONVEYANCE_FRAMEWORK.md`
- ArangoDB Graphs & Traversals: <https://www.arangodb.com/docs/stable/graphs.html>

---

### Appendix A — Graph Creation Snippet (arangosh)

```js
// db is bound to the target database (arxiv_repository)
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

### Appendix B — Path Prompt Template

```text
Hierarchy context (HiRAG):
- Topic ▸ Subtopic ▸ Section summaries: <N lines>

Relational paths (PathRAG):
1) A --[relation; snippet: "..."]--> B --[...]--> C
2) ...

Instructions: Answer using only the information in these paths. Cite path # and node ids.
```
