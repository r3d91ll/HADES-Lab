# PathRAG — Path-Based Retrieval on ArangoDB (Phase 2)

## Summary

Define traversal and scoring methods that navigate the unified HiRAG graph to extract a small set of high‑value relational paths for prompting. PathRAG consumes structural (HiRAG), semantic (ANN edges), and learned (GraphSAGE) signals; it does not change the graph’s structure.

## Retrieval Stages

- Stage A — Candidate narrowing (uses hierarchy and ANN views)
  - Embed query; fetch top‑K topics (L2/L1) by cosine; descend with bounded beams (e.g., 8→4→2) to build a candidate subgraph (bounded by hops/degree).
- Stage B — Flow‑pruned relational paths (PathRAG)
  - From candidates, extract M best paths (length 2–5) emphasizing relation types and learned weights; export path prompts with rationale/citations.

## Weighted Shortest Path (AQL sketch)

```aql
FOR v, e, p IN OUTBOUND SHORTEST_PATH @s TO @t
  GRAPH @graph_name
  OPTIONS { weightAttribute: "weight", defaultWeight: 0.1 }
  LET raw = SUM(
    FOR ed IN p.edges
      LET boost = @rel_boost[ed.type] ? @rel_boost[ed.type] : 1.0
      RETURN LOG(GREATEST(ed.weight * boost, @epsilon))
  )
  LET score = raw / (1 + @lambda * (LENGTH(p.edges) - 2))
  RETURN { nodes: p.vertices, edges: p.edges, score }
```

## Freshness & Time Decay

- Compute per-edge freshness using decay constants stored in `weight_config.lambda_half_life[type]`:

  \[
  \text{freshness}(\text{age}_\text{days}) = \exp(-\lambda_{type} · \text{age}_\text{days}), \quad \lambda = \ln 2 / \text{half\_life}_\text{days}
  \]

- Default half-lives: citations 720 d, hierarchy edges 540 d, code calls/imports 180 d, semantic edges 90 d; tune during Conveyance reviews.
- Builders persist the computed freshness in `weight_components.freshness`; runtime scoring reads the same constants to avoid drift between offline and online behavior.

## Candidate Expansion (ranked beams)

```aql
LET topics = (
  FOR c IN clusters
    FILTER c.level IN [1,2]
    FILTER COSINE_DISTANCE(c.centroid, @qEmb) < @topicThresh
    SORT COSINE_DISTANCE(c.centroid, @qEmb) ASC
    LIMIT @kTopic
    RETURN c._id
)

FOR v, e, p IN 1..@depth OUTBOUND topics GRAPH @graph_name
  FILTER v.type IN @allowed_types
  LET hscore = (1 - COSINE_DISTANCE(v.embedding, @qEmb)) * (1.0 / (1 + @deg_alpha * NVL(v.deg_out, 0)))
  SORT hscore DESC
  LIMIT @beam
  RETURN DISTINCT v
```

## Edge Weighting & Learning

- Initialize `relations.weight` via logistic squashing \(\sigma(β₀ + β₁·sim + β₂·type\_prior + β₃·evidence + β₄·freshness)\); coefficients live in `weight_config` and `weight_components` persist raw contributors for audits.
- Online updates from `query_logs` (pairwise ranking) with bounded learning rate; nightly refresh for hot clusters. `bridge_cache` stores hot paths with provenance.
- Emit `path_confidence = MIN(edge.confidence)` with each path so downstream consumers can gate low-trust outputs; if confidence or required factors vanish, flag the path with `path_zero_propagation=1` and exclude it from prompts.

## Prompting & Budgets

- Path‑based prompting with a brief hierarchy ledger; require citations `(path #, node ids)`.
- Token budgets: `M` paths × ≤120 tokens/path; hierarchy summary ≤300 tokens; enforce ≤60% of baseline tokens.
- Deterministic ordering by path score then recency.
- Zero-propagation: when Conveyance components are missing (e.g., GNN unavailable and deterministic substitute disabled), drop affected paths and log `path_zero_propagation` for diagnostics.

## Driver‑Level Safeguards

- Set driver timeouts and AQL `maxRuntime`; cap `batchSize` appropriately.
- For RW jobs (weights), retry on conflicts; circuit breakers mirror memory client.

## APIs

- Python: `get_hierarchy_candidates(q_emb, k_topic, thresholds) -> CandidateGraph`
- Python: `get_best_paths(candidate_graph, M, L, relation_boosts) -> List[Path]`
- MemGPT tools: `graph_query_paths(query, k, L)` (RO) and guarded `graph_entity_update` (RW).

## Acceptance (PathRAG)

- On 10k-doc / 2k-query, win‑rate ≥+10% vs. baseline; Recall@20 ≥+8%; traversal p99 ≤900 ms; prompt ≤60% tokens.
