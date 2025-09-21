# Phase 2 Graph Pipeline (HiRAG → GraphSAGE → PathRAG)

## Summary

Consolidate the remaining graph roadmap into a single Phase 2 initiative covering:

1. HiRAG graph population and semantic structure enhancements.
2. GraphSAGE training + inference loops feeding weights/embeddings back into Arango.
3. PathRAG traversal updates that consume the learned signals.

This issue replaces GitHub Issues #19 and #32, merging their scope into one tracked epic.

Phase 2 operates a unified Arango graph with typed edges: `HIERARCHY_OF`, `CONTAINS`, `CALLS/IMPORTS`, `REFS/DERIVES_FROM`, and `SEMANTIC_SIMILAR_TO`. Retrieval layers (HiRAG/semantic) are expressed as view filters or weights over the single graph so we avoid drift between parallel stores.

## Motivation

- Avoid fragmented ownership between graph construction, GNN training, and retrieval.
- Ensure telemetry, deployment, and testing requirements align across layers.
- Provide a single Funnel for Phase 2 milestone tracking.
- Protect $R$ by keeping structural and semantic edges co-resident while allowing configurable views per product workflow.

## Milestones

1. **HiRAG Enhancements**
   - Finalize entity/edge builders, hierarchy refinement, metadata validation.
   - Implement embedding aggregation for paper-level vectors.
   - Backfill degree stats and provenance fields.
   - Normalize arXiv identifiers across historical formats: persist `arxiv_id_raw`, canonical `arxiv_id` (lowercased, may include `vN`), versionless `arxiv_id_root`, `version`, and `id_variant` (old/new). Derive temporal buckets from `published` or YYMM in the ID when missing.
   - Freeze Phase 2 graph schema (`entities`, `chunks`; edges listed above) and add `snapshot_id`, `source`, `ts`, `quality`, `recency` to edge metadata with versioning.
   - Build `SEMANTIC_SIMILAR_TO` edges with configurable top-k per node (`k_sem ≤ 20`, `τ_sem` threshold) persisted as first-class edges alongside structural relations.
   - Define ANN index strategy per collection and standardize AQL patterns for hybrid + ANN queries; add CI lint to block anti-patterns (no `FILTER` between `FOR` and `LIMIT` when using `APPROX_NEAR_*`; apply filters in a post-filter subquery).

2. **GraphSAGE Integration**
   - Implement export pipeline, trainer, checkpoint registry (see `docs/prd/phase2/graphsage_integration_prd.md`).
   - Deploy inference service and Arango writers for embeddings/weights.
   - Add telemetry & rollback hooks.
   - Train GraphSAGE-2L (mean aggregator, `d=64`) with fanout `[25, 10]`, unsupervised loss, and five negative samples per batch; evaluate GraphSAINT/LADIES if variance spikes.
   - Build node features `[text_emb || struct_feats || type_emb]` with normalization, version tags, and explicit drift checks on dimensionality or embedding source.
   - Persist `sage_emb_vN` on vertices plus `embedding_ledger(node_id, snapshot_id, model_version)` for provenance and rollback; write edge weights from pairwise z-similarity when applicable.
   - Run inductive inference on create/update events and batch refresh per snapshot cadence; ensure backpressure, retry, and failure alerts are wired into the service.

3. **PathRAG Upgrades**
   - Update path scoring to blend GraphSAGE outputs with existing priors.
   - Refresh bridge cache and query tooling.
   - Add end-to-end validation on a 10k-document subset, then full corpus.
   - Implement `path_score` = λ₁·mean GraphSAGE z-similarity + λ₂·semantic support + λ₃·recency + λ₄·quality − λ₅·path length, with tunable λ owned by Retrieval.
   - Enforce prompt packing token budgets (≤60 % of baseline top-k tokens) and deterministic ordering by score, then recency.
   - Define Bridge cache contract `{query_signature → {paths, snippets, tokens}}`, TTL, warm/cold handling, and hit-rate instrumentation; cap semantic hops per path.
   - Register per-relation freshness decay constants (`lambda_half_life`) in `weight_config` so builders and runtime scoring share the same recency math.

## Deliverables

- Updated code: builders, GNN trainer modules, inference service, CLI commands (`gnn-train`, `gnn-embed`, `graph-export`, `retriever-serve`, `retriever-benchmark`, `graph-validate`).
- Documentation: GraphSAGE PRD, runbooks, monitoring dashboards, and model registry manifests (hyperparameters, data snapshot, metrics, checksums).
- Tests/benchmarks: unit + integration coverage, Conveyance metrics, acceptance experiment harness for 10k-doc / 2k-query benchmark.
- Decision log on fallback behaviour (GNN unavailable) and recorded λ/token budget decisions.
- CI additions: AQL lint for ANN queries, feature registry validation, snapshot parity checks.
- Conveyance instrumentation note + dashboards proving $C=\frac{W·R·H}{T}·Ctx^{\alpha}$ (efficiency view) with `T` reported separately and Context components logged individually.
- ANN fallback runbook (Arango vector index → external ANN) including migration steps, canary criteria, and feature flag ownership.

## Acceptance Criteria

- **AC1 – HiRAG completeness:** vertex/edge counts match spec within ±2 %, schema conformance ≥99.5 %, provenance present ≥99.9 % on the target subset.
- **AC2 – GraphSAGE model:** training completes within the agreed snapshot window; produces `sage_emb_vN (d=64)` with link-prediction AUC ≥0.80 on held-out edges or contrastive loss ≤τ on dev; inference p99 latency ≤50 ms/node for inductive inserts at target throughput.
- **AC3 – PathRAG quality & efficiency:** on the 10k-doc / 2k-query benchmark, win-rate ≥+10 % vs. flat top-k baseline, Recall@20 ≥+8 %, and answer token budget ≤60 % of baseline median.
- **AC4 – Latency SLOs:** hybrid top-k retrieval p99 ≤750 ms; PathRAG traversal (≤4 hops, ≤8 paths) p99 ≤900 ms end-to-end; monthly error budget ≤5 % before feature work resumes.
- **AC5 – Observability & rollback:** dashboards expose $W,R,H,T$ and context metrics; one-click rollback to prior `sage_emb` version verified; degraded (no-GNN) mode auto-switch documented and tested.
- **AC6 – Conveyance compliance:** telemetry and reports compute $C=\frac{W·R·H}{T}·Ctx^{\alpha}$ (efficiency view) with $α∈[1.5,2.0]$; missing factors trigger zero-propagation (`C=0`) for affected queries/paths.

## Conveyance Math Alignment

We adopt the HADES efficiency view for all Phase 2 metrics:

\[
C = \frac{W·R·H}{T}·Ctx^{\alpha}, \quad α ∈ [1.5, 2.0]
\]

- `T` (time to converge) divides the product; it is never exponentiated or combined with Context multipliers.
- `Ctx` amplifies outcomes through tracked components `{L, I, A, G}` that dashboards display individually.
- Zero-propagation guard: when any base factor (`W`, `R`, `H`) or required context component drops to zero (e.g., GNN unavailable without deterministic substitute), affected candidates are removed and reported with `C=0`.

## Telemetry & Alerts

- **W (What):** `retriever.topk_precision`, `path.win_rate`, `answer.token_budget_used`, `doc.quality_distribution`, `judge_consistency`.
- **R (Where):** `graph.avg_degree`, `path.length_histogram`, `semantic_edge_coverage`, `connected_component_sizes`.
- **H (How):** `gnn.training_time`, `gnn.inference_latency_p99`, `feature_build_time`, `export_time`, `artifact_registry_status`.
- **T (Time):** `retriever.hybrid_latency_p99`, `retriever.path_latency_p99`, `ann.latency_p99`, `bridge.cache_hit_rate`, `queue.backpressure`.
- **Ctx (L/I/A/G):** `schema_parity_errors` (I), `provenance_missing_rate` (G), `prompt_pack_over_budget` (A), `idempotency_conflicts` (L).
- Configure alerts for ANN slowdown, Bridge cache miss spikes, SAGE inference backlog, schema drift, and snapshot skew.
- Zero-propagation alerts fire when Conveyance hits zero for ≥1 % of queries in a 15 min window.

## Resilience & Fallback Policies

- **ANN fallback:** maintain external FAISS/HNSW index capability. On Arango vector index regression (dimension/metric change, feature flag disabled, or latency SLO breach), rebuild semantic edges against the external index, require ≥95 % coverage + ANN p99 ≤200 ms during canary, then flip traffic via feature flag.
- **Index migrations:** any change to `embed_dim`/`metric` provisions a new index, rebuilds in parallel, and swaps readers only after 48 h of canary validation meeting coverage/latency gates.
- **GNN blue/green:** promote new `sage_emb` via shadow scoring for ≥48 h; cut over only if win-rate/Recall improvements hold and drift metrics remain within thresholds; retain last three checkpoints for rollback.
- **Error budgets:** if retrieval or ANN SLO error budgets exceed 5 % in a month, freeze feature releases and run a tuning sprint until budgets recover.

## Risks & Mitigations

- SAGE over-smoothing on dense hierarchies → cap fanout, add residual/pooling variants if validation AUC stalls.
- Semantic edges over-emphasize hubs → apply degree-aware down-weighting and cap semantic hops per path.
- ANN query anti-patterns degrade latency → enforce AQL lint + canned query templates.
- Feature drift between text embeddings and GNN → maintain feature registry with blocking checks on embedding dimension/version.
- GNN unavailability at runtime → feature flag PathRAG SAGE term, expose degraded mode metrics, and document rollback.

## Tracking

- Owner: Graph ML + Knowledge Graph teams.
- Target release: Phase 2 (Q3).
- Replace GitHub Issues #19 and #32; close them once this epic is accepted.
- Leads: Graph ML – Alexis Park; Knowledge Graph – Omar Singh; PathRAG – Mei Chen; Observability – Dan Rivera.
- Decision owners: Retrieval (λ weights, token budgets), Data Platform (snapshot cadence), Graph ML (feature registry + model versions).

## Decision Gates (must confirm before coding)

- Vector index: Arango experimental vector index (3.12.4+) vs. external ANN; if Arango, enforce AQL template/linting policy.
- ANN fallback: maintain external ANN infrastructure + migration playbook; confirm coverage and latency gates before switching providers.
- GNN stack: PyTorch Geometric vs. DGL; embedding dimension (64 vs. 128); fanout [25,10]; negatives per batch (5); snapshot cadence (nightly + on-demand).
- PathRAG scoring: λ weights and token budgets; degraded mode behavior when GNN unavailable; bridge cache TTL/size.
- Identifier policy: canonicalization rules for `arxiv_id`/`arxiv_id_root` and temporal bucketing from YYMM fallback.

## Proposed Defaults (for review)

- ANN backend: Arango vector index (3.12.4+ experimental) with cosine metric; prefetch `@prefetch=200`; apply filters post‑ANN loop.
- Text embeddings: Jina v4, d=2048; paper‑level vector = length‑weighted mean of chunk vectors; `embed_source=jina_v4`.
- GNN: PyTorch Geometric; GraphSAGE‑2L (mean agg.), d=64, fanout `[25,10]`, negatives per positive = 5, AdamW, early stop on AUC; nightly snapshot + on‑demand inductive inserts; retain last 3 checkpoints.
- PathRAG scoring weights (initial): λ₁=0.4 (z‑sim), λ₂=0.3 (semantic support), λ₃=0.2 (recency), λ₄=0.1 (quality), λ₅=0.05 per extra hop; tune after 10k/2k benchmark.
- Bridge cache: TTL 24h or invalidation on new `snapshot_id` (embeddings); cap 10k entries; LRU eviction.
- Token budgets: prompt packing ≤60% of baseline top‑k tokens; deterministic ordering by score then recency.
- Snapshot cadence: nightly rebuild; embedding_ledger persisted with `{node_id, snapshot_id, model_version, ts}`.
- ID policy: persist `arxiv_id_raw`, canonical `arxiv_id`, versionless `arxiv_id_root`, `version`, `id_variant`; derive YYMM from ID when `published` absent.

## Task Checklist (Phase 2)

- [ ] Ingest/schema: add arXiv ID normalization fields and persist `published`/`update_date`.
- [ ] HiRAG builders: finalize clusters, hierarchy edges, degree stats, journaling.
- [ ] Semantic PRD and builders: define `SEMANTIC_SIMILAR_TO`, ANN config, AQL templates, CI lint.
- [ ] GraphSAGE export → trainer → inference service; registry and telemetry.
- [ ] PathRAG scoring update; bridge cache contract; benchmark harness (10k-doc / 2k-query).
- [ ] MemGPT hooks: candidate/paths tools, heartbeat paging based on drift.
- [ ] Dashboards + alerts; degraded mode and rollback validation.
- [ ] Register freshness decay constants in `weight_config` and plumb into builders/runtime scoring.
- [ ] Implement ANN fallback runbook + feature flag switch, and stage canary validation tests.
- [ ] Wire Conveyance efficiency view dashboards + zero-propagation alerts.
