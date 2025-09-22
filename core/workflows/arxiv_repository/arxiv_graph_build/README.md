# Arxiv Graph Build Workflow (Design)

## Purpose

Construct the Phase 2 Arango graph for the `arxiv_repository` database by orchestrating HiRAG + PathRAG bootstrap steps over sharded partitions. The workflow reuses the sharding toolkit (`core.tools.sharding`) to backfill entities, relations, hierarchy edges, and semantic edges while respecting Conveyance requirements for throughput, telemetry, and idempotency.

## Inputs & Outputs

- **Inputs**
  - `arxiv_metadata`, `arxiv_abstract_chunks`, `arxiv_abstract_embeddings` collections populated by the initial ingest workflow.
  - Sharding configuration (`hash_mod`, shard count) aligned with Arango graph shards.
  - Optional resume manifest / lease store state.
- **Outputs**
  - `entities`, `relations`, `clusters`, `cluster_edges` populated per PRD.
  - Semantic edge collection entries (`type='semantic_similar_to'`).
  - Run metadata (partition results, telemetry events, failure manifests).

## Phases

1. **Schema Ensure**
   - Idempotent call to `GraphSchemaManager.ensure_schema()`.

2. **Entity Ingest (HiRAG)**
   - Sharded via `HiragEntityIngestJob` with `ArxivPartitionAdapter`.
   - Deterministic `_key` ensures idempotent UPSERT.

3. **Bootstrap Relations**
   - Sharded `HiragRelationsJob` to seed co-category edges.
   - Later replaced/refined by semantic builder.

4. **Hierarchy Build**
   - `HiragHierarchyJob` per partition (hash bucket) updating clusters and membership edges.

5. **Semantic Edge Construction**
   - `HiragSemanticEdgesJob` per partition calling `build_semantic_similarity_edges` with ANN-like cosine search.
   - Parameters (k, threshold, embed source, snapshot) passed via workflow config; edges written as `semantic_similar_to`.

6. **Post-Processing & Telemetry**
   - Aggregate partition metrics, publish Conveyance stats, produce resume manifests.

## Orchestration Components

- **Adapters**: `ArxivPartitionAdapter` (hash buckets aligned with shard count).
- **Jobs**: `HiragEntityIngestJob`, `HiragRelationsJob`, `HiragHierarchyJob`, (upcoming) `SemanticEdgesJob`.
- **Runner**: `ShardRunner` with
  - Durable lease store (TODO: implement Arango-backed store).
  - Token buckets for cursor + write throttling.
  - Metrics callback emitting `shard.started/succeeded/failed` events.
- **Configuration**
  - JSON/YAML describing shard settings, concurrency, retry policy.
  - CLI wrappers under `core/tools/sharding/cli.py` for ad-hoc execution.

## Failure & Resume Strategy

- Leases capture attempts + last error; failed partitions remain in lease store.
- Resume run filters partitions by status (`FAILED`/expired) and replays.
- Invariants detect data drift; failure halts run and surfaces partition ID for inspection.

## Integration Points

- Called from higher-level workflow orchestrator (e.g., Airflow/Dagster) that sequences initial ingest → graph build → semantic builder → GraphSAGE export.
- Telemetry forwarded to Conveyance dashboards (TODO: integrate metrics sink).
- Semantic stage depends on ANN index provisioning; gate before GraphSAGE training.

## Implementation Checklist

- [x] Sharding toolkit & adapter doc (`docs/1a_sharding_toolkit_usage.md`).
- [x] Partition-aware HiRAG jobs (`core/tools/sharding/hirag_jobs.py`).
- [ ] Durable lease/token backends (Arango/Redis).
- [ ] Semantic edges sharded job.
- [ ] Workflow class + orchestrator script.
- [ ] Runbook: command invocations, telemetry ingestion.

## Notes on Directory Layout

- `core/workflows/arxiv_repository/initial_data_ingestion/` – handles raw pipeline into `arxiv_*` collections.
- `core/workflows/arxiv_repository/arxiv_graph_build/` – this workflow builds the graph atop those collections.
- Future directories may host GraphSAGE training (`graphsage_export`) and PathRAG runtime workflows, keeping all database-specific orchestration co-located.

This organization allows database-specific workflows to evolve independently while sharing toolkit modules under `core/tools` and PRDs under `docs/prd/phase2`.
