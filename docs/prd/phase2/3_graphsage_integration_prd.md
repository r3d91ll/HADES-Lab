# GraphSAGE Integration for HiRAG × PathRAG

## Summary

Train and deploy an inductive GraphSAGE model over the HiRAG knowledge graph so we can embed new nodes, refine edge weights, and provide Graph Neural Network (GNN) signals to the PathRAG query layer. This PRD defines data export pipelines, GraphSAGE architectures, training/evaluation loops, model registries, and runtime inference hooks that feed back into `arxiv_repository` collections.

Companion PRDs:

- HiRAG: `docs/prd/phase2/hirag_arango_prd.md`
- PathRAG: `docs/prd/phase2/pathrag_arango_prd.md`
- Memory integration: `docs/prd/phase3/memgpt_virtual_context_arango_prd.md`
- Model serving: `docs/prd/phase3/model_manager_vllm_prd.md`

## Conveyance Scorecard (Integration stance)

| Dimension | Score | Evidence |
|-----------|-------|----------|
| **W** | 0.84 | GraphSAGE learns structural + semantic neighborhoods, improving edge weights and bridge detection. |
| **R** | 0.86 | Inference operates on-demand against Arango exports and cached embeddings, keeping late-bound nodes inductive. |
| **H** | 0.82 | Shared PyTorch/DGL stack with existing ML infrastructure; ownership split between graph ML and infra teams. |
| **T** | 0.78 | Training amortized nightly; online inference cached; integration adds <15 ms per query (T is a cost term dividing W·R·H in the efficiency view). |
| **Ctx** | 0.87 | Ctx logging extends W/R/H/T metrics with GNN health signals; aligns with Conveyance telemetry. |

Using α = 1.7 → Conveyance ≈ **0.58**. If GraphSAGE fails to converge or embeddings drift beyond tolerance, fall back to deterministic weights and mark Conveyance `C = 0` for GNN-dependent flows.

## Problem Statement

HiRAG builds the hierarchical structure and PathRAG navigates it, but both rely on static similarity heuristics. We need an inductive GNN that can embed new papers/entities without retraining from scratch and provide richer edge weights for traversal.

## Goals

1. Export HiRAG graph data (features, adjacency, negatives) for GraphSAGE training.
2. Train, validate, and version a GraphSAGE model (PyTorch Geometric or DGL) with nightly retraining on the `arxiv_repository` subset.
3. Provide batch and streaming inference APIs that update `entities` embeddings, `relations` weights, and `bridge_cache` entries.
4. Surface GraphSAGE metrics (loss, ROC-AUC, embedding drift, freshness) in the Conveyance telemetry stream.
5. Expose the GNN outputs to PathRAG scoring (weights/boosts) without breaking existing fallbacks.

## Non-Goals

- End-to-end active learning or reinforcement loops.
- Serving the GNN through vLLM (GNN stays on the graph service path).
- Architecting a generalized feature store beyond HiRAG/PathRAG requirements.

## Stakeholders & Users

- Graph ML engineering (model training & monitoring).
- Knowledge graph team (feature export, schema evolution).
- Agent runtime (consumes updated weights/embeddings).

## Architecture Overview

1. **Data export pipeline**
   - Nightly job extracts node features from `entities` (text embeddings, cluster context, metadata) and edge lists from `relations`/`cluster_edges`.
   - Generates train/val/test splits with negative sampling for link prediction.

2. **GraphSAGE trainer**
   - PyTorch (torch-geometric preferred) running with configurable neighborhood sampling.
   - Supports single-GPU and multi-GPU regimes via DDP; uses ML Ops bucket for checkpoints.
   - Stores metadata (dataset hash, hyperparams, metrics) in `weight_config`.

3. **Inference service**
   - Batch inference: ingest new nodes, compute embeddings, and push results back to `entities`/`relations`.
   - Streaming API: Python service with UDS endpoint for low-latency embedding requests (backed by cached model weights).

4. **Integration with Arango**
   - Updated `relations.weight` and `weight_components.sim` fields from GraphSAGE outputs.
   - `bridge_cache` refresh uses learned embeddings for top bridge candidates.

5. **PathRAG updates**
   - Path scoring uses learned weights (GraphSAGE-sim + priors). Fallback to deterministic scores if GNN unavailable.

### Data Flow Diagram (High-Level)

```
HiRAG Graph (Arango)
   │
   ├─ nightly export (Parquet/Arrow)
   │      │
   │      └─ GraphSAGE Trainer (PyG/DGL)
   │               │
   │               ├─ checkpoints → model registry
   │               └─ metrics → Conveyance telemetry
   │
   └─ Inference Service (TorchScript/ONNX)
           │
           ├─ embeddings → entities.embedding_gnn
           ├─ weights → relations.weight_components.sim
           └─ bridges → bridge_cache
```

## Feature Engineering

- Node text features: mean pooled `arxiv_abstract_embeddings`, cluster summaries, categories.
- Structural features: degree stats, cluster memberships, hierarchical depth.
- Edge features: relation type priors, timestamp recency.
- Negative sampling: random walks avoiding positives, controlled ratio.

## Training Regimens

- Loss: link prediction (binary cross entropy with logits) using positive/negative edges.
- Optimizer: AdamW, cyclical learning rates.
- Early stopping on validation AUC; checkpoint best models.
- Rebuild nightly; horizon jobs triggered via Airflow/Argo.

## Inference & Deployment

- Package best checkpoint as TorchScript; load into `graph-inference` microservice with AF_UNIX interface.
- Batch job re-embedding schedule: hourly for new nodes; full refresh weekly.
- Update Arango via HTTP/2 client with idempotent writes, logging metrics to `query_logs`.

## Rollout & Guardrails

- Auto-rollback policy: reject a candidate model when validation AUC drops >3 percentage points vs. last accepted snapshot **or** embedding drift ΔL2 >0.15. Trigger on-call, keep `model_version_active` pinned, and open an incident note.
- Blue/green deployment: promote `model_version_candidate` via shadow scoring in PathRAG for ≥48 h; cut over when win-rate/Recall gains are non-negative, latency stays within SLO, and drift metrics pass. Retain last three checkpoints for immediate rollback.
- Zero-propagation: when GraphSAGE is unavailable or fails health gates, PathRAG removes GNN-weighted paths and marks Conveyance `C=0` for GNN-dependent flows until deterministic weights take over.

## Telemetry & Observability

- Loss, AUC, embedding drift ΔL2, inference latency.
- Alerts fire on drift threshold breaches (ΔL2 >0.15), validation AUC drop >3 p.p., or inference latency p99 >50 ms.
- Emit metrics to Prometheus (`graphsage_loss`, `graphsage_inference_latency_ms`, etc.).

## Security & Compliance

- Training data stays in same security domain as Arango; artifact registry locked to infra accounts.
- Inference service binds to UDS (`/run/hades/graphsage.sock`) with ACLs.

## Validation & Benchmarks

- Compare against baseline (semantic-only weights): improvements in PathRAG path quality (Bridge hits, Recall@M).
- Run ablation: GNN-only, heuristics-only, hybrid.
- Stress tests: embedding growth, online inference throughput (>200 QPS goal).

## Timeline & Milestones

1. Export pipeline + schema updates.
2. MVP GraphSAGE training loop, single-GPU.
3. Model registry + batch inference integration.
4. PathRAG scoring update + A/B metrics.
5. Production hardening (autoscaling, monitoring, documentation).

## Risks & Mitigations

- **Model drift** → monitor drift metrics; fallback to heuristics.
- **Training instability** → hyperparameter sweeps, gradient clipping.
- **Inference latency** → cache embeddings, micro-batching.

## Open Questions

- Do we require multi-relational GraphSAGE (edge-type aware) vs. standard aggregator?
- Where do we store historical embeddings (Arango, object store, both)?
- Should PathRAG expose GNN confidence to downstream consumers?

## References

- Original GraphSAGE paper: <https://arxiv.org/abs/1706.02216>
- PyTorch Geometric docs: <https://pytorch-geometric.readthedocs.io/>
- HiRAG PRD: `docs/prd/phase2/hirag_arango_prd.md`
- PathRAG PRD: `docs/prd/phase2/pathrag_arango_prd.md`
