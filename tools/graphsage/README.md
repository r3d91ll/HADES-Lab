# GraphSAGE Module - GNN Training on Pre-built Graphs

## Overview

This module provides GraphSAGE training capabilities for pre-built academic graphs. It works in conjunction with:
- **graph_construction**: Builds and exports the graph (20+ hour process)
- **gnn_models**: Provides multiple GNN architectures (GraphSAGE, GAT, GCN)

The separation allows training multiple models on the same graph without rebuilding.

## Philosophy: Death of the Author

Inspired by Roland Barthes' literary theory, we treat each paper as an independent knowledge actant in the academic network. Papers connect through:
- **Categorical relationships** (same field of study)
- **Temporal proximity** (published within same time window)
- **Keyword similarity** (shared conceptual space)
- **Citation networks** (explicit knowledge references)

Notably absent: author-based edges. This eliminates the "John Smith problem" while emphasizing knowledge flow over social networks.

## Architecture

```
graphsage/
├── pipelines/           # Main pipeline scripts
│   ├── graphsage_pipeline.py    # Primary training pipeline
│   └── cache/                    # Graph caching
├── configs/             # Configuration files
│   └── graphsage_config.yaml    # Training configuration
├── training/            # Training scripts
│   ├── train_graphsage.py       # Main training script
│   └── train_graphsage_distributed.py  # Multi-GPU training
├── monitoring/          # Progress monitoring
│   ├── monitor_training.py      # Training metrics
│   ├── monitor_graphsage.py     # Model monitoring
│   └── monitor_graph_build.py   # Graph construction monitoring
├── utils/              # Utility scripts
│   ├── build_paper_centric_graph.py  # Paper-centric edge builder
│   ├── build_additional_edges.py     # Supplementary edges
│   ├── build_category_edges_fast.py  # Category edge optimization
│   ├── build_temporal_citations.py   # Temporal edge builder
│   ├── discover_bridges.py           # Cross-disciplinary connections
│   └── sample_graph_for_training.py  # Graph sampling
├── gephi/              # Visualization exports
├── build_graph_parallel.py      # Parallel graph construction (40 workers)
├── check_graph_progress.py      # Non-invasive progress checker
└── rebuild_complete_graph.py    # Full graph rebuild script
```

## Current Status (2025-01-11)

### Graph Statistics
- **Papers**: 2,824,688 (complete ArXiv corpus)
- **Edge Collections**:
  - `same_field`: ~5.6M edges (category-based)
  - `temporal_proximity`: ~5.6M edges (7-day window)
  - `keyword_similarity`: In progress
  - `citations`: Pending
- **Average Degree**: ~8 (will increase with keyword/citation edges)

### Performance Metrics
- **Parallel Processing**: 36-40 CPU workers
- **Category Edges**: 862K → 5.6M edges (bug fix applied)
- **Temporal Processing**: ~15 hours for 2.82M papers
- **Memory Usage**: ~120GB RAM during peak processing
- **Storage**: ~1.6GB per million edges

## Quick Start

### 1. Build the Graph (One-time, 20+ hours)

```bash
cd /home/todd/olympus/HADES-Lab/tools/graph_construction/
python builders/build_graph_parallel.py

# Monitor progress (in separate terminal)
python monitoring/check_graph_progress.py
```

### 2. Export Graph for Training

```bash
# Export the built graph to PyTorch Geometric format
python graph_manager.py  # Exports to /data/graphs/
```

### 3. Train GNN Models

```bash
cd /home/todd/olympus/HADES-Lab/tools/gnn_models/

# Train GraphSAGE
python train_gnn.py --config configs/training_config.yaml --model graphsage

# Train GAT on the same graph
python train_gnn.py --config configs/training_config.yaml --model gat

# Train GCN on the same graph
python train_gnn.py --config configs/training_config.yaml --model gcn
```

### 4. Monitor Training

```bash
# Real-time training metrics
python monitoring/monitor_training.py
```

## Configuration

### graphsage_config.yaml

```yaml
model:
  hidden_dim: 256
  num_layers: 2
  dropout: 0.5
  aggregator: mean  # mean, gcn, pool, lstm

training:
  batch_size: 2048  # Reduced for memory efficiency
  learning_rate: 0.01
  epochs: 200
  num_neighbors: [25, 10]  # Sampling per layer
  
data:
  num_features: 2048  # Jina v4 embedding dimension
  num_classes: 176    # ArXiv categories
```

## Edge Building Process

### Phase 1: Category Edges (Complete)
- Papers in same ArXiv category
- Batched processing to avoid O(n²) explosion
- ~5.6M edges for 2.82M papers

### Phase 2: Temporal Edges (Running)
- Papers within 7-day publication window
- Same category requirement
- ~5.6M expected edges

### Phase 3: Keyword Similarity (Pending)
- Cosine similarity > 0.7 on keyword embeddings
- Cross-category connections
- Expected: ~10M edges

### Phase 4: Citations (Pending)
- Direct citation relationships
- Bidirectional edges
- Expected: ~30M edges

## Thermal Management Discovery

During the 15+ hour graph build, we discovered CPU thermal throttling at 95°C. Solution:
- Removed case side panel
- Added external fan (high setting)
- Temperature dropped to 80-85°C
- CPU frequency increased from 4.6GHz to 4.7GHz

**ANT Framework Insight**: The cooling system acts as a "translation device" in the Actor-Network, enabling higher conveyance (C) by reducing time (T) through sustained computational throughput.

## Conveyance Framework Analysis

Using C = (W·R·H/T)·Ctx^α:

- **W (What)**: Paper content quality via Jina embeddings
- **R (Where)**: Graph topology (4 edge types)
- **H (Who)**: GraphSAGE model capability
- **T (Time)**: 15+ hours for full graph build
- **Ctx**: Context quality through neighborhood aggregation
- **α**: 1.5-2.0 (super-linear context amplification)

### System Optimization
- Parallel processing: T ↓ by 88x (22 min → 15 sec per category)
- Bug fix: R ↑ by 6.5x (862K → 5.6M edges)
- Thermal management: H ↑ through sustained performance

## Known Issues

1. **Temporal edge processing is slow** (~15 hours)
   - CPU-bound, not GPU-acceleratable
   - Required for temporal dynamics research

2. **Memory usage during peak** (~120GB)
   - ZFS ARC cache helps significantly
   - Consider batch processing for larger graphs

3. **ArangoDB timeout on large batches**
   - Fixed: Increased timeout to 300s
   - Batch size: 5000 edges optimal

## Deprecated Scripts (Moved to Acheron)

Following the Acheron protocol, deprecated scripts are preserved with timestamps:
- `train_fast.py`, `train_simple.py`, `train_ultra.py` → Training experiments
- `build_coauthorship_correct.py` → Author-based approach (abandoned)
- `analyze_author_ambiguity.py` → Author disambiguation analysis

## Future Work

1. **Complete keyword similarity edges** (~2 hours estimated)
2. **Build citation network** (~4 hours estimated)
3. **Train on complete 2.82M graph** (48 hours estimated)
4. **Implement HiRAG integration** for retrieval
5. **Bridge discovery** for interdisciplinary connections

## Citations

If using this implementation, please cite:
- GraphSAGE: Hamilton et al., "Inductive Representation Learning on Large Graphs" (2017)
- Death of the Author: Barthes, Roland, "La mort de l'auteur" (1967)
- Actor-Network Theory: Latour, Bruno, "Reassembling the Social" (2005)

## Contact

For questions about the "Death of the Author" implementation or thermal management as translation devices, contact Todd @ HADES-Lab.
