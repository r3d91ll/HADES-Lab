# Graph Neural Network Architecture - Modular Design

## Philosophy: Separation of Concerns

We've separated graph construction from model training because:

1. **Graph building is expensive** (20+ hours for 2.82M papers)
2. **Multiple models can train on the same graph** (GraphSAGE, GAT, GCN, etc.)
3. **Different experiments need different architectures** but same topology
4. **Iterative development** - tune models without rebuilding graphs

## Architecture Overview

```
core/
└── graph/                 # CORE INFRASTRUCTURE (reusable)
    ├── graph_manager.py   # Central graph interface
    ├── builders/          # Graph building scripts
    │   ├── build_graph_parallel.py      # Main parallel builder
    │   ├── build_paper_centric_graph.py # Paper-centric edges
    │   ├── build_category_edges_fast.py # Category edges
    │   ├── build_temporal_citations.py  # Temporal edges
    │   └── build_additional_edges.py    # Keyword/citation edges
    ├── monitoring/        # Progress tracking
    │   ├── check_graph_progress.py      # Non-invasive monitor
    │   └── monitor_graph_build.py       # Detailed monitoring
    ├── utils/             # Helper utilities
    └── exports/           # Export utilities

tools/
├── gnn_training/          # GNN MODEL TRAINING
│   ├── train_gnn.py       # Unified training script
│   ├── graphsage/         # GraphSAGE-specific
│   ├── gat/               # Graph Attention Networks
│   ├── gcn/               # Graph Convolutional Networks
│   ├── configs/           # Training configurations
│   └── evaluation/        # Model evaluation
│
├── graphsage/             # GRAPHSAGE-SPECIFIC TOOLS
│   ├── training/          # Training scripts
│   ├── monitoring/        # Training monitors
│   └── pipelines/         # Pipeline scripts
│
├── graph_analytics/       # GRAPH ANALYSIS (future)
│   └── analyze.py         # Statistical analysis
│
└── hirag/                 # HIERARCHICAL RETRIEVAL
    └── graph_retrieval.py # Uses graph for retrieval
```

## Workflow

### Phase 1: Graph Construction (Once)

```bash
# Build the complete graph (20+ hours)
cd core/graph/
python builders/build_graph_parallel.py --workers 36

# Monitor progress
python monitoring/check_graph_progress.py

# Export to PyTorch Geometric format
python graph_manager.py
# Creates: /data/graphs/arxiv_graph_20250111_120000/
#   ├── graph.pt         # PyTorch Geometric Data object
#   ├── metadata.json    # Graph statistics and info
#   └── node_mapping.pkl # Paper ID to node index mapping
```

### Phase 2: Model Training (Many times)

```bash
# Train different architectures on the SAME graph
cd tools/gnn_models/

# GraphSAGE
python train_gnn.py \
    --config configs/graphsage_config.yaml \
    --graph arxiv_graph \
    --model graphsage

# Graph Attention Network
python train_gnn.py \
    --config configs/gat_config.yaml \
    --graph arxiv_graph \
    --model gat

# Graph Convolutional Network  
python train_gnn.py \
    --config configs/gcn_config.yaml \
    --graph arxiv_graph \
    --model gcn
```

### Phase 3: Comparison & Evaluation

```bash
# Compare model performances
python evaluation/compare_models.py \
    --models graphsage_model.pt gat_model.pt gcn_model.pt

# Generate performance report
python evaluation/benchmark.py --graph arxiv_graph
```

## Benefits of This Architecture

### 1. **Efficiency**
- Build graph once, train many models
- 20+ hour build → 5 minute load
- Parallel experiments on same topology

### 2. **Flexibility**
- Swap architectures without rebuilding
- Test hyperparameters quickly
- A/B testing of model designs

### 3. **Reproducibility**
- Exported graphs are versioned
- Same graph = fair comparison
- Metadata tracks build parameters

### 4. **Scalability**
- Add new model architectures easily
- Train on subgraphs for testing
- Scale to larger graphs without code changes

## Current Graph Statistics (2025-01-11)

```
Papers: 2,824,688
Edges: 31,693,470 (and growing)
  - same_field: 25,706,082
  - temporal_proximity: 5,987,388
  - keyword_similarity: In progress...
  - citations: Pending

Average Degree: ~22.4
Build Time: 20+ hours (36 CPU workers)
Export Size: ~8GB (compressed)
```

## Example Configuration

```yaml
# configs/graphsage_config.yaml
model:
  architecture: graphsage
  hidden_dim: 256
  num_layers: 2
  dropout: 0.5
  aggregator: mean

training:
  batch_size: 2048
  learning_rate: 0.01
  epochs: 200
  num_neighbors: [25, 10]
  device: cuda

data:
  graph_name: arxiv_graph  # Loads pre-built graph
  split_ratio: [0.8, 0.1, 0.1]  # train/val/test
```

## Future Extensions

1. **More Architectures**
   - HGT (Heterogeneous Graph Transformer)
   - RGCN (Relational GCN)
   - Custom hybrid models

2. **Graph Variants**
   - Filtered graphs (by year, field)
   - Sampled subgraphs for testing
   - Multi-relational representations

3. **Advanced Features**
   - Edge type attention
   - Temporal dynamics
   - Cross-modal learning

## Key Insight: "Death of the Author" Preserved

The paper-centric graph topology (no author edges) remains central to all models. This philosophical choice is built into the graph construction phase, ensuring all downstream models operate on the same "authorless" knowledge network.

## Conveyance Framework Analysis

C = (W·R·H/T)·Ctx^α

- **Graph Construction**: Optimizes R (topology) once
- **Model Training**: Iterates on H (model capability)
- **Separation**: Reduces T (time) for experimentation
- **Result**: Higher overall conveyance through modularity
