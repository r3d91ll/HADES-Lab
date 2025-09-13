# Core Graph Infrastructure

## Overview

Core reusable graph construction and management infrastructure used by multiple tools:
- **GraphSAGE** (`tools/graphsage/`) - Graph neural network training
- **GNN Training** (`tools/gnn_training/`) - Multiple GNN architectures
- **HiRAG** (`tools/hirag/`) - Hierarchical retrieval with graph context
- **Graph Analytics** - Statistical analysis and visualization

## Architecture

```
core/graph/
├── graph_manager.py      # Main interface for graph operations
├── builders/             # Graph construction scripts
│   ├── build_graph_parallel.py        # Main parallel builder (36 workers)
│   ├── build_paper_centric_graph.py   # Paper-centric edges
│   ├── build_category_edges_fast.py   # Category relationships
│   ├── build_temporal_citations.py    # Temporal edges
│   └── build_additional_edges.py      # Keyword/citation edges
├── monitoring/           # Progress tracking
│   ├── check_graph_progress.py        # Non-invasive monitor
│   └── monitor_graph_build.py         # Detailed monitoring
├── utils/               # Helper utilities
│   ├── sample_graph_for_training.py   # Graph sampling
│   ├── explore_graph.py               # Graph exploration
│   └── discover_bridges.py            # Cross-disciplinary connections
└── exports/             # Export utilities
```

## Key Components

### GraphManager

Central interface for all graph operations:

```python
from core.graph.graph_manager import GraphManager

# Initialize manager
manager = GraphManager()

# Get current graph statistics
stats = manager.get_graph_stats()
print(f"Papers: {stats['papers']:,}")
print(f"Edges: {stats['total_edges']:,}")

# Export graph for training
paths = manager.export_graph(name="arxiv_graph")

# Load graph in any tool
data, metadata = manager.load_graph(name="arxiv_graph")
```

### Graph Building (One-time, 20+ hours)

```bash
cd core/graph/builders/
python build_graph_parallel.py --workers 36

# Monitor in separate terminal
cd ../monitoring/
python check_graph_progress.py
```

### Export Format

Exported graphs are stored in `/data/graphs/` with:
- `graph.pt` - PyTorch Geometric Data object
- `metadata.json` - Graph statistics and configuration
- `node_mapping.pkl` - Paper ID to node index mapping

## Philosophy: "Death of the Author"

The graph construction implements paper-centric topology:
- **NO author edges** - eliminates disambiguation issues
- **Category edges** - papers in same field
- **Temporal edges** - papers published nearby in time
- **Keyword edges** - semantic similarity
- **Citation edges** - explicit knowledge references

## Information Reconstructionism Framework

```
Information = WHERE × WHAT × CONVEYANCE × TIME
```

- **WHERE**: Graph topology (this module)
- **WHAT**: Paper content (embeddings)
- **CONVEYANCE**: Model capability (GNN architectures)
- **TIME**: Temporal relationships (edges)

## Current Graph Statistics (2025-01-11)

```
Papers: 2,824,688
Total Edges: 31,693,470+
  - same_field: 25,706,082
  - temporal_proximity: 5,987,388
  - keyword_similarity: In progress...
  - citations: Pending

Average Degree: ~22.4
Build Time: 20+ hours
Storage: ~8GB exported
```

## Usage in Tools

Any tool can use the core graph infrastructure:

```python
# In any tool (e.g., tools/graphsage/train.py)
from core.graph.graph_manager import GraphManager

manager = GraphManager()
data, metadata = manager.load_graph("arxiv_graph")

# Use data for training, analysis, etc.
print(f"Loaded {metadata['num_nodes']} nodes")
print(f"Loaded {metadata['num_edges']} edges")
```

## API Reference

### GraphManager Methods

- `get_graph_stats()` - Get current database statistics
- `export_graph(name, include_features)` - Export to PyTorch format
- `load_graph(name)` - Load exported graph
- `list_graphs()` - List all available graphs

### Builder Scripts

- `build_graph_parallel.py` - Main parallel builder
  - `--workers`: Number of CPU workers (default: 36)
  - `--batch-size`: Edge batch size (default: 5000)
  
- `build_paper_centric_graph.py` - Build all edge types
  - `--edge-types`: Which edges to build
  - `--limit`: Paper limit for testing

## Performance Optimization

1. **Parallel Processing**: 36-40 CPU workers
2. **Batch Inserts**: 5000 edges per batch
3. **Memory Management**: ~120GB RAM usage
4. **Thermal Management**: Monitor CPU temps (keep <85°C)

## Future Extensions

1. **Heterogeneous Graphs** - Multiple node types
2. **Dynamic Graphs** - Time-evolving topology
3. **Filtered Exports** - Domain-specific subgraphs
4. **Incremental Updates** - Add new papers without rebuild
