# Production Graph Building System

## Overview
This directory contains the PRODUCTION-READY graph building system for ArXiv papers.
All experimental and deprecated code has been moved to Acheron/ for historical reference.

## Directory Structure

```
core/graph/
├── PRODUCTION_README.md          # This file
├── configs/                       # Configuration files
│   └── arxiv_graph_config.yaml  # Main graph configuration
├── builders/                      # Clean, production builders
│   ├── build_similarity_edges_universal.py  # Universal similarity edge builder
│   ├── build_structural_edges.py            # Category and temporal edges
│   └── build_version_edges.py               # Paper version connections
├── orchestration/                 # High-level orchestration
│   └── orchestrate_graph_build.py          # Complete pipeline orchestrator
├── experiments/                   # Research and experiments
│   └── keyword_vs_abstract_comparison.py   # Embedding comparison study
└── docs/                         # Documentation
    └── GRAPH_BUILD_ARCHITECTURE.md        # Detailed architecture

```

## Production Components

### 1. Universal Similarity Edge Builder
**File**: `builders/build_similarity_edges_universal.py`
- Handles ANY embedding type (keyword, abstract, etc.)
- FAISS-accelerated with GPU support
- Configurable thresholds and parameters
- No redundant code for different embedding types

### 2. Structural Edge Builders
**File**: `builders/build_structural_edges.py`
- `same_field`: Papers in same category
- `temporal_proximity`: Papers published within time window
- Fast, deterministic connections

### 3. Version Edge Builder  
**File**: `builders/build_version_edges.py`
- Connects different versions of same paper
- Uses ArXiv version metadata

### 4. Orchestration System
**File**: `orchestration/orchestrate_graph_build.py`
- Manages complete graph build pipeline
- Configuration-driven
- Checkpointing and recovery
- Validation and statistics

## Usage

### Complete Graph Build
```bash
cd core/graph/orchestration/
python orchestrate_graph_build.py --config ../configs/arxiv_graph_config.yaml
```

### Build Specific Edge Type
```bash
# Similarity edges (choose embedding type)
cd core/graph/builders/
python build_similarity_edges_universal.py abstract  # or keyword

# Structural edges
python build_structural_edges.py --edge-type same_field
python build_structural_edges.py --edge-type temporal_proximity

# Version edges
python build_version_edges.py
```

### Database Management
```bash
# View current state
cd core/utils/
python arango_manager.py stats

# Prepare for rebuild
python arango_manager.py arxiv-rebuild

# Drop specific collection
python arango_manager.py drop-collection keyword_similarity
```

## Configuration

Edit `configs/arxiv_graph_config.yaml` to adjust:
- Similarity thresholds
- Batch sizes
- GPU settings
- Edge type priorities
- Validation parameters

## Edge Types

### Based on Research Results

After comparing keyword vs abstract embeddings, we determined:
- **Abstract embeddings** provide superior graph quality
- **Keywords** are redundant (derived from abstracts)
- Single semantic edge type reduces complexity

### Final Graph Structure
1. **same_field**: ~4.1M edges (categorical)
2. **temporal_proximity**: ~40.6M edges (time-based)
3. **abstract_similarity**: ~10-15M edges (semantic)
4. **paper_versions**: ~1.7M edges (versioning)

Total: ~60M edges for 2.8M papers

## Performance

- **Full graph build**: <2 hours with dual GPU
- **Memory usage**: <48GB GPU RAM
- **Batch processing**: 15k papers at a time
- **Checkpointing**: Automatic resume on failure

## Validation

The orchestrator automatically validates:
- Edge counts within expected ranges
- Graph connectivity (no orphans)
- Clustering coefficient
- Component structure

## Maintenance

### Adding New Embedding Types
1. Generate embeddings in `arxiv_embeddings` collection
2. Use Universal Builder with new field name:
```python
builder = UniversalSimilarityBuilder(
    embedding_field='your_new_embedding',
    edge_collection='your_similarity_edges',
    threshold=0.7
)
```

### Deprecating Code
Never delete! Move to Acheron with timestamp:
```bash
mv old_builder.py ../../Acheron/graph/old_builder_2025-09-12_11-30-00.py
```

## Philosophy

"Death of the Author" - Papers connect through content, not authorship.
No author-based edges. Papers stand as autonomous knowledge entities.