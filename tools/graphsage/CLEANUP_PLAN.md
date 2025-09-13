# GraphSAGE Cleanup Plan

## Files to Keep (Essential)

### In core/graph/:
1. `graph_manager.py` - Central interface ✓
2. `builders/build_graph_parallel.py` - Main builder (currently running) ✓
3. `monitoring/check_graph_progress.py` - Progress checker ✓
4. `__init__.py` - Module init ✓

### In tools/graphsage/:
1. `training/train_graphsage.py` - Main training script
2. `pipelines/graphsage_pipeline.py` - Pipeline orchestration
3. `configs/graphsage_config.yaml` - Configuration
4. `monitoring/monitor_training.py` - Training monitor

### In tools/gnn_training/:
1. `train_gnn.py` - Unified training for all GNNs ✓

## Files to Archive (Move to Acheron)

### Deprecated/Redundant:
1. `gephi/` - Entire directory (visualization not needed now)
2. `theory_practice_finder.py` - Experimental
3. `graph_visualizer.py` - Not used
4. `neighborhood_sampler.py` - Duplicate functionality
5. `train_distributed.py` - Redundant with train_graphsage_distributed.py
6. `analyze_training_logs.py` - Can recreate if needed

### In core/graph (redundant builders):
1. `rebuild_complete_graph.py` - Replaced by build_graph_parallel.py
2. `build_paper_centric_graph.py` - Old version
3. `build_additional_edges.py` - Integrated into parallel builder
4. `build_category_edges_fast.py` - Integrated into parallel builder
5. `build_temporal_citations.py` - Integrated into parallel builder
6. `graph_export.py` - Functionality in graph_manager.py
7. `preprocess_graph.py` - Old preprocessing

## Files to Consolidate

### Monitoring (keep best, archive rest):
- Keep: `monitor_training.py`
- Archive: `monitor_graphsage.py`, `monitor_graph_build.py`

### Utils:
- `discover_bridges.py` - Keep (useful for research)
- `explore_graph.py` - Keep (useful for exploration)
- `sample_graph_for_training.py` - Keep (needed for sampling)
- `node_features.py` - Check if functionality is in graph_manager

## Final Structure Goal

```
core/graph/
├── graph_manager.py         # Main interface
├── builders/
│   └── build_graph_parallel.py  # Only active builder
├── monitoring/
│   └── check_graph_progress.py  # Simple progress check
├── utils/
│   ├── discover_bridges.py      # Research utility
│   ├── explore_graph.py         # Exploration utility
│   └── sample_graph.py          # Sampling utility
└── __init__.py

tools/graphsage/
├── train_graphsage.py       # Main training
├── configs/
│   └── graphsage_config.yaml
├── monitoring/
│   └── monitor_training.py
└── README.md

tools/gnn_training/
├── train_gnn.py            # Multi-architecture training
└── configs/
    └── training_config.yaml
```

## Estimated Reduction
- Current: 31+ files
- After cleanup: ~12 essential files
- Reduction: 60%+ file count
