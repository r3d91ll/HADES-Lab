# Graph Build Architecture for ArXiv Papers

## Philosophy: "Death of the Author"
Papers are treated as autonomous knowledge entities. No author-based connections.
Each paper stands on its own merit and connects through content, time, and field.

## Edge Types and Their Purpose

### 1. Content-Based Edges (Semantic)

#### Abstract Similarity (SINGLE SEMANTIC EDGE TYPE)
- **Purpose**: Connect papers with similar research topics/approaches
- **Method**: Cosine similarity of abstract embeddings (384-dim)
- **Threshold**: 0.75 (high threshold for quality connections)
- **Expected edges**: ~10-15M (sparse, high-quality connections)
- **Implementation**: GPU-accelerated with batching
- **Rationale**: Abstracts contain all keyword information plus context.
  Keywords are just extracted features FROM abstracts - building both
  would be redundant and computationally wasteful.

### 2. Structural Edges (Categorical)

#### a) Same Field
- **Purpose**: Connect papers in the same research area
- **Method**: Exact match on primary category (e.g., "cs.LG")
- **Edges**: 4.1M ✅ COMPLETE
- **Note**: Creates dense clusters within disciplines

### 3. Temporal Edges (Time-based)

#### a) Temporal Proximity
- **Purpose**: Connect papers published around the same time
- **Method**: Papers within 30 days of each other
- **Edges**: 40.6M ✅ COMPLETE
- **Note**: ArXiv-specific optimization using YYMM.NNNNN format

#### b) Paper Versions
- **Purpose**: Connect different versions of the same paper
- **Method**: Group by base arxiv_id
- **Expected edges**: ~1.7M (39.5% of papers have multiple versions)
- **Implementation**: Direct from versions field

## Build Pipeline

### Phase 1: Data Preparation ✅
1. Import metadata from Kaggle dataset
2. Extract keywords using TF-IDF
3. Generate keyword embeddings
4. Generate abstract embeddings

### Phase 2: Edge Construction
```python
# Execution order (optimized for memory and performance)
1. same_field edges (✅ COMPLETE - 4.1M)
2. temporal_proximity edges (✅ COMPLETE - 40.6M)  
3. abstract_similarity edges (PENDING - after embeddings complete)
4. paper_versions edges (PENDING - quick to build)

# REMOVED: keyword_similarity edges (redundant with abstract_similarity)
```

### Phase 3: Graph Optimization
1. Create indices on edge collections
2. Build neighbor lookup tables
3. Calculate graph statistics (degree distribution, clustering coefficient)
4. Identify connected components

## Memory Requirements

### For 2.8M papers:
- Abstract similarity matrix: 31.4TB (float32) - MUST batch
- Keyword similarity matrix: 31.4TB (float32) - MUST batch
- GPU memory per batch: ~7-8GB (15k papers)
- System RAM for embeddings: ~4.3GB per embedding type

### Batching Strategy:
```python
# Process 15k papers at a time against all 2.8M
batch_size = 15000  # Fits in 48GB GPU with headroom
total_batches = ceil(2.8M / 15000) = 187 batches
time_per_batch = ~30 seconds
total_time = ~1.5 hours per edge type
```

## Quality Metrics

### Edge Quality Thresholds:
- Abstract similarity: 0.75 (high confidence)
- Keyword similarity: 0.65 (medium confidence)
- Keep top-k edges per node: 50 (prevent super-nodes)

### Expected Graph Properties:
- Average degree: 20-30 edges per paper
- Clustering coefficient: 0.3-0.4 (academic communities)
- Largest component: 95%+ of papers
- Diameter: 15-20 hops

## Reproducibility Requirements

1. **Configuration file** with all parameters
2. **Checksums** of input data
3. **Random seeds** for any sampling
4. **Timestamp** all edge creations
5. **Version** embedding models used

## Build Script Structure

```bash
# Complete graph build sequence
python orchestrate_graph_build.py \
    --config graph_config.yaml \
    --verify-data \
    --build-edges all \
    --validate-graph \
    --export-stats
```

## Validation Steps

1. **Edge count validation**: Each edge type within expected range
2. **Symmetry check**: Undirected edges are symmetric
3. **Orphan check**: No isolated nodes (except truly unique papers)
4. **Duplicate check**: No duplicate edges
5. **Weight range**: All similarities in [0,1]

## Extensions for Non-ArXiv Data

### Universal Edge Types:
- keyword_similarity (TF-IDF based)
- abstract_similarity (if abstracts available)
- same_field (if categories available)

### Data-Specific Edge Types:
- temporal_proximity (needs timestamp format consideration)
- citation_edges (if citation data available)
- author_collaboration (if not following "Death of the Author")

## Performance Optimizations

1. **GPU Utilization**:
   - Use both A6000 GPUs with NVLink
   - Mixed precision (fp16) for similarity computation
   - Batch size optimization per GPU memory

2. **Memory Management**:
   - Process in chunks, never load full matrix
   - Clear GPU cache between batches
   - Use memory-mapped files for large embeddings

3. **Parallelization**:
   - Edge types can be built in parallel
   - Within each type, batch processing can be parallelized
   - Use multiprocessing for CPU-bound operations

## Final Graph Statistics Target

```
Total nodes: 2,798,038 papers
Total edges: ~60-70M (all types combined)
Average degree: 25-30
Graph density: 0.00001 (very sparse)
Components: 1 giant component (95%+), few small islands
```