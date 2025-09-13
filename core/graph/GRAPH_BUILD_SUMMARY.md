# Graph Building Summary

## Problem Solved: GPU OOM with 2.8M Papers

### Initial Issue
- Building keyword similarity edges for 2.8M papers was causing GPU OOM
- Tried to allocate 52.52 GiB for similarity matrix on 44.42 GiB GPU
- Loading embeddings one-by-one was inefficient (2000 papers/sec)

### Solution Implemented
1. **Faiss Library Integration**: Used Facebook's Faiss for billion-scale similarity search
2. **Batch Loading**: Load embeddings in 50k chunks with optimized AQL queries
3. **Memory-Efficient Indexing**: 
   - IVF (Inverted File) index with clustering
   - IVFPQ for datasets >100k papers (Product Quantization)
   - Normalized embeddings for cosine similarity
4. **Batched Search**: Process similarities in 5k batches to avoid OOM

## Current Graph Statistics

### Edge Collections
- **same_field**: 4,119,634 edges (papers in same ArXiv category)
- **temporal_proximity**: 40,622,701 edges (papers within ±1 month)
  - Optimized using ArXiv ID format (YYMM.NNNNN)
  - Reduced from 98+ hours to 14 minutes
- **keyword_similarity**: 3,640,254 edges (from 100k test run)
  - 58.8% are cross-category (interdisciplinary)
  - Building full 2.8M paper edges in progress

### Interdisciplinary Insights
Top cross-category connections found:
1. **Physics Bridge**: gr-qc ↔ hep-th (12.6k edges, 78.3% similarity)
2. **Experiment-Theory**: hep-ph ↔ hep-ex (11k edges, 79% similarity)
3. **Astro-Particle**: astro-ph ↔ hep-ph (10k edges, 79.2% similarity)
4. **Quantum-Materials**: quant-ph ↔ cond-mat (5.5k edges, 78.3% similarity)

### Theory-Practice Bridges
Found papers that connect theoretical domains (math, physics, statistics) with applied domains (CS, engineering, economics). Examples:
- SAT solving (CS) ↔ Polynomial theory (Math)
- Curvature flows (Math.DG) ↔ Computational geometry (CS.CG)

## Performance Metrics
- **Embedding Loading**: ~8-9 seconds per 50k batch
- **Faiss Index Building**: <2 seconds for 100k vectors
- **Similarity Search**: ~4 seconds per 5k query batch
- **Total for 100k papers**: ~2 minutes
- **Estimated for 2.8M papers**: ~45-60 minutes

## Files Created
- `/core/graph/builders/build_keyword_edges_faiss.py` - Faiss-based builder
- `/core/graph/builders/build_keyword_edges_batched.py` - GPU batched approach
- `/core/graph/builders/analyze_interdisciplinary.py` - Analysis tool
- `/core/graph/exports/` - Exported analysis results

## Key Optimizations
1. **No GPU for embeddings**: Load to CPU, only use GPU for final similarity computation
2. **Faiss advantages**: 
   - Optimized C++ implementation
   - SIMD instructions (AVX512)
   - Approximate nearest neighbor (faster than exact)
3. **Database batching**: 50k edges inserted at once
4. **Memory management**: Clear and truncate collections before rebuilding

## Next Steps
- [ ] Wait for full 2.8M keyword edge building to complete
- [ ] Consider citation edge building (currently only 4 edges)
- [ ] Build GNN training pipeline on top of completed graph
- [ ] Explore more sophisticated edge weighting schemes