# Graph Building Performance Optimization Notes

## Current Build Performance (First Run)

### Timing Breakdown (21.5+ hours so far)
1. **Category Edges** (same_field): ~1-2 hours
   - 25.7M edges created
   - Fast with 36 parallel workers
   - ~15 seconds per category batch

2. **Temporal Edges** (temporal_proximity): ~15-16 hours ⚠️
   - 5.98M edges created
   - MAJOR BOTTLENECK
   - 7-day window searches are expensive
   - CPU-bound, not parallelizable by paper

3. **Keyword Edges** (keyword_similarity): In progress (5+ hours so far)
   - 0 edges created yet (still processing)
   - Cosine similarity calculations
   - Threshold: 0.7

4. **Citation Edges**: Not started yet

### Bottleneck Analysis

#### Temporal Edges (CRITICAL PATH)
- **Problem**: O(n²) comparisons within each category
- **Current**: Check all papers within 7 days
- **Issue**: Even with batching, comparing timestamps is expensive

#### Potential Optimizations

1. **Temporal Edge Optimization**
   ```python
   # Current: Naive approach
   for paper1 in category_papers:
       for paper2 in category_papers:
           if abs(date1 - date2) <= 7 days:
               create_edge()
   
   # Better: Sort and sliding window
   sorted_papers = sort_by_date(category_papers)
   for i, paper in enumerate(sorted_papers):
       j = i + 1
       while j < len(sorted_papers) and within_7_days(paper, sorted_papers[j]):
           create_edge(paper, sorted_papers[j])
           j += 1
   ```

2. **Keyword Similarity Optimization**
   - Use approximate nearest neighbors (Faiss, Annoy)
   - Lower threshold might be too restrictive
   - Batch similarity computations

3. **Citation Edge Optimization**
   - These should be fast (direct lookups)
   - Ensure proper indexing on citation fields

### Memory Usage Patterns
- Peak: ~120GB during temporal edge building
- Stable: ~60-80GB during category edges
- Consider chunking strategies for large categories

### Parallelization Improvements

1. **Current**: 36 workers for category edges ✓
2. **Missing**: Parallel temporal processing
   - Split by date ranges?
   - Process different categories in parallel?

3. **Database Optimizations**
   - Batch size: 5000 seems optimal
   - Consider bulk import methods
   - Ensure proper indexes

## Recommendations for Next Build

### High Priority
1. **Optimize temporal edge algorithm** (sliding window)
2. **Add progress bars** with ETA calculations
3. **Implement checkpointing** for resume capability

### Medium Priority
1. **Profile keyword similarity** computation
2. **Consider approximate algorithms** for similarity
3. **Add memory monitoring** to prevent OOM

### Low Priority
1. **Experiment with different worker counts**
2. **Try different batch sizes**
3. **Consider GPU acceleration** for similarity

## Expected Improvements

With optimizations:
- Temporal edges: 15 hours → 2-3 hours (5-7x speedup)
- Keyword edges: Unknown → Estimate 4-5 hours
- Total time: 30+ hours → 10-12 hours

## Code Changes Needed

1. In `build_graph_parallel.py`:
   - Implement sorted sliding window for temporal
   - Add detailed progress tracking
   - Implement checkpointing

2. New utilities:
   - Progress estimator
   - Memory monitor
   - Performance profiler

## Research Value

Having two builds allows us to:
1. Verify reproducibility
2. Measure optimization impact
3. Document computational cost (ANT framework)
4. Validate "Death of the Author" topology consistency
