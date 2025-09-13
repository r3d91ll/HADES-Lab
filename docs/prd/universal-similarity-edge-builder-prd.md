# Product Requirements Document: Universal Similarity Edge Builder

**Version:** 1.0  
**Date:** September 12, 2025  
**Author:** HADES Team  
**Status:** Draft

## 1. Executive Summary

The Universal Similarity Edge Builder is a flexible, embedding-agnostic module for constructing similarity edges in knowledge graphs. It replaces multiple specialized builders (keyword, abstract, etc.) with a single configurable system that can process any embedding type stored in ArangoDB.

## 2. Problem Statement

### Current Challenges

- **Code Duplication**: Separate builders for keyword_similarity, abstract_similarity, etc.
- **Maintenance Overhead**: Each builder needs individual updates and bug fixes
- **Inflexibility**: Adding new embedding types requires new builder implementations
- **Inconsistent Behavior**: Different builders may have subtle algorithmic differences
- **Research Bottleneck**: Difficult to compare different embedding strategies

### Impact

- 3+ separate builder modules with 80% code overlap
- Hours of redundant development for each new embedding type
- Inconsistent graph quality across different edge types
- Blocked research on optimal embedding strategies

## 3. Proposed Solution

### Core Concept

A single, parameterized builder that accepts:

- **embedding_field**: Which embedding to use (e.g., 'keyword_embedding', 'abstract_embedding')
- **edge_collection**: Where to store edges (e.g., 'keyword_similarity', 'abstract_similarity')
- **threshold**: Similarity cutoff (configurable per embedding type)
- **algorithm**: FAISS or direct GPU computation (auto-selected based on size)

### Key Innovation

FAISS and similarity algorithms don't care about embedding semantics - they just process vectors. By parameterizing the input/output, we can use the same code for any embedding type.

## 4. Requirements

### 4.1 Functional Requirements

#### MUST Have

- **FR1**: Support any embedding field from arxiv_embeddings collection
- **FR2**: Create edges in any specified edge collection
- **FR3**: Configurable similarity threshold per embedding type
- **FR4**: Support both GPU and CPU computation
- **FR5**: Handle 2.8M+ papers efficiently
- **FR6**: Prevent super-nodes with top-k limiting
- **FR7**: Ensure edge uniqueness (no duplicates)
- **FR8**: Support batch processing for memory efficiency

#### SHOULD Have

- **FR9**: Auto-select optimal algorithm (FAISS vs direct GPU)
- **FR10**: Progress tracking and logging
- **FR11**: Checkpoint/resume capability for long runs
- **FR12**: Memory management with periodic cleanup

#### NICE to Have

- **FR13**: Multi-GPU support with NVLink
- **FR14**: Approximate algorithms for huge datasets
- **FR15**: Incremental updates for new papers

### 4.2 Non-Functional Requirements

#### Performance

- **NFR1**: Process 2.8M papers in <2 hours with GPU
- **NFR2**: Memory usage <48GB GPU RAM
- **NFR3**: Support datasets up to 10M papers

#### Reliability

- **NFR4**: Atomic edge creation (no partial graphs)
- **NFR5**: Graceful handling of OOM conditions
- **NFR6**: Validation of embedding dimensions

#### Usability

- **NFR7**: Simple API requiring only 3 parameters
- **NFR8**: Clear error messages for missing embeddings
- **NFR9**: Comprehensive logging of progress

#### Maintainability

- **NFR10**: Single codebase for all similarity edges
- **NFR11**: Well-documented configuration options
- **NFR12**: Unit tests for core algorithms

## 5. Technical Architecture

### 5.1 System Design

```python
UniversalSimilarityBuilder
├── __init__(embedding_field, edge_collection, threshold)
├── load_embeddings() -> (paper_ids, vectors)
├── build_faiss_index(vectors) -> index
├── find_similar_papers(index, batch) -> edges
├── build_edges_faiss() -> edge_count
└── build_edges_gpu_direct() -> edge_count
```

### 5.1a API Specification

```python
class UniversalSimilarityBuilder:
    def __init__(self,
                 embedding_field: str,      # e.g., 'abstract', 'keyword'
                 edge_collection: str,       # e.g., 'abstract_similarity'
                 threshold: float = 0.7,     # Similarity threshold
                 batch_size: int = 50000,    # Papers per batch
                 use_gpu: bool = True,       # GPU acceleration
                 top_k: int = 50,           # Max edges per node
                 db_config: dict = None):    # Database configuration
        """
        Initialize the universal similarity builder.
        
        Args:
            embedding_field: Name of the embedding field to use
            edge_collection: Name of the edge collection to create
            threshold: Minimum similarity score for edge creation
            batch_size: Number of papers to process per batch
            use_gpu: Whether to use GPU acceleration if available
            top_k: Maximum number of edges per node (prevents super-nodes)
            db_config: ArangoDB connection configuration
        """
    
    def build_edges_faiss(self) -> int:
        """
        Build similarity edges using FAISS for efficient similarity search.
        
        Returns:
            Number of edges created
        """
    
    def get_stats(self) -> dict:
        """
        Get statistics about the edge building process.
        
        Returns:
            Dictionary with stats (papers_processed, edges_created, duration, etc.)
        """
```

### 5.2 Algorithm Selection

```python
if dataset_size < 100K and fits_in_gpu_memory:
    use_direct_gpu()  # Fastest for small datasets
elif dataset_size < 1M:
    use_faiss_gpu_flat()  # Good for medium datasets
else:
    use_faiss_gpu_ivf()  # Necessary for large datasets
```

### 5.3 Data Flow

1. Load embeddings from ArangoDB
2. Build FAISS index (or load to GPU)
3. Process in batches to find similarities
4. Filter by threshold and top-k
5. Create unique edges in edge collection
6. Return statistics

## 6. Use Cases

### 6.1 Research: Keyword vs Abstract Comparison

```python
# Build keyword graph
keyword_builder = UniversalSimilarityBuilder(
    embedding_field='keyword_embedding',
    edge_collection='keyword_similarity_experiment',
    threshold=0.65
)
keyword_edges = keyword_builder.build_edges_faiss()

# Build abstract graph with same parameters
abstract_builder = UniversalSimilarityBuilder(
    embedding_field='abstract_embedding',
    edge_collection='abstract_similarity_experiment',
    threshold=0.65  # Same threshold for fair comparison
)
abstract_edges = abstract_builder.build_edges_faiss()

# Compare graph properties
```

### 6.2 Production: Optimal Configuration

```python
# After research determines abstracts are better
builder = UniversalSimilarityBuilder(
    embedding_field='abstract_embedding',
    edge_collection='similarity',
    threshold=0.75,
    top_k=50
)
builder.build_edges_faiss()
```

### 6.3 Future: Multi-Modal Embeddings

```python
# Combine multiple embedding types
for embedding_type in ['title', 'abstract', 'conclusion']:
    builder = UniversalSimilarityBuilder(
        embedding_field=f'{embedding_type}_embedding',
        edge_collection=f'{embedding_type}_similarity',
        threshold=thresholds[embedding_type]
    )
    builder.build_edges_faiss()
```

## 7. Success Metrics

### 7.1 Technical Metrics

- **Code Reduction**: 70% less code than separate builders
- **Performance**: <2 hours for 2.8M papers
- **Memory Efficiency**: <48GB GPU RAM usage
- **Edge Quality**: Consistent similarity scores across runs

### 7.2 Research Metrics

- **Experiment Speed**: 5x faster A/B testing
- **Comparison Fairness**: Identical algorithms for all embeddings
- **Reproducibility**: 100% deterministic with same inputs

### 7.3 Operational Metrics

- **Development Time**: 1 hour to add new embedding type (vs 1 day)
- **Bug Fix Time**: Single fix propagates to all edge types
- **Testing Coverage**: One test suite covers all edge types

## 8. Implementation Plan

### Phase 1: Core Implementation ✅

- [x] Create UniversalSimilarityBuilder class (`core/graph/builders/build_similarity_edges_universal.py`)
- [x] Implement FAISS-based algorithm with IVF indexing for large datasets
- [x] Add GPU acceleration with automatic CPU fallback
- [x] Support configurable parameters (threshold, top_k, batch_size)

### Phase 2: Testing & Validation

- [ ] Unit tests for core functions
- [ ] Integration test with real data (test with 10k paper subset)
- [ ] Performance benchmarking (compare GPU vs CPU, different batch sizes)
- [ ] Memory profiling (track GPU memory usage patterns)

### Phase 3: Research Experiments

- [ ] Keyword vs Abstract comparison
- [ ] Threshold sensitivity analysis
- [ ] Top-k impact study
- [ ] Embedding dimension analysis

### Phase 4: Production Deployment

- [ ] Migrate existing builders
- [ ] Update orchestration scripts
- [ ] Documentation and examples
- [ ] Monitoring and alerts

## 9. Risks and Mitigations

### Risk 1: Memory Overflow

- **Mitigation**: Batch processing, automatic algorithm selection

### Risk 2: Inconsistent Embeddings

- **Mitigation**: Validation of dimensions, null checks

### Risk 3: Performance Regression

- **Mitigation**: Benchmark against specialized builders

### Risk 4: Configuration Complexity

- **Mitigation**: Sensible defaults, clear documentation

## 10. Architecture Decision: Abstract-Only with GraphSAGE

### Final Decision: Abstract Embeddings Only

After careful analysis, we've determined that abstract embeddings alone provide the optimal solution:

#### Why Abstract-Only Wins

1. **GraphSAGE Handles Complexity**
   - Designed for massive, dense graphs through neighborhood sampling
   - Learns which connections matter from rich training signal
   - Converts "hairball" problem into feature-rich training data

2. **Incremental Updates Are Cheap**
   - Initial build: ~30 hours (one-time cost)
   - Daily updates: ~7 minutes (process only new papers)
   - No need for full rebuilds ever

3. **Rich Semantic Signal**
   - Abstract embeddings capture full paper context
   - Better for discovering non-obvious connections
   - GraphSAGE learns to filter noise

4. **Keywords Not Justified**
   - Extraction cost: 24-48 hours
   - Only saves time if rebuilding >26 times
   - With incremental updates, we never rebuild
   - Would reduce semantic richness

### Architecture Overview

```python
# 1. One-time graph build
builder = UniversalSimilarityBuilder(
    embedding_field='abstract',
    edge_collection='abstract_similarity',
    threshold=0.7,  # Rich connections for training
    top_k=100  # Prevent infinite edges
)

# 2. GraphSAGE training (samples from dense graph)
model = GraphSAGE(
    input_dim=2048,
    hidden_dim=256,
    num_layers=2,
    sampling_strategy=[25, 10]  # Sample neighborhoods
)

# 3. Daily incremental updates (~7 min)
updater = IncrementalGraphUpdater()
updater.add_new_papers(todays_papers)

# 4. Weekly model updates (~30 min)
model.fit_incremental(new_nodes, epochs=5)
```

### Other Open Questions

1. **Embedding Combination**: Should we support weighted combination of multiple embeddings?
   - Decision: Test after establishing baseline with individual embeddings
   
2. **Distance Metrics**: Should we support metrics beyond cosine similarity?
   - Decision: Start with cosine, add others if needed
   
3. **Directed Edges**: Should similarity edges be directed based on asymmetric measures?
   - Decision: Keep undirected for simplicity
   
4. **Temporal Decay**: Should older papers have lower similarity weights?
   - Decision: Consider for future enhancement

## 11. Detailed Experiment Design

### 11.1 Keyword Extraction Strategy

**Critical Issue**: ArXiv Kaggle dataset does not include keywords - they must be extracted.

#### Extraction Methods to Evaluate

1. **TF-IDF (Current Baseline)**
   - Pros: Simple, deterministic, no additional models needed
   - Cons: Slow (6+ seconds per batch), CPU-bound, ignores semantic meaning
   - Quality: Extracts frequent terms, may include stop words and non-meaningful terms

2. **KeyBERT (Recommended)**
   ```python
   from keybert import KeyBERT
   
   kw_model = KeyBERT('all-MiniLM-L6-v2')
   keywords = kw_model.extract_keywords(
       abstract,
       keyphrase_ngram_range=(1, 2),
       stop_words='english',
       top_n=10,
       use_mmr=True,  # Maximal Marginal Relevance for diversity
       diversity=0.5
   )
   ```
   - Pros: Semantic understanding, better quality keywords, GPU acceleration
   - Cons: Requires additional model, slower than TF-IDF
   - Quality: Extracts semantically meaningful phrases

3. **YAKE (Yet Another Keyword Extractor)**
   ```python
   import yake
   
   kw_extractor = yake.KeywordExtractor(
       lan="en",
       n=2,  # max ngram size
       dedupLim=0.7,
       top=10
   )
   keywords = kw_extractor.extract_keywords(abstract)
   ```
   - Pros: Fast, unsupervised, no model needed, language-agnostic
   - Cons: Statistical approach, no semantic understanding
   - Quality: Good balance of speed and quality

4. **RAKE (Rapid Automatic Keyword Extraction)**
   - Pros: Very fast, good for technical documents
   - Cons: Can produce long phrases, no semantic understanding
   - Quality: Works well for scientific papers

5. **SciBERT/PubMedBERT** (Domain-Specific)
   - Pros: Trained on scientific text, best quality for ArXiv
   - Cons: Requires specialized models, slower
   - Quality: Highest quality for scientific keywords

#### Recommended Approach

```python
class KeywordExtractor:
    def __init__(self, method='keybert'):
        if method == 'keybert':
            self.model = KeyBERT('sentence-transformers/all-MiniLM-L6-v2')
        elif method == 'yake':
            self.model = yake.KeywordExtractor()
        # etc.
    
    def extract_batch(self, abstracts: List[str]) -> List[List[str]]:
        """Extract keywords from batch of abstracts."""
        # Parallel processing with multiple workers
        # GPU acceleration where possible
        # Return list of keyword lists
```

#### Storage Strategy

```python
# New collection for keyword mappings
'arxiv_keywords': {
    '_key': 'paper_id',
    'keywords': ['machine learning', 'neural networks', ...],
    'extraction_method': 'keybert',
    'extraction_date': '2025-09-12'
}

# Then generate embeddings
'arxiv_keyword_embeddings': {
    '_key': 'paper_id',
    'embedding': [...],  # Jina v4 embedding of concatenated keywords
    'keywords_used': 10
}
```

### 11.2 Keyword vs Abstract Comparison Experiment

**Hypothesis**: Abstract embeddings will produce higher-quality similarity edges than keyword embeddings due to richer semantic content, but keyword embeddings will be dramatically faster for graph operations.

#### Performance Comparison (Expected)

| Metric | Keywords | Abstracts | Ratio |
|--------|----------|-----------|-------|
| Tokens per paper | 6-10 | 150-200 | 20x smaller |
| Embedding size | 2048-dim | 2048-dim | Same |
| FAISS index build | ~1 min | ~15 min | 15x faster |
| Similarity search | ~5 min | ~60 min | 12x faster |
| GPU memory usage | ~2 GB | ~40 GB | 20x less |
| Total graph build | ~10 min | ~2 hours | 12x faster |
| Edge quality | Focused | Comprehensive | Trade-off |

#### Break-even Analysis

```python
# When does keyword extraction pay off?
keyword_extraction_time = 48  # hours (one-time)
abstract_graph_time = 2  # hours per build
keyword_graph_time = 0.17  # hours per build (10 min)

# Break-even point
builds_to_break_even = keyword_extraction_time / (abstract_graph_time - keyword_graph_time)
# = 48 / 1.83 = ~26 graph rebuilds

# If you experiment with >26 different configurations, keywords save time overall
```

#### Experimental Setup

```python
# Parameters to test
configurations = [
    {'embedding': 'keyword', 'thresholds': [0.5, 0.6, 0.65, 0.7, 0.75]},
    {'embedding': 'abstract', 'thresholds': [0.65, 0.7, 0.75, 0.8, 0.85]}
]

# Metrics to collect
metrics = {
    'edge_count': 'Total number of edges created',
    'avg_degree': 'Average connections per node',
    'clustering_coefficient': 'Graph clustering measure',
    'connected_components': 'Number of isolated subgraphs',
    'retrieval_quality': 'Precision@10 for known paper sets'
}
```

#### Evaluation Criteria

1. **Graph Density**: Not too sparse, not too dense (target: 10-50 edges/node)
2. **Component Size**: Fewer isolated components is better
3. **Semantic Coherence**: Papers in same cluster should share topics
4. **Processing Time**: Faster is better for same quality

### 11.2 Hybrid Embedding Experiment

**Hypothesis**: Combining embeddings with different weights can improve quality.

```python
# Weighted combination approach
combined_embedding = (
    0.3 * keyword_embedding +
    0.7 * abstract_embedding
)

# Test different weight combinations
weight_configs = [
    (1.0, 0.0),  # Keywords only
    (0.7, 0.3),  # Keyword-heavy
    (0.5, 0.5),  # Balanced
    (0.3, 0.7),  # Abstract-heavy
    (0.0, 1.0),  # Abstracts only
]
```

## 12. Implementation Strategy

### GraphSAGE-Centric Architecture

Our architecture is designed around GraphSAGE's capabilities to handle dense graphs through sampling:

#### Phase 1: Initial Graph Construction

1. **Build Dense Abstract Graph**
   ```python
   builder = UniversalSimilarityBuilder(
       embedding_field='abstract',
       edge_collection='abstract_similarity',
       threshold=0.7,  # Lower threshold for richer training signal
       top_k=100,  # High but bounded connectivity
       batch_size=50000
   )
   stats = builder.build_edges_faiss()
   ```

2. **Expected Characteristics**
   - ~50-100 edges per node average
   - Some super-nodes with 100+ connections (research hubs)
   - Rich semantic neighborhoods for GraphSAGE training

#### Phase 2: GraphSAGE Training

```python
# GraphSAGE handles the complexity through sampling
graphsage = GraphSAGE(
    input_dim=2048,  # Jina v4 embedding dimension
    hidden_dim=256,
    output_dim=128,
    num_layers=2,
    dropout=0.5
)

# Neighborhood sampling strategy
sampler = NeighborhoodSampler(
    fanouts=[25, 10],  # Sample 25 first-hop, 10 second-hop
    batch_size=1024
)

# Train on sampled subgraphs, not full graph
graphsage.train(sampler, epochs=100)
```

#### Phase 3: Incremental Updates

```python
class IncrementalUpdater:
    def daily_update(self, new_papers):
        # 1. Generate embeddings for new papers (~100/day)
        new_embeddings = embedder.embed(new_papers)  # 1 min
        
        # 2. Find similarities using existing FAISS index
        faiss_index = self.load_index()
        similarities = faiss_index.search(new_embeddings)  # 5 min
        
        # 3. Add edges to ArangoDB
        self.add_edges(similarities)  # 1 min
        
        # 4. Update FAISS index
        faiss_index.add(new_embeddings)
        self.save_index(faiss_index)
        
        # Total: ~7 minutes per day
    
    def weekly_model_update(self):
        # Incremental GraphSAGE training on new nodes
        graphsage.fit_incremental(
            new_nodes=self.weeks_new_papers,
            epochs=5
        )
        # Total: ~30 minutes per week
```

### WHO Variable Implications

In the Conveyance Framework (C = W·R·H/T · Ctx^α):

- **W (WHAT)**: Rich abstract embeddings provide maximum semantic signal
- **R (WHERE)**: Dense graph captures all relevant relationships
- **H (WHO)**: GraphSAGE acts as intelligent agent, learning to navigate complexity
- **T (TIME)**: One-time cost amortized over years of cheap updates
- **Ctx**: Preserved through full abstract context, not reduced to keywords

**The abstract-only approach maximizes C by:**
- Maximizing W (full semantic content)
- Maximizing R (rich connections)
- Optimizing H (GraphSAGE intelligence)
- Minimizing T (through incremental updates)

## 13. Implementation Roadmap

### Immediate Actions (After Ingestion Completes)

#### Day 1: Graph Construction
- Build abstract similarity graph with threshold=0.7, top_k=100
- Monitor GPU memory usage and processing time
- Document edge count and density metrics

#### Day 2-3: GraphSAGE Implementation
- Implement neighborhood sampling
- Train initial GraphSAGE model
- Validate on known paper relationships

#### Day 4: Incremental Update System
- Build FAISS index persistence
- Implement daily update pipeline
- Test with simulated new papers

#### Day 5: Production Deployment
- Deploy incremental updater
- Set up monitoring and alerts
- Document operational procedures

### Keyword Extraction (NOT NEEDED)

Based on our analysis, keyword extraction is not justified because:

1. **Incremental updates eliminate rebuild costs** - We build once, update daily
2. **GraphSAGE handles dense graphs** - Complexity becomes training signal
3. **Abstract embeddings provide richer signal** - Better for research discovery
4. **Time investment not justified** - 24-48 hours for marginal speed gains

### Performance Expectations

Based on current ingestion rate (36.3 papers/sec):
- **Ingestion completion**: ~21 hours total
- **Graph building**: ~2 hours (one-time)
- **GraphSAGE training**: ~4 hours
- **Daily updates**: ~7 minutes
- **Total initial investment**: ~27 hours

After this one-time cost, the system runs with minimal overhead forever.


### Long-term Roadmap

**Q4 2025**
- Multi-modal embeddings (title + abstract + citations)
- Cross-collection similarity (papers to authors)
- Temporal similarity decay

**Q1 2026**
- Incremental update system
- Active learning for threshold tuning
- Graph neural network integration

## 13. Future Enhancements

### 13.1 Advanced Similarity Metrics

Beyond cosine similarity:
- **Euclidean Distance**: For dense embeddings
- **Manhattan Distance**: For sparse features
- **Learned Metrics**: Neural network-based similarity

### 13.2 Incremental Updates

```python
class IncrementalSimilarityBuilder(UniversalSimilarityBuilder):
    def add_new_papers(self, new_paper_ids):
        # Load existing FAISS index
        # Add new embeddings
        # Find similarities only for new papers
        # Merge with existing edges
```

### 13.3 Multi-GPU Scaling

```python
class MultiGPUSimilarityBuilder(UniversalSimilarityBuilder):
    def build_edges_distributed(self):
        # Split embeddings across GPUs
        # Build partial indices
        # Merge results with deduplication
```

### 13.4 Embedding Fusion Strategies

1. **Early Fusion**: Concatenate embeddings before similarity
2. **Late Fusion**: Average similarity scores from different embeddings
3. **Learned Fusion**: Train a model to combine embeddings optimally

### 13.5 Cross-Collection Edges

Build heterogeneous graphs:
- Paper → Author similarity (based on embedding overlap)
- Paper → Institution similarity
- Paper → Topic similarity

## 12. Conclusion

The Universal Similarity Edge Builder represents a significant architectural improvement, replacing multiple specialized builders with a single, flexible module. This enables faster research iteration, easier maintenance, and consistent behavior across all embedding types.

### Key Benefits

- **70% code reduction**
- **5x faster experimentation**
- **Consistent algorithms across all edge types**
- **Future-proof for new embedding types**

### Next Steps

1. Complete testing phase
2. Run keyword vs abstract experiment
3. Deploy optimal configuration to production
4. Deprecate specialized builders
