# HiRAG Implementation Assessment & Plan

## Current Database State Analysis

### Existing Collections (Database: academy_store)

**ArXiv Collections (Papers)**:
- `arxiv_papers`: 2,493 papers
- `arxiv_chunks`: 42,554 text chunks  
- `arxiv_embeddings`: 43,554 Jina v4 embeddings
- `arxiv_structures`: Document structures
- Supporting: `arxiv_tables`, `arxiv_images`, `arxiv_equations`

**GitHub Collections (Code)**:
- `github_repositories`: 4 repositories (HiRAG, Graph-R1, etc.)
- `github_papers`: 524 code files
- `github_chunks`: 2,177 code chunks
- `github_embeddings`: 2,177 Jina v4 embeddings with coding LoRA
- Edges: `github_repo_files`, `github_has_chunks`, `github_has_embeddings`

**Theory-Practice Bridge Collections (Existing)**:
- `theory_practice_bridges`: Bridge detection storage
- `conveyance_scores`: Conveyance calculations
- `semantic_similarity`: Semantic relationship edges
- `paper_implements_theory`: Implementation relationships

### Gap Analysis vs HiRAG PRD Requirements

#### ✅ **Already Available**:
1. **Raw Papers Collection**: `arxiv_papers` (2,493 papers)
2. **Text Chunks**: `arxiv_chunks` (42,554 chunks)
3. **Embeddings**: Jina v4 embeddings ready (43,554)
4. **Code Repositories**: GitHub integration complete
5. **Bridge Detection Infrastructure**: Basic collections exist

#### ❌ **Missing for HiRAG Implementation**:
1. **Entities Collection**: No extracted entities from papers
2. **Hierarchical Clusters**: No Level 1/Level 2 cluster structure
3. **Relations Collection**: No entity-to-entity relationship edges
4. **Cluster Edges**: No cluster membership/hierarchy edges
5. **Conveyance Weighting**: No materialized edge weights
6. **HiRetrieval Engine**: No three-level retrieval implementation

## Implementation Plan

### Phase 1: Entity Extraction & Relations (Weeks 1-2)

**Goal**: Extract entities from existing paper chunks and establish relationships

**Tasks**:
1. **Create Missing Collections**:
   ```aql
   // New collections needed
   entities          // paper concepts, methods, people
   relations         // entity-to-entity relationships  
   clusters          // hierarchical summary nodes
   cluster_edges     // membership and hierarchy
   query_logs        // performance tracking
   bridge_cache      // precomputed hot bridges
   ```

2. **Entity Extraction Pipeline**:
   - Process existing `arxiv_chunks` (42,554 chunks)
   - Extract: concepts, methods, algorithms, people, datasets
   - Use NER + domain-specific extraction (ML/AI focused)
   - Store in `entities` collection with embeddings

3. **Relationship Discovery**:
   - Citation analysis from paper references
   - Semantic similarity between entities (>0.8 threshold)
   - Implementation relationships (paper→code matches)
   - Store in `relations` with initial weights

### Phase 2: Hierarchical Clustering (HiIndex) (Weeks 3-4)

**Goal**: Build Level 1 and Level 2 hierarchical clusters

**Tasks**:
1. **Level 0 → Level 1 Clustering**:
   - Hybrid clustering: 60% semantic + 40% structural
   - HDBSCAN on entity embeddings (cosine similarity)
   - Parameters: `min_cluster_size=5`, `min_samples=3`
   - Generate cluster summaries (150-200 tokens)

2. **Level 1 → Level 2 Super-Clustering**:
   - Meta-embeddings from L1 summaries
   - Agglomerative clustering (Ward linkage)
   - Target: `n ≈ √(level1_count)` clusters

3. **Cluster Storage**:
   - Store clusters with `layer∈{1,2}`
   - Create `cluster_edges` for membership/hierarchy
   - Generate semantic embeddings for cluster summaries

### Phase 3: Three-Level Retrieval (HiRetrieval) (Weeks 5-6)

**Goal**: Implement Local ↔ Bridge ↔ Global retrieval engine

**Tasks**:
1. **Local Retrieval**:
   ```aql
   // Entity-level matching
   FOR e IN entities
     FILTER e.layer == 0
     FILTER ANALYZER(e.description, "text_en") LIKE @query
     RETURN e
   ```

2. **Global Retrieval**:
   ```aql
   // Cluster expansion from local entities
   FOR e IN @local_entities
     FOR c IN OUTBOUND e cluster_edges
       FILTER c.layer >= 1
       RETURN c
   ```

3. **Bridge Retrieval**:
   ```aql
   // Weighted shortest paths
   FOR s IN @locals
     FOR t IN @globals
       FOR p IN OUTBOUND s relations
         OPTIONS { weightAttribute: 'conveyance_weight' }
         SHORTEST_PATH TO t
         LIMIT 5
         RETURN p
   ```

### Phase 4: Conveyance Scoring (Weeks 7-8)

**Goal**: Implement C = (W·R·H/T)·Ctx^α framework

**Tasks**:
1. **Edge Weight Materialization**:
   - Base weights by relation type
   - Temporal decay: `exp(-age_days/365)`
   - Embedding similarity: `sim²`
   - Cross-layer bonus: `×1.5`

2. **Conveyance Metrics**:
   - **W (Semantic)**: Weighted cosine similarity
   - **R (Graph)**: Inverse path length + edge weights
   - **H/T (Efficiency)**: Model capability / latency
   - **Ctx = ¼(L+I+A+G)**: Context components
   - **α = 1.6**: Empirical amplification factor

3. **Query Engine Integration**:
   - Real-time conveyance scoring
   - Performance optimization (caching, indexing)
   - Feedback loop for weight adaptation

## Expected Outcomes

### Performance Targets (from PRD):
- **Bridge Discovery Rate**: ≥ 0.80 (vs 0.45 baseline)
- **Context Coherence**: ≥ 0.85 (vs 0.65 baseline)  
- **Query Latency**: ≤ 1.8s (vs 1.2s baseline)
- **Semantic Quality**: > 0.85 top-k relevance
- **Alpha Observation**: α ≈ 1.6 context amplification

### Database Scale Projections:
- **Entities**: ~50,000 from 42,554 chunks
- **Relations**: ~200,000 relationships
- **Level 1 Clusters**: ~500 topic clusters
- **Level 2 Clusters**: ~25 super-clusters
- **Total Storage**: ~500GB including indexes

### Theory-Practice Bridge Examples:
- PageRank paper (1998) → NetworkX implementation
- Transformer paper (2017) → HuggingFace implementation  
- GAT paper (2017) → PyTorch Geometric implementation

## Risk Mitigation

1. **Performance**: Implement caching, materialized views, strategic indexing
2. **Quality**: Human validation on curated ground truth (500+ pairs)
3. **Scale**: Incremental processing, efficient AQL queries
4. **Maintenance**: Scheduled reclustering, weight decay updates

## Success Criteria

✅ **MVP (Week 8)**:
- Three-level retrieval working on 10k subset
- Bridge discovery > baseline
- P95 latency < 2s

✅ **Full Release (Week 12)**:
- Full corpus processing (2,493 papers)
- 25% uplift in bridge discovery
- α ≈ 1.6 empirically observed
- Production-ready performance