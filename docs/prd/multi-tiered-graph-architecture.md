# Multi-Tiered Graph Architecture for HADES

## Overview

The HADES graph should reflect the natural structure of academic knowledge, not force artificial connections. Theory-practice bridges emerge from semantic similarity in the embedding space, while the graph structure captures the social, institutional, and intellectual topology of research.

## Graph Tiers

### Tier 1: Social Network
**Captures the human collaboration network**

- **Coauthorship**: Direct collaboration edges
- **Advisory relationships**: PhD advisor-student connections
- **Research groups**: Lab/group membership edges
- **Conference co-attendance**: Shared venue participation

### Tier 2: Institutional Network
**Organizational and geographical structure**

- **Same institution**: Papers from same university/lab
- **Institutional collaborations**: Cross-institution partnerships
- **Funding networks**: Shared grants/funding sources
- **Geographic proximity**: Regional research clusters

### Tier 3: Intellectual Network
**Knowledge structure and flow**

- **Citations**: Direct reference edges (most important)
- **Shared references**: Papers citing same sources
- **Academic fields**: Same ArXiv category/subcategory
- **Keyword overlap**: Shared technical terms
- **Method similarity**: Using same techniques/datasets

### Tier 4: Temporal Network
**Evolution of ideas over time**

- **Temporal proximity**: Papers published within same period
- **Version updates**: ArXiv v1->v2->v3 chains
- **Follow-up work**: Explicit continuation papers
- **Response papers**: Comments, rebuttals, extensions

### Tier 5: Influence Network
**Impact and attention flow**

- **ArXiv Influence Flowers**: Their influence metric visualization
- **Download patterns**: Co-downloaded papers
- **Reading sessions**: Sequential access patterns
- **Social media mentions**: Twitter/Reddit co-discussion

## Implementation Strategy

### Phase 1: Core Academic Graph
```python
def build_core_graph():
    # 1. Citations (from references)
    citation_edges = extract_citations_from_latex()
    
    # 2. Coauthorship (already have)
    coauthor_edges = build_coauthorship()
    
    # 3. Academic fields
    category_edges = build_category_edges()
    
    # 4. Temporal proximity
    temporal_edges = build_temporal_edges(window_days=30)
```

### Phase 2: Institutional Layer
```python
def extract_affiliations():
    # Parse author affiliations from papers
    # Link to institution database (ROR, GRID)
    # Build institutional collaboration network
```

### Phase 3: Influence Integration
```python
def integrate_influence_flowers():
    # ArXiv provides influence metrics
    # Convert to edge weights
    # Add as influence edges
```

## Edge Weighting Strategy

Different edge types should have different weights based on their importance:

```python
EDGE_WEIGHTS = {
    'cites': 1.0,           # Direct citations most important
    'coauthor': 0.8,        # Strong collaboration signal
    'same_institution': 0.4, # Moderate connection
    'same_field': 0.3,      # Weak topical connection
    'temporal': 0.2,        # Temporal coincidence
}
```

## Semantic Discovery Layer

**Critical**: Theory-practice bridges are NOT graph edges. They emerge from:

1. **Embedding similarity**: Cosine distance in Jina space
2. **Cross-domain retrieval**: Query theoretical papers, retrieve practical
3. **Semantic mapping**: Project papers into conveyance space

```python
def find_theory_practice_bridges(paper_id):
    """
    Find bridges through embedding space, not graph edges
    """
    paper_embedding = get_embedding(paper_id)
    
    # Search in different conveyance regions
    theoretical_region = conveyance < 0.3
    practical_region = conveyance > 0.7
    
    if is_theoretical(paper_id):
        # Find practical implementations
        bridges = search_embeddings(
            paper_embedding,
            filter=practical_region
        )
    else:
        # Find theoretical foundations
        bridges = search_embeddings(
            paper_embedding,
            filter=theoretical_region
        )
    
    return bridges
```

## GraphSAGE Configuration

With proper multi-tiered edges, GraphSAGE can learn meaningful representations:

```python
class MultiTierGraphSAGE:
    def __init__(self):
        self.edge_types = [
            'cites', 'coauthor', 'same_field',
            'same_institution', 'temporal'
        ]
        
    def aggregate(self, node, layer):
        # Type-specific aggregation
        aggregations = []
        for edge_type in self.edge_types:
            neighbors = get_neighbors(node, edge_type)
            agg = self.aggregators[edge_type](neighbors)
            aggregations.append(agg)
        
        # Combine with attention mechanism
        return self.combine(aggregations)
```

## Expected Graph Statistics

With proper edge building:

```
Nodes: 2,825,818
Edges by type:
- Citations: ~30M (avg 10-15 per paper)
- Coauthorship: ~5M (already built)
- Same field: ~50M (within categories)
- Temporal: ~20M (30-day windows)
- Institutional: ~10M (estimated)

Total edges: ~115M
Average degree: ~40
Connected components: ~50,000
Largest component: ~2M nodes (70% of graph)
```

## Storage Optimization

Multi-tiered edges in ArangoDB:

```javascript
// Edge collections by tier
db._create("citations", {type: 3});
db._create("coauthorship", {type: 3});
db._create("same_field", {type: 3});
db._create("institutional", {type: 3});
db._create("temporal", {type: 3});
db._create("influence", {type: 3});

// Indexes for fast traversal
db.citations.ensureIndex({type: "persistent", fields: ["_from"]});
db.citations.ensureIndex({type: "persistent", fields: ["_to"]});
```

## Conveyance Framework Alignment

- **W (what)**: Rich multi-tiered structure captures full context
- **R (where)**: Natural topology of academic knowledge
- **H (who)**: GraphSAGE can now learn from proper neighborhoods
- **T (time)**: Efficient traversal with typed edges
- **Ctx**: Each tier adds different context dimension
- **α**: Multi-tier context should achieve α > 1.7

## Next Steps

1. Build citation edges from LaTeX source files
2. Expand category-based edges
3. Extract institutional affiliations
4. Integrate ArXiv influence flowers API
5. Test GraphSAGE on connected subgraph
6. Measure conveyance amplification

The key insight: Let the graph represent the natural academic network structure. Theory-practice bridges emerge from semantic search, not forced graph edges.