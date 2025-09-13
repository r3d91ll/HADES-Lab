# Available Edge Types for HADES Graph

## Currently Available

### 1. **Coauthorship** (12.9M edges available)
- Source: `paper_authors` edge collection
- 2.1M unique authors
- Average 6 papers per author
- Most prolific: ~3000 papers

### 2. **Category/Field Connections** (estimated ~50M edges)
- Source: `categories` field in papers
- 176 unique categories
- Every paper has 1-3 categories
- Major fields: hep-ph, astro-ph, cs.*, math.*

### 3. **Journal Co-publication** (907K papers)
- Source: `journal_ref` field
- Papers published in same journal
- Strong indicator of field similarity

### 4. **DOI Connections** (1.26M papers)
- Source: `doi` field
- Can link to external citation databases
- Published vs preprint versions

### 5. **Submitter Networks** (2.8M papers)
- Source: `submitter` field
- Often the corresponding author
- Different from full author list

### 6. **Version Evolution** (1.1M papers with v2+)
- Source: `versions` field
- Connect different versions of same paper
- Shows paper evolution over time
- Some papers have 100+ versions!

### 7. **License Type Groups** (2.37M papers)
- Source: `license` field
- Papers with same license terms
- Indicates institutional policies

### 8. **Comment-based Connections** (2.07M papers)
- Source: `comments` field
- Contains conference info, page counts
- Can extract:
  - Conference co-presentation
  - Workshop participation
  - "Accepted to NIPS 2023" etc.

### 9. **Citations** (Limited - 112 raw, 10 resolved)
- Source: `citations_raw`, `citations_resolved`
- Need to extract from LaTeX/PDFs
- Most valuable but requires processing

### 10. **Bibliography Connections** (64 entries)
- Source: `bibliography_entries`
- Papers citing same references
- Limited data currently

## Missing But Potentially Extractable

### 1. **Institutional Affiliations**
- Collections exist but empty
- Must extract from:
  - Author names (sometimes include affiliation)
  - Paper text/headers
  - Submitter email domains

### 2. **Funding/Grant Connections**
- Extract from acknowledgments
- NSF, NIH, ERC grant numbers
- Report numbers might indicate labs

### 3. **Topic/Keyword Connections**
- Extract from abstracts
- Use NLP to find key concepts
- Link papers with similar terminology

### 4. **Temporal Proximity** (Implementable)
- Use `update_date` or version dates
- Connect papers published within time windows
- Captures "zeitgeist" effects

## Recommended Build Priority

1. **Fix coauthorship** using `paper_authors` ✓
2. **Category edges** for all 2.8M papers (in progress)
3. **Journal co-publication** (easy, high value)
4. **Submitter networks** (easy, adds connections)
5. **Version evolution** (connects paper versions)
6. **Temporal proximity** (captures trends)
7. **Extract conference from comments** (medium effort)
8. **Extract institutions** (high effort, high value)

## Edge Weight Strategy

```python
EDGE_WEIGHTS = {
    'coauthorship': 0.8,        # Strong collaboration
    'same_category': 0.3,        # Field similarity
    'same_journal': 0.5,         # Publication venue
    'same_submitter': 0.6,       # Usually corresponding author
    'version_evolution': 1.0,    # Same paper!
    'temporal_proximity': 0.2,   # Weak time correlation
    'same_conference': 0.4,      # Event co-occurrence
    'citation': 1.0,             # Direct knowledge flow
}
```

## Expected Graph Statistics

With all edge types:
- **Nodes**: 2.8M papers
- **Edges**: ~100M+ estimated
- **Average degree**: ~70
- **Connected components**: Very few isolated nodes
- **Largest component**: >99% of graph

This rich multi-relational graph will enable GraphSAGE to learn nuanced representations capturing the full academic knowledge structure!