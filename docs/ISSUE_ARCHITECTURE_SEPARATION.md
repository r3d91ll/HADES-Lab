# Architecture Issue: Separate Generic PDF Processing from ArXiv-Specific Logic

## Problem Statement

The current implementation tightly couples ArXiv-specific functionality with generic PDF/LaTeX processing capabilities. This prevents us from easily adding other paper sources (Semantic Scholar, PubMed, bioRxiv, local PDFs, etc.) without duplicating the entire processing pipeline.

## Current Situation

### Mixed Responsibilities in `tools/arxiv/`

1. **ArXiv-Specific Logic**:
   - Metadata extraction from ArXiv JSON
   - Version tracking (v1, v2, etc.)
   - Category management (cs.LG, math.CO, etc.)
   - ArXiv ID formatting and parsing
   - OAI-PMH protocol handling

2. **Generic Processing** (Should be reusable):
   - PDF extraction via Docling
   - LaTeX source processing
   - Text chunking strategies
   - Embedding generation with Jina v4
   - ACID-compliant database storage
   - GPU worker pool management

## Proposed Solution

### 1. Create Generic Document Processor

**Location**: `core/processors/document_processor.py`

```python
class DocumentProcessor:
    """
    Generic document processor for any PDF/LaTeX source.
    Handles extraction, chunking, and embedding.
    """
    
    def process_pdf(self, pdf_path: str, latex_path: Optional[str] = None) -> ProcessingResult:
        """Process any PDF with optional LaTeX source."""
        # 1. Extract with Docling
        # 2. Merge LaTeX if provided
        # 3. Generate chunks
        # 4. Create embeddings
        # 5. Return structured result
        pass
```

### 2. Simplify ArXiv Tool

**Location**: `tools/arxiv/arxiv_manager.py`

```python
class ArXivManager:
    """
    ArXiv-specific management only.
    Delegates processing to generic DocumentProcessor.
    """
    
    def __init__(self):
        self.processor = DocumentProcessor()
        
    def process_arxiv_paper(self, arxiv_id: str) -> Result:
        # 1. Get ArXiv metadata
        # 2. Find PDF/LaTeX paths
        # 3. Call generic processor
        # 4. Add ArXiv-specific metadata
        # 5. Store with ArXiv context
        pass
```

### 3. Enable Other Sources

**Example**: `tools/semantic_scholar/`

```python
class SemanticScholarManager:
    """
    Semantic Scholar paper management.
    Uses same DocumentProcessor as ArXiv.
    """
    
    def __init__(self):
        self.processor = DocumentProcessor()  # Same processor!
        
    def process_s2_paper(self, s2_id: str) -> Result:
        # Different metadata source, same processing pipeline
        pass
```

## Database Schema Extension

### Current SQLite Schema
```sql
-- paper_tracking table (current)
CREATE TABLE paper_tracking (
    arxiv_id TEXT PRIMARY KEY,  -- ArXiv-specific!
    title TEXT,
    pdf_path TEXT,
    ...
);
```

### Proposed Generic Schema
```sql
-- papers table (generic)
CREATE TABLE papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,  -- 'arxiv', 'semantic_scholar', 'pubmed', 'local'
    source_id TEXT NOT NULL,  -- Original ID from source
    title TEXT,
    pdf_path TEXT,
    latex_path TEXT,
    process_date TIMESTAMP,
    processing_status TEXT,
    UNIQUE(source, source_id)
);

-- source_metadata table (source-specific data)
CREATE TABLE source_metadata (
    paper_id INTEGER REFERENCES papers(id),
    source TEXT,
    metadata JSON,  -- Flexible storage for source-specific fields
    PRIMARY KEY(paper_id, source)
);
```

## Implementation Plan

### Phase 1: Extract Generic Components (Week 1)
1. Create `core/processors/document_processor.py`
2. Move Docling extraction logic
3. Move chunking strategies
4. Move embedding generation
5. Create clean interfaces

### Phase 2: Refactor ArXiv Tool (Week 2)
1. Create `ArXivManager` wrapper
2. Remove generic logic from arxiv_pipeline.py
3. Update configs to use new structure
4. Test with existing workflows

### Phase 3: Add New Source Example (Week 3)
1. Create `tools/semantic_scholar/` as proof of concept
2. Implement S2 API integration
3. Use same DocumentProcessor
4. Verify shared processing works

### Phase 4: Database Migration (Week 4)
1. Create new generic schema
2. Write migration script from old to new
3. Update all queries
4. Test thoroughly

## Benefits

1. **Reusability**: One processing pipeline for all PDF sources
2. **Maintainability**: Fix bugs once, benefit everywhere
3. **Extensibility**: Add new sources easily
4. **Clarity**: Clear separation of concerns
5. **Testing**: Test generic processing independently
6. **Performance**: Optimize one pipeline, all sources benefit

## Risks and Mitigations

### Risk 1: Breaking Existing Workflows
**Mitigation**: Keep backward-compatible wrapper during transition

### Risk 2: Performance Regression
**Mitigation**: Benchmark before/after, maintain same optimizations

### Risk 3: Loss of ArXiv-Specific Optimizations
**Mitigation**: Allow source-specific hints to processor

## Success Metrics

1. Can process ArXiv papers with same performance
2. Can add new source in < 1 day
3. Code duplication reduced by > 60%
4. Single test suite covers all sources
5. Clear documentation for adding sources

## Example Usage After Refactor

```python
# ArXiv processing
arxiv_mgr = ArXivManager()
result = arxiv_mgr.process("2301.00303")  # Handles ArXiv specifics

# Semantic Scholar processing
s2_mgr = SemanticScholarManager()
result = s2_mgr.process("649def34f8be52c8b66281af98ae884c09aef38b")  # S2 paper ID

# Local PDF processing
local_mgr = LocalPDFManager()
result = local_mgr.process("/path/to/paper.pdf")  # Just a file path

# All use the same underlying DocumentProcessor!
```

## Decision Required

Should we proceed with this architectural separation? 

**Pros**:
- Clean architecture
- Future-proof for multiple sources
- Easier to maintain
- Better testing

**Cons**:
- Refactoring effort required
- Risk of breaking existing code
- Need to update documentation
- Team needs to learn new structure

## Recommendation

**YES** - Proceed with separation. The current coupling will become a major bottleneck as we add more paper sources. Better to refactor now while the codebase is manageable than wait until we have multiple sources with duplicated code.

---

*Issue Created: January 2025*  
*Priority: HIGH*  
*Estimated Effort: 4 weeks*  
*Impact: Architectural improvement enabling multi-source paper processing*