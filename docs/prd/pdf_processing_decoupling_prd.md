# PRD: PDF Processing Decoupling

## Problem Statement

PDF processing logic is currently tightly coupled with ArXiv-specific code in `tools/arxiv/pipelines/arxiv_pipeline.py`. This prevents other data sources (GitHub, HiRAG, future tools) from leveraging the same PDF processing capabilities without importing ArXiv-specific dependencies.

## Solution

Create a generic `PDFProcessor` in `core/processors/` that handles:
1. PDF extraction (via Docling)
2. Document chunking with late chunking
3. Embedding generation (via Jina v4)
4. Storage to ArangoDB

The ArXiv pipeline will then use this generic processor, passing ArXiv-specific metadata as parameters.

## Requirements

### Functional Requirements

1. **Generic PDF Processing Pipeline**
   - Accept PDF file path as input
   - Extract content using DoclingExtractor
   - Generate embeddings using JinaV4Embedder
   - Store results in ArangoDB collections
   - Return processing status and metadata

2. **Phase Separation**
   - Maintain extraction → embedding phase separation
   - Support GPU memory management between phases
   - Enable parallel processing with worker pools

3. **Configuration-Driven**
   - Accept configuration for batch sizes, workers, GPU settings
   - Support both YAML config files and programmatic configuration

4. **Source Agnostic**
   - No hardcoded references to ArXiv IDs or metadata
   - Accept generic document metadata as parameters
   - Support custom collection names in ArangoDB

### Non-Functional Requirements

1. **Performance**
   - Maintain current processing rates (11.3 papers/minute)
   - Support GPU acceleration for both phases
   - Efficient memory management with staging

2. **Reusability**
   - Clean API for use by multiple tools
   - No dependencies on specific data sources
   - Pluggable storage backends (future)

3. **Maintainability**
   - Clear separation of concerns
   - Comprehensive logging
   - Error recovery and checkpointing

## Architecture

### Core Components

```python
# core/processors/pdf_processor.py
class PDFProcessor:
    """Generic PDF processing pipeline.
    
    Implements WHAT dimension of Conveyance Framework:
    - Extracts content (DoclingExtractor)
    - Generates embeddings (JinaV4Embedder)
    - Stores to backend (ArangoStorage)
    """
    
    def __init__(self, config: ProcessorConfig):
        self.extractor = DoclingExtractor(...)
        self.embedder = JinaV4Embedder(...)
        self.storage = ArangoStorage(...)
    
    def process_pdf(self, 
                    pdf_path: str, 
                    document_id: str,
                    metadata: Dict[str, Any]) -> ProcessingResult:
        """Process single PDF through full pipeline."""
        
    def process_batch(self,
                     pdf_batch: List[PDFTask]) -> List[ProcessingResult]:
        """Process batch of PDFs with parallel workers."""
```

### Integration Points

```python
# tools/arxiv/pipelines/arxiv_pipeline.py
from core.processors.pdf_processor import PDFProcessor

class ArxivPipeline:
    def __init__(self):
        self.pdf_processor = PDFProcessor(config)
    
    def process_arxiv_paper(self, arxiv_id: str):
        pdf_path = f"/bulk-store/arxiv-data/pdf/{arxiv_id}.pdf"
        metadata = self.get_arxiv_metadata(arxiv_id)
        
        result = self.pdf_processor.process_pdf(
            pdf_path=pdf_path,
            document_id=arxiv_id,
            metadata=metadata
        )
```

## Success Metrics

1. **Functional Success**
   - ArXiv pipeline continues working with same performance
   - GitHub tool can process PDFs without ArXiv dependencies
   - HiRAG can process arbitrary PDFs

2. **Performance Metrics**
   - Processing rate: ≥11.3 documents/minute
   - GPU memory usage: ≤8GB per worker
   - Zero performance regression

3. **Code Quality**
   - 70%+ code reuse across tools
   - Clean separation of concerns
   - Comprehensive test coverage

## Implementation Plan

### Phase 1: Core PDFProcessor
1. Create `core/processors/pdf_processor.py`
2. Extract generic logic from `arxiv_pipeline.py`
3. Implement phase separation and worker pools
4. Add configuration management

### Phase 2: Storage Abstraction
1. Create storage interface in `core/framework/storage/`
2. Support collection name customization
3. Implement atomic operations

### Phase 3: ArXiv Integration
1. Update `arxiv_pipeline.py` to use PDFProcessor
2. Pass ArXiv-specific metadata
3. Maintain backward compatibility

### Phase 4: Testing & Validation
1. Unit tests for PDFProcessor
2. Integration tests with ArXiv pipeline
3. Performance benchmarking
4. GitHub tool integration test

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regression | High | Benchmark before/after, profile code |
| Breaking ArXiv pipeline | High | Comprehensive testing, gradual rollout |
| GPU memory issues | Medium | Maintain phase separation, monitor usage |
| API complexity | Medium | Simple, intuitive interface design |

## Dependencies

- `core/framework/extractors/docling_extractor.py` (existing)
- `core/framework/embedders.py` (existing)
- `core/framework/storage.py` (existing)
- `tools/arxiv/pipelines/arango_db_manager.py` (to be generalized)

## Timeline

- **Week 1**: Core PDFProcessor implementation
- **Week 2**: Storage abstraction and ArXiv integration
- **Week 3**: Testing and performance validation
- **Week 4**: Documentation and rollout

## Open Questions

1. Should we support multiple storage backends immediately or just ArangoDB?
2. How should we handle document-specific preprocessing (e.g., LaTeX for ArXiv)?
3. Should checkpoint/resume functionality be in core or tool-specific?

## Approval

- [ ] Engineering Lead
- [ ] Architecture Review
- [ ] Performance Team