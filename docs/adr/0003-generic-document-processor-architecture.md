# Architecture Decision Record: Generic Document Processor Architecture

**Status**: Implemented  
**Date**: 2025-08-27  
**Version**: 1.0  
**Authors**: System Architecture, Claude Code

## Context

The HADES-Lab system required a fundamental architectural separation between source-specific document management (ArXiv, Semantic Scholar, etc.) and generic document processing operations (extraction, embedding, storage). The previous monolithic architecture tightly coupled ArXiv-specific logic with processing operations, creating barriers to:

- Supporting multiple document sources
- Independent testing of processing components  
- Reusing processing optimizations across sources
- Scaling processing operations independently

The existing ArXiv pipeline achieved excellent performance (11.3 papers/minute, 100% success rate), but this performance was locked within ArXiv-specific code paths.

## Decision

We have implemented a **Generic Document Processor** architecture that creates a clear separation of concerns:

### 1. Generic Document Processor (`core.processors.generic_document_processor`)

A source-agnostic processing engine that handles:
- PDF extraction using RobustExtractor with timeout protection
- Jina v4 embedding generation with late chunking
- ArangoDB storage with atomic transactions
- Phase-separated architecture (Extraction → Embedding → Storage)

### 2. Source-Specific Document Managers  

Pattern exemplified by `ArXivDocumentManager`:
- Source-specific document discovery and validation
- Metadata extraction and preparation
- Creation of `DocumentTask` objects for generic processing
- No knowledge of processing internals

### 3. Configurable Collection Prefixes

The processor accepts a `collection_prefix` parameter enabling:
- ArXiv documents: `arxiv_papers`, `arxiv_chunks`, `arxiv_embeddings`
- Future sources: `s2_papers`, `local_papers`, etc.
- Complete isolation between source datasets

## Implementation Details

### Core Architecture Components

```python
@dataclass
class DocumentTask:
    """Universal document task representation."""
    document_id: str          # Unique identifier (arxiv_id, DOI, filename)
    pdf_path: str            # Absolute path to PDF
    latex_path: Optional[str] # Optional LaTeX source
    metadata: Dict[str, Any] # Source-specific metadata

class GenericDocumentProcessor:
    """Source-agnostic document processing engine."""
    def __init__(self, config: Dict[str, Any], collection_prefix: str):
        self.collections = {
            'papers': f'{collection_prefix}_papers',
            'chunks': f'{collection_prefix}_chunks', 
            'embeddings': f'{collection_prefix}_embeddings',
            'structures': f'{collection_prefix}_structures'
        }
    
    def process_documents(self, tasks: List[DocumentTask]) -> Dict[str, Any]:
        # Phase 1: Extraction (with RobustExtractor)
        # Phase 2: Embedding (with Jina v4)  
        # Returns comprehensive processing results
```

### Process Flow Architecture

```
DocumentManager → DocumentTask[] → GenericProcessor → ArangoDB
     ↑                ↑                    ↑             ↑
Source-Specific   Universal     Source-Agnostic    Collection-
 Discovery       Interface        Processing        Specific
```

### Worker Process Architecture

The processor implements a dual-phase worker model:

**Extraction Phase Workers:**
- Initialize `RobustExtractor` with timeout protection
- GPU assignment via `CUDA_VISIBLE_DEVICES` 
- Process documents to staging directory (`/dev/shm/acid_staging`)
- Configurable worker count (default: 8 workers)

**Embedding Phase Workers:**
- Initialize `JinaV4Embedder` with fp16 optimization
- Process staged documents with late chunking
- Store to ArangoDB with atomic transactions
- GPU memory management and cache clearing

### Configuration Flexibility

The processor supports multiple configuration patterns:
```yaml
# Option 1: New nested format
phases:
  extraction:
    workers: 8
    gpu_devices: [0, 1]
  embedding:
    workers: 4
    
# Option 2: Legacy flat format  
extraction:
  workers: 8
embedding:
  workers: 4
```

## Consequences

### Positive Outcomes

1. **Code Reusability**: 70%+ of processing code is now source-agnostic
2. **Performance Preservation**: All optimizations (parallel processing, GPU utilization, atomic transactions) remain available to all sources
3. **Independent Testing**: Generic processor can be tested with synthetic documents
4. **Memory Efficiency**: Staging directory configuration allows RamFS utilization
5. **Scalability**: Worker counts and GPU assignment configurable per deployment

### Implementation Evidence

Performance validation with ArXiv documents (100 papers processed):
- **Extraction Phase**: 36+ papers/minute with 8 CPU workers  
- **Embedding Phase**: 8+ papers/minute with 8 GPU workers
- **End-to-End**: 11.3 papers/minute maintained
- **Success Rate**: 100% with atomic transactions
- **Chunks Generated**: ~1585 chunks (15.8 avg per document)

### Architectural Benefits

1. **Source Independence**: Adding new sources requires only implementing a document manager
2. **Processing Isolation**: Source-specific bugs cannot affect core processing
3. **Configuration Granularity**: Different sources can use different processing parameters
4. **Resource Management**: GPU and memory allocation handled at generic level

### Trade-offs

1. **Increased Abstraction**: More layers between source and processing
2. **Configuration Complexity**: Multiple config formats supported for compatibility  
3. **Documentation Requirements**: Clear patterns needed for new source implementations

## Technical Implementation Details

### RobustExtractor Integration

The processor integrates with `RobustExtractor` providing:
- Timeout protection (configurable, default 30s)
- PyMuPDF fallback for problematic PDFs
- Process isolation preventing crashes from affecting other workers

### Jina v4 Embedding Strategy

- **Late Chunking**: Process full documents (32k tokens) before chunking
- **Chunk Configuration**: 1000 tokens with 200 token overlap
- **Memory Management**: Explicit GPU cache clearing between documents
- **FP16 Optimization**: Reduces memory usage without quality loss

### ArangoDB Integration

- **Atomic Transactions**: All-or-nothing storage guarantees
- **Lock Management**: Prevents concurrent processing of same document
- **Collection Naming**: Prefix-based separation enabling multi-source support

## Migration and Usage Patterns

### For New Document Sources

```python
# 1. Implement source-specific manager
class NewSourceManager:
    def prepare_documents(self) -> List[DocumentTask]:
        # Source-specific document discovery
        return [DocumentTask(...), ...]

# 2. Instantiate generic processor
processor = GenericDocumentProcessor(
    config=load_config(),
    collection_prefix="newsource"
)

# 3. Process documents
results = processor.process_documents(manager.prepare_documents())
```

### For Existing ArXiv Pipeline

The ArXiv pipeline now uses this architecture via `ArXivPipelineV2`, maintaining full compatibility with existing configurations while gaining all architectural benefits.

## Theoretical Framework Alignment

This architecture strengthens our Information Reconstructionism implementation:

- **WHERE Dimension**: Preserved through collection prefixes and staging directories
- **WHAT Dimension**: Standardized through generic embedding pipeline
- **CONVEYANCE Dimension**: Maximized through robust extraction with fallbacks
- **Observer Boundaries**: Source managers serve as translation points between external data and internal processing

In Actor-Network Theory terms, the `GenericDocumentProcessor` serves as a universal translator, enabling heterogeneous document sources to participate in a standardized processing network.

## Future Extensions

1. **Multi-Source Processing**: Single processor instance handling multiple sources simultaneously
2. **Processing Strategies**: Pluggable extraction and embedding strategies per source
3. **Resource Optimization**: Dynamic worker allocation based on document characteristics
4. **Monitoring Integration**: Source-specific metrics while maintaining processing efficiency

## Recommendations

1. **Standardize on New Architecture**: All new document sources should use this pattern
2. **Gradual Migration**: Existing pipelines should migrate when convenient  
3. **Configuration Consolidation**: Eventually deprecate legacy flat configuration format
4. **Documentation Priority**: Create templates and examples for new source implementation

## References

- Implementation: `core/processors/generic_document_processor.py`
- ArXiv Manager: `tools/arxiv/arxiv_document_manager.py` 
- Pipeline v2: `tools/arxiv/pipelines/arxiv_pipeline_v2.py`
- Test Results: `arxiv_pipeline_v2_results_20250826_162353.json`

---

*This ADR documents the fundamental architectural separation enabling HADES-Lab to process documents from multiple sources while maintaining the performance and reliability characteristics of the original ArXiv-specific implementation.*