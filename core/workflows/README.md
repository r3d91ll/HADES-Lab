# Workflows - Pipeline Orchestration System

The workflows module provides orchestration for complex, multi-phase document processing pipelines, managing state, coordinating components, and ensuring atomic operations while optimizing the Conveyance equation through efficient pipeline execution.

## Overview

Workflows orchestrate the interaction between extractors, embedders, storage systems, and monitoring components, implementing the complete Conveyance Framework pipeline. They manage the flow of information through all dimensions (W×R×H×T) while maintaining Context coherence.

## Architecture

```
workflows/
├── workflow_base.py       # Abstract base workflow class
├── workflow_pdf.py        # PDF processing workflow
├── workflow_pdf_batch.py  # Batch PDF processing
├── state/                # State management
│   └── state_manager.py  # Pipeline state tracking
├── storage/              # Storage abstractions
│   ├── storage_base.py   # Storage interface
│   └── storage_local.py  # Local filesystem storage
└── __init__.py          # Public API exports
```

## Theoretical Foundation

### Conveyance Framework Implementation

```python
C = (W·R·H/T)·Ctx^α
```

Workflows optimize all dimensions simultaneously:
- **WHAT (W)**: Quality extraction and processing
- **WHERE (R)**: Structured storage and retrieval
- **WHO (H)**: Optimal component selection
- **TIME (T)**: Pipeline efficiency and parallelization
- **Context (Ctx)**: State coherence across phases

### Actor-Network Theory

Workflows act as "intermediaries" that translate between different actors (extractors, embedders, storage) while maintaining information integrity across network boundaries.

## Core Components

### WorkflowBase

Abstract base class for all workflows:

```python
from core.workflows import WorkflowBase, WorkflowConfig

class CustomWorkflow(WorkflowBase):
    """Custom processing workflow."""

    def setup(self):
        """Initialize components."""
        self.extractor = self.create_extractor()
        self.embedder = self.create_embedder()
        self.storage = self.create_storage()

    def process(self, input_path: str) -> WorkflowResult:
        """Process single item."""
        # Extract
        content = self.extractor.extract(input_path)

        # Process
        embeddings = self.embedder.embed(content.chunks)

        # Store
        self.storage.save(content, embeddings)

        return WorkflowResult(success=True, data=content)

    def process_batch(self, paths: List[str]) -> List[WorkflowResult]:
        """Process multiple items."""
        return [self.process(path) for path in paths]
```

### PDFWorkflow

Specialized workflow for PDF processing:

```python
from core.workflows import PDFWorkflow

# Initialize workflow
workflow = PDFWorkflow(
    config={
        "extraction": {
            "type": "docling",
            "use_ocr": True,
            "chunk_size": 1000
        },
        "embedding": {
            "model": "jinaai/jina-embeddings-v3",
            "device": "cuda",
            "batch_size": 32
        },
        "storage": {
            "type": "arango",
            "database": "academy_store",
            "collection": "papers"
        }
    }
)

# Process single PDF
result = workflow.process("research_paper.pdf")

print(f"Success: {result.success}")
print(f"Chunks: {len(result.chunks)}")
print(f"Processing time: {result.duration:.2f}s")

# Process batch
results = workflow.process_batch(
    pdf_files,
    num_workers=8,
    show_progress=True
)
```

### BatchWorkflow

High-throughput batch processing:

```python
from core.workflows import BatchWorkflow

workflow = BatchWorkflow(
    max_workers=32,
    batch_size=24,
    use_gpu=True,
    checkpoint_interval=100  # Save state every 100 items
)

# Process large dataset with checkpointing
results = workflow.process_dataset(
    input_dir="/data/papers",
    output_dir="/data/processed",
    resume_from_checkpoint=True  # Resume if interrupted
)

print(f"Processed: {results.total_processed}")
print(f"Failed: {results.total_failed}")
print(f"Throughput: {results.items_per_minute:.2f} items/min")
```

### StateManager

Pipeline state tracking and recovery:

```python
from core.workflows.state import StateManager

state = StateManager(
    checkpoint_dir="/tmp/pipeline_state",
    save_interval=60  # Save every 60 seconds
)

# Track processing state
state.mark_started("doc_123")
state.mark_completed("doc_123", result_data)

# Check status
if state.is_completed("doc_123"):
    print("Already processed")

# Get pending items
pending = state.get_pending()
print(f"{len(pending)} items pending")

# Recovery after crash
state.load_checkpoint()
resumed_items = state.get_in_progress()
for item in resumed_items:
    reprocess(item)
```

## Workflow Patterns

### Basic Pipeline

```python
from core.workflows import PDFWorkflow

# Simple sequential processing
workflow = PDFWorkflow()

for pdf_file in pdf_files:
    try:
        result = workflow.process(pdf_file)
        print(f"Processed: {pdf_file}")
    except Exception as e:
        print(f"Failed: {pdf_file} - {e}")
```

### Parallel Processing

```python
from core.workflows import ParallelWorkflow
from concurrent.futures import ProcessPoolExecutor

workflow = ParallelWorkflow(
    num_workers=16,
    chunk_size=10  # Items per worker
)

# Process in parallel
with ProcessPoolExecutor(max_workers=16) as executor:
    futures = workflow.submit_batch(executor, documents)
    results = workflow.collect_results(futures)

print(f"Success rate: {results.success_rate:.2%}")
```

### Phased Execution

```python
from core.workflows import PhasedWorkflow

# Separate extraction and embedding phases
workflow = PhasedWorkflow(
    phases=["extraction", "embedding", "storage"]
)

# Phase 1: Extract all documents
with workflow.phase("extraction") as phase:
    extracted = phase.process_all(
        documents,
        workers=32,  # CPU-intensive
        batch_size=24
    )

# Phase 2: Generate embeddings
with workflow.phase("embedding") as phase:
    embeddings = phase.process_all(
        extracted,
        workers=8,  # GPU-limited
        batch_size=32
    )

# Phase 3: Store results
with workflow.phase("storage") as phase:
    phase.process_all(embeddings, workers=4)
```

### Error Recovery

```python
from core.workflows import ResilientWorkflow

workflow = ResilientWorkflow(
    max_retries=3,
    retry_delay=5.0,
    fallback_strategies=["docling", "pypdf", "ocr"]
)

# Process with automatic retry and fallback
results = workflow.process_with_recovery(
    documents,
    on_error="continue",  # or "stop", "retry"
    save_failed="/tmp/failed_docs"
)

# Review failed items
for failed in results.failed_items:
    print(f"Failed: {failed.path}")
    print(f"Error: {failed.error}")
    print(f"Attempts: {failed.attempts}")
```

### Streaming Workflow

```python
from core.workflows import StreamingWorkflow

# Process documents as stream
workflow = StreamingWorkflow(
    buffer_size=100,
    flush_interval=30  # Flush every 30 seconds
)

# Stream processing
for document in workflow.stream_process(document_source):
    # Results available immediately
    print(f"Processed: {document.id}")

    # Periodic flush to storage
    if workflow.should_flush():
        workflow.flush_to_storage()
```

## Advanced Features

### Custom Components

```python
from core.workflows import WorkflowBase
from core.extractors import ExtractorBase
from core.embedders import EmbedderBase

class CustomWorkflow(WorkflowBase):
    """Workflow with custom components."""

    def create_extractor(self) -> ExtractorBase:
        """Create custom extractor."""
        if self.config.get("use_custom_extractor"):
            return MyCustomExtractor(self.config.extraction)
        return super().create_extractor()

    def create_embedder(self) -> EmbedderBase:
        """Create custom embedder."""
        if self.config.get("use_custom_embedder"):
            return MyCustomEmbedder(self.config.embedding)
        return super().create_embedder()

    def post_process(self, result: WorkflowResult) -> WorkflowResult:
        """Custom post-processing."""
        # Add custom metadata
        result.metadata["processed_by"] = "CustomWorkflow"
        result.metadata["version"] = "1.0.0"

        # Custom validation
        if not self.validate_result(result):
            raise ValueError("Result validation failed")

        return result
```

### Conditional Routing

```python
from core.workflows import ConditionalWorkflow

workflow = ConditionalWorkflow()

# Define routing rules
workflow.add_route(
    condition=lambda doc: doc.page_count > 100,
    handler=LargeDocumentWorkflow()
)

workflow.add_route(
    condition=lambda doc: doc.has_equations,
    handler=MathDocumentWorkflow()
)

workflow.add_route(
    condition=lambda doc: doc.is_scanned,
    handler=OCRWorkflow()
)

# Process with automatic routing
for document in documents:
    result = workflow.process(document)
    print(f"Processed via: {result.workflow_used}")
```

### Pipeline Composition

```python
from core.workflows import PipelineComposer

# Compose complex pipeline
pipeline = PipelineComposer()

pipeline.add_stage("preprocessing", PreprocessWorkflow())
pipeline.add_stage("extraction", ExtractionWorkflow())
pipeline.add_stage("enrichment", EnrichmentWorkflow())
pipeline.add_stage("embedding", EmbeddingWorkflow())
pipeline.add_stage("storage", StorageWorkflow())

# Add transformations between stages
pipeline.add_transform(
    from_stage="extraction",
    to_stage="enrichment",
    transform=lambda x: enrich_with_metadata(x)
)

# Execute pipeline
results = pipeline.execute(documents)
```

### Workflow Monitoring

```python
from core.workflows import MonitoredWorkflow
from core.monitoring import WorkflowMonitor

# Workflow with integrated monitoring
workflow = MonitoredWorkflow(
    base_workflow=PDFWorkflow(),
    monitor=WorkflowMonitor(
        metrics_file="workflow_metrics.json",
        log_interval=10
    )
)

# Process with monitoring
results = workflow.process_batch(documents)

# Get performance metrics
metrics = workflow.get_metrics()
print(f"Average processing time: {metrics.avg_time:.2f}s")
print(f"Throughput: {metrics.throughput:.2f} docs/min")
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Resource usage: {metrics.resource_usage}")
```

## Configuration

### YAML Configuration

```yaml
# workflow_config.yaml
workflow:
  type: pdf_batch
  max_workers: 32
  checkpoint_interval: 100

  extraction:
    type: docling
    use_ocr: true
    chunk_size: 1000
    chunk_overlap: 200

  embedding:
    model: jinaai/jina-embeddings-v3
    device: cuda
    batch_size: 32
    use_fp16: true

  storage:
    type: arango
    database: academy_store
    collections:
      papers: arxiv_papers
      chunks: arxiv_chunks
      embeddings: arxiv_embeddings

  error_handling:
    max_retries: 3
    retry_delay: 5.0
    save_failed: /tmp/failed_docs

  monitoring:
    enabled: true
    log_interval: 10
    metrics_file: metrics.json
```

### Dynamic Configuration

```python
from core.workflows import WorkflowFactory

# Load from configuration
workflow = WorkflowFactory.from_config("workflow_config.yaml")

# Override at runtime
workflow.update_config({
    "max_workers": 64,  # More workers
    "embedding.batch_size": 48  # Larger batches
})

# Conditional configuration
if gpu_available():
    workflow.enable_gpu()
else:
    workflow.use_cpu_fallback()
```

## State Management

### Checkpoint System

```python
from core.workflows import CheckpointedWorkflow

workflow = CheckpointedWorkflow(
    checkpoint_dir="/data/checkpoints",
    checkpoint_interval=100,  # Every 100 items
    keep_checkpoints=5  # Keep last 5 checkpoints
)

# Process with checkpointing
try:
    results = workflow.process_batch(large_dataset)
except KeyboardInterrupt:
    print("Interrupted - state saved")

# Resume from checkpoint
results = workflow.resume_from_checkpoint()
print(f"Resumed from item {results.resume_index}")
```

### Transaction Support

```python
from core.workflows import TransactionalWorkflow

workflow = TransactionalWorkflow(
    isolation_level="read_committed"
)

# Atomic batch processing
with workflow.transaction() as txn:
    for document in documents:
        result = workflow.process(document)
        txn.add_result(result)

    # All-or-nothing commit
    if txn.validate():
        txn.commit()
    else:
        txn.rollback()
```

## Performance Optimization

### Memory Management

```python
from core.workflows import MemoryOptimizedWorkflow

workflow = MemoryOptimizedWorkflow(
    max_memory_gb=16,
    gc_interval=100,  # Garbage collect every 100 items
    use_memory_mapping=True
)

# Process with memory limits
workflow.process_large_dataset(
    dataset,
    stream_mode=True,  # Don't load all into memory
    chunk_results=True  # Write results incrementally
)
```

### GPU Optimization

```python
from core.workflows import GPUOptimizedWorkflow

workflow = GPUOptimizedWorkflow(
    gpu_devices=[0, 1],  # Multi-GPU
    mixed_precision=True,  # FP16
    cuda_graphs=True  # CUDA graph optimization
)

# Optimize batch sizes for GPU memory
workflow.auto_tune_batch_size(sample_data)
print(f"Optimal batch size: {workflow.batch_size}")

# Process with GPU optimization
results = workflow.process_batch(
    documents,
    pin_memory=True,
    prefetch_factor=2
)
```

### Caching Strategy

```python
from core.workflows import CachedWorkflow

workflow = CachedWorkflow(
    cache_dir="/data/cache",
    cache_size_gb=100,
    cache_policy="lru"  # or "fifo", "lfu"
)

# Process with caching
for document in documents:
    # Check cache first
    if workflow.is_cached(document):
        result = workflow.get_from_cache(document)
    else:
        result = workflow.process(document)
        workflow.cache_result(document, result)
```

## Error Handling

### Comprehensive Error Management

```python
from core.workflows import WorkflowError, WorkflowResult

class RobustWorkflow(WorkflowBase):
    """Workflow with comprehensive error handling."""

    def process(self, input_path: str) -> WorkflowResult:
        try:
            # Main processing
            return self._process_internal(input_path)

        except ExtractorError as e:
            # Try fallback extractor
            return self._process_with_fallback(input_path)

        except EmbeddingError as e:
            # Skip embedding, return partial result
            return WorkflowResult(
                success=False,
                partial=True,
                error=str(e)
            )

        except StorageError as e:
            # Retry with exponential backoff
            return self._retry_with_backoff(
                lambda: self._process_internal(input_path),
                max_retries=3
            )

        except Exception as e:
            # Log and save for manual review
            self.log_error(input_path, e)
            self.save_failed_item(input_path)
            return WorkflowResult(
                success=False,
                error=str(e)
            )
```

## Testing

```python
import pytest
from core.workflows import PDFWorkflow, WorkflowResult

def test_pdf_workflow():
    """Test PDF processing workflow."""
    workflow = PDFWorkflow()

    result = workflow.process("test.pdf")

    assert result.success
    assert len(result.chunks) > 0
    assert result.duration > 0

def test_batch_processing():
    """Test batch processing."""
    workflow = PDFWorkflow()

    results = workflow.process_batch(
        ["test1.pdf", "test2.pdf"],
        num_workers=2
    )

    assert len(results) == 2
    assert all(r.success for r in results)

def test_error_recovery():
    """Test error recovery."""
    workflow = ResilientWorkflow(max_retries=2)

    # Simulate intermittent failure
    with mock.patch("extract", side_effect=[Exception, Success]):
        result = workflow.process("test.pdf")

    assert result.success
    assert result.metadata["attempts"] == 2
```

## Best Practices

### 1. Use Appropriate Workflow

```python
# For single documents
workflow = PDFWorkflow()

# For batch processing
workflow = BatchWorkflow(max_workers=32)

# For streaming
workflow = StreamingWorkflow(buffer_size=100)

# For reliability
workflow = ResilientWorkflow(max_retries=3)
```

### 2. Configure Properly

```python
# Optimize for your hardware
workflow = PDFWorkflow(
    max_workers=os.cpu_count(),
    gpu_batch_size=get_optimal_gpu_batch(),
    use_fp16=gpu_supports_fp16()
)
```

### 3. Monitor Performance

```python
# Always monitor production workflows
workflow = MonitoredWorkflow(
    base_workflow=your_workflow,
    alert_on_error_rate=0.05,
    alert_on_low_throughput=10
)
```

### 4. Handle Errors Gracefully

```python
# Implement proper error handling
try:
    results = workflow.process_batch(documents)
except WorkflowError as e:
    handle_workflow_error(e)
    save_partial_results(e.partial_results)
```

## Migration Guide

### From Sequential Processing

```python
# Old approach
for pdf in pdfs:
    text = extract(pdf)
    chunks = chunk(text)
    embeddings = embed(chunks)
    store(embeddings)

# New approach
workflow = PDFWorkflow()
results = workflow.process_batch(pdfs, num_workers=8)
```

### From Custom Pipelines

```python
# Old custom pipeline
def process_document(doc):
    # Complex custom logic
    pass

# New approach - extend WorkflowBase
class CustomWorkflow(WorkflowBase):
    def process(self, doc):
        # Same logic, better structure
        pass
```

## Related Components

- [Extractors](../extractors/README.md) - Content extraction
- [Embedders](../embedders/README.md) - Vector generation
- [Monitoring](../monitoring/README.md) - Performance tracking
- [Config](../config/README.md) - Workflow configuration
- [Database](../database/README.md) - Result storage