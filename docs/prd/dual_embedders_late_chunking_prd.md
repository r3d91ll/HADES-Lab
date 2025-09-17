# Product Requirements Document: Dual Embedders with Mandatory Late Chunking

## 1. Executive Summary

### Problem Statement

During the recent code reorganization, we lost the high-performance embedding implementation that achieved 48 papers/second throughput using sentence-transformers. The current implementation uses only the transformers library, achieving ~8-15 papers/second. Additionally, the codebase lacks a clear standard for chunking methodology, risking information loss through naive chunking approaches.

### Solution

Implement two optimized embedding solutions with mandatory late chunking as the standard:

1. **Sentence-transformers implementation** for high-throughput batch processing (target: 48+ papers/sec)
2. **Transformers implementation** for sophisticated context handling (current: 8-15 papers/sec)

Both implementations will enforce late chunking as the only acceptable chunking method, ensuring context preservation and preventing information loss.

## 2. Background and Context

### Historical Performance

- **Previous achievement**: 48 papers/second on 2x NVIDIA A6000 GPUs (100% utilization)
- **Current performance**: 8-15 papers/second with transformers implementation
- **Performance gap**: ~3-6x slower than optimal

### Conveyance Framework Alignment

According to our Conveyance Framework: `C = (W·R·H/T)·Ctx^α`

- **Late chunking preserves Ctx**: Maintains context awareness across chunk boundaries
- **Naive chunking causes Ctx→0**: Breaking context causes zero-propagation
- **Performance (T) vs Quality (W)**: Both implementations optimize different dimensions

## 3. Requirements

### 3.1 Functional Requirements

#### Core Requirements

1. **FR-1**: Implement `SentenceTransformersEmbedder` class using sentence-transformers library
2. **FR-2**: Maintain existing `JinaV4Embedder` class using transformers library
3. **FR-3**: Both implementations MUST use late chunking exclusively
4. **FR-4**: Support Jina models (v3 and v4) in both implementations
5. **FR-5**: Implement common `EmbedderBase` interface for both

#### Late Chunking Requirements

6. **FR-6**: Process full text context before creating chunks (up to model limit)
7. **FR-7**: Maintain context overlap between chunks
8. **FR-8**: Preserve chunk position and context metadata
9. **FR-9**: Support single-chunk texts without degradation
10. **FR-10**: Handle edge cases (very short/long texts) correctly

#### Integration Requirements

11. **FR-11**: Update `EmbedderFactory` to support both implementations
12. **FR-12**: Configuration-based selection between implementations
13. **FR-13**: Backward compatibility with existing code
14. **FR-14**: Update CLAUDE.md to mandate late chunking

### 3.2 Performance Requirements

#### Throughput Targets

1. **PR-1**: Sentence-transformers: ≥48 papers/second for abstracts
2. **PR-2**: Transformers: Maintain current 8-15 papers/second
3. **PR-3**: Support batch sizes of 128-256 for sentence-transformers
4. **PR-4**: Efficient GPU memory utilization (≤80% VRAM)

#### Scalability

5. **PR-5**: Multi-GPU support for both implementations
6. **PR-6**: Linear scaling with batch size increases
7. **PR-7**: Support for 2.8M document processing runs

### 3.3 Quality Requirements

1. **QR-1**: Embedding similarity ≥0.95 between implementations for same text
2. **QR-2**: No information loss at chunk boundaries
3. **QR-3**: Consistent embeddings for repeated processing
4. **QR-4**: Proper error handling and recovery

## 4. Success Metrics

### Primary Metrics

1. **Throughput**: Achieve 48+ papers/second with sentence-transformers
2. **Quality**: Maintain embedding quality (cosine similarity ≥0.95)
3. **Reliability**: 99.9% success rate for document processing
4. **Memory**: ≤80% GPU memory utilization under load

### Secondary Metrics

5. **Code Coverage**: ≥90% test coverage for new code
6. **Documentation**: 100% of public methods documented
7. **Performance Regression**: No degradation in existing pipeline
8. **Integration Time**: <1 day to integrate into existing workflows

## 5. Technical Design

### 5.1 Architecture

```
core/embedders/
├── embedders_base.py          # Base interface (existing)
├── embedders_factory.py       # Updated factory with dual support
├── embedders_jina.py          # Transformers implementation (existing, optimized)
├── embedders_sentence.py      # NEW: Sentence-transformers implementation
└── README.md                  # Updated documentation

tools/arxiv/workflows/
├── workflow_metadata_transformers.py  # Using transformers version
└── workflow_metadata_sentence.py      # Using sentence-transformers version
```

### 5.2 Late Chunking Algorithm

```python
def late_chunk(text: str, max_tokens: int, chunk_size: int, overlap: int):
    """
    Mandatory late chunking algorithm:
    1. Tokenize full text (up to max_tokens)
    2. Process entire context through model
    3. Create overlapping chunks maintaining context
    4. Each chunk knows its position and neighbors
    """
```

### 5.3 Configuration

```yaml
embedders:
  sentence_transformers:
    model: "jinaai/jina-embeddings-v3-base-en"
    batch_size: 128
    max_length: 8192
    use_late_chunking: true  # Always true, non-configurable

  transformers:
    model: "jinaai/jina-embeddings-v4"
    batch_size: 24
    max_length: 32768
    use_late_chunking: true  # Always true, non-configurable
```

## 6. Implementation Plan

### Phase 1: Foundation (Day 1-2)

1. Create PRD and GitHub issue
2. Create feature branch
3. Update CLAUDE.md with late chunking mandate
4. Implement `embedders_sentence.py`

### Phase 2: Integration (Day 3-4)

5. Update `embedders_factory.py`
6. Optimize `embedders_jina.py`
7. Create benchmark script
8. Implement workflows

### Phase 3: Testing (Day 5)

9. Unit tests for both implementations
10. Integration tests
11. Performance benchmarks
12. 1000-document test run

### Phase 4: Deployment (Day 6)

13. Documentation updates
14. PR creation and review
15. CodeRabbit review integration
16. Merge to main

## 7. Testing Strategy

### 7.1 Unit Tests

- Test late chunking with various text lengths
- Verify context preservation
- Test batch processing
- Edge case handling

### 7.2 Integration Tests

- End-to-end workflow tests
- Database storage verification
- Multi-GPU scaling tests

### 7.3 Performance Tests

- Benchmark both implementations
- Memory profiling
- Throughput measurements
- Comparison with historical performance

### 7.4 Quality Tests

- Embedding similarity comparison
- Consistency verification
- Context preservation validation

## 8. Risks and Mitigations

### Risk 1: Performance Target Not Met

- **Mitigation**: Iterative optimization, profiling, batch size tuning

### Risk 2: Memory Issues with Large Batches

- **Mitigation**: Dynamic batch sizing, memory monitoring

### Risk 3: API Changes in Libraries

- **Mitigation**: Pin library versions, comprehensive testing

### Risk 4: Breaking Existing Code

- **Mitigation**: Backward compatibility layer, extensive integration tests

## 9. Documentation Requirements

1. Update CLAUDE.md with late chunking principle
2. API documentation for both embedders
3. Performance comparison guide
4. Migration guide from naive to late chunking
5. Troubleshooting guide

## 10. Rollout Plan

1. **Week 1**: Implementation and testing
2. **Week 2**: Integration with existing workflows
3. **Week 3**: Full 2.8M document processing test
4. **Week 4**: Performance optimization and documentation

## 11. Appendix

### A. Late Chunking Principle

Late chunking is mandatory because:

1. **Context Preservation**: Every chunk maintains awareness of surrounding context
2. **Semantic Coherence**: Chunk boundaries don't break semantic units
3. **Consistency**: Same processing pipeline for all text lengths
4. **Quality**: Even single-chunk texts benefit from full context processing
5. **Zero-Propagation Prevention**: Naive chunking causes information loss (W→0)

### B. Historical Context

The original high-performance implementation was lost during the September 2024 reorganization. Analysis reveals it used sentence-transformers library with optimized batch processing, achieving 48 papers/second on dual A6000 GPUs.

### C. Success Criteria Checklist

- [ ] Sentence-transformers implementation complete
- [ ] 48+ papers/second throughput achieved
- [ ] Late chunking enforced in both implementations
- [ ] CLAUDE.md updated with mandatory late chunking
- [ ] All tests passing
- [ ] Documentation complete
- [ ] CodeRabbit review passed
- [ ] PR approved and merged

---

**Document Version**: 1.0
**Date**: January 2025
**Author**: HADES Team
**Status**: Draft for Review
