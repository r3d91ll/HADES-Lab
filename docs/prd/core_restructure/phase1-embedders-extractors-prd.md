# Phase 1: Embedders and Extractors Restructure PRD

**Parent Issue**: #35 (Master Core Infrastructure Restructure)
**Timeline**: Days 1-3 of 21-day sprint
**Status**: Ready for Implementation

## Problem Statement

Currently, embedders and extractors are buried in `core/framework/`, creating an unnecessary layer of abstraction. Everything in `core/` IS the framework, making the `framework/` subdirectory redundant.

## Solution Overview

Move embedders and extractors to top-level directories within `core/`, establishing clear module boundaries and consistent naming conventions.

## Detailed Requirements

### 1. Embedders Module (`core/embedders/`)

**Current Location**: `core/framework/embedders.py`, `core/framework/sentence_embedder.py`, etc.

**New Structure**:
```
core/embedders/
├── __init__.py
├── embedders_base.py           # Base class for all embedders
├── embedders_jina.py           # Jina v4 embedder (transformers)
├── embedders_sentence.py       # Sentence-transformers implementation
├── embedders_vllm.py           # VLLM embedder (if salvageable)
├── embedders_retrieval.py     # Retrieval-specific embedder
└── embedders_factory.py       # Factory for creating embedders
```

**Requirements**:
- Maintain backward compatibility during transition
- Preserve all optimization flags (fp16, batch_size, etc.)
- Keep performance metrics (48.3 papers/sec baseline)
- Update all imports in tools/ and experiments/

### 2. Extractors Module (`core/extractors/`)

**Current Location**: `core/framework/extractors.py`

**New Structure**:
```
core/extractors/
├── __init__.py
├── extractors_base.py          # Base class for all extractors
├── extractors_docling.py       # GPU-accelerated Docling
├── extractors_pdfplumber.py   # PDFPlumber fallback
├── extractors_latex.py         # LaTeX source extraction
└── extractors_factory.py      # Factory for creating extractors
```

**Requirements**:
- Preserve GPU acceleration capabilities
- Maintain extraction quality metrics
- Keep all error handling and recovery
- Update references in ACID pipeline

### 3. Import Updates

**Files to Update**:
- `tools/arxiv/pipelines/arxiv_pipeline.py`
- `tools/arxiv/utils/arxiv_metadata_ingestion_sentence.py`
- `tools/github/pipelines/github_pipeline.py`
- `experiments/*/src/*.py`
- Any other files importing from `core.framework`

**Import Changes**:
```python
# Old
from core.framework.embedders import JinaV4Embedder
from core.framework.extractors import DoclingExtractor

# New
from core.embedders.embedders_jina import JinaV4Embedder
from core.extractors.extractors_docling import DoclingExtractor
```

## Implementation Steps

### Day 1: Embedders Migration
1. Create `core/embedders/` directory structure
2. Move and rename embedder files with proper prefix
3. Update internal imports within embedders
4. Create `__init__.py` with backward compatibility imports
5. Test with sample embedding generation

### Day 2: Extractors Migration
1. Create `core/extractors/` directory structure
2. Move and rename extractor files with proper prefix
3. Update internal imports within extractors
4. Create `__init__.py` with backward compatibility imports
5. Test with sample PDF extraction

### Day 3: Import Updates and Testing
1. Update all imports across codebase
2. Run integration tests
3. Verify ACID pipeline still works
4. Check performance metrics match baseline
5. Document any issues or deviations

## Success Criteria

### Functional
- [ ] All embedders working with same performance
- [ ] All extractors working with same quality
- [ ] ACID pipeline runs without errors
- [ ] Backward compatibility maintained

### Performance
- [ ] Maintain 48.3 papers/second throughput
- [ ] GPU memory usage unchanged (7-8GB per worker)
- [ ] Batch processing efficiency preserved

### Code Quality
- [ ] Consistent naming (files prefixed with module name)
- [ ] Clear module boundaries
- [ ] All tests passing
- [ ] Documentation updated

## Rollback Plan

If issues arise:
1. Git revert to previous commit
2. Restore from Acheron/ archives
3. Use backward compatibility imports
4. Document issues for resolution

## Dependencies

- Must be completed before Phase 2 (database/workflows)
- Blocks all other core restructuring work
- Required for configuration centralization

## Notes

- Keep `core/framework/` during transition with deprecation warnings
- Archive old structure to `Acheron/` with timestamps
- Update CLAUDE.md with new import patterns
- Ensure sentence-transformers optimization preserved (12% faster)

## Validation

Run these commands to validate:
```bash
# Test embedders
python -c "from core.embedders.embedders_jina import JinaV4Embedder; e = JinaV4Embedder(); print('✓ Embedders working')"

# Test extractors
python -c "from core.extractors.extractors_docling import DoclingExtractor; e = DoclingExtractor(); print('✓ Extractors working')"

# Test ACID pipeline
cd tools/arxiv/pipelines/
python arxiv_pipeline.py --config ../configs/acid_pipeline_phased.yaml --count 10
```

## References

- Baseline metrics: `/docs/baseline/baseline-2025-01-13.md`
- Master PRD: `/docs/prd/core_restructure/master-core-restructure-prd.md`
- Issue #35: Master Core Infrastructure Restructure