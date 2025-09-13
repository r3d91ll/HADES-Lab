# Phase 2: Database and Workflows Restructure PRD

**Parent Issue**: #35 (Master Core Infrastructure Restructure)
**Timeline**: Days 4-7 of 21-day sprint
**Status**: Ready for Implementation

## Problem Statement

Database interfaces are scattered and inconsistent. Workflow orchestration is mixed with storage concerns. The `core/utils/` directory contains workflow components that should be in a dedicated workflows module.

## Solution Overview

Consolidate database interfaces under `core/database/`, create a dedicated `core/workflows/` module with integrated storage, and establish clear separation between orchestration and data access.

## Detailed Requirements

### 1. Database Module (`core/database/`)

**Current Structure** (mostly complete):
```
core/database/
├── arango/
│   ├── arango_client.py
│   ├── arango_unix_client.py
│   └── arango_batch_processor.py
└── postgres/
    └── (to be added)
```

**New Additions**:
```
core/database/
├── __init__.py
├── database_factory.py         # Factory for database connections
├── arango/
│   ├── __init__.py
│   ├── arango_client.py       # Existing
│   ├── arango_unix_client.py  # Existing (Unix socket optimization)
│   ├── arango_batch.py        # Batch operations
│   └── arango_graph.py        # Graph-specific operations
└── postgres/
    ├── __init__.py
    ├── postgres_client.py      # PostgreSQL client
    └── postgres_metadata.py    # ArXiv metadata operations
```

**Requirements**:
- Preserve Unix socket optimization (faster performance)
- Maintain atomic transaction support
- Keep batch processing capabilities
- Add PostgreSQL interface for metadata

### 2. Workflows Module (`core/workflows/`)

**Move from `core/utils/` and `core/processors/`:
- `workflow_pdf.py` (from document_processor.py)
- `workflow_pdf_batch.py` (from batch_processor.py)
- `workflow_state_manager.py` (from state_manager.py)

**New Structure**:
```
core/workflows/
├── __init__.py
├── workflow_base.py            # Base workflow class
├── workflow_pdf.py             # Single PDF processing
├── workflow_pdf_batch.py       # Batch PDF processing
├── workflow_latex.py           # LaTeX processing
├── workflow_github.py          # GitHub repository processing
├── state/
│   ├── __init__.py
│   ├── state_manager.py       # State management
│   └── state_checkpoint.py    # Checkpoint handling
└── storage/
    ├── __init__.py
    ├── storage_base.py         # Storage interface
    ├── storage_local.py        # Local filesystem
    ├── storage_s3.py           # S3 storage
    └── storage_ramfs.py        # RamFS for staging
```

**Requirements**:
- Maintain phase separation (extraction → embedding)
- Preserve checkpoint/resume functionality
- Keep atomic operations
- Integrate storage backends seamlessly

### 3. Integration Points

**ACID Pipeline Updates**:
```python
# Old
from core.framework.storage import LocalStorage
from core.utils.state_manager import StateManager

# New
from core.workflows.storage.storage_local import LocalStorage
from core.workflows.state.state_manager import StateManager
```

**Database Access Pattern**:
```python
# Centralized factory
from core.database.database_factory import DatabaseFactory

# Get appropriate database
arango_db = DatabaseFactory.get_arango(use_unix=True)
postgres_db = DatabaseFactory.get_postgres()
```

## Implementation Steps

### Day 4: Database Consolidation
1. Complete `core/database/arango/` organization
2. Add PostgreSQL client implementation
3. Create database factory
4. Test Unix socket optimization preserved
5. Verify atomic transactions working

### Day 5: Workflows Creation
1. Create `core/workflows/` structure
2. Move workflow files from `core/utils/`
3. Move document processors to workflows
4. Update internal imports
5. Test workflow execution

### Day 6: Storage Integration
1. Create `workflows/storage/` subdirectory
2. Move storage implementations
3. Implement storage factory
4. Test RamFS staging still works
5. Verify S3 compatibility

### Day 7: State Management
1. Create `workflows/state/` subdirectory
2. Move state management code
3. Enhance checkpoint functionality
4. Test resume capabilities
5. Validate atomic saves

## Success Criteria

### Functional
- [ ] Database connections working (Unix socket preserved)
- [ ] Workflows execute without errors
- [ ] Storage backends functional
- [ ] State management operational
- [ ] Checkpoint/resume working

### Performance
- [ ] Unix socket performance maintained
- [ ] Batch processing efficiency preserved
- [ ] RamFS staging speed unchanged
- [ ] Atomic operations still atomic

### Architecture
- [ ] Clear separation: database vs workflows
- [ ] Storage integrated with workflows
- [ ] Consistent naming throughout
- [ ] Factory patterns implemented

## Rollback Plan

1. Preserve existing code in Acheron/
2. Maintain backward compatibility imports
3. Test thoroughly before removing old code
4. Document migration issues

## Dependencies

- Requires Phase 1 completion (embedders/extractors)
- Blocks Phase 3 (monitoring/utils cleanup)
- Critical for configuration centralization

## Migration Checklist

### Files to Move
- [x] `core/database/arango_unix_client.py` → `core/database/arango/arango_unix_client.py` (already done)
- [ ] `core/utils/state_manager.py` → `core/workflows/state/state_manager.py`
- [ ] `core/processors/document_processor.py` → `core/workflows/workflow_pdf.py`
- [ ] `core/processors/batch_processor.py` → `core/workflows/workflow_pdf_batch.py`
- [ ] `core/framework/storage.py` → `core/workflows/storage/`

### Imports to Update
- [ ] `tools/arxiv/pipelines/arxiv_pipeline.py`
- [ ] `tools/arxiv/utils/lifecycle.py`
- [ ] `tools/github/pipelines/github_pipeline.py`
- [ ] All experiment files

## Validation

```bash
# Test database connections
python -c "from core.database.database_factory import DatabaseFactory; db = DatabaseFactory.get_arango(use_unix=True); print('✓ Database working')"

# Test workflows
python -c "from core.workflows.workflow_pdf import PDFWorkflow; print('✓ Workflows working')"

# Test storage
python -c "from core.workflows.storage.storage_local import LocalStorage; print('✓ Storage working')"

# Run integration test
cd tools/arxiv/pipelines/
python arxiv_pipeline.py --config ../configs/acid_pipeline_phased.yaml --count 5
```

## Notes

- Database module is largely complete, focus on organization
- Workflows module is new, needs careful design
- Storage must stay with workflows (not separate module)
- State management critical for checkpoint/resume
- Maintain all optimizations (Unix socket, RamFS, etc.)

## References

- Phase 1 PRD: `/docs/prd/core_restructure/phase1-embedders-extractors-prd.md`
- Master PRD: `/docs/prd/core_restructure/master-core-restructure-prd.md`
- Issue #35: Master Core Infrastructure Restructure