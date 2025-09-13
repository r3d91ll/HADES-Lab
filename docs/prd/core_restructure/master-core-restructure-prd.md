# Master PRD: Core Infrastructure Restructure & Standardization

**Version:** 1.0
**Date:** January 13, 2025
**Author:** HADES Development Team
**Status:** Draft
**Type:** Master PRD
**Supersedes:** Issue #34 (too narrow in scope)

## Executive Summary

Complete restructuring of the `core/` directory to eliminate the redundant `framework/` layer, establish clear single-purpose modules, and create a production-ready, immutable infrastructure layer. This master PRD defines the overall architecture and spawns individual PRDs for each core module.

## Problem Statement

### Current State
The `core/` directory has evolved organically without architectural governance, resulting in:

1. **Redundant Nesting**: `core/framework/` is redundant - everything in `core/` IS the framework
2. **Mixed Concerns**: Single directories handling multiple unrelated responsibilities
3. **Scattered Components**: Related functionality spread across multiple directories
4. **Inconsistent Patterns**: No naming conventions or organizational principles
5. **Hidden Dependencies**: Unclear relationships between components
6. **Duplicate Infrastructure**: Multiple implementations of monitoring, config, logging
7. **Graph Chaos**: `core/graph/` contains its own configs, monitoring, experiments, orchestrations

### Impact
- **Cognitive Overload**: Developers cannot intuit where functionality lives
- **Import Hell**: Deep, inconsistent import paths (`core.framework.extractors.docling_extractor`)
- **Maintenance Nightmare**: Changes require hunting across multiple directories
- **Onboarding Barrier**: New developers struggle to understand the architecture
- **Technical Debt**: Accumulating cruft makes the codebase harder to evolve
- **Production Risk**: Unclear boundaries between infrastructure and application code

## Vision

Transform `core/` into a flat, single-purpose module structure where:
- Each directory has ONE clear responsibility
- No redundant nesting (no `framework/` subdirectory)
- Consistent naming conventions (module_prefix pattern)
- Clear separation of infrastructure vs application code
- All modules are config-driven and testable

## Key Design Decisions

### 1. No Separate Storage Module
**Decision**: Place storage functionality under `workflows/storage/` instead of creating a top-level `storage/` module.

**Rationale**:
- `storage.py` is actually workflow state management, not abstract storage
- `memory_store.py` is for caching during active workflows
- Database connections are already handled by `database/`
- Abstract storage interfaces are unnecessary complexity

**Result**: Cleaner separation where:
- `database/` = External database connections
- `workflows/storage/` = Workflow state and caching

### 2. Remove Search Module
**Decision**: No `core/search/` module.

**Rationale**:
- Search is inherently tool-specific
- Each tool has different search requirements
- Generic search abstractions add no value

### 3. Database Module is Complete
**Decision**: `core/database/` only needs documentation, not restructuring.

**Rationale**:
- Current implementation is solid
- Unix socket optimizations working
- Connection pooling implemented
- Only missing comprehensive documentation

### 4. Graph Module Requires Special Handling
**Decision**: Graph refactor is DEFERRED until after core restructure validation.

**Rationale**:
- Most complex module with 12+ duplicate builders
- Has its own configs, monitoring, experiments, utils (all need removal)
- Touches everything - needs clean modules to build on
- Current implementation works (though messy) for validation rebuild
- Requires separate comprehensive refactor PRD

**Current Graph Issues**:
- `graph/builders/` - 12+ scripts doing similar things
- `graph/configs/` - Should use central config
- `graph/monitoring/` - Should use central monitoring
- `graph/experiments/` - Should move to `experiments/graph/`
- `graph/utils/` - Should consolidate with `core/utils/`
- `graph/orchestrations/` - Should move to `workflows/`

**Future Graph Structure** (Post-Validation):
```
core/graph/
├── graph_algorithms.py     # Core algorithms (PageRank, etc.)
├── graph_builder.py        # ONE unified builder
├── graph_manager.py        # Management operations
├── graph_memory_store.py   # RAM storage
├── graph_traversal.py      # Navigation/queries
└── graph_types.py         # Type definitions
```

## Architectural Principles

### 1. Single Responsibility
Each directory under `core/` serves exactly one purpose, clearly indicated by its name.

### 2. Flat Structure
Eliminate unnecessary nesting. `core/` modules are peers, not hierarchies.

### 3. Consistent Naming
- Directories: Plural nouns (`embedders/`, `extractors/`)
- Files: `<module>_<specific>.py` pattern
- Classes: Follow file names in PascalCase

### 4. Config-Driven
Every module accepts configuration through the centralized config system.

### 5. Test Coverage
Each module has corresponding tests in `tests/core/<module>/`.

### 6. No Application Logic
`core/` contains only infrastructure. Application logic goes in `tools/` or `experiments/`.

## Proposed Structure

```
core/
├── config/           # Configuration management
├── database/         # Database connections and managers (COMPLETE - needs docs)
├── embedders/        # All embedding implementations
├── extractors/       # All extraction implementations
├── graph/            # Graph operations and algorithms
├── logging/          # Logging infrastructure
├── monitoring/       # System and workflow monitoring
├── processors/       # Data processing pipelines
├── utils/            # Shared utilities
└── workflows/        # Workflow orchestration
    ├── monitoring/   # Workflow-specific monitoring
    └── storage/      # Workflow state management and caching
```

**REMOVED**:
- `framework/` directory entirely (redundant - everything in core IS the framework)
- `search/` module (search is tool-specific, not core infrastructure)
- `storage/` as top-level (moved to workflows/storage/ where it belongs)

## Module Definitions

### Core Modules (Infrastructure)

| Module | Purpose | Source | Sub-PRD |
|--------|---------|--------|---------|
| `config/` | Configuration management | `framework/config.py` | PRD-001 |
| `database/` | Database connections | **COMPLETE** - needs documentation only | PRD-002 |
| `embedders/` | Embedding implementations | `framework/*embedder*.py` | PRD-003 |
| `extractors/` | Content extraction | `framework/extractors/` | PRD-004 |
| `graph/` | Graph algorithms | Existing + reorganize + `memory_store.py` | PRD-005 |
| `logging/` | Logging infrastructure | `framework/logging.py` | PRD-006 |
| `monitoring/` | System monitoring | Existing + `framework/metrics.py` | PRD-007 |
| `processors/` | Processing pipelines | Existing + `framework/base_processor.py` | PRD-008 |
| `utils/` | Shared utilities | Existing + audit | PRD-009 |
| `workflows/` | Orchestration | Existing + expand | PRD-010 |
| `workflows/monitoring/` | Workflow monitoring | `tools/*/monitoring/` workflow monitors | PRD-010a |
| `workflows/storage/` | Workflow state & caching | `framework/storage.py`, `memory_store.py` | PRD-010b |

## Implementation Strategy: Stop & Restructure

### Critical Decision: Full Stop After Current Ingestion
**Rationale**: The current data ingestion is about to complete, providing us with a known-good baseline. This is the PERFECT time to restructure because:
1. We have a complete, working database state to validate against
2. Rebuilding the database after restructure serves as comprehensive integration testing
3. Prevents accumulating technical debt in graph building phase
4. No wasted work - rebuild proves the restructure succeeded

### Pre-Implementation Phase (Immediate)
1. **Complete current ingestion** - Let running processes finish
2. **Document current state** - Exact counts of papers, embeddings, chunks
3. **Create database snapshot** - Backup for comparison
4. **FULL STOP** - No new development until restructure complete

### Phase 1: Planning & Documentation (Days 1-3)
1. Create all sub-PRDs in `docs/prd/core_restructure/`
2. Document current pipeline behavior precisely
3. Create test criteria for validation
4. Create master GitHub issue and sub-issues
5. Archive Issue #34 as superseded

### Phase 2: Foundation Modules (Days 4-7)
Build modules that everything else depends on:
1. `config/` - Central configuration system
2. `logging/` - Unified logging infrastructure
3. `database/` - Documentation only (already complete)
4. `utils/` - Audit and clean up utilities

### Phase 3: Processing Infrastructure (Days 8-11)
Core processing components:
1. `embedders/` - Consolidate all embedding implementations
2. `extractors/` - Organize all extractors with consistent naming
3. `processors/` - Integrate base_processor.py
4. `workflows/` - Expand with monitoring/ and storage/

### Phase 4: Support Systems (Days 12-14)
Supporting infrastructure:
1. `monitoring/` - Consolidate system and workflow monitoring
2. Remove `framework/` directory completely
3. Update all imports across codebase
4. Comprehensive testing of each module

### Phase 5: Graph Module (DEFERRED)
**IMPORTANT**: Graph refactor happens AFTER validation
1. Continue using current graph/ as-is during rebuild
2. Full graph refactor only after core is proven stable
3. Separate comprehensive PRD for graph refactor

### Phase 6: Validation Through Rebuild (Days 15-21)
**THE CRITICAL TEST**: Rebuild entire database with restructured code
1. Run exact same ingestion pipeline
2. Compare results with snapshot:
   - Same number of papers processed
   - Identical embeddings generated
   - Same chunks created
   - Matching error patterns
3. Performance should be same or better
4. Document any differences

### Success Criteria for Rebuild
- [ ] 100% of papers reprocessed successfully
- [ ] Embeddings match dimensionality and quality
- [ ] Chunk counts within 0.1% of original
- [ ] No new error types introduced
- [ ] Performance metrics equal or better
- [ ] All tests passing

## Success Metrics

### Quantitative
- **Import Depth**: Average 2 levels (from 4)
- **Directory Count**: 12 focused modules (from 15+ scattered)
- **Code Duplication**: 0% (from ~30%)
- **Test Coverage**: 80% minimum
- **Import Statements**: 30% reduction in length

### Qualitative
- **Intuitive Structure**: New developers understand in <30 minutes
- **Clear Boundaries**: Obvious what belongs where
- **Easy Extension**: Adding new functionality follows clear patterns
- **Production Ready**: `core/` can be versioned and released

## Risk Management

### Risk 1: Breaking Changes
**Mitigation**:
- Incremental migration with compatibility layers
- Comprehensive test suite before migration
- Feature flags for gradual rollout

### Risk 2: Scope Creep
**Mitigation**:
- Strict adherence to sub-PRDs
- No feature additions during restructure
- Clear acceptance criteria

### Risk 3: Import Update Complexity
**Mitigation**:
- Automated import rewriting scripts
- Backward compatibility through `__init__.py`
- Phased migration by module

## Sub-PRD Structure

Each sub-PRD will follow this template:
1. Module Purpose & Scope
2. Current State Analysis
3. Proposed Structure
4. File Mapping (old → new)
5. API Design
6. Configuration Schema
7. Testing Strategy
8. Migration Plan
9. Success Criteria

## Dependencies

### Upstream
- None (this is foundational)

### Downstream
- All `tools/*` modules
- All `experiments/*`
- All tests

### Cross-Module
- Config system affects all modules
- Logging system affects all modules
- Storage interfaces affect embedders, extractors, processors

## Migration Strategy

### Step 1: Parallel Structure
Create new structure alongside old without breaking anything.

### Step 2: Dual Support
Support imports from both old and new locations with deprecation warnings.

### Step 3: Gradual Migration
Migrate one module at a time, testing thoroughly.

### Step 4: Cleanup
Remove old structure, archive to Acheron.

## Example: Before and After

### Before (Deep, Confusing)
```python
from core.framework.extractors.docling_extractor import DoclingExtractor
from core.framework.embedders import JinaV4Embedder
from core.database.arango_db_manager import ArangoDBManager
from core.framework.config import ConfigManager
from core.processors.document_processor import DocumentProcessor
```

### After (Flat, Clear)
```python
from core.extractors import DoclingExtractor
from core.embedders import JinaEmbedder
from core.database import ArangoManager
from core.config import ConfigManager
from core.processors import DocumentProcessor
```

## Sub-PRDs to Create

1. `PRD-001-config.md` - Configuration Management System ✅ (Created)
2. `PRD-002-database.md` - Database Documentation (module complete)
3. `PRD-003-embedders.md` - Embedding Infrastructure
4. `PRD-004-extractors.md` - Extraction Framework
5. `PRD-005-graph.md` - Graph Operations Consolidation
6. `PRD-006-logging.md` - Logging Infrastructure
7. `PRD-007-monitoring.md` - System Monitoring Infrastructure
8. `PRD-008-processors.md` - Processing Pipeline Framework
9. `PRD-009-utils.md` - Shared Utilities Audit
10. `PRD-010-workflows.md` - Workflow Orchestration
    - `PRD-010a-workflow-monitoring.md` - Workflow-specific monitoring
    - `PRD-010b-workflow-storage.md` - Workflow state management

## Acceptance Criteria

### Must Have
- [ ] All functionality preserved
- [ ] All tests passing
- [ ] No `framework/` directory
- [ ] Consistent naming throughout
- [ ] 80% test coverage
- [ ] All modules config-driven

### Should Have
- [ ] Performance improvements
- [ ] Reduced memory footprint
- [ ] Better error messages
- [ ] Comprehensive documentation

### Nice to Have
- [ ] API versioning
- [ ] Module health checks
- [ ] Automatic dependency injection

## Timeline

### Sprint Timeline (21 Days Total)
- **Days 1-3**: Planning, PRDs, documentation of current state
- **Days 4-7**: Foundation modules (config, logging, utils)
- **Days 8-11**: Processing modules (embedders, extractors, processors, workflows)
- **Days 12-14**: Support systems (monitoring, cleanup, imports)
- **Days 15-21**: VALIDATION - Complete database rebuild
- **Post-Validation**: Graph refactor (separate project)

### Why This Timeline Works
1. **Current ingestion completing** - Natural break point
2. **3-week sprint** - Focused, achievable
3. **Validation week** - Proves everything works
4. **Graph deferred** - Reduces complexity, allows focus
5. **Database rebuild** - Perfect integration test

## Conclusion

This restructure represents a critical architectural decision: **STOP NOW to build it right**. With our current data ingestion completing, we have a perfect validation baseline - a known-good database state that the restructured code must replicate exactly.

The validation strategy is elegant: rebuilding the entire database with the restructured code serves as comprehensive integration testing. If we can reproduce the exact same results with clean, well-architected code, we've proven the restructure succeeded.

Key insights:
1. **Perfect timing** - Natural break between ingestion and graph building
2. **No wasted work** - Database rebuild is the test, not overhead
3. **Graph deferred** - Most complex module saved for after validation
4. **Clean foundation** - Future development on solid architecture
5. **Known baseline** - Current database state validates the restructure

Once complete, `core/` becomes truly immutable - a production-ready, well-documented, fully-tested foundation that can support HADES growth indefinitely without accumulating technical debt.