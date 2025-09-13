# Product Requirements Document: Monitoring & Framework Organization

**Version:** 1.0
**Date:** January 13, 2025
**Author:** HADES Development Team
**Status:** Draft
**Related Issues:** #34 (Config System)

## Executive Summary

Reorganize monitoring and framework components based on demand-driven principles: understanding WHEN monitoring is needed, WHY it's required, and WHAT should be monitored. This creates clear separation between infrastructure monitoring, workflow monitoring, and tool-specific monitoring while establishing consistent naming conventions across the entire `core/` directory.

## Problem Statement

### Current State
- **Misplaced Monitoring**: Workflow monitors in `tools/*/monitoring/` that aren't tool-specific
- **Scattered Embedders**: 5 embedder files loose in `core/framework/` without organization
- **Inconsistent Naming**: Files don't follow parent-directory prefix convention
- **Duplicate Code**: GPU monitoring, progress tracking repeated across monitors
- **Unclear Ownership**: Confusion about what belongs in core vs tools
- **Unused Components**: `metrics.py` and `config.py` sitting idle

### Impact
- **Cognitive Load**: Developers unsure where to find/place monitoring code
- **Code Duplication**: Same monitoring logic reimplemented multiple times
- **Import Confusion**: Inconsistent paths and naming makes imports difficult
- **Maintenance Burden**: Changes require updates in multiple locations
- **Scaling Issues**: Hard to add new monitoring without copying existing code

## Solution Overview

Implement demand-driven monitoring architecture with three clear layers:
1. **Infrastructure Monitoring** (`core/monitoring/`) - System health
2. **Workflow Monitoring** (`core/workflows/monitoring/`) - Job progress
3. **Tool Monitoring** (`tools/*/monitoring/`) - Tool-specific metrics

Simultaneously reorganize `core/framework/` with consistent naming and logical grouping.

## Monitoring Philosophy

### When We Need Intensive Monitoring
1. **Long-running batch processes** - Weekend/overnight runs
2. **Multi-GPU operations** - Training, embedding generation
3. **Debugging/optimization** - Finding bottlenecks
4. **Production deployments** - Ensuring stability
5. **Research experiments** - Collecting publication metrics

### Why We Need Monitoring (Demand Drivers)
1. **Resource Optimization** - GPU/CPU efficiency
2. **Failure Detection** - Catch crashes in unattended runs
3. **Performance Tracking** - Throughput targets and SLAs
4. **Cost Analysis** - Compute resource consumption
5. **Academic Documentation** - Metrics for papers/grants

### What We Monitor (Demand-Driven)
- **Long runs** → Progress, checkpoints, ETA
- **GPU work** → Memory, utilization, temperature
- **Debugging** → Detailed logs, errors, bottlenecks
- **Production** → Health checks, alerts, recovery
- **Research** → Statistical metrics, reproducibility data

## Requirements

### Phase 1: Deep Dive Analysis (Week 1)

#### FR1.1: Monitoring Inventory
- **FR1.1.1**: Catalog all monitoring code across the codebase
- **FR1.1.2**: Classify each monitor by type (infrastructure/workflow/tool)
- **FR1.1.3**: Identify duplicate functionality
- **FR1.1.4**: Map dependencies and imports
- **FR1.1.5**: Document monitoring demand patterns

#### FR1.2: Framework Analysis
- **FR1.2.1**: Inventory all `core/framework/` components
- **FR1.2.2**: Trace import dependencies for embedders
- **FR1.2.3**: Identify config system integration points
- **FR1.2.4**: Document naming inconsistencies

### Phase 2: Monitoring Architecture (Week 1-2)

#### FR2.1: Infrastructure Monitoring (`core/monitoring/`)
```
core/monitoring/
├── __init__.py
├── monitoring_metrics.py          # From framework/metrics.py
├── monitoring_system.py           # Existing system_monitor.py
├── monitoring_power.py            # Existing power_monitor.py
├── monitoring_analyzer.py         # Existing metrics_analyzer.py
└── base/
    ├── monitor_base.py           # Abstract base class
    └── monitor_types.py          # Monitoring type definitions
```

#### FR2.2: Workflow Monitoring (`core/workflows/monitoring/`)
```
core/workflows/monitoring/
├── __init__.py
├── workflow_monitor_base.py      # Base class for all workflow monitors
├── workflow_monitor_batch.py     # Long-running batch jobs
├── workflow_monitor_phased.py    # Phase-separated pipelines
├── workflow_monitor_checkpoint.py # Checkpoint-based monitoring
└── workflow_monitor_progress.py  # Generic progress tracking
```

#### FR2.3: Tool-Specific Monitoring (stays in `tools/*/monitoring/`)
- Only truly tool-specific monitors remain
- Example: GraphSAGE loss curves, ArXiv metadata validation
- Must import from `core/monitoring/` for shared functionality

### Phase 3: Framework Organization (Week 2)

#### FR3.1: Embedder Organization
```
core/framework/embedders/
├── __init__.py                    # Backward compatible exports
├── embedder_base.py              # Abstract base class
├── embedder_jina_v4.py          # From embedders.py
├── embedder_sentence.py         # From sentence_embedder.py
├── embedder_vllm.py             # From vllm_embedder.py
├── embedder_vllm_retrieval.py   # From vllm_retrieval_embedder.py
└── embedder_graph.py            # From graph_embedders.py
```

#### FR3.2: Extractor Renaming
```
core/framework/extractors/
├── extractor_base.py            # Abstract base class
├── extractor_code.py            # From code_extractor.py
├── extractor_docling.py         # From docling_extractor.py
├── extractor_latex.py           # From latex_extractor.py
├── extractor_robust.py          # From robust_extractor.py
└── extractor_tree_sitter.py     # From tree_sitter_extractor.py
```

#### FR3.3: Framework File Renaming
- `config.py` → `framework_config.py`
- `logging.py` → `framework_logging.py`
- `storage.py` → `framework_storage.py`
- `memory_store.py` → `framework_memory_store.py`
- `base_processor.py` → `framework_base_processor.py`

### Phase 4: Config Integration (Week 2-3)

#### FR4.1: Monitoring Config
- All monitors accept ConfigManager configuration
- Default configs in `/core/configs/monitoring/`
- Runtime override support

#### FR4.2: Embedder/Extractor Config
- Each embedder/extractor has config schema
- Configs in `/core/configs/embedders/` and `/core/configs/extractors/`
- Validation through Pydantic models

### Phase 5: Migration & Testing (Week 3-4)

#### FR5.1: Import Updates
- Update all imports to new paths
- Maintain backward compatibility via `__init__.py`
- Deprecation warnings for old paths

#### FR5.2: Testing
- Unit tests for all monitors
- Integration tests for workflow monitoring
- Performance benchmarks

## Success Metrics

### Quantitative
- **Zero duplication**: No repeated monitoring code
- **100% classification**: Every monitor clearly categorized
- **Import reduction**: 30% fewer import statements
- **Code reduction**: 40% less monitoring code overall
- **Test coverage**: 80% coverage on monitoring code

### Qualitative
- **Clear ownership**: Obvious where each monitor belongs
- **Easy extension**: New monitors follow clear patterns
- **Better debugging**: Easier to find relevant monitors
- **Improved reusability**: Shared monitoring components

## Migration Strategy

### Step 1: Non-Breaking Preparation
1. Create new directory structures
2. Copy (don't move) files to new locations
3. Add deprecation warnings to old locations
4. Update documentation

### Step 2: Gradual Migration
1. Update high-traffic pipelines first
2. Migrate one tool at a time
3. Run parallel testing
4. Monitor for issues

### Step 3: Cleanup
1. Remove deprecated files
2. Archive to Acheron
3. Final testing
4. Documentation update

## Monitoring Demand Matrix

| Demand | When | Why | What | Location |
|--------|------|-----|------|----------|
| System Health | Always | Resource optimization | CPU, GPU, Memory | `core/monitoring/` |
| Batch Progress | Long runs | Failure detection | Checkpoints, ETA | `core/workflows/monitoring/` |
| Pipeline Phases | Multi-stage | Performance tracking | Phase timing, throughput | `core/workflows/monitoring/` |
| Training Metrics | Model training | Research documentation | Loss, accuracy, epochs | `tools/*/monitoring/` |
| API Limits | External calls | Cost control | Rate limits, quotas | `tools/*/monitoring/` |

## Integration with Config System (Issue #34)

This PRD complements the Config System implementation by:
1. **Shared Config Schemas**: Monitoring configs part of base.yaml
2. **Consistent Patterns**: Both use hierarchical configuration
3. **Unified Approach**: Same naming conventions and organization
4. **Config-Driven Monitoring**: All monitors configured through ConfigManager

```yaml
# /core/configs/base.yaml
monitoring:
  enabled: true
  interval: 60  # seconds
  infrastructure:
    gpu: true
    cpu: true
    memory: true
    power: false
  workflow:
    checkpoint_interval: 100
    progress_reporting: true
    eta_calculation: true
  metrics:
    file_output: true
    database_output: false
    retention_days: 7
```

## Example Usage

### Infrastructure Monitoring
```python
from core.monitoring import SystemMonitor
from core.framework.framework_config import ConfigManager

config = ConfigManager.load('monitoring')
monitor = SystemMonitor(config.monitoring.infrastructure)
monitor.start()
```

### Workflow Monitoring
```python
from core.workflows.monitoring import BatchMonitor

monitor = BatchMonitor(
    checkpoint_file="checkpoint.json",
    log_file="pipeline.log",
    config=config.monitoring.workflow
)
monitor.track_progress()
```

### Tool-Specific Monitoring
```python
from tools.graphsage.monitoring import TrainingMonitor
from core.monitoring import SystemMonitor

# Compose tool-specific with infrastructure
system_monitor = SystemMonitor(config)
training_monitor = TrainingMonitor(config)

# Both run in parallel
system_monitor.start()
training_monitor.track_loss()
```

## Risks and Mitigations

### Risk 1: Breaking Changes
**Risk**: Import changes break existing code
**Mitigation**:
- Backward compatible `__init__.py` files
- Gradual migration with deprecation warnings
- Comprehensive testing before removal

### Risk 2: Performance Impact
**Risk**: Monitoring overhead affects performance
**Mitigation**:
- Configurable monitoring levels
- Lazy loading of monitors
- Async monitoring where possible

### Risk 3: Complexity
**Risk**: Three-layer monitoring too complex
**Mitigation**:
- Clear documentation and examples
- Base classes enforce patterns
- Config-driven setup reduces boilerplate

## Open Questions

1. Should we support hot-swapping of monitoring levels?
2. Do we need a monitoring dashboard/UI?
3. Should metrics go to a time-series database?
4. How do we handle distributed monitoring (multi-node)?
5. Should we integrate with external monitoring (Prometheus/Grafana)?

## Appendix

### A. Current Monitoring Inventory

**Infrastructure Monitoring** (`core/monitoring/`):
- system_monitor.py - General system resources
- power_monitor.py - Power consumption
- metrics_analyzer.py - Metrics analysis

**Misplaced Workflow Monitors**:
- tools/arxiv/monitoring/monitor_overnight.py → core/workflows/monitoring/
- tools/arxiv/monitoring/monitor_phased.py → core/workflows/monitoring/
- tools/arxiv/monitoring/weekend_monitor.py → core/workflows/monitoring/

**Tool-Specific (stays in place)**:
- tools/graphsage/monitoring/monitor_training.py - GraphSAGE specific
- core/graph/monitoring/check_graph_progress.py - Graph building specific

### B. Import Impact Analysis

**High Impact** (>10 imports):
- core.framework.embedders - 17 files
- core.framework.extractors.docling_extractor - 12 files

**Medium Impact** (5-10 imports):
- core.framework.logging - 8 files
- core.framework.storage - 6 files

**Low Impact** (<5 imports):
- core.framework.metrics - 0 files (unused)
- core.framework.config - 0 files (unused)

### C. Dependencies

- Depends on: Issue #34 (Config System)
- Blocks: Full production readiness
- Related to: Test infrastructure setup