# Phase 3: Configuration and Monitoring Restructure PRD

**Parent Issue**: #35 (Master Core Infrastructure Restructure)
**Timeline**: Days 8-10 of 21-day sprint
**Status**: Ready for Implementation

## Problem Statement

Configuration is scattered across multiple systems (YAML files, environment variables, hardcoded values). Monitoring is inconsistently implemented with duplicate code in tools/arxiv/monitoring/ and core/framework/metrics.py.

## Solution Overview

Create a centralized configuration system using Pydantic for validation and a unified monitoring module that follows demand-driven principles (monitor when needed, not always).

## Detailed Requirements

### 1. Configuration Module (`core/config/`)

**New Structure**:
```
core/config/
├── __init__.py
├── config_base.py              # Base configuration classes
├── config_manager.py           # Central configuration manager
├── config_loader.py            # YAML/JSON/ENV loader
├── schemas/
│   ├── __init__.py
│   ├── schema_pipeline.py     # Pipeline configuration schema
│   ├── schema_database.py     # Database configuration schema
│   ├── schema_embedding.py    # Embedding configuration schema
│   └── schema_monitoring.py   # Monitoring configuration schema
└── defaults/
    ├── default_pipeline.yaml  # Default pipeline settings
    ├── default_database.yaml  # Default database settings
    └── default_embedding.yaml # Default embedding settings
```

**Configuration Hierarchy** (highest to lowest priority):
1. Command-line arguments
2. Environment variables
3. User config file
4. Default config file

**Pydantic Models Example**:
```python
from pydantic import BaseModel, Field, validator

class EmbeddingConfig(BaseModel):
    model_name: str = Field(default="jinaai/jina-embeddings-v4")
    batch_size: int = Field(default=80, ge=1, le=256)
    use_fp16: bool = Field(default=True)
    max_seq_length: int = Field(default=8192, ge=512, le=32768)
    num_workers: int = Field(default=2, ge=1, le=8)

    @validator('num_workers')
    def validate_workers(cls, v):
        import torch
        max_gpus = torch.cuda.device_count()
        if v > max_gpus:
            raise ValueError(f"num_workers ({v}) exceeds available GPUs ({max_gpus})")
        return v

class DatabaseConfig(BaseModel):
    arango_host: str = Field(default="localhost")
    arango_port: int = Field(default=8529)
    arango_database: str = Field(default="academy_store")
    use_unix_socket: bool = Field(default=True)
    unix_socket_path: str = Field(default="/tmp/arangodb.sock")
```

**Requirements**:
- Validate all configuration at startup
- Support hot-reloading for development
- Maintain backward compatibility with existing YAML configs
- Provide clear error messages for invalid config
- Support configuration inheritance/composition

### 2. Monitoring Module (`core/monitoring/`)

**Consolidate from**:
- `core/framework/metrics.py`
- `tools/arxiv/monitoring/`
- Various ad-hoc logging

**New Structure**:
```
core/monitoring/
├── __init__.py
├── monitoring_base.py          # Base monitoring classes
├── monitoring_metrics.py       # Performance metrics
├── monitoring_pipeline.py      # Pipeline-specific monitoring
├── monitoring_gpu.py           # GPU utilization tracking
├── monitoring_database.py      # Database performance
└── exporters/
    ├── __init__.py
    ├── exporter_console.py     # Console output
    ├── exporter_file.py        # File logging
    ├── exporter_grafana.py     # Grafana/Prometheus
    └── exporter_tensorboard.py # TensorBoard integration
```

**Demand-Driven Monitoring Principles**:
1. **When**: Only collect metrics when explicitly enabled
2. **Why**: Clear purpose for each metric (debugging, optimization, alerting)
3. **What**: Specific, actionable metrics (not everything)

**Key Metrics**:
```python
class PipelineMetrics:
    # Performance
    papers_per_second: float
    gpu_utilization: float
    memory_usage_gb: float

    # Quality
    extraction_success_rate: float
    embedding_coverage: float
    checkpoint_saves: int

    # Errors
    failed_papers: List[str]
    retry_attempts: int
    timeout_count: int
```

**Requirements**:
- Lazy initialization (only when monitoring enabled)
- Minimal overhead when disabled
- Structured logging with context
- Export to multiple backends
- Real-time dashboard capability

### 3. Utils Cleanup (`core/utils/`)

**Current Contents** (to be moved):
- Workflow components → `core/workflows/`
- Database helpers → `core/database/`
- Monitoring utilities → `core/monitoring/`

**Keep in Utils** (true utilities only):
```
core/utils/
├── __init__.py
├── utils_file.py               # File operations
├── utils_text.py               # Text processing
├── utils_math.py               # Mathematical helpers
├── utils_time.py               # Time/date utilities
└── utils_validation.py         # Input validation
```

**Requirements**:
- Only pure utility functions
- No business logic
- No external dependencies (except stdlib)
- Well-tested and documented
- Reusable across all modules

## Implementation Steps

### Day 8: Configuration System
1. Create `core/config/` structure
2. Implement Pydantic schemas
3. Build configuration manager
4. Add validation and loading
5. Test with existing YAML configs

### Day 9: Monitoring Consolidation
1. Create `core/monitoring/` structure
2. Consolidate metrics from all sources
3. Implement demand-driven collection
4. Add exporters for different backends
5. Test performance overhead

### Day 10: Utils Cleanup
1. Audit current `core/utils/` contents
2. Move non-utility code to appropriate modules
3. Organize remaining utilities
4. Update all imports
5. Verify nothing broken

## Success Criteria

### Configuration
- [ ] All settings centralized
- [ ] Validation working with clear errors
- [ ] Existing YAML configs still work
- [ ] Environment variables respected
- [ ] Hot-reload in development mode

### Monitoring
- [ ] Metrics consolidated in one place
- [ ] Minimal overhead when disabled
- [ ] Real-time metrics available
- [ ] Export to file/console/Grafana
- [ ] GPU monitoring working

### Utils
- [ ] Only true utilities remain
- [ ] All code moved to proper modules
- [ ] No broken imports
- [ ] Clear documentation

## Performance Impact

- Configuration validation: <100ms at startup
- Monitoring overhead: <1% when enabled, 0% when disabled
- No impact on processing throughput (48.3 papers/sec)

## Migration Guide

### Configuration Migration
```python
# Old (scattered)
batch_size = os.environ.get('BATCH_SIZE', 80)
config = yaml.load('config.yaml')

# New (centralized)
from core.config import ConfigManager
config = ConfigManager.load('pipeline')
batch_size = config.embedding.batch_size
```

### Monitoring Migration
```python
# Old (ad-hoc)
print(f"Processed {count} papers")
logger.info(f"GPU memory: {mem}")

# New (structured)
from core.monitoring import PipelineMonitor
monitor = PipelineMonitor(enabled=True)
monitor.record('papers_processed', count)
monitor.record('gpu_memory_gb', mem)
```

## Validation

```bash
# Test configuration
python -c "from core.config import ConfigManager; c = ConfigManager.load('pipeline'); print('✓ Config working')"

# Test monitoring
python -c "from core.monitoring import PipelineMonitor; m = PipelineMonitor(); m.record('test', 1); print('✓ Monitoring working')"

# Test utils
python -c "from core.utils.utils_text import clean_text; print('✓ Utils working')"

# Integration test
cd tools/arxiv/pipelines/
python arxiv_pipeline.py --config ../configs/acid_pipeline_phased.yaml --count 5 --monitor
```

## Notes

- Configuration must support both old YAML and new Pydantic models
- Monitoring should be disabled by default (demand-driven)
- Utils should be minimal - resist the urge to add
- Consider using structlog for structured logging
- GPU monitoring requires nvidia-ml-py

## References

- Configuration PRD: `/docs/prd/centralized-config-prd.md`
- Phase 2 PRD: `/docs/prd/core_restructure/phase2-database-workflows-prd.md`
- Master PRD: `/docs/prd/core_restructure/master-core-restructure-prd.md`
- Issue #35: Master Core Infrastructure Restructure