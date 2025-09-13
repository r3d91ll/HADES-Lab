# Product Requirements Document: Centralized Configuration System

**Version:** 1.0
**Date:** January 13, 2025
**Author:** HADES Development Team
**Status:** Draft

## Executive Summary

Implement a centralized, hierarchical configuration management system for HADES-Lab to eliminate configuration duplication, standardize environment variable handling, and provide a single source of truth for all processing pipelines.

## Problem Statement

### Current State
- **Configuration Fragmentation**: Configuration files scattered across `/tools/*/configs/` directories
- **Code Duplication**: Every pipeline manually loads YAML files with `yaml.safe_load()`
- **Inconsistent Environment Variables**: Direct `os.getenv()` calls throughout codebase
- **No Validation**: Raw dictionary access without type checking or validation
- **Missing Defaults**: Each file must define its own defaults
- **Dead Code**: Existing `core/framework/config.py` ConfigManager is unused

### Impact
- **Maintenance Burden**: Changes require updates in multiple locations
- **Error Prone**: No validation leads to runtime failures from typos or missing keys
- **Inconsistency**: Different pipelines handle configuration differently
- **Technical Debt**: Configuration logic mixed with business logic
- **Testing Difficulty**: Hard to mock configurations for unit tests

## Solution Overview

Activate and enhance the existing ConfigManager system in `core/framework/config.py` to provide:
1. Hierarchical configuration with clear precedence rules
2. Pydantic-based validation and type safety
3. Centralized environment variable handling
4. Separation of configuration from business logic

## Requirements

### Functional Requirements

#### FR1: Configuration Hierarchy
- **FR1.1**: Support 4-level hierarchy (highest to lowest priority):
  1. Runtime overrides (passed to functions)
  2. Environment variables (`HADES_*` and legacy `ARANGO_*`)
  3. Processor-specific configs (`/core/configs/processors/*.yaml`)
  4. Base configuration (`/core/configs/base.yaml`)
- **FR1.2**: Deep merge configurations preserving nested structures
- **FR1.3**: Support both HADES_ and ARANGO_ environment variable prefixes

#### FR2: Configuration Structure
- **FR2.1**: Create `/core/configs/` directory structure:
  ```
  core/configs/
  ├── base.yaml                 # Base configuration
  ├── processors/               # Processor-specific overrides
  │   ├── arxiv_pipeline.yaml
  │   ├── github_pipeline.yaml
  │   └── pdf_workflow.yaml
  └── README.md                 # Configuration documentation
  ```
- **FR2.2**: Define standard configuration sections:
  - Database (ArangoDB, PostgreSQL)
  - Processing (batch sizes, timeouts, workers)
  - Embeddings (models, dimensions)
  - Logging (levels, formats, destinations)
  - Metrics (collection, reporting)

#### FR3: Type Safety and Validation
- **FR3.1**: Use Pydantic models for all configuration objects
- **FR3.2**: Validate configuration at load time, not runtime
- **FR3.3**: Provide clear error messages for validation failures
- **FR3.4**: Support optional fields with sensible defaults

#### FR4: Migration Path
- **FR4.1**: Maintain backward compatibility during migration
- **FR4.2**: Support both old (direct YAML loading) and new (ConfigManager) patterns initially
- **FR4.3**: Provide migration utilities to convert existing configs

### Non-Functional Requirements

#### NFR1: Performance
- **NFR1.1**: Configuration loading < 100ms
- **NFR1.2**: Cache parsed configurations in memory
- **NFR1.3**: Lazy load processor-specific configs only when needed

#### NFR2: Developer Experience
- **NFR2.1**: Simple API: `config = ConfigManager.load('arxiv_pipeline')`
- **NFR2.2**: IDE autocomplete support through type hints
- **NFR2.3**: Clear documentation and examples
- **NFR2.4**: Helpful error messages with suggestions

#### NFR3: Testing
- **NFR3.1**: Easy to mock for unit tests
- **NFR3.2**: Support test-specific configuration overrides
- **NFR3.3**: Validate all configs in CI/CD pipeline

#### NFR4: Observability
- **NFR4.1**: Log configuration sources (which level provided each value)
- **NFR4.2**: Support configuration dump for debugging
- **NFR4.3**: Detect and warn about unused configuration keys

## Implementation Plan

### Phase 1: Infrastructure Setup (Week 1)
1. Create `/core/configs/` directory structure
2. Write `base.yaml` with all current defaults
3. Create processor-specific configs from existing YAML files
4. Add configuration documentation

### Phase 2: Core Integration (Week 1-2)
1. Update `core/framework/config.py` if needed
2. Integrate ConfigManager into `core/database/arango/arango_client.py`
3. Create configuration factory for common patterns
4. Add configuration validation tests

### Phase 3: Pipeline Migration (Week 2-3)
Priority order:
1. `tools/arxiv/pipelines/arxiv_pipeline.py` - Most complex, highest impact
2. `tools/github/github_pipeline_manager.py` - Second data source
3. `core/workflows/workflow_pdf_batch.py` - Shared by multiple tools
4. Other tools and utilities

### Phase 4: Cleanup and Documentation (Week 3-4)
1. Remove duplicate configuration loading code
2. Archive old configuration files to Acheron
3. Update CLAUDE.md with configuration guidelines
4. Write comprehensive configuration guide
5. Add configuration examples to each tool's README

## Success Metrics

### Quantitative
- **Zero configuration duplication**: Single source of truth for each setting
- **100% validation coverage**: All configs validated before use
- **50% code reduction**: Remove configuration boilerplate from pipelines
- **< 5 minute migration**: New pipelines can adopt config system quickly

### Qualitative
- **Developer Satisfaction**: Easier to understand and modify configurations
- **Reduced Errors**: Fewer runtime failures from configuration issues
- **Better Testing**: Easier to test with different configurations
- **Improved Onboarding**: New developers understand system configuration quickly

## Configuration Schema

### Base Configuration Structure
```yaml
# /core/configs/base.yaml
database:
  arango:
    host: "localhost"
    port: 8529
    username: "root"
    password: ""  # From ARANGO_PASSWORD env var
    database: "academy_store"
    pool_size: 10
    retry_max: 3
    retry_delay: 1.0

  postgres:
    host: "localhost"
    port: 5432
    database: "arxiv"
    username: "postgres"
    password: ""  # From PGPASSWORD env var

processing:
  batch_size: 24
  num_workers: 8
  timeout_seconds: 300
  checkpoint_interval: 100
  staging_dir: "/dev/shm/acid_staging"

embeddings:
  model: "jinaai/jina-embeddings-v4"
  dimension: 2048
  batch_size: 10
  use_fp16: true
  device: "cuda"

chunking:
  strategy: "late"  # "late" or "sliding"
  chunk_size: 8192
  chunk_overlap: 512
  max_tokens: 32768

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: null  # Optional log file path

metrics:
  enabled: true
  report_interval: 60  # seconds
  include_gpu: true
```

### Processor-Specific Override Example
```yaml
# /core/configs/processors/arxiv_pipeline.yaml
processing:
  batch_size: 24
  num_workers: 32  # More extraction workers

embeddings:
  batch_size: 24  # Optimized for dual A6000

phases:
  extraction:
    workers: 32
    batch_size: 24

  embedding:
    workers: 8
    batch_size: 24
```

## API Examples

### Basic Usage
```python
from core.framework.config import ConfigManager

# Load configuration for specific processor
config = ConfigManager.load('arxiv_pipeline')

# Access typed configuration
db_config = config.database.arango
print(f"Connecting to {db_config.host}:{db_config.port}")

# Initialize components with config
db_manager = ArangoDBManager(config.database.arango.dict())
```

### Runtime Override
```python
# Override specific values at runtime
config = ConfigManager.load('arxiv_pipeline', override={
    'processing': {'batch_size': 48},
    'logging': {'level': 'DEBUG'}
})
```

### Testing Support
```python
# In tests, use minimal config
def test_pipeline():
    test_config = ConfigManager.load('test', override={
        'database': {'arango': {'database': 'test_db'}},
        'processing': {'num_workers': 1}
    })
    pipeline = Pipeline(test_config)
    # ... test code
```

## Risks and Mitigations

### Risk 1: Breaking Changes
**Risk**: Migration could break existing pipelines
**Mitigation**:
- Implement backward compatibility layer
- Gradual migration with feature flags
- Comprehensive testing before each migration step

### Risk 2: Configuration Complexity
**Risk**: Hierarchical config might be confusing
**Mitigation**:
- Clear documentation with examples
- Configuration debugging tools
- Validation with helpful error messages

### Risk 3: Performance Impact
**Risk**: Configuration loading could slow down startup
**Mitigation**:
- Cache parsed configurations
- Lazy load processor-specific configs
- Profile and optimize hot paths

## Open Questions

1. Should we support hot-reloading of configuration during runtime?
2. How should we handle secrets (passwords, API keys)?
3. Should we version configuration schemas for backward compatibility?
4. Do we need configuration profiles (dev, staging, prod)?

## Appendix

### A. File Migration List

**High Priority:**
- tools/arxiv/pipelines/arxiv_pipeline.py
- tools/github/github_pipeline_manager.py
- core/workflows/workflow_pdf_batch.py

**Medium Priority:**
- core/database/arango/arango_manager.py
- core/graph/orchestrate_graph_build.py
- tools/arxiv/utils/arxiv_metadata_ingestion_sentence.py

**Low Priority:**
- Test files (can continue using simple configs)
- One-off scripts
- Experimental code

### B. Environment Variable Mapping

| Legacy Variable | New Variable | Description |
|----------------|--------------|-------------|
| ARANGO_PASSWORD | HADES_DB_ARANGO_PASSWORD | ArangoDB password |
| ARANGO_HOST | HADES_DB_ARANGO_HOST | ArangoDB host |
| PGPASSWORD | HADES_DB_POSTGRES_PASSWORD | PostgreSQL password |
| CUDA_VISIBLE_DEVICES | HADES_COMPUTE_GPUS | GPU device list |

### C. Related Documents
- [CLAUDE.md](/CLAUDE.md) - Coding guidelines
- [Actor-Network Theory Integration](/docs/theory/ant.md)
- [Information Reconstructionism](/docs/theory/reconstructionism.md)