# PRD-001: Core Config Module

**Version:** 1.0
**Date:** January 13, 2025
**Parent PRD:** Master Core Restructure
**Module:** `core/config/`
**Status:** Draft

## Module Purpose & Scope

Centralized configuration management for all HADES components, providing hierarchical configuration with validation, environment variable support, and runtime overrides.

## Current State Analysis

### What Exists
- `core/framework/config.py` - Unused but well-designed ConfigManager
- Scattered YAML files in `tools/*/configs/`
- Direct `os.getenv()` calls throughout codebase
- Manual `yaml.safe_load()` in every pipeline

### Problems
- No single source of truth
- No validation
- No type safety
- Duplicate configuration logic

## Proposed Structure

```
core/config/
├── __init__.py              # Public API exports
├── config_manager.py        # Main ConfigManager class
├── config_schemas.py        # Pydantic schemas
├── config_loader.py         # YAML/env loading logic
├── config_validator.py      # Validation logic
└── defaults/               # Default configurations
    ├── base.yaml          # System-wide defaults
    ├── embedders.yaml     # Embedder defaults
    ├── extractors.yaml    # Extractor defaults
    ├── monitoring.yaml    # Monitoring defaults
    └── workflows.yaml     # Workflow defaults
```

## File Mapping

| Old Location | New Location |
|-------------|--------------|
| `core/framework/config.py` | `core/config/config_manager.py` |
| `tools/*/configs/*.yaml` | `core/config/defaults/` |
| New | `core/config/config_schemas.py` |
| New | `core/config/config_validator.py` |

## API Design

### Public Interface
```python
from core.config import ConfigManager, Config

# Load configuration
config = ConfigManager.load('arxiv_pipeline')

# Access typed configuration
db_config = config.database.arango
embedder_config = config.embedders.jina

# Runtime override
config = ConfigManager.load('arxiv_pipeline', overrides={
    'database.arango.host': 'remote-host'
})

# Validate configuration
ConfigManager.validate(config)
```

### Configuration Hierarchy
1. Runtime overrides (highest)
2. Environment variables
3. Module-specific configs
4. Default configurations (lowest)

## Configuration Schema

```python
# config_schemas.py
from pydantic import BaseModel, Field

class DatabaseConfig(BaseModel):
    """Database configuration."""

    class ArangoConfig(BaseModel):
        host: str = Field(default="localhost")
        port: int = Field(default=8529)
        username: str = Field(default="root")
        password: str = Field(default="")
        database: str = Field(default="academy_store")
        pool_size: int = Field(default=10)

    class PostgresConfig(BaseModel):
        host: str = Field(default="localhost")
        port: int = Field(default=5432)
        database: str = Field(default="arxiv")
        username: str = Field(default="postgres")
        password: str = Field(default="")

    arango: ArangoConfig = Field(default_factory=ArangoConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)

class Config(BaseModel):
    """Root configuration object."""
    database: DatabaseConfig
    embedders: EmbeddersConfig
    extractors: ExtractorsConfig
    monitoring: MonitoringConfig
    workflows: WorkflowsConfig
```

## Testing Strategy

### Unit Tests
- Config loading from files
- Environment variable parsing
- Hierarchy merging
- Validation errors

### Integration Tests
- Full pipeline configuration
- Multi-module configuration
- Override scenarios

### Test Files
```
tests/core/config/
├── test_config_manager.py
├── test_config_loader.py
├── test_config_validator.py
├── test_config_schemas.py
└── fixtures/
    └── test_configs.yaml
```

## Migration Plan

### Phase 1: Setup (Day 1)
1. Create `core/config/` directory structure
2. Move and refactor `config.py`
3. Create Pydantic schemas
4. Create default configurations

### Phase 2: Integration (Day 2-3)
1. Update high-priority pipelines
2. Add deprecation warnings
3. Create migration guide

### Phase 3: Cleanup (Day 4-5)
1. Remove old config loading code
2. Archive old config files
3. Update documentation

## Success Criteria

### Must Have
- [ ] All pipelines use ConfigManager
- [ ] 100% validation coverage
- [ ] Environment variable support
- [ ] Backward compatibility

### Should Have
- [ ] Hot reload support
- [ ] Config versioning
- [ ] Automatic documentation

### Metrics
- Zero configuration duplication
- 50% reduction in config-related code
- < 100ms config load time
- 100% type safety

## Dependencies

### Upstream
- None (foundational module)

### Downstream
- All other core modules
- All tools
- All experiments

## Example Usage

### Basic Pipeline
```python
from core.config import ConfigManager
from core.database import ArangoManager

# Load configuration
config = ConfigManager.load('arxiv_pipeline')

# Use configuration
db = ArangoManager(config.database.arango)
```

### With Overrides
```python
# Override for testing
test_config = ConfigManager.load('arxiv_pipeline', overrides={
    'database.arango.database': 'test_db',
    'monitoring.enabled': False
})
```

### Custom Configuration
```python
# Create custom config
custom_config = ConfigManager.create({
    'database': {'arango': {'host': 'custom-host'}},
    'embedders': {'model': 'custom-model'}
})
```

## Notes

- This module is foundational - must be completed first
- All other modules will depend on this
- Must maintain backward compatibility during migration
- Consider using Hydra or similar if config gets complex