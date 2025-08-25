# Directory Structure Update - January 2025

## Summary of Changes

This document outlines the major directory structure reorganization implemented in the HADES-Lab project to improve code organization, maintain historical context, and clarify the separation between core framework and specific tools.

## Major Structural Changes

### 1. Tools Organization
**Before:** ArXiv tools scattered at root level (`arxiv/`)  
**After:** Consolidated under `tools/arxiv/`

This change clearly separates domain-specific tools from the core framework, making the codebase more modular and maintainable.

### 2. Core Framework Relocation
**Before:** `core_framework/` at root level  
**After:** `core/framework/`

The core framework is now properly namespaced under `core/`, with additional subdirectories for:
- `core/mcp_server/` - MCP server implementation
- `core/processors/` - Processing modules  
- `core/utils/` - Core utilities
- `core/logs/` - Centralized logging

### 3. Acheron Protocol Implementation
**New:** `Acheron/` directory for deprecated code

Following the mythological principle that "code never dies, it flows to Acheron," all deprecated code is now preserved with timestamps:
- Format: `filename_YYYY-MM-DD_HH-MM-SS.ext`
- Current contents:
  - `Acheron/test_scripts/` - Legacy test scripts
  - `Acheron/configs/` - Deprecated pipeline configurations

### 4. Test Consolidation
**Before:** `test_scripts/` at root level  
**After:** 
- Active tests: `tools/arxiv/tests/`
- Deprecated tests: `Acheron/test_scripts/` (timestamped)

## Updated Directory Tree

```
HADES-Lab/
├── tools/
│   └── arxiv/
│       ├── pipelines/          # ACID-compliant processing
│       ├── monitoring/         # Real-time monitoring
│       ├── database/           # PostgreSQL setup
│       ├── scripts/            # Utility scripts
│       ├── utils/              # Database utilities
│       ├── tests/              # Active tests
│       ├── configs/            # Pipeline configurations
│       └── logs/               # ArXiv-specific logs
├── core/
│   ├── framework/
│   │   ├── embedders.py       # Jina v4 embeddings
│   │   ├── extractors/        # Docling extraction
│   │   └── storage.py         # ArangoDB management
│   ├── mcp_server/            # MCP server
│   ├── processors/            # Processing modules
│   ├── utils/                 # Core utilities
│   └── logs/                  # Core logs
├── docs/
│   ├── agents/                # Agent configurations
│   ├── experiments/           # Research results
│   ├── theory/                # Theoretical framework
│   └── adr/                   # Architecture decisions
├── Acheron/                   # Deprecated code (timestamped)
│   ├── test_scripts/
│   └── configs/
└── pyproject.toml            # Poetry dependencies
```

## Import Path Updates

### Core Framework Imports
```python
# Old
from core_framework.embedders import JinaV4Embedder
from core_framework.extractors.docling_extractor import DoclingExtractor

# New
from core.framework.embedders import JinaV4Embedder
from core.framework.extractors.docling_extractor import DoclingExtractor
```

### ArXiv Tools Imports
```python
# Old
from arxiv.pipelines.arxiv_pipeline import ProcessingTask

# New
from tools.arxiv.pipelines.arxiv_pipeline import ProcessingTask
```

## Command Path Updates

### Pipeline Execution
```bash
# Old
cd arxiv/pipelines/
python arxiv_pipeline.py --config ../configs/acid_pipeline_phased.yaml

# New
cd tools/arxiv/pipelines/
python arxiv_pipeline.py --config ../configs/acid_pipeline_phased.yaml
```

### Database Operations
```bash
# Old
cd arxiv/utils/
python check_db_status.py

# New
cd tools/arxiv/utils/
python check_db_status.py
```

## Configuration File Updates

- Active configs: `tools/arxiv/configs/`
- Deprecated configs: `Acheron/configs/` (preserved for reference)

## Benefits of New Structure

1. **Clear Separation**: Tools vs. core framework
2. **Historical Preservation**: Acheron maintains our development history
3. **Better Organization**: Related components grouped together
4. **Easier Navigation**: Logical hierarchy
5. **Import Clarity**: Unambiguous module paths

## Migration Notes

- All existing functionality preserved
- No breaking changes to APIs
- Documentation updated to reflect new paths
- Agent configurations updated with correct paths

## Acheron Philosophy

"In Greek mythology, Acheron is the river of sorrow where souls cross into the underworld. In HADES-Lab, it's where deprecated code flows - never deleted, always preserved with timestamps for archaeological analysis of our development journey."

This approach ensures we can:
- Study failed experiments
- Understand design evolution
- Recover useful patterns from deprecated code
- Maintain complete development history

## Next Steps

1. Update any external scripts referencing old paths
2. Verify all imports in notebooks and utilities
3. Update CI/CD pipelines if applicable
4. Communicate changes to team members

---

*Last Updated: January 2025*