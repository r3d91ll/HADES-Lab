# ArXiv Tools Directory Organization

## 📅 Latest Consolidation: 2025-08-25
## 📅 Previous Organization: 2025-08-23

This document describes the organized structure of the tools/arxiv directory following our cleanup and reorganization effort.

## 📁 Directory Structure

```
tools/arxiv/
├── pipelines/           # Main processing pipelines
├── monitoring/          # Real-time monitoring tools  
├── scripts/             # Utility and management scripts
├── database/            # Database setup and schemas
├── utils/               # Verification and checking tools
├── logs/                # Processing logs
├── checkpoints/         # Resume points
├── configs/             # Local configurations
├── README.md           # User guide
├── CLAUDE.md           # Developer guidance
├── ORGANIZATION.md     # This file
└── __init__.py         # Package initialization
```

## 📂 Directory Purposes

### `pipelines/` - Main Processing Pipelines

**Purpose**: Contains the ACID-compliant high-performance processing engine.

**Key Files**:
- `arxiv_pipeline_acid.py` - ACID pipeline achieving 6.2 papers/minute
- `arango_acid_processor.py` - ACID transaction handler with stream transactions
- `worker_pool.py` - Parallel worker management for E3 architecture
- `hybrid_search.py` - Cross-database semantic search

**Usage**: Single entry point for all ArXiv paper processing.

### `monitoring/` - Real-time Monitoring Tools

**Purpose**: Tools for tracking pipeline progress, system resources, and processing status.

**Key Files**:
- `monitor_overnight.py` - Production monitor with checkpoint awareness
- `monitor_unified_embeddings.py` - Database status and rate monitoring
- `monitor_pipeline_live.py` - Live queue and worker monitoring
- `pipeline_status_reporter.py` - JSON status reporting
- `demo_queue_monitoring.py` - Queue visualization demo
- `MONITOR_README.md` - Monitoring documentation

**Usage**: Run these alongside pipelines to track progress and diagnose issues.

### `scripts/` - Utility and Management Scripts

**Purpose**: Helper scripts for database management, running pipelines, and system maintenance.

**Key Files**:
- `reset_databases.py` - Clean slate utility for fresh starts
- `run_overnight_unified.sh` - Automated overnight batch runner
- `run_unified_hysteresis.sh` - Standard pipeline launcher
- `conveyance_measurement.py` - Theory validation measurements

**Usage**: Administrative tasks and automated runs.

### `database/` - Database Setup and Management

**Purpose**: SQL schemas, import scripts, and database management tools.

**Key Files**:
- `setup_arxiv_datalake.sql` - PostgreSQL schema
- `build_arxiv_datalake_v2.py` - Metadata import tool
- `database_tools.py` - Database utilities

**Usage**: Initial setup and database maintenance.

### `utils/` - Verification and Checking Tools

**Purpose**: Tools for checking system health, database status, and data integrity.

**Key Files**:
- `check_db_status.py` - Comprehensive database health check
- `verify_database_sync.py` - PostgreSQL-ArangoDB sync verification
- `daily_arxiv_update.py` - Daily metadata update from ArXiv

**Usage**: Regular health checks and updates.

### `logs/` - Processing Logs

**Purpose**: Centralized location for all pipeline and monitoring logs.

**Contents**:
- `unified_pipeline.log` - Main pipeline log
- `monitor_unified.log` - Monitor output
- Timestamped overnight run logs

**Usage**: Debugging and performance analysis.

### `checkpoints/` - Resume Points

**Purpose**: Stores checkpoint files for resuming interrupted processing.

**Contents**:
- `unified_checkpoint.json` - Current pipeline state
- Backup checkpoints with timestamps

**Usage**: Automatic resume after failures.

## 🗄️ Archived Files

The following files were moved to `../../Acheron/HADES/tools/arxiv/` during cleanup:

### Deprecated Pipeline Variants (12 files)
- `arxiv_pipeline_dataparallel.py` - Early GPU parallelization
- `arxiv_pipeline_hysteresis.py` - First hysteresis implementation
- `arxiv_pipeline_memory_fix.py` - Memory optimization experiments
- `arxiv_pipeline_nvlink.py` - NVLink-specific optimizations
- `arxiv_pipeline_unified.py` - First unified approach
- `arxiv_pipeline_unified_flow.py` - Flow control experiments
- Plus various test and experimental versions

### Test Files
- `test_status.json`
- `test_status.tmp`
- `pipeline_status.json`

## 🔄 Organization Benefits

1. **Clear Separation of Concerns**: Each directory has a specific purpose
2. **Easy Navigation**: Logical grouping makes files easy to find
3. **Reduced Clutter**: Root directory only contains essential files
4. **Preserved History**: All deprecated code archived in Acheron
5. **Better Maintainability**: Related files grouped together

## 📝 Guidelines for Future Development

### Adding New Files

- **New pipeline**: Add to `pipelines/`
- **New monitor**: Add to `monitoring/`
- **New utility script**: Add to `scripts/`
- **New database tool**: Add to `database/`
- **New verification tool**: Add to `utils/`

### Deprecating Code

1. Never delete code directly
2. Move to `../../Acheron/HADES/tools/arxiv/` with timestamp
3. Format: `filename_YYYY-MM-DD_HH-MM-SS.ext`
4. Update this document to reflect changes

### Naming Conventions

- Pipeline files: `arxiv_pipeline_*.py`
- Monitor files: `monitor_*.py`
- Script files: Descriptive names (e.g., `reset_databases.py`)
- Shell scripts: `run_*.sh` or `*.sh`

## 🚀 Quick Navigation

### For Processing Papers
```bash
cd pipelines/
python arxiv_pipeline_unified_hysteresis.py --config ../../configs/processors/arxiv_unified.yaml
```

### For Monitoring
```bash
cd monitoring/
python monitor_overnight.py --arango-password "$ARANGO_PASSWORD"
```

### For Database Management
```bash
cd scripts/
python reset_databases.py
```

### For Health Checks
```bash
cd utils/
python check_db_status.py --detailed
```

## 📊 Statistics

- **Active Python files**: 15
- **Shell scripts**: 3
- **Documentation files**: 4
- **Archived files**: 12+
- **Total directories**: 8

## ⚙️ Optimal Configuration (Discovered 2025-08-23)

After extensive testing, we've identified the optimal worker configuration:

### Resource Distribution
- **CPU**: Docling PDF extraction (79-80% utilization)
- **GPU**: Jina v4 embeddings only (99% compute, 80% VRAM)
- **Workers**: 8 parallel (4 per GPU with fp16)
- **Processing Rate**: 1.9-2.5 papers/minute sustained

### Why This Works
1. **Separation of Concerns**: CPU for extraction, GPU for embeddings
2. **No Resource Contention**: Each worker has dedicated resources
3. **Stable VRAM**: ~80% usage prevents OOM errors
4. **Sustainable**: Can run for hours without degradation

### Configuration Settings
```yaml
# In configs/processors/arxiv_unified.yaml
num_workers: 8
device_map:
  embedder: "cuda"      # GPU for embeddings
  extractor: "cpu"      # CPU for Docling
batch_size: 32
use_fp16: true
```

This configuration prioritizes **stability over speed**, achieving consistent throughput without crashes.

---

*This organization follows the HADES project structure guidelines and Actor-Network Theory principles of clear actant separation and obligatory passage points.*