# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Run the ACID Pipeline (Most Common)
```bash
# Set environment variables
export PGPASSWORD="your-postgres-password"
export ARANGO_PASSWORD="your-arango-password"
export CUDA_VISIBLE_DEVICES=0,1  # or single GPU: 1

# Run the production-ready ACID pipeline
cd tools/arxiv/pipelines/
python arxiv_pipeline.py \
    --config ../configs/acid_pipeline_phased.yaml \
    --count 100 \
    --arango-password "$ARANGO_PASSWORD"

# Monitor pipeline progress
cd ../monitoring/
python acid_monitoring.py
```

### Database Operations
```bash
# Check database status
cd tools/arxiv/utils/
python check_db_status.py --detailed

# Verify Jina v4 deployment (if script exists)
python verify_jina_v4_deployment.py

# Reset databases (nuclear option)
cd ../scripts/
python reset_databases.py --pg-password "$PGPASSWORD" --arango-password "$ARANGO_PASSWORD"
```

### Testing
```bash
# Run integration tests
cd tools/arxiv/tests/acid/
python test_acid_pipeline.py
python test_integration.py
python test_performance.py

# Note: Legacy test scripts have been moved to Acheron/test_scripts/
# with timestamps for historical reference
```

### Build & Development
```bash
# Install dependencies with Poetry
poetry install
poetry shell

# Format code
black tools/arxiv/ core/framework/

# Type checking
mypy core/framework/

# Lint
ruff check tools/arxiv/ core/framework/
```

## High-Level Architecture

HADES-Lab implements a hybrid PostgreSQL-ArangoDB architecture following Actor-Network Theory principles, where HADES orchestrates between external database actants without data duplication.

### Core Components

1. **PostgreSQL Data Lake** (`arxiv_datalake`)
   - Source of truth for all metadata (2.79M ArXiv papers)
   - Tables: `arxiv_papers`, `arxiv_versions`, `arxiv_authors`, `arxiv_paper_authors`
   - Never duplicated in ArangoDB

2. **ArangoDB Graph Store** (`academy_store`)
   - Stores only expensive computations (embeddings, extracted structures)
   - Collections: `arxiv_embeddings`, `arxiv_structures`
   - Atomic transactions ensure ACID compliance

3. **ACID Pipeline** (`tools/arxiv/pipelines/arxiv_pipeline.py`)
   - Phase-separated architecture: Extraction (CPU) → Embedding (GPU)
   - 11.3 papers/minute processing rate with 100% success
   - Dual A6000 GPUs with NVLink for parallel processing

### Processing Flow

```
ArXiv Papers → PostgreSQL (metadata) → HADES Processing → ArangoDB (embeddings)
                                              ↓
                                    Phase 1: Extract (Docling)
                                    Phase 2: Embed (Jina v4)
```

### Key Technologies

- **Jina v4 Embeddings**: 2048-dimensional with late chunking (32k token context)
- **Docling v2**: GPU-accelerated PDF extraction
- **ProcessPoolExecutor**: Parallel worker management
- **Atomic Transactions**: All-or-nothing database operations

## Mathematical Framework Implementation

The system implements Information Reconstructionism: **C = (W·R·H)/T · Ctx^α**

- **W (WHAT)**: Semantic content via Jina v4 embeddings
- **R (WHERE)**: Relational positioning in dual databases  
- **H (WHO)**: Processing capability (8 GPU workers optimal)
- **T (Time)**: Minimized through parallel processing
- **Ctx**: Context preserved through late chunking
- **α ≈ 1.5-2.0**: Super-linear context amplification

## Directory Structure

```
HADES-Lab/
├── tools/
│   └── arxiv/                  # ArXiv processing tools
│       ├── pipelines/          # ACID-compliant processing pipelines
│       ├── monitoring/         # Real-time monitoring tools
│       ├── database/           # PostgreSQL setup and management
│       ├── scripts/            # Utility scripts
│       ├── utils/              # Database status and daily updates
│       ├── tests/              # Integration and unit tests
│       └── configs/            # Pipeline configurations
├── core/                       # Core framework components
│   ├── framework/              # Shared framework modules
│   │   ├── embedders.py       # Jina v4 with late chunking
│   │   ├── extractors/        # Docling PDF extraction
│   │   └── storage.py         # ArangoDB management
│   ├── mcp_server/            # MCP server implementation
│   ├── processors/            # Processing modules
│   └── utils/                 # Core utilities
├── experiments/                # Research and experimentation
│   ├── README.md              # Experiment guidelines
│   ├── experiment_template/   # Template for new experiments
│   ├── datasets/              # Shared experimental datasets
│   ├── documentation/         # Experiment-specific docs
│   └── experiment_1/          # Active experiments
├── docs/                       # System documentation
│   ├── agents/                # Agent prompts and configs
│   ├── adr/                   # Architecture Decision Records
│   ├── theory/                # Theoretical framework docs
│   └── methodology/           # Implementation methodology
├── Acheron/                    # Deprecated code (timestamped)
│   ├── test_scripts/          # Legacy test scripts
│   └── configs/               # Deprecated configurations
├── .claude/                    # Claude Code configurations
│   └── agents/                # Custom agent definitions
└── pyproject.toml             # Poetry dependency management
```

## Configuration Management

Pipelines are config-driven via YAML files:

- `tools/arxiv/configs/acid_pipeline_phased.yaml` - Production ACID pipeline
- `Acheron/configs/arxiv_unified.yaml` - Deprecated unified processing (moved to Acheron)
- `Acheron/configs/arxiv_memory_optimized.yaml` - Deprecated low-memory config

Key configuration parameters:
- `batch_size`: Papers per batch (32 optimal)
- `phases.extraction.workers`: Number of extraction workers (8 optimal)
- `phases.embedding.workers`: Number of embedding workers (8 optimal)
- `staging.directory`: RamFS staging location (`/dev/shm/acid_staging`)

## Import Conventions

```python
# From core framework
from core.framework.embedders import JinaV4Embedder
from core.framework.extractors.docling_extractor import DoclingExtractor
from core.framework.storage import ArangoDBManager

# From ArXiv tools
from tools.arxiv.pipelines.arango_db_manager import ArangoDBManager, retry_with_backoff
from tools.arxiv.pipelines.arxiv_pipeline import ProcessingTask, PhaseManager
```

## Common Issues & Solutions

### Pipeline Stuck/Slow
```bash
# Check GPU memory
nvidia-smi
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
# Check logs
tail -f tools/arxiv/logs/acid_phased.log
```

### Database Connection Issues
```bash
# Verify ArangoDB is running
curl http://localhost:8529/_api/version
# Check PostgreSQL
psql -h localhost -U postgres -d arxiv_datalake -c "SELECT COUNT(*) FROM arxiv_papers;"
```

### Out of Memory
- Reduce `batch_size` in configuration
- Reduce number of workers
- Use `arxiv_memory_optimized.yaml` config

## Performance Baselines

- **Extraction Phase**: ~36 papers/minute with 8 CPU workers
- **Embedding Phase**: ~8 papers/minute with 8 GPU workers  
- **End-to-end**: 11.3 papers/minute (validated on 1000+ papers)
- **GPU Memory**: 7-8GB per worker with fp16
- **Success Rate**: 100% with atomic transactions

## Critical Implementation Details

1. **Late Chunking**: Process full documents (32k tokens) before chunking to preserve context
2. **No Duplication**: PostgreSQL has metadata, ArangoDB has only computations
3. **Atomic Operations**: All database writes are transactional
4. **Phase Separation**: Extract phase completes before embedding phase starts
5. **Worker Isolation**: Each worker is completely independent (no inter-worker communication)

## Acheron Protocol - Deprecated Code Management

**CRITICAL**: Never delete code. All deprecated code is moved to the `Acheron/` directory with timestamps.

### Deprecation Process

When code becomes obsolete:
1. Move to appropriate Acheron subdirectory
2. Add timestamp suffix: `filename_YYYY-MM-DD_HH-MM-SS.ext`
3. Preserve directory structure within Acheron
4. Document reason for deprecation

### Current Acheron Contents

- `Acheron/test_scripts/` - Legacy test scripts with timestamps
- `Acheron/configs/` - Deprecated pipeline configurations
- `Acheron/test_scripts/core_framework/` - Old framework version

This archaeological approach preserves our development history and failed experiments for future analysis.

## Environment Variables

```bash
# Required
export PGPASSWORD="your-postgres-password"
export ARANGO_PASSWORD="your-arango-password"

# Optional
export ARANGO_HOST="192.168.1.69"  # Default: localhost
export CUDA_VISIBLE_DEVICES=0,1    # GPUs to use
export USE_GPU=true                # Enable GPU acceleration
```

## Data Locations

- **ArXiv Metadata**: `/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json`
- **PDF Repository**: `/bulk-store/arxiv-data/pdf/` (organized by YYMM/arxiv_id.pdf)
- **Staging Directory**: `/dev/shm/acid_staging/` (RamFS for speed)
- **Checkpoints**: `tools/arxiv/pipelines/acid_phased_checkpoint.json`
- **Logs**: `tools/arxiv/logs/acid_phased.log`
- **Experiment Datasets**: `experiments/datasets/` (curated JSON datasets for research)

## Experiments Directory

The `experiments/` directory is separate from infrastructure code and contains:

- **experiment_template/**: Template for creating new experiments
- **datasets/**: Shared datasets (cs_papers.json, graph_papers.json, etc.)
- **documentation/**: Experiment-specific documentation and analysis
- **experiment_*/**: Individual experiment directories

### Creating a New Experiment

```bash
# Copy template
cp -r experiments/experiment_template experiments/my_experiment

# Update configuration
vim experiments/my_experiment/config/experiment_config.yaml

# Run experiment
cd experiments/my_experiment
python src/run_experiment.py --config config/experiment_config.yaml
```

### Infrastructure vs Experiments

- **Infrastructure** (`core/`, `tools/`): Reusable, production-ready components
- **Experiments** (`experiments/`): Research code, one-off analyses, prototypes

Experiments can import infrastructure but should never modify it directly.