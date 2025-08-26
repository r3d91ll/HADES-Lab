# HADES - Heterogeneous Adaptive Dimensional Embedding System

Production infrastructure for Information Reconstructionism - processing, storing, and serving embedded knowledge through a hybrid PostgreSQL-ArangoDB architecture.

## Overview

HADES is a distributed network infrastructure following Actor-Network Theory principles:

- **PostgreSQL Data Lake**: Source of truth for metadata (2.79M ArXiv papers)
- **ArangoDB Graph Store**: Expensive computations only (embeddings, structures)
- **Jina v4 Embeddings**: 2048-dimensional with late chunking (32k token context)
- **MCP Server**: Model Context Protocol interface for Claude integration
- **Hybrid Architecture**: HADES orchestrates between databases without duplication

## Architecture

```
HADES-Lab/
├── core/                       # Core HADES infrastructure
│   ├── mcp_server/            # MCP interface for Claude integration
│   ├── framework/             # Shared framework
│   │   ├── embedders.py      # Jina v4 embeddings
│   │   ├── extractors/       # Docling PDF extraction
│   │   └── storage.py        # ArangoDB management
│   ├── processors/            # Base processor classes
│   ├── utils/                 # Core utilities
│   └── logs/                  # Core system logs
│
├── tools/                      # Processing tools (data sources)
│   └── arxiv/                 # ArXiv paper processing
│       ├── pipelines/         # ACID-compliant pipelines
│       ├── monitoring/        # Real-time monitoring
│       ├── database/          # PostgreSQL setup
│       ├── scripts/           # Utility scripts
│       ├── utils/             # Database utilities
│       ├── tests/             # Integration tests
│       ├── configs/           # Pipeline configurations
│       └── logs/              # ArXiv processing logs
│
├── experiments/                # Research and experimentation
│   ├── README.md              # Experiment guidelines
│   ├── experiment_template/   # Template for new experiments
│   ├── datasets/              # Shared experimental datasets
│   ├── documentation/         # Experiment-specific docs
│   └── experiment_1/          # Active experiments
│
├── docs/                       # System documentation
│   ├── adr/                   # Architecture Decision Records
│   ├── agents/                # Agent configurations
│   ├── theory/                # Theoretical framework
│   └── methodology/           # Implementation methodology
│
├── Acheron/                    # Deprecated code (timestamped)
│   ├── test_scripts/          # Legacy test scripts
│   └── configs/               # Deprecated configurations
│
└── .claude/                    # Claude Code configurations
    └── agents/                # Custom agent definitions
```

## Features

### Current (Production Ready)

- **ACID Pipeline**: 11.3 papers/minute with 100% success rate (validated on 1000+ papers)
- **E3 Architecture**: Extract (36 CPU) → Encode (Jina v4) → Embed (8 GPU workers)
- **ArXiv Processing**: 375k papers in experiment window (Dec 2012 - Aug 2016)
- **Jina v4 Late Chunking**: Context-aware embeddings preserving document structure (32k tokens)
- **Multi-collection Storage**: Separate collections for embeddings, equations, tables, images
- **MCP Server**: Async tools for progressive document processing

### In Development

- **Additional Data Sources**: GitHub repositories, web documentation
- **Cross-source Graphs**: Theory-practice bridges across multiple sources
- **Enhanced Embeddings**: Domain-specific fine-tuning

## Installation

```bash
# Clone the repository
git clone git@github.com:r3d91ll/HADES-Lab.git
cd HADES-Lab

# Install dependencies with Poetry
poetry install
# Optional: activate the virtual environment
# poetry shell

# Configure database connections
export PGPASSWORD="your-postgres-password"
export ARANGO_PASSWORD="your-arango-password"
export ARANGO_HOST="192.168.1.69"  # or your host

# GPU settings (optional)
export CUDA_VISIBLE_DEVICES=1  # or 0,1 for dual GPU
```

## Usage

### ACID Pipeline (PostgreSQL → ArangoDB)

```bash
# Run the ACID-compliant pipeline (11.3 papers/minute)
cd tools/arxiv/pipelines/
python arxiv_pipeline.py \
    --config ../configs/acid_pipeline_phased.yaml \
    --count 100 \
    --arango-password "$ARANGO_PASSWORD"

# Monitor progress (from repo root)
python core/utils/monitor_pipeline.py \
  --checkpoint-file tools/arxiv/pipelines/acid_phased_checkpoint.json
```

### MCP Server

```bash
# Install in Claude Code (from repo root)
claude mcp add hades-arxiv python "$(pwd)/core/mcp_server/launch.py" \
  -e ARANGO_PASSWORD="$ARANGO_PASSWORD"

# Or run standalone for testing
python core/mcp_server/launch.py
```

### Check Status

```bash
# Database status
python utils/check_db_status.py --detailed

# Jina v4 deployment verification
python tools/arxiv/verify_jina_v4_deployment.py
```

## Philosophy

Following Actor-Network Theory (ANT) principles:

- **HADES is the network**, not any single component
- **PostgreSQL**: External actant providing metadata
- **ArangoDB**: External actant storing computations
- **Power through translation**: HADES translates between actants
- **No duplication**: Each piece of data has one authoritative source

## Import Conventions

```python
# From tools (e.g., in tools/arxiv/)
from core.framework.embedders import JinaV4Embedder
from core.framework.extractors import DoclingExtractor

# From core components
from core.mcp_server import server
from core.processors.base_processor import BaseProcessor
```

## Key Innovations

1. **Hybrid Architecture**: PostgreSQL for metadata, ArangoDB for expensive computations only
2. **Late Chunking**: Process full documents (32k tokens) before chunking for context preservation
3. **Multi-source Integration**: Unified framework for ArXiv, GitHub, and Web data
4. **Information Reconstructionism**: Implementing WHERE × WHAT × CONVEYANCE × TIME theory

## License

Apache License 2.0 - See LICENSE file for details
