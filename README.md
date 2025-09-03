# HADES - Heterogeneous Adaptive Dimensional Embedding System

Production infrastructure for Information Reconstructionism - processing, storing, and serving embedded knowledge through ArangoDB with local SQLite caching.

## Overview

HADES is a distributed network infrastructure following Actor-Network Theory principles:

- **ArangoDB Graph Store**: Primary storage for embeddings and document structures
- **SQLite Local Cache**: Fast local indexing of PDF locations and metadata
- **Jina v4 Embeddings**: 2048-dimensional with late chunking (32k token context)
- **Direct PDF Processing**: Processes papers directly from `/bulk-store/arxiv-data/pdf/`
- **ACID Compliance**: Atomic transactions ensure data consistency
- **Acheron Protocol**: Deprecated code preservation with timestamps (never delete, always preserve)

## Architecture

```
HADES-Lab/
├── core/                       # Core HADES infrastructure
│   ├── mcp_server/            # MCP interface for Claude integration
│   ├── framework/             # Shared framework
│   │   ├── embedders.py      # Jina v4 embeddings
│   │   ├── extractors/       # Content extraction
│   │   │   ├── docling_extractor.py    # PDF extraction
│   │   │   ├── code_extractor.py       # Code file extraction
│   │   │   └── tree_sitter_extractor.py # Symbol extraction
│   │   └── storage.py        # ArangoDB management
│   ├── processors/            # Base processor classes
│   ├── utils/                 # Core utilities
│   └── logs/                  # Core system logs
│
├── tools/                      # Processing tools (data sources)
│   ├── arxiv/                 # ArXiv paper processing
│   │   ├── pipelines/         # ACID-compliant pipelines
│   │   ├── monitoring/        # Real-time monitoring
│   │   ├── database/          # Database utilities
│   │   ├── scripts/           # Utility scripts
│   │   ├── utils/             # Database utilities
│   │   ├── tests/             # Integration tests
│   │   ├── configs/           # Pipeline configurations
│   │   └── logs/              # ArXiv processing logs
│   └── github/                # GitHub repository processing
│       ├── configs/           # GitHub pipeline configurations
│       ├── github_pipeline_manager.py  # Graph-based processing
│       ├── setup_github_graph.py       # Graph collection setup
│       └── test_treesitter_simple.py   # Tree-sitter testing
│
├── experiments/                # Research and experimentation
│   ├── README.md              # Experiment guidelines
│   ├── experiment_template/   # Template for new experiments
│   ├── datasets/              # Shared experimental datasets
│   │   ├── cs_papers.json     # Computer Science papers
│   │   ├── graph_papers.json  # Graph theory papers
│   │   ├── ml_ai_papers.json  # ML/AI papers
│   │   └── sample_10k.json    # Quick testing sample
│   ├── documentation/         # Experiment-specific analysis
│   │   └── experiments/       # Research notes and findings
│   └── experiment_1/          # Individual experiments
│       ├── config/            # Experiment configurations
│       ├── src/               # Experiment source code
│       └── analysis/          # Results and analysis
│
├── docs/                       # System documentation
│   ├── adr/                   # Architecture Decision Records
│   ├── agents/                # Agent configurations
│   ├── theory/                # Theoretical framework
│   └── methodology/           # Implementation methodology
│
├── Acheron/                    # Deprecated code (timestamped)
│   ├── test_scripts/          # Legacy test scripts (timestamped)
│   ├── configs/               # Deprecated pipeline configurations
│   ├── acid_monitoring.py     # Legacy monitoring script
│   └── pipeline_status_reporter.py  # Legacy pipeline reporter
│
└── .claude/                    # Claude Code configurations
    └── agents/                # Custom agent definitions
```

## Features

### Current (Production Ready)

- **ACID Pipeline**: 11.3 papers/minute with 100% success rate (validated on 1000+ papers)
- **Phase-Separated Architecture**: Extract (GPU-accelerated Docling) → Embed (Jina v4)
- **Direct PDF Processing**: No database dependencies, processes from local filesystem
- **GitHub Repository Processing**: Clone, extract, embed code with Tree-sitter symbol extraction
- **Graph-Based Storage**: Repository → File → Chunk → Embedding relationships in ArangoDB
- **Tree-sitter Integration**: Symbol extraction for Python, JavaScript, TypeScript, Java, C/C++, Go, Rust
- **Jina v4 Late Chunking**: Context-aware embeddings preserving document structure (32k tokens)
- **Multi-collection Storage**: Separate ArangoDB collections for embeddings, equations, tables, images
- **SQLite Caching**: Optional local cache for PDF indexing and metadata
- **Experiments Framework**: Structured research environment with curated datasets
- **Acheron Protocol**: Archaeological preservation of all deprecated code

### In Development

- **Cross-Repository Analysis**: Theory-practice bridge detection across repositories
- **Enhanced Config Understanding**: Leveraging Jina v4's coding LoRA for config semantics
- **Incremental Repository Updates**: Only process changed files
- **Active Monitoring**: Real-time pipeline monitoring system

## Installation

```bash
# Clone the repository
git clone git@github.com:r3d91ll/HADES-Lab.git
cd HADES-Lab

# Install dependencies with Poetry
poetry install
# Optional: activate the virtual environment
# poetry shell

# Configure ArangoDB connection
export ARANGO_PASSWORD="your-arango-password"
export ARANGO_HOST="192.168.1.69"  # or your host

# GPU settings (optional)
export CUDA_VISIBLE_DEVICES=1  # or 0,1 for dual GPU
```

## Usage

### ACID Pipeline (Direct PDF Processing)

```bash
# Run the ACID-compliant pipeline (11.3 papers/minute)
cd tools/arxiv/pipelines/
python arxiv_pipeline.py \
    --config ../configs/acid_pipeline_phased.yaml \
    --count 100 \
    --arango-password "$ARANGO_PASSWORD"

# Monitor progress
tail -f tools/arxiv/logs/acid_phased.log
```

### GitHub Repository Processing

```bash
# Setup graph collections (first time only)
cd tools/github/
python setup_github_graph.py

# Process a single repository
python github_pipeline_manager.py --repo "owner/repo"

# Example: Process word2vec repository
python github_pipeline_manager.py --repo "dav/word2vec"

# Query processed repositories (in ArangoDB)
# Find all embeddings for a repository:
# FOR v, e, p IN 1..3 OUTBOUND 'github_repositories/owner_repo'
#   GRAPH 'github_graph'
#   FILTER IS_SAME_COLLECTION('github_embeddings', v)
#   RETURN v
```

### Creating Experiments

```bash
# Copy template
cp -r experiments/experiment_template experiments/my_experiment

# Update configuration
vim experiments/my_experiment/config/experiment_config.yaml

# Run experiment
cd experiments/my_experiment
python src/run_experiment.py --config config/experiment_config.yaml
```

### Check Status

```bash
# Database status
python tools/arxiv/utils/check_db_status.py --detailed

# Verify GPU availability
nvidia-smi
```

## Philosophy

Following Actor-Network Theory (ANT) principles:

- **HADES is the network**, not any single component
- **Local filesystem**: External actant providing PDF documents
- **ArangoDB**: Primary actant for graph-based knowledge storage
- **Power through translation**: HADES translates documents into embedded knowledge
- **Direct processing**: No intermediate databases, straight from PDF to embeddings
- **Acheron Protocol**: "Code never dies, it flows to Acheron" - preserving development archaeology

## Import Conventions

```python
# From tools (e.g., in tools/arxiv/)
from core.framework.embedders import JinaV4Embedder
from core.framework.extractors import DoclingExtractor

# From core components
from core.mcp_server import server
from core.processors.base_processor import BaseProcessor
```

### Infrastructure vs Experiments

- **Infrastructure** (`core/`, `tools/`): Reusable, production-ready components
- **Experiments** (`experiments/`): Research code, one-off analyses, prototypes

```bash
# Run experiment
cd experiments/my_experiment
python src/run_experiment.py --config config/experiment_config.yaml

# Use shared datasets
python -c "import json; papers = json.load(open('../datasets/cs_papers.json'))"
```

## Key Innovations

1. **Streamlined Architecture**: Direct PDF processing with ArangoDB storage, optional SQLite caching
2. **Late Chunking**: Process full documents (32k tokens) before chunking for context preservation  
3. **Multi-source Integration**: Unified framework for ArXiv, GitHub, and Web data
4. **Graph-Based Code Storage**: Repository → File → Chunk → Embedding relationships enable cross-repo analysis
5. **Tree-sitter Symbol Extraction**: AST-based symbol extraction without semantic interpretation (let Jina handle that)
6. **Information Reconstructionism**: Implementing WHERE × WHAT × CONVEYANCE × TIME theory
7. **Experiments Framework**: Structured research environment with infrastructure separation
8. **Archaeological Code Preservation**: Acheron protocol maintains complete development history

## License

Apache License 2.0 - See LICENSE file for details
