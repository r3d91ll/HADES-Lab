# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## HADES — Conveyance Framework (System Prompt)

**Mission:**
All reasoning, evaluation, and design choices must be framed as applications of the **Conveyance Framework** below. Unless explicitly instructed otherwise, use the **efficiency view** and keep the interpretation of **α** restricted to context amplification (do **not** exponentiate time except in the "monolithic" alternative).

## Core variables

* **W** = What (signal/content quality)
* **R** = Where (relational/topological positioning)
* **H** = Who (agent/model capability & access patterns)
* **T** = Time to converge (latency/cost)
* **L, I, A, G** = Context components

  * **L**: local coherence
  * **I**: instruction fit
  * **A**: actionability
  * **G**: grounding
* **Ctx** = wL·L + wI·I + wA·A + wG·G  (0 ≤ each component ≤ 1; weights default to 0.25 unless specified)
* **α** ∈ [1.5, 2.0] (super-linear amplification exponent applied to **Ctx only**)

## Canonical equations

### 1) Conveyance — Efficiency view (default)

```math
C = (W · R · H / T) · Ctx^α
```

Interpretation: outcome per unit time, boosted super-linearly by context quality.

### 2) Conveyance — Capability view (when T is fixed/controlled)

```math
C_cap = (W · R · H) · Ctx^α
```

Use for apples-to-apples capability comparisons at a fixed time budget.

### 3) Monolithic alternative (use sparingly)

```math
C = ((W · R · H / T) · Ctx)^α
```

Note: puts time inside the exponent and muddies α's interpretation. Only use if explicitly requested.

### 4) Self-optimization (sleep cycle)

Given a target conveyance C_target:

```math
H = (C_target · T) / (W · R · Ctx^α)
```

Raise **H**, lower **T**, improve **W/R**, or strengthen **Ctx** to hit the target.

### 5) Zero-propagation gate

If any of {W, R, H} = 0 or T → ∞ ⇒ C = 0.

## Operational rules

1. **α applies only to Ctx.** Never exponentiate **T** in the default/capability views.
2. **Choose the time stance deliberately:**

   * Use **efficiency** when latency/throughput matters or when T varies.
   * Use **capability** for controlled comparisons at fixed T.
3. **Avoid double-counting time.** If better Ctx requires extra retrieval/rerank/citation work, charge that cost to **T**, not to **α**.
4. **Mapping requirement (for any study/system):**
   Map reported variables to {C, W, R, H, T, L, I, A, G}. Compute Ctx from L/I/A/G and stated weights.
5. **Estimating α (if applicable):**

   * Prefer within-item contrasts holding W/R/H/T fixed or measured.
   * Compute:  α̂ = Δlog(C) / Δlog(Ctx).
   * If T varies, include log T explicitly (efficiency view) and keep α on log Ctx.
6. **Reporting:**
   When time varies, report both **efficiency** and **capability** views if possible, and state any confounders (e.g., outer-loop effects, retrieval policy changes).
7. **Zero-propagation:**
   If any base dimension collapses (W, R, H → 0 or T → ∞), declare C = 0 and explain which factor failed.

## What to log (so results are estimable)

For each run/condition:

* Outcome: **C** (e.g., EM/F1/pass@k/quality).
* Factors: **W, R, H, T**, and **L, I, A, G** (→ **Ctx**).
* Protocol: model/decoding params, retrieval policy, steps/halting, dataset split.

**All analyses, critiques, and designs must conform to this framework and explicitly state which view (efficiency vs capability) is used and why.**

## 🚨 CRITICAL: Late Chunking Principle

**MANDATORY**: All text chunking MUST use late chunking. Never use naive chunking.

### Why Late Chunking is Required

From the Conveyance Framework: **C = (W·R·H/T)·Ctx^α**

- **Naive chunking** breaks context awareness → Ctx approaches 0 → **C = 0** (zero-propagation)
- **Late chunking** preserves full document context → Ctx remains high → **C is maximized**

### Implementation Requirements

1. **Process full text first**: Always tokenize and encode the complete document (up to model limits)
2. **Then chunk with context**: Create chunks from the contextualized representation
3. **Never chunk before encoding**: This loses critical semantic relationships

```python
# ❌ WRONG - Naive chunking (forbidden)
chunks = split_text(document, chunk_size=512)
embeddings = [embed(chunk) for chunk in chunks]  # Each chunk lacks context

# ✅ CORRECT - Late chunking (mandatory)
full_encoding = encode_full_document(document)  # Process entire document
chunks = create_chunks_with_context(full_encoding, chunk_size=512)  # Context-aware chunks
```

### Embedder Selection

- **High throughput (48+ papers/sec)**: Use `SentenceTransformersEmbedder`
- **Sophisticated processing**: Use `JinaV4Embedder` (transformers)
- **Both MUST use late chunking**: This is non-negotiable

## 🚨 CRITICAL: Development Cycle

**ALWAYS follow this development cycle for new features:**

### PRD → Issue → Branch → Code → Test → PR

1. **PRD (Product Requirements Document)**
   * Create comprehensive PRD in `docs/prd/`
   * Define problem, solution, requirements, success metrics
   * Get alignment on what we're building BEFORE coding

2. **Issue**
   * Create GitHub issue from PRD: `gh issue create`
   * Reference issue number in all commits
   * Link PRD in issue description

3. **Branch**
   * Create feature branch: `git checkout -b feature/feature-name`
   * NEVER develop directly on main
   * Keep branches focused on single feature

4. **Code**
   * Implement based on PRD requirements
   * Follow existing patterns and architecture
   * Reuse components where possible (70%+ target)

5. **Test**
   * Write and run tests BEFORE creating PR
   * Ensure all tests pass
   * Document test results

6. **PR (Pull Request)**
   * Create PR with comprehensive description
   * Reference issue number (Closes #XX)
   * Wait for CodeRabbit review
   * Address review comments

### Example Workflow

```bash
# 1. Create PRD
vim docs/prd/new_feature_prd.md

# 2. Create Issue
gh issue create --title "Feature: New Feature" --body "$(cat docs/prd/new_feature_prd.md)"

# 3. Create Branch
git checkout -b feature/new-feature

# 4. Code (implement feature)
# ... development work ...

# 5. Test
python tests/test_new_feature.py

# 6. Commit and Push
git add .
git commit -m "feat: Add new feature (Issue #XX)"
git push -u origin feature/new-feature

# 7. Create PR
gh pr create --title "feat: New Feature (Issue #XX)" --body "..."
```

**NO CODING WITHOUT A PRD!** This ensures we build the right thing with clear requirements.

## Overview

HADES-Lab is a research infrastructure implementing Information Reconstructionism theory through production-grade document processing and embedding systems. The system processes ArXiv papers and GitHub repositories using the mathematical framework WHERE × WHAT × CONVEYANCE × TIME = Information, with Actor-Network Theory principles guiding the architecture.

## Quick Start Commands

### Run the ACID Pipeline (Most Common)

```bash
# Set environment variables
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

# GPU verification
nvidia-smi

# Clear ArangoDB collections (nuclear option)
cd ../utils/
# Use ArangoDB web interface or arangosh to clear collections
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

### Development Environment Setup

```bash
# Install with Poetry (recommended)
poetry install
poetry shell  # Activate virtual environment

# Set required environment variables
export ARANGO_PASSWORD="your-arango-password"
export ARANGO_HOST="localhost"  # or your ArangoDB host
export PGPASSWORD="your-postgres-password"

# Verify GPU availability
nvidia-smi
```

### Other Operations

```bash
# GitHub repository processing
cd tools/github/
python setup_github_graph.py  # First time setup
python github_pipeline_manager.py --repo "owner/repo"

# Run experiments
cd experiments/my_experiment/
python src/run_experiment.py --config config/experiment_config.yaml
```

## High-Level Architecture

### Core Concept: Multi-Dimensional Information Theory

The system implements the equation **Information = WHERE × WHAT × CONVEYANCE × TIME** where:

* **WHERE**: File system location, ArangoDB graph proximity (64 dimensions)
* **WHAT**: Semantic content via Jina v4 embeddings (1024 dimensions)
* **CONVEYANCE**: Actionability/implementability (936 dimensions)
* **TIME**: Temporal positioning (24 dimensions)
* **Context**: Exponential amplifier (Context^α where α ≈ 1.5-2.0)

If any dimension = 0, then Information = 0 (multiplicative dependency).

### Module Structure

```dir
HADES-Lab/
├── core/                      # Reusable infrastructure
│   ├── framework/            # Embedders, extractors, storage
│   ├── processors/           # Document processing
│   ├── mcp_server/          # MCP interface for Claude integration
│   └── database/            # ArangoDB management
├── tools/                    # Data source processors
│   ├── arxiv/               # ArXiv paper processing (production-ready)
│   ├── github/              # GitHub repository processing
│   └── hirag/               # Hierarchical retrieval system
├── experiments/             # Research code and analyses
└── docs/                    # Architecture decisions and theory
```

### Dual Storage Architecture

1. **PostgreSQL** (`arxiv` database): Complete ArXiv metadata (2.7M+ papers)
   * Authors, categories, abstracts, submission dates
   * File tracking: `has_pdf`, `pdf_path`, `has_latex`, `latex_path`

2. **ArangoDB** (`academy_store` database): Processed knowledge graph
   * `arxiv_papers`: Processing status and embeddings
   * `arxiv_chunks`: Text segments with context preservation
   * `arxiv_embeddings`: 2048-dimensional Jina v4 vectors
   * `arxiv_structures`: Equations, tables, images

3. **Local Storage**: Direct PDF processing from `/bulk-store/arxiv-data/pdf/`

## Key Technical Features

### ACID Pipeline Performance

* **11.3 papers/minute** end-to-end processing rate

* **Phase-separated architecture**: Extraction → Embedding
* **Late chunking**: Process full documents (32k tokens) before chunking
* **Direct PDF processing**: No database dependencies
* **Atomic transactions**: All-or-nothing consistency

### Advanced Processing

* **Jina v4 embeddings**: 2048-dimensional with late chunking

* **Tree-sitter integration**: Symbol extraction for 7+ languages
* **GPU acceleration**: Dual-GPU support with memory management
* **HiRAG integration**: Hierarchical retrieval-augmented generation

## Configuration-Driven Architecture

Most operations use YAML configuration files:

```yaml
# Example: tools/arxiv/configs/acid_pipeline_phased.yaml
phases:
  extraction:
    workers: 32
    batch_size: 24
    timeout_seconds: 300
  embedding:
    workers: 8
    batch_size: 24
    use_fp16: true
```

Configuration files are located in:

* `tools/arxiv/configs/` - ArXiv processing
* `tools/github/configs/` - GitHub processing
* `experiments/*/config/` - Experiment-specific

## Import Conventions

```python
# From core framework (when in tools/ or experiments/)
from core.framework.embedders import JinaV4Embedder
from core.framework.extractors import DoclingExtractor
from core.framework.storage import ArangoStorage

# From tools (when in experiments/ or other tools/)
from core.tools.arxiv.pipelines.arxiv_pipeline import AcidPipeline

# Within same module
from .utils import helper_function
```

## Common Development Workflows

### Processing ArXiv Papers

1. **Setup database** (one-time):

   ```bash
   cd tools/arxiv/utils/
   python rebuild_database.py
   ```

2. **Run ACID pipeline**:

   ```bash
   cd tools/arxiv/pipelines/
   python arxiv_pipeline.py --config ../configs/acid_pipeline_phased.yaml --count 100
   ```

3. **Monitor progress**:

   ```bash
   tail -f ../logs/acid_phased.log
   ```

### Creating Experiments

1. **Copy template**:

   ```bash
   cp -r experiments/experiment_template experiments/my_experiment
   ```

2. **Configure experiment**:

   ```bash
   vim experiments/my_experiment/config/experiment_config.yaml
   ```

3. **Use curated datasets**:
   * `experiments/datasets/cs_papers.json` - Computer Science papers
   * `experiments/datasets/ml_ai_papers.json` - ML/AI papers
   * `experiments/datasets/sample_10k.json` - Quick testing sample

### Processing GitHub Repositories

```bash
cd tools/github/
python setup_github_graph.py  # First time setup
python github_pipeline_manager.py --repo "owner/repository"
```

## Performance Expectations

### Processing Rates

* **Extraction**: ~36 papers/minute (32 workers)

* **Embedding**: ~8 papers/minute (8 GPU workers)
* **End-to-end**: ~11.3 papers/minute
* **GPU Memory**: 7-8GB per worker with fp16

### Memory Requirements

* **System RAM**: 64GB minimum, 128GB recommended

* **GPU Memory**: RTX 3090/4090 (24GB) or RTX A6000 (48GB)
* **Storage**: 2TB NVMe SSD for staging operations

## Theoretical Integration Requirements

**CRITICAL**: All code must include docstrings connecting implementation to theoretical framework:

```python
def calculate_conveyance(document):
    """
    Calculate the CONVEYANCE dimension - actionability of information.

    From Information Reconstructionism theory: measures how readily
    information transforms from theory to practice. High conveyance
    indicates clear procedural pathways with implementation instructions.

    In Actor-Network Theory terms, this quantifies the "translation"
    potential - how easily knowledge crosses network boundaries.
    """
```

This makes the codebase a living demonstration of interdisciplinary theory.

## Environment Variables

```bash
# Database connections
export ARANGO_PASSWORD="your-password"
export ARANGO_HOST="localhost"
export PGPASSWORD="your-postgres-password"

# GPU settings
export USE_GPU=true
export CUDA_VISIBLE_DEVICES=0,1
```

## Acheron Protocol - Code Preservation

**Never delete code**. Move deprecated code to `Acheron/` with timestamps:

```bash
# Always add timestamp when moving deprecated code
mv old_file.py Acheron/module_name/old_file_2025-01-20_14-30-25.py
```

This preserves the archaeological record of development decisions.

## Infrastructure vs Research Separation

* **Infrastructure** (`core/`, `tools/`): Production-ready, reusable components
* **Research** (`experiments/`): One-off analyses, prototypes, research code

Experiments can import from infrastructure, but infrastructure should not depend on experiments.

## Critical Implementation Notes

1. **Late Chunking**: Always process full documents before chunking to preserve context
2. **Atomic Operations**: All database operations must be atomic (success or rollback)
3. **Phase Separation**: Complete extraction before embedding to optimize GPU memory
4. **Direct Processing**: Process files from filesystem without database queries
5. **Error Recovery**: All pipelines must support resumption from checkpoints
6. **Context Preservation**: Maintain document structure through processing pipeline

## MCP Server Integration

HADES includes an MCP server for Claude Code integration:

```bash
# Add to Claude Code
claude mcp add hades-arxiv python /home/todd/olympus/HADES-Lab/core/mcp_server/launch.py \
  -e ARANGO_PASSWORD="${ARANGO_PASSWORD}" \
  -e ARANGO_HOST="${ARANGO_HOST}"
```

Provides tools for paper processing, semantic search, and GPU monitoring.