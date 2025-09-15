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

## Commands

### Environment Setup

```bash
# Activate virtual environment (REQUIRED before any Python commands)
source /home/todd/.cache/pypoetry/virtualenvs/hades-z5jmQstn-py3.12/bin/activate

# Install dependencies with Poetry
poetry install
poetry shell

# Set environment variables
export ARANGO_PASSWORD="your-arango-password"
export ARANGO_HOST="localhost"
export CUDA_VISIBLE_DEVICES=0,1  # Configure GPUs
export PGPASSWORD="your-postgres-password"
```

### Development Commands

```bash
# Format code
black core/ tools/ tests/

# Type checking
mypy core/

# Lint
ruff check core/ tools/ tests/

# Run tests
python -m pytest tests/
python -m pytest tests/test_embedders_phase1.py  # Single test
python -m pytest tests/arxiv/acid/  # Test directory

# Run integration tests
cd tests/arxiv/
python test_large_scale_processing.py --config ../configs/large_scale_test.yaml
```

### Workflow Commands

```bash
# Run ArXiv parallel workflow (primary production workflow)
cd core/workflows/
python workflow_arxiv_parallel.py \
    --metadata-file /path/to/arxiv_metadata.json \
    --batch-size 1000 \
    --embedding-batch-size 128 \
    --num-workers 8 \
    --arango-password "$ARANGO_PASSWORD"

# Run PDF batch processing
python workflow_pdf_batch.py \
    --pdf-directory /bulk-store/arxiv-data/pdf/ \
    --batch-size 24 \
    --num-workers 32

# Run ArXiv metadata workflow
python workflow_arxiv_metadata.py \
    --config ../config/workflows/arxiv_metadata_default.yaml
```

### Database Operations

```bash
# Check database status
cd core/tools/arxiv/utils/
python check_db_status.py --detailed

# Monitor GPU usage
nvidia-smi -l 1  # Update every second
watch -n 1 gpustat  # Alternative with better formatting
```

### Monitoring Commands

```bash
# Monitor workflow progress
cd core/monitoring/
python progress_tracker.py

# View performance metrics
python performance_monitor.py --show-metrics

# Check logs
tail -f logs/workflow_*.log
tail -f core/logs/*.log
```

## High-Level Architecture

### Core Processing Pipeline

The HADES system implements a parallel processing architecture optimized for the Conveyance Framework equation **C = (W·R·H/T)·Ctx^α**:

1. **Workflow Layer** (`core/workflows/`)
   - `workflow_base.py`: Abstract base class for all workflows
   - `workflow_arxiv_parallel.py`: Production multi-GPU parallel processing
   - `workflow_pdf_batch.py`: Direct PDF processing without database dependencies
   - `workflow_arxiv_memory.py`: Memory-optimized processing for large documents

2. **Embedder Layer** (`core/embedders/`)
   - `JinaV4Embedder`: 2048-dimensional embeddings with 32k context window
   - `SentenceTransformersEmbedder`: High-throughput embeddings
   - All embedders implement late chunking (mandatory)

3. **Storage Layer** (`core/database/`)
   - **ArangoDB**: Graph database for processed documents and embeddings
   - **PostgreSQL**: Complete ArXiv metadata (2.7M+ papers)
   - **LMDB**: High-performance key-value storage for caching

4. **Monitoring Layer** (`core/monitoring/`)
   - Real-time progress tracking
   - Performance metrics collection
   - GPU utilization monitoring
   - Memory usage tracking

### Module Organization

```
HADES-Lab/
├── core/                      # Core infrastructure (reusable)
│   ├── workflows/            # Processing workflows
│   ├── embedders/           # Embedding models
│   ├── extractors/          # Document extraction
│   ├── database/            # Storage backends
│   ├── monitoring/          # Performance tracking
│   └── config/              # Configuration management
├── tools/                    # Specific implementations
│   ├── arxiv/               # ArXiv paper processing
│   ├── github/              # GitHub repository processing
│   └── rag_utils/           # RAG utilities
├── tests/                    # Test suites
│   └── arxiv/               # ArXiv-specific tests
├── Acheron/                  # Deprecated code archive (timestamped)
└── docs/                     # Documentation
    └── prd/                 # Product requirements documents
```

### Key Technical Features

- **Parallel GPU Processing**: Multi-worker architecture with GPU isolation
- **Late Chunking**: Preserves context across chunk boundaries (mandatory)
- **Atomic Transactions**: All-or-nothing database operations
- **Memory Optimization**: Streaming processing for large documents
- **Error Recovery**: Checkpoint-based resumption
- **Phase Separation**: Extraction → Embedding pipeline

### Performance Characteristics

- **Throughput**: 40+ papers/second with parallel processing
- **GPU Memory**: 7-8GB per worker with fp16
- **Batch Sizes**: 1000 records (I/O), 128 embeddings (GPU)
- **Context Window**: 32k tokens (Jina v4)
- **Embedding Dimensions**: 2048 (Jina v4), 768 (Sentence Transformers)

## Acheron Protocol - Code Preservation

**Never delete code**. Move deprecated code to `Acheron/` with timestamps:

```bash
# Always add timestamp when moving deprecated code
mv old_file.py Acheron/module_name/old_file_2025-01-20_14-30-25.py
```

This preserves the archaeological record of development decisions.