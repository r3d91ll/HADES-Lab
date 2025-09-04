# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the ArXiv tools directory.

## Quick Start (ArXiv Processing)

### Most Common Commands

```bash
# NEW: ArXiv Lifecycle Manager (Recommended)
cd scripts/
python lifecycle_cli.py process 2508.21038  # Single paper
python lifecycle_cli.py batch papers.txt    # Multiple papers
python lifecycle_cli.py status 2508.21038   # Check status

# Traditional: ACID pipeline with phase separation
cd pipelines/
python arxiv_pipeline.py \
    --config ../configs/acid_pipeline_phased.yaml \
    --count 1000 \
    --arango-password "$ARANGO_PASSWORD"

# Monitor processing in real-time
tail -f ../logs/acid_phased.log

# Database operations
cd ../utils/
python check_db_status.py --detailed  # Database status
python rebuild_postgresql_complete.py # Full database rebuild

# Check GPU usage
nvidia-smi
```

### Emergency Fixes

```bash
# Pipeline stuck/slow - check logs
tail -f ../logs/acid_phased.log | grep -E "Phase|Worker"

# Out of memory
python -c "import torch; torch.cuda.empty_cache()"

# Clear staging directory
rm -rf /dev/shm/acid_staging/*

# Worker distribution issues
ps aux | grep python | grep worker | wc -l
```

### What You Need to Know (Critical Points)

1. **Workers process COMPLETE papers** - not individual chunks (late chunking requires full context)
2. **No inter-worker communication** - each worker is completely isolated
3. **8 GPU workers optimal** - 4 per A6000 with fp16 (~7-8GB VRAM each)
4. **Phase separation** - Extraction completes before embedding starts
5. **Direct PDF processing** - No database dependencies, straight from filesystem
6. **ArangoDB only** - All storage goes to ArangoDB collections

## ArXiv Lifecycle Manager (NEW)

### Unified Paper Processing Workflow

The **ArXiv Lifecycle Manager** replaces scattered single-purpose scripts with a comprehensive workflow that handles:

```bash
# Complete paper lifecycle in one command
python lifecycle_cli.py process 2508.21038
```

**What it does:**
1. **Checks PostgreSQL** - Queries database for existing metadata and files
2. **Downloads missing content** - Fetches PDF/LaTeX from ArXiv API if needed
3. **Updates databases** - Keeps PostgreSQL and ArangoDB synchronized  
4. **Processes through ACID** - Extracts text, equations, tables, images
5. **Generates embeddings** - Creates Jina v4 vectors with late chunking
6. **Integrates HiRAG** - Updates hierarchical retrieval collections

### Key Features

- **Idempotent operations** - Safe to run multiple times
- **Complete audit trail** - Track what was processed when
- **LaTeX intelligence** - Automatically detects and downloads LaTeX sources
- **Batch processing** - Handle hundreds of papers efficiently
- **Status tracking** - Monitor paper processing states
- **Error recovery** - Resume from failures

### Commands

```bash
# Process single paper
python lifecycle_cli.py process 2508.21038

# Check paper status  
python lifecycle_cli.py status 2508.21038 --json

# Process batch of papers
python lifecycle_cli.py batch paper_list.txt --output results.json

# Fetch metadata only (no processing)
python lifecycle_cli.py metadata 2508.21038
```

### Status Levels

- **NOT_FOUND**: Paper not in system
- **METADATA_ONLY**: Metadata available, no files
- **DOWNLOADED**: PDF/LaTeX downloaded, not processed
- **PROCESSED**: Fully processed through ACID pipeline  
- **HIRAG_INTEGRATED**: Available in HiRAG collections

## Project Overview

ArXiv Tools provides infrastructure for processing ArXiv papers with dual storage (PostgreSQL + ArangoDB), implementing the mathematical framework where **C = (W·R·H)/T · Ctx^α**. Following Actor-Network Theory principles, the system maintains complete metadata in PostgreSQL while storing embeddings and extracted structures in ArangoDB.

## High-Level Architecture

### Dual Storage Architecture

1. **PostgreSQL** (`arxiv` database) - Complete Metadata Repository
   - 2.7M papers with full ArXiv metadata (Kaggle dataset)
   - Authors, categories, abstracts, submission dates, versions
   - File tracking: `has_pdf`, `pdf_path`, `has_latex`, `latex_path`
   - Query interface for creating targeted paper lists

2. **Local File Storage** (`/bulk-store/arxiv-data/`)
   - `pdf/YYMM/`: 1.8M+ PDF files organized by year-month
   - `latex/YYMM/`: LaTeX source files (.tar.gz format)  
   - `metadata/`: Kaggle ArXiv metadata snapshot (4.5GB JSON)

3. **ArangoDB Graph Store** (`academy_store` database) - Processed Knowledge
   - `arxiv_papers`: Paper processing status and document embeddings
   - `arxiv_chunks`: Text segments with context preservation
   - `arxiv_embeddings`: Jina v4 vectors (2048-dim) with late chunking
   - `arxiv_equations`: Mathematical formulas with LaTeX rendering
   - `arxiv_tables`: Table content and structure
   - `arxiv_images`: Image metadata and descriptions
   - `arxiv_structures`: Combined structural elements
   - Atomic transactions ensure consistency

3. **ACID Pipeline** (`arxiv_pipeline.py`)
   - Config-driven via `configs/acid_pipeline_phased.yaml`
   - Processes PDFs directly from filesystem
   - Phase 1: Extract with GPU-accelerated Docling
   - Phase 2: Generate Jina v4 embeddings (2048-dim)
   - Stores embeddings and structures in ArangoDB

## Common Development Commands

### Pipeline Processing

```bash
# Run the ACID pipeline (config-driven)
python arxiv_pipeline.py \
    --config ../configs/acid_pipeline_phased.yaml \
    --arango-password "$ARANGO_PASSWORD" \
    --count 100  # Number of papers to process

# Run with specific source
python arxiv_pipeline.py \
    --config ../configs/acid_pipeline_phased.yaml \
    --source local \
    --count 50

# Monitor pipeline
tail -f ../logs/acid_phased.log
```

### Database Status Checks

```bash
# Check database status
python check_db_status.py --detailed

# Connect to ArangoDB web interface
# Browse to: http://192.168.1.69:8529
# Database: academy_store
```

## Key Components

### ACID Pipeline (`arxiv_pipeline.py`)

- **Phase-separated**: Extraction phase completes before embedding phase
- **Checkpointing**: Automatic resume on failure with atomic checkpoint saves
- **Batch processing**: Configurable batch sizes for GPU efficiency
- **Error recovery**: Tracks failed papers for retry

### Phase Manager

- **Phase 1 - Extraction**: GPU-accelerated Docling extracts PDFs to staged JSON
- **Phase 2 - Embedding**: Jina v4 processes staged JSONs with late chunking
- **Staging**: Uses RamFS (`/dev/shm/acid_staging`) for inter-phase data transfer
- **GPU management**: Cleans GPU memory between phases

### Critical Implementation Details

**Direct Processing**: PDFs are processed directly from filesystem without database queries.

**Late Chunking with Jina v4**: Process full documents (up to 32k tokens) BEFORE chunking to preserve context.

**Phase Separation**: Complete extraction phase ensures all GPU memory is available for embedding phase.

**Atomic Operations**: All database operations are atomic - either fully succeed or fully rollback.

## Database Schema

### ArangoDB Collections

- `arxiv_papers`: Paper metadata and processing status
  - `_key`: Sanitized arxiv_id
  - `status`: Processing status (PROCESSED, FAILED)
  - `num_chunks`, `num_equations`, `num_tables`, `num_images`: Counts

- `arxiv_chunks`: Text chunks from papers
  - `paper_id`: Link to arxiv_papers
  - `text`: Chunk text
  - `chunk_index`: Position in document
  - `context_window_used`: Tokens of context

- `arxiv_embeddings`: Vector embeddings
  - `paper_id`: Link to arxiv_papers
  - `chunk_id`: Link to arxiv_chunks
  - `vector`: 2048-dimensional embedding
  - `model`: 'jina-v4'

- `arxiv_structures`: Extracted structures
  - Equations, tables, images with metadata

## Expected Performance

### Processing Rates

- **Extraction Phase**: ~36 papers/minute with 32 workers
- **Embedding Phase**: ~8 papers/minute with 8 GPU workers
- **End-to-end**: ~11.3 papers/minute overall
- **GPU Memory**: 7-8GB per embedding worker with fp16

## Environment Variables

```bash
# ArangoDB
export ARANGO_PASSWORD='YOUR_ARANGO_PASSWORD'
export ARANGO_HOST='192.168.1.69'  # Default host

# Processing
export USE_GPU=true
export CUDA_VISIBLE_DEVICES=0,1  # Dual GPU
```

## File Locations

- **ArXiv PDFs**: `/bulk-store/arxiv-data/pdf/` (organized by YYMM)
- **Staging directory**: `/dev/shm/acid_staging/` (RamFS for speed)
- **Checkpoint file**: `acid_phased_checkpoint.json`
- **Log files**: `tools/arxiv/logs/acid_phased.log`

## Mathematical Framework Implementation

This toolkit directly implements the conveyance equation **C = (W·R·H)/T · Ctx^α**:

### Variable Mappings

- **W (WHAT)**: Jina v4 embeddings (2048-dim) capturing semantic content
- **R (WHERE)**: Direct file access + ArangoDB graph proximity
- **H (WHO)**: Pipeline processing capability, GPU acceleration
- **T (Time)**: Processing latency, batch efficiency metrics
- **Ctx**: Context preserved through late chunking
- **α ≈ 1.5-2.0**: Measured through retrieval performance

## Common Problems & Solutions

### "Pipeline Processing Slow/Stuck"

```bash
# Check what phase is running
grep "PHASE" ../logs/acid_phased.log | tail -5

# Check GPU memory
nvidia-smi

# Check staging directory size
du -sh /dev/shm/acid_staging/

# Restart with fresh GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

### "Out of Memory Errors"

```bash
# Check memory usage
free -h
nvidia-smi

# Reduce workers in config
# Edit configs/acid_pipeline_phased.yaml
# phases.extraction.workers: 32 → 16
# phases.embedding.workers: 8 → 4
```

### "Database Connection Issues"

```bash
# Verify ArangoDB is running
curl http://192.168.1.69:8529/_api/version

# Check collections exist
# Browse to http://192.168.1.69:8529
# Select academy_store database
```

## Performance Optimization

- Use batch size 24 for both extraction and embedding
- Keep staging in RamFS for speed
- Use GPU acceleration for both phases
- Monitor GPU memory between phases
- Process papers alphabetically for cache efficiency