# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the ArXiv tools directory.

## Quick Start (ArXiv Processing)

### Most Common Commands

```bash
# Run ACID pipeline with phase separation
cd pipelines/
python arxiv_pipeline.py \
    --config ../configs/acid_pipeline_phased.yaml \
    --count 1000 \
    --arango-password "$ARANGO_PASSWORD"

# Monitor processing in real-time
tail -f ../logs/acid_phased.log

# Check database status
cd ../utils/
python check_db_status.py --detailed

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

## Project Overview

ArXiv Tools provides infrastructure for processing ArXiv papers directly from the filesystem through ArangoDB, implementing the mathematical framework where **C = (W·R·H)/T · Ctx^α**. Following Actor-Network Theory principles, these tools process PDFs directly without intermediate databases, optimizing for maximum conveyance while minimizing time T.

## High-Level Architecture

### Streamlined Pipeline Architecture

1. **Local PDF Repository** (`/bulk-store/arxiv-data/pdf/`)
   - Direct access to ArXiv papers
   - Organized by YYMM/arxiv_id.pdf structure
   - No database dependencies for processing
   - Optional SQLite cache for indexing

2. **ArangoDB Graph Store** (`academy_store` database)
   - Collections for all extracted data:
     - `arxiv_embeddings`: Jina v4 embeddings with late chunking
     - `arxiv_chunks`: Text chunks with context windows
     - `arxiv_papers`: Paper metadata and processing status
     - `arxiv_structures`: Equations, tables, images from PDFs
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