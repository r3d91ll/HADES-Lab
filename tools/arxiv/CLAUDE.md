# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the ArXiv tools directory.

## Quick Start (ArXiv Processing)

### Most Common Commands (Updated 2025-08-23)

```bash
# Run unified pipeline with hysteresis control
cd pipelines/
python arxiv_pipeline_unified_hysteresis.py \
    --config ../../configs/processors/arxiv_unified.yaml \
    --count 1000 \
    --pg-password "$PGPASSWORD" \
    --arango-password "$ARANGO_PASSWORD"

# Monitor processing in real-time
cd ../monitoring/
python monitor_overnight.py --arango-password "$ARANGO_PASSWORD" --refresh 5

# Check database status
cd ../utils/
python check_db_status.py --detailed

# Reset for fresh start
cd ../scripts/
python reset_databases.py --pg-password "$PGPASSWORD" --arango-password "$ARANGO_PASSWORD"

# Check GPU usage
nvidia-smi
```

### Emergency Fixes

```bash
# Pipeline stuck/slow - check queue status
tail -f ../logs/unified_pipeline.log | grep -E "Queue|Hysteresis|Worker"

# Out of memory
python -c "import torch; torch.cuda.empty_cache()"

# Database corruption - nuclear reset
cd ../scripts/
python reset_databases.py --pg-password "$PGPASSWORD" --arango-password "$ARANGO_PASSWORD"

# Worker distribution issues
ps aux | grep python | grep worker | wc -l
```

### What You Need to Know (Critical Points)

1. **Workers process COMPLETE papers** - not individual chunks (late chunking requires full context)
2. **No inter-worker communication** - each worker is completely isolated
3. **8 GPU workers optimal** - 4 per A6000 with fp16 (~7-8GB VRAM each)
4. **Hysteresis thresholds** - 1500/1000 prevents queue overflow
5. **Unified documents** - PDF+LaTeX combined BEFORE embedding
6. **PostgreSQL = metadata** - ArangoDB = embeddings only (never duplicate)

## Project Overview

ArXiv Tools provides infrastructure for processing ArXiv papers through a hybrid PostgreSQL-ArangoDB pipeline, implementing the mathematical framework where **C = (W·R·H)/T · Ctx^α**. Following Actor-Network Theory principles, these tools orchestrate between PostgreSQL (metadata source of truth) and ArangoDB (expensive computations only), optimizing for maximum conveyance while minimizing time T.

## High-Level Architecture

### Hybrid Pipeline Architecture

1. **PostgreSQL Data Lake** (`arxiv_datalake` database)
   - Source of truth for all metadata (2.79M papers)
   - Normalized schema: papers, versions, authors, paper_authors
   - Never duplicated in ArangoDB
   - Experiment window: 375k papers (Dec 2012 - Aug 2016)

2. **ArangoDB Graph Store** (`academy_store` database)
   - Minimal collections (expensive computations only):
     - `arxiv_embeddings`: Jina v4 embeddings with late chunking
     - `arxiv_structures`: Equations, tables, images from PDFs
   - No metadata duplication from PostgreSQL
   - Atomic transactions ensure consistency

3. **Hybrid Pipeline** (`hybrid_pipeline.py`)
   - Config-driven via `configs/processors/arxiv_hybrid.yaml`
   - Queries PostgreSQL for papers to process
   - Extracts text/structures with Docling v2
   - Generates Jina v4 embeddings (2048-dim)
   - Stores only computations in ArangoDB

## Common Development Commands

### Database Setup

```bash
# Create PostgreSQL data lake
export PGPASSWORD='YOUR_PASSWORD'
psql -h localhost -U postgres -d arxiv_datalake < setup_arxiv_datalake.sql

# Import ArXiv metadata (use bulletproof script only!)
python3 import_arxiv_to_postgres_bulletproof.py \
    --password 'YOUR_PASSWORD' \
    --batch-size 5000

# Check import progress
python3 import_arxiv_to_postgres_bulletproof.py \
    --password 'YOUR_PASSWORD' \
    --stats-only
```

### PDF Coverage Analysis

```bash
# Verify PDF coverage in experiment window
python3 verify_pdf_coverage.py --password 'YOUR_PASSWORD' --export

# Check what PDFs are available locally
ls -la /bulk-store/arxiv-data/pdf/ | head -20
```

### Hybrid Pipeline Processing

```bash
# Run the hybrid pipeline (config-driven)
python3 hybrid_pipeline.py \
    --config ../../configs/processors/arxiv_hybrid.yaml \
    --pg-password "$PGPASSWORD" \
    --arango-password "$ARANGO_PASSWORD" \
    --max-papers 100  # For testing

# Run with fresh start (ignore checkpoint)
python3 hybrid_pipeline.py \
    --config ../../configs/processors/arxiv_hybrid.yaml \
    --pg-password "$PGPASSWORD" \
    --arango-password "$ARANGO_PASSWORD" \
    --no-resume

# Monitor running pipeline (from HADES root)
python3 core/utils/monitor_pipeline.py \
    --checkpoint tools/arxiv/hybrid_checkpoint.json \
    --password "$PGPASSWORD" \
    --database Avernus
```

### Database Status Checks

```bash
# Check Jina v4 deployment status
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 verify_jina_v4_deployment.py

# Check database status (from utils directory)
cd ../../utils
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 check_db_status.py --detailed

# Clean ArangoDB for fresh start
cd ../tools/arxiv
python3 clean_arango_for_hybrid.py
```

## Key Components

### Hybrid Pipeline (`hybrid_pipeline.py`)

- **Config-driven**: All settings in `configs/processors/arxiv_hybrid.yaml`
- **Checkpointing**: Automatic resume on failure with atomic checkpoint saves
- **Batch processing**: Configurable batch sizes for GPU efficiency
- **Error recovery**: Tracks failed papers with retry attempts

### PostgreSQL Import (`import_arxiv_to_postgres_bulletproof.py`)

- **Production-ready**: Handles foreign keys correctly with `INSERT ... ON CONFLICT`
- **Savepoint isolation**: Per-record transaction isolation prevents corruption
- **Resumable**: Can restart from any line number after interruption
- **Note**: Never use the non-bulletproof variants (they have FK bugs)

### Critical Implementation Details

**No Duplication Philosophy**: PostgreSQL holds all metadata, ArangoDB only stores what can't be quickly recomputed (embeddings, extracted structures).

**Late Chunking with Jina v4**: Process full documents (up to 32k tokens) BEFORE chunking to preserve context, resulting in dramatically better semantic understanding.

**Minimal Collections**: Only two ArangoDB collections - `arxiv_embeddings` and `arxiv_structures` - following the "expensive computations only" principle.

**Atomic Operations**: All database operations are atomic - either fully succeed or fully rollback, preventing partial states.

## Database Schema

### PostgreSQL Tables (Source of Truth)

- `arxiv_papers`: Core paper metadata (title, abstract, categories)
- `arxiv_versions`: Version history with dates (v1, v2, etc.)
- `arxiv_authors`: Unique author names (normalized)
- `arxiv_paper_authors`: Many-to-many relationships

### ArangoDB Collections (Minimal Design)

- `arxiv_embeddings`: Just embeddings and processing metadata
  - `_key`: Sanitized arxiv_id
  - `abstract_embedding`: 2048-dim vector for abstract
  - `chunk_embeddings`: Array of chunk embeddings with context info
  - `processing_date`: When processed
- `arxiv_structures`: Extracted structures from PDFs
  - `equations`: LaTeX equations from papers
  - `tables`: Structured table data with headers
  - `images`: Image metadata and captions

## Expected Data Volumes

### Experiment Window (Dec 2012 - Aug 2016)

- Total papers: ~376,000
- Papers with PDFs locally: ~185,000 (49.3%)
- ML/AI papers: ~10,000
- Graph-related papers: ~28,000

### Processing Performance

- Import speed: 600-1,200 papers/second
- Total import time: ~40-80 minutes for 2.79M papers
- Database size: ~10-15GB for complete PostgreSQL dataset

## Environment Variables

```bash
# PostgreSQL
export PGPASSWORD='YOUR_POSTGRES_PASSWORD'

# ArangoDB
export ARANGO_PASSWORD='YOUR_ARANGO_PASSWORD'
export ARANGO_HOST='192.168.1.69'  # Default host

# Processing
export USE_GPU=true
export CUDA_VISIBLE_DEVICES=1  # or 0,1 for dual GPU
```

## File Locations

- **ArXiv metadata**: `/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json`
- **PDF repository**: `/bulk-store/arxiv-data/pdf/` (organized by YYMM format)
- **Checkpoint file**: `hybrid_checkpoint.json` (auto-resume on failure)
- **Failed imports log**: `failed_imports.json` (PostgreSQL import issues)

## Integration with HADES

The hybrid pipeline enables:

- Direct PDF processing from `/bulk-store/arxiv-data/pdf/`
- Jina v4 embeddings with late chunking (32k token context)
- Minimal ArangoDB storage (computations only, no duplication)
- Cross-database queries joining PostgreSQL metadata with ArangoDB embeddings
- Foundation for theory-practice bridge discovery across multiple sources

## Mathematical Framework Implementation

This toolkit directly implements the conveyance equation **C = (W·R·H)/T · Ctx^α**:

### Variable Mappings

- **W (WHAT)**: Jina v4 embeddings (2048-dim) capturing semantic content
- **R (WHERE)**: PostgreSQL relations + ArangoDB graph proximity
- **H (WHO)**: Pipeline processing capability, GPU acceleration
- **T (Time)**: Processing latency, batch efficiency metrics
- **Ctx**: Context preserved through late chunking (L), config alignment (I), extracted structures (A), citations (G)
- **α ≈ 1.5-2.0**: Measured through retrieval performance

### Optimization Strategy

The hybrid pipeline optimizes for high C by:

1. **Maximizing W**: High-quality embeddings with late chunking
2. **Maximizing R**: Rich relational structure in dual databases
3. **Maximizing H**: GPU acceleration, efficient batching
4. **Minimizing T**: Checkpoint resume, parallel processing
5. **Maximizing Ctx^α**: Preserving document coherence, extracting actionable content

## Common Problems & Solutions

### "Pipeline Processing Slow/Stuck"

```bash
# Check what's happening
python ../../core/utils/monitor_pipeline.py --checkpoint hybrid_checkpoint.json

# Check GPU memory
nvidia-smi

# Check database status
python ../../utils/check_db_status.py --json

# Restart with fresh GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

### "Import Errors/Foreign Key Violations"

```bash
# ALWAYS use bulletproof script
python import_arxiv_to_postgres_bulletproof.py --password 'PASSWORD' --batch-size 5000

# If still fails, nuclear option
dropdb arxiv_datalake
createdb arxiv_datalake
psql -h localhost -U postgres -d arxiv_datalake < database/setup_arxiv_datalake_complete.sql
```

### "Out of Memory Errors"

```bash
# Check memory usage
free -h
nvidia-smi

# Reduce batch sizes in config
# Edit configs/processors/arxiv_hybrid.yaml
# batch_size: 32 → 16 → 8
```

### "Slow Database Queries"

```sql
-- Refresh materialized views
REFRESH MATERIALIZED VIEW experiment_papers_2012_2016;
ANALYZE arxiv_papers;

-- Check query performance
EXPLAIN ANALYZE SELECT COUNT(*) FROM arxiv_papers;
```

### "Resume Interrupted Processing"

```bash
# Pipeline auto-resumes from checkpoint
python hybrid_pipeline.py --config ../../configs/processors/arxiv_hybrid.yaml

# Force fresh start (ignore checkpoint)
python hybrid_pipeline.py --config ../../configs/processors/arxiv_hybrid.yaml --no-resume

# Resume import from specific line
python import_arxiv_to_postgres_bulletproof.py --password 'PASSWORD' --resume-from 1500000
```

## Troubleshooting

### Check System Status

```bash
# PostgreSQL
systemctl status postgresql@14-main
psql -h localhost -U postgres -d arxiv_datalake -c "SELECT COUNT(*) FROM arxiv_papers;"

# ArangoDB  
curl http://localhost:8529/_api/version
python ../../utils/check_db_status.py --detailed

# GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Performance Optimization

- Use batch size 5000-10000 for imports
- Increase PostgreSQL `shared_buffers` for better performance
- Refresh materialized views after large imports
- Use GPU acceleration for embeddings (CUDA_VISIBLE_DEVICES)
