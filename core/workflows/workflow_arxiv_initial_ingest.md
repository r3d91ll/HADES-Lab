# ArXiv Initial Ingest Workflow Documentation

## Overview

The ArXiv initial ingest workflow processes 2.8+ million academic papers from the ArXiv dataset, generating embeddings and storing them in ArangoDB for semantic search and retrieval. This is a one-time batch job optimized for simplicity and performance.

## Key Design Decisions

### No Complex Recovery
This is a **one-time batch job** on static data. The workflow prioritizes simplicity:
- No checkpointing or complex recovery mechanisms
- If interrupted, simply restart with `--drop-collections`
- Total runtime: ~15-16 hours (down from 24 hours in earlier versions)

### Processing Strategy
- **Size-sorted processing**: Papers processed smallest to largest for optimal GPU batching
- **Late chunking**: Full document context preserved before chunking (mandatory per Conveyance Framework)
- **PHP bridge**: Database operations use PHP to bypass python-arango limitations

## Architecture

### Processing Pipeline

```text
JSON Metadata File (2.8M records on NVME)
        ↓
[Main Process: Metadata Loader]
        ↓
    Input Queue
    ↙        ↘
[Worker 0]  [Worker 1]  (GPU-isolated processes)
[GPU 0]     [GPU 1]
    ↘        ↙
   Output Queue
        ↓
[Storage Thread: Database Writer]
        ↓
    ArangoDB (via PHP bridge)
    ├── arxiv_metadata (papers)
    ├── arxiv_abstract_chunks (text segments)
    └── arxiv_abstract_embeddings (vectors)
```

### Key Components

1. **`workflow_arxiv_initial_ingest.py`** - Main workflow (formerly workflow_arxiv_sorted_simple.py)
2. **`core/database/arango/php_unix_bridge.php`** - PHP bridge for database operations
3. **`core/embedders/embedders_jina.py`** - Jina v4 embedder with proper late chunking

## Quick Start

### Prerequisites

```bash
# Set environment variable
export ARANGO_PASSWORD="your-password"

# Verify GPUs available
nvidia-smi

# Check PHP bridge works
php core/database/arango/php_unix_bridge.php test
```

### Production Run (Fresh Start)

```bash
# Full production run with clean start
python core/workflows/workflow_arxiv_initial_ingest.py \
    --drop-collections \
    --workers 2 \
    --embedding-batch-size 50 \
    --chunk-size-tokens 500 \
    --chunk-overlap-tokens 200 \
    --batch-size 100
```

### Test Run

```bash
# Quick test with 1000 records
python core/workflows/workflow_arxiv_initial_ingest.py \
    --drop-collections \
    --count 1000 \
    --workers 2 \
    --embedding-batch-size 32 \
    --batch-size 100
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--count` | all | Number of records to process |
| `--batch-size` | 100 | Records per I/O batch |
| `--embedding-batch-size` | 48 | Embeddings per GPU batch |
| `--workers` | 2 | Number of GPU workers |
| `--drop-collections` | False | Drop and recreate collections (uses PHP bridge) |
| `--chunk-size-tokens` | 500 | Tokens per chunk |
| `--chunk-overlap-tokens` | 200 | Overlap between chunks |

## Configuration

### Data Location
- **Metadata file**: `data/arxiv-kaggle-latest.json` (NVME for fast loading)
- **Original location**: `/bulk-store/arxiv-data/metadata/` (slower HDD)

### Database
- **Database**: `arxiv_repository`
- **Collections**:
  - `arxiv_metadata` - Paper metadata
  - `arxiv_abstract_embeddings` - Vector embeddings
  - `arxiv_abstract_chunks` - Text chunks

### Embedder
- **Model**: Jina v4 (jinaai/jina-embeddings-v4)
- **Dimensions**: 2048
- **Context window**: 32k tokens
- **Late chunking**: Mandatory (preserves context)

## Performance Metrics

### Current Performance (RTX A6000 x2)

| Metric | Value |
|--------|-------|
| Processing rate | ~47 papers/second |
| Total runtime | ~15-16 hours for 2.8M papers |
| GPU memory usage | 7-8GB per worker |
| Batch efficiency | 95%+ GPU utilization |

### Optimization History
- **Original**: 24 hours
- **After late chunking fix**: 15-16 hours (37.5% improvement)
- **Size-sorted batching**: Consistent throughput

## Monitoring

### Real-time Progress

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor workflow output (shows progress every 30 seconds)
# The workflow prints progress directly to console
```

### Database Verification

```bash
# Check collection counts via PHP
php core/database/arango/php_unix_bridge.php check_collections

# Get database statistics
php core/database/arango/php_unix_bridge.php stats
```

## PHP Bridge

The workflow uses PHP for database operations because python-arango cannot use Unix sockets:

```bash
# Test connection
php core/database/arango/php_unix_bridge.php test

# Check collections
php core/database/arango/php_unix_bridge.php check_collections

# Drop collections (for fresh start)
php core/database/arango/php_unix_bridge.php drop_collections

# Create collections
php core/database/arango/php_unix_bridge.php create_collections
```

Currently using TCP (`tcp://localhost:8529`) until Unix socket permissions are resolved.

## Troubleshooting

### Common Issues

1. **Collection creation fails**
   - Use PHP bridge: `php core/database/arango/php_unix_bridge.php create_collections`
   - Python-arango has issues with collection visibility

2. **CUDA Out of Memory**
   - Reduce `--embedding-batch-size` (safe: 32-48 for 24GB GPUs)

3. **Slow metadata loading**
   - Ensure using NVME copy: `data/arxiv-kaggle-latest.json`
   - Not the HDD version in `/bulk-store/`

4. **Process interrupted**
   - Just restart with `--drop-collections`
   - No complex recovery needed for one-time job

## Theory Connection (Conveyance Framework)

**C = (W·R·H/T)·Ctx^α**

- **W**: Jina v4 embeddings (2048 dimensions)
- **R**: ArangoDB graph structure
- **H**: Dual-GPU processing
- **T**: ~47 papers/second (reduced from 24 to 15 hours)
- **Ctx**: Preserved via mandatory late chunking
- **α**: 1.5-2.0 (context amplification)

### Critical: Late Chunking
- Process full document first → Then chunk with context
- Early chunking (chunk → embed) violates framework (Ctx→0)
- This fix alone improved performance significantly

## File Organization

### Active Files
- `core/workflows/workflow_arxiv_initial_ingest.py` - Main workflow
- `core/database/arango/php_unix_bridge.php` - PHP database bridge
- `core/embedders/embedders_jina.py` - Embedder with late chunking

### Archived to Acheron
- `workflow_arxiv_sorted.py` - Deprecated complex version
- `workflow_arxiv_parallel.py` - Older parallel implementation
- `workflow_arxiv_metadata.py` - Metadata-only workflow
- `core/database/arango/unix_client.py` - Failed Python Unix socket attempt
- `core/database/arango/arango_client.py` - HTTP connection pooling

## Support

For issues:
1. Check GPU status: `nvidia-smi`
2. Verify PHP bridge: `php core/database/arango/php_unix_bridge.php test`
3. For interrupted runs: Restart with `--drop-collections`

---

*Last updated: 2025-01-21*
*Version: 2.0.0 - Simplified one-time ingest with PHP bridge*