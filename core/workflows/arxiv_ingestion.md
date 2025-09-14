# ArXiv Ingestion Workflow Documentation

## Overview

The ArXiv ingestion system processes 2.8+ million academic papers from the ArXiv dataset, generating embeddings and storing them in ArangoDB for semantic search and retrieval. The system uses dual-GPU parallel processing to achieve high throughput while maintaining data consistency.

## Architecture

### Workflow Components

1. **`workflow_arxiv_parallel.py`** - Multi-GPU parallel processing workflow
2. **`workflow_arxiv_metadata.py`** - Single-GPU reference implementation
3. **`arxiv_metadata_config.py`** - Configuration management with validation

### Processing Pipeline

```text
JSON Metadata File (2.8M records)
        ↓
[Main Process: Metadata Reader]
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
    ArangoDB
    ├── arxiv_metadata (papers)
    ├── arxiv_abstract_chunks (text segments)
    └── arxiv_abstract_embeddings (vectors)
```

### Key Features

- **Dual-GPU Processing**: Each worker process gets exclusive GPU access
- **Late Chunking**: Preserves context by processing full documents before chunking
- **Atomic Transactions**: All-or-nothing database writes for consistency
- **Progress Tracking**: Real-time monitoring of all processing phases
- **Checkpoint/Resume**: Can safely interrupt and resume processing

## Quick Start

### Prerequisites

```bash
# Set environment variable
export ARANGO_PASSWORD="your-password"

# Verify GPUs available
nvidia-smi

# Check database connection
python dev-utils/check_arxiv_db.py
```

### Production Run (2.8M Records)

```bash
cd /home/todd/olympus/HADES-Lab/core/workflows/

# Full production run with optimal settings
python workflow_arxiv_parallel.py \
    --count 2828998 \
    --batch-size 1000 \
    --embedding-batch-size 48 \
    --workers 2 \
    --drop-collections  # Only on first run
```

### Test Run (10K Records)

```bash
# Quick test with smaller dataset
python workflow_arxiv_parallel.py \
    --count 10000 \
    --batch-size 100 \
    --embedding-batch-size 32 \
    --workers 2 \
    --drop-collections
```

### Resume After Interruption

```bash
# Resume from checkpoint (DO NOT use --drop-collections)
python workflow_arxiv_parallel.py \
    --count 2828998 \
    --batch-size 1000 \
    --embedding-batch-size 48 \
    --workers 2 \
    --resume
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--count` | 100 | Number of records to process |
| `--batch-size` | 100 | Records per batch for queueing |
| `--embedding-batch-size` | 32 | Embeddings per GPU batch |
| `--workers` | 2 | Number of GPU workers (1-4) |
| `--drop-collections` | False | Drop existing collections before starting |
| `--resume` | False | Resume from last checkpoint |

## Configuration

The workflow uses `ArxivMetadataConfig` from `core/tools/arxiv/arxiv_metadata_config.py`:

```python
# Key configuration parameters
metadata_file: Path = "/bulk-store/arxiv-data/metadata/arxiv-kaggle-latest.json"
batch_size: int = 1000  # Main batch size
embedding_batch_size: int = 48  # Per-GPU batch size
num_workers: int = 2  # GPU workers
checkpoint_interval: int = 1000  # Auto-checkpoint frequency

# Database
arango_database: str = "academy_store"
metadata_collection: str = "arxiv_metadata"
chunks_collection: str = "arxiv_abstract_chunks"
embeddings_collection: str = "arxiv_abstract_embeddings"

# Embedder
embedder_model: str = "jinaai/jina-embeddings-v4"
use_fp16: bool = True  # Half precision for memory efficiency
```

## Test/Development System Specifications

### Hardware Configuration

The workflow was developed and tested on the following system:

- **Motherboard**: ASRock TRX50-WS
- **CPU**: AMD Threadripper 7960X (24 cores / 48 threads)
- **RAM**: 256GB ECC DDR5 RDIMM
- **GPUs**: 2× NVIDIA RTX A6000 Ampere (48GB VRAM each)
- **Total GPU Memory**: 96GB VRAM
- **Operating System**: Linux (Ubuntu/Debian based)

This configuration provides:
- Massive parallel processing capability with 24 physical cores
- ECC memory for data integrity during long runs
- 96GB total VRAM for large batch processing
- Professional-grade GPUs with excellent FP16 performance

## Performance Metrics

### Observed Performance (RTX A6000 x2)

| Batch Size | Throughput | GPU Memory | Time for 2.8M |
|------------|------------|------------|---------------|
| 32 | ~35 papers/sec | 60-65% | ~22 hours |
| 48 | ~39 papers/sec | 70-75% | ~20 hours |
| 64 | ~42 papers/sec | 85-90% | ~18 hours |

### Memory Requirements

- **System RAM**: 64GB minimum (128GB recommended)
- **GPU VRAM**: 24GB+ per GPU (RTX 3090/4090 or A6000)
- **Disk Space**: ~50GB for full dataset with embeddings

### Optimization Tips

1. **Batch Size Tuning**
   - Start conservative (32-48) to avoid OOM
   - Monitor GPU memory with `nvidia-smi`
   - Increase gradually if memory allows

2. **Worker Count**
   - Use 1 worker per available GPU
   - More workers don't help if GPUs are saturated

3. **Checkpoint Frequency**
   - Default: every 1000 records
   - Increase for less I/O overhead
   - Decrease for more frequent save points

## Monitoring

### Real-time Progress

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor database growth
watch -n 30 "python dev-utils/check_arxiv_db.py --sample-size 0"

# Check logs (if using metadata_processor.py wrapper)
tail -f /tmp/arxiv_metadata_*.log
```

### Database Verification

```bash
# Full verification with samples
python dev-utils/check_arxiv_db.py --detailed --sample-size 5

# Quick count check
python dev-utils/check_arxiv_db.py --sample-size 0
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--embedding-batch-size`
   - Current safe values: 32-48 for 24GB GPUs

2. **Progress Tracker Error**
   - Fixed in latest version
   - Old error "Unknown step: metadata_loading" is harmless

3. **Database Connection Failed**
   - Ensure ARANGO_PASSWORD is set
   - Check ArangoDB is running
   - Verify Unix socket at `/tmp/arangodb.sock`

4. **Slow Processing**
   - Check GPU utilization (should be 90%+)
   - Verify both GPUs are being used
   - Consider increasing batch sizes if memory allows

### Recovery Procedures

```bash
# If processing fails mid-run:

# 1. Check how many records were processed
python dev-utils/check_arxiv_db.py --sample-size 0

# 2. Check checkpoint status
cat /tmp/arxiv_metadata_checkpoint.json | python -m json.tool | head -20

# 3. Resume from checkpoint
python workflow_arxiv_parallel.py \
    --count 2828998 \
    --batch-size 1000 \
    --embedding-batch-size 48 \
    --workers 2 \
    --resume
```

## Theory Connection (Conveyance Framework)

This workflow implements the Conveyance equation: **C = (W·R·H/T)·Ctx^α**

- **W (WHAT)**: Semantic content via Jina v4 embeddings (2048 dimensions)
- **R (WHERE)**: Graph relationships in ArangoDB collections
- **H (WHO)**: Dual-GPU parallel processing capability
- **T (TIME)**: Optimized for ~40 papers/second throughput
- **Ctx**: Preserved through mandatory late chunking (α ≈ 1.5-2.0)

Late chunking is critical: processing full documents before chunking maintains context awareness (high Ctx), preventing zero-propagation in the Conveyance equation.

## Development Notes

### Adding Custom Processing

To extend the workflow with additional processing steps:

1. Add new collection in config
2. Update `_store_batch()` method to include new data
3. Add progress tracking step in `_initialize_components()`
4. Update transaction scope to include new collection

### Testing Changes

```bash
# Test with small dataset first
python workflow_arxiv_parallel.py --count 100 --drop-collections

# Verify data structure
python dev-utils/check_arxiv_db.py --detailed

# Scale up gradually: 100 → 1,000 → 10,000 → full dataset
```

## Related Documentation

- `/home/todd/olympus/HADES-Lab/core/tools/arxiv/README.md` - ArXiv tools overview
- `/home/todd/olympus/HADES-Lab/CLAUDE.md` - System architecture and theory
- `/home/todd/olympus/HADES-Lab/dev-utils/check_arxiv_db.py` - Database verification tool

## Support

For issues or questions:

1. Check logs in `/tmp/arxiv_metadata_*.log`
2. Verify GPU status with `nvidia-smi`
3. Review checkpoint file for recovery options
4. Consult CLAUDE.md for theoretical framework

---

*Last updated: 2025-01-14*
*Version: 1.0.0 - Initial dual-GPU implementation with progress tracking*
