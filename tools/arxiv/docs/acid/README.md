# ACID Pipeline for ArXiv Processing

**E3 Architecture Implementation: Extract → Encode → Embed**

A phase-separated, ACID-compliant pipeline achieving **5.9 papers/minute** (2-3x target performance).

## 🎯 Performance Achievements

- **100 Paper Test**: 100% success rate, 5.9 papers/min end-to-end
- **Extraction Phase**: 10.0 papers/min (Docling v2)
- **Embedding Phase**: 15.0 papers/min (Jina v4 with late chunking)
- **1000 Paper Test**: Currently running (34% complete at time of writing)

## 🚀 Production Files

### Core Pipeline
- **`acid_pipeline_phased.py`** - Main production pipeline with E3 architecture
- **`configs/acid_pipeline_phased.yaml`** - Production config (36 extraction workers, batch_size 24)
- **`run_phased_pipeline.sh`** - Production run script

### Supporting Infrastructure
- **`arango_acid_processor.py`** - ArangoDB ACID transaction handler
- **`on_demand_processor.py`** - On-demand paper processing
- **`process_papers_utility.py`** - Paper processing utilities
- **`import_arxiv_to_sqlite.py`** - SQLite cache importer
- **`setup_sqlite_cache.py`** - SQLite cache setup
- **`update_arxiv_daily.py`** - Daily update cron job handler

## Quick Start

```bash
# 1. Quick test with 1000 papers
python setup_and_test.py --quick

# 2. Full setup (2.7M papers - takes 30-60 minutes)
python setup_and_test.py

# 3. Setup without tests (if you want to test manually later)
python setup_and_test.py --skip-tests
```

## Architecture Overview

This pipeline simplifies the previous dual-database (PostgreSQL + ArangoDB) architecture into a single ArangoDB instance with ACID guarantees.

### Key Components

1. **SQLite Cache** (`/bulk-store/arxiv-cache.db`)
   - Lightweight metadata storage
   - Paper tracking and status
   - Fast lookups and queries

2. **ArangoDB** (`academy_store` database)
   - All content storage (embeddings, chunks, structures)
   - Stream transactions for ACID guarantees
   - TTL-based lock management

3. **Processing Pattern**
   - **RESERVE**: Acquire lock (prevents conflicts)
   - **COMPUTE**: Extract content (outside transaction)
   - **COMMIT**: Store atomically (fast transaction)
   - **RELEASE**: Release lock (enables next worker)

## Manual Setup Steps

### 1. Import ArXiv Metadata to SQLite

```bash
# Import full dataset (2.7M papers)
python import_arxiv_to_sqlite.py

# Or import with limit for testing
python import_arxiv_to_sqlite.py --limit 10000

# Check statistics
python import_arxiv_to_sqlite.py --stats-only

# Search papers
python import_arxiv_to_sqlite.py --search "transformer attention"
```

### 2. Test ACID Guarantees

```bash
# Run comprehensive ACID tests
python test_acid_pipeline.py

# Tests include:
# - Atomicity: Transaction rollback on failure
# - Consistency: Constraint maintenance
# - Isolation: Concurrent access control
# - Durability: Data persistence
```

### 3. Integration Testing

```bash
# Test with real ArXiv papers
python test_integration.py

# Processes test papers:
# - 1212.1432
# - 1301.0007
# - 1506.01094
```

### 4. Daily Updates

```bash
# Run manual update (last 7 days)
python update_arxiv_daily.py --days-back 7

# Update specific categories
python update_arxiv_daily.py --categories cs.AI cs.LG math.CO

# Add to crontab for automatic updates (runs at 2 AM daily)
crontab -e
# Add: 0 2 * * * /usr/bin/python3 /path/to/update_arxiv_daily.py
```

## On-Demand Processing

Process specific papers when needed:

```bash
# Python API
from on_demand_processor import OnDemandProcessor

processor = OnDemandProcessor(config)
results = processor.process_papers(['2310.08560', '2310.06825'])

# Results show status for each paper:
# - 'processed': Successfully processed
# - 'already_processed': Previously completed
# - 'not_found': PDF not available
# - 'failed': Processing error
```

## Worker Pool for Batch Processing

Process multiple papers in parallel:

```bash
# Python API
from worker_pool import ArangoWorkerPool

pool = ArangoWorkerPool(num_workers=4, config=config)
batch_result = pool.process_batch(paper_paths, timeout_per_paper=60)

print(f"Processed: {batch_result.successful}/{batch_result.total}")
print(f"Rate: {batch_result.papers_per_minute:.1f} papers/minute")
```

## Monitoring

Real-time monitoring of the system:

```bash
# Python API
from monitoring import ArangoMonitor

monitor = ArangoMonitor(config)

# Get metrics
metrics = monitor.get_overall_metrics()
print(f"Total papers: {metrics.total_papers}")
print(f"Processed: {metrics.processed_papers}")
print(f"Active locks: {metrics.active_locks}")

# Check health
health = monitor.check_health()
print(f"Status: {health['status']}")  # healthy, warning, or critical

# Clean up expired locks
monitor.cleanup_expired_locks()
```

## Configuration

Create a config dictionary:

```python
config = {
    'cache_root': '/bulk-store/arxiv-data',
    'sqlite_db': '/bulk-store/arxiv-cache.db',
    'arango': {
        'host': ['http://192.168.1.69:8529'],
        'database': 'academy_store',
        'username': 'root',
        'password': os.environ.get('ARANGO_PASSWORD')
    },
    'embedder_config': {
        'device': 'cuda',  # or 'cpu'
        'use_fp16': True,
        'chunk_size_tokens': 1000,
        'chunk_overlap_tokens': 200
    },
    'extractor_config': {
        'use_ocr': False,
        'extract_tables': True,
        'use_fallback': False
    }
}
```

## Database Schema

### SQLite Tables

- `papers`: Main paper tracking
  - `arxiv_id`: Primary key
  - `title`, `abstract`: Metadata
  - `pdf_status`: not_checked, found, downloaded
  - `processing_status`: pending, processing, processed, failed
  - `in_arango`: Boolean flag

- `import_stats`: Import tracking
- `update_log`: Daily update history

### ArangoDB Collections

- `papers`: Paper metadata and status
- `chunks`: Text chunks with embeddings
- `embeddings`: Additional embeddings
- `equations`: Extracted equations
- `tables`: Extracted tables
- `images`: Extracted images
- `locks`: Processing locks (TTL-enabled)

## Migration from Old Architecture

If migrating from the dual-database setup:

```bash
# Use migration strategy
from migration_strategy import MigrationStrategy

strategy = MigrationStrategy(old_config, new_config)

# Phase 1: Setup new collections
strategy.phase1_setup()

# Phase 2: Enable dual writes
strategy.phase2_dual_write()

# Phase 3: Switch reads to new system
strategy.phase3_switch_reads()

# Phase 4: Cleanup old system
strategy.phase4_cleanup()
```

## Performance Expectations

- **Import**: 600-1,200 papers/second
- **Processing**: 2-5 papers/minute (with GPU)
- **Lock acquisition**: <100ms
- **Transaction commit**: <500ms
- **Worker pool**: 10-20 papers/minute (4 workers)

## Troubleshooting

### "Database locked" errors
- Check active locks: `monitor.get_lock_status()`
- Clean expired: `monitor.cleanup_expired_locks()`

### "Transaction failed" 
- Check ArangoDB logs: `/var/log/arangodb3/`
- Verify stream transactions enabled

### "Out of memory"
- Reduce batch sizes in config
- Clear GPU cache: `torch.cuda.empty_cache()`

### "Slow processing"
- Check GPU utilization: `nvidia-smi`
- Verify worker distribution
- Monitor lock contention

## Environment Variables

```bash
export ARANGO_PASSWORD="your_password"
export CUDA_VISIBLE_DEVICES=0,1  # For GPU processing
```

## Testing Checklist

- [ ] SQLite database created and populated
- [ ] ACID tests pass (atomicity, consistency, isolation, durability)
- [ ] Integration tests with real papers succeed
- [ ] Daily updater fetches new papers
- [ ] Worker pool processes batches correctly
- [ ] Monitoring shows healthy status
- [ ] Lock cleanup works properly
- [ ] On-demand processing functions

## Next Steps

1. Run `setup_and_test.py` to validate everything works
2. Process a batch of papers to verify performance
3. Set up cron job for daily updates
4. Monitor system health during production use
5. Scale workers based on load requirements

## 📁 Directory Organization

### Production Code
```
acid/
├── acid_pipeline_phased.py      # Main E3 pipeline
├── configs/
│   └── acid_pipeline_phased.yaml # Production configuration
├── run_phased_pipeline.sh       # Production run script
└── Supporting files...           # See list above
```

### Test Infrastructure (Has Dependencies - Keep)
```
acid/
├── test_integration.py          # Integration tests
├── test_acid_pipeline.py        # ACID guarantee tests
├── setup_and_test.py            # Test orchestration
├── monitoring.py                # Performance monitoring
├── worker_pool.py               # Worker pool management
└── migration_strategy.py        # Migration utilities
```

### Archived Files
```
archive/
├── test_scripts/                # One-time test scripts
├── old_pipelines/               # Previous pipeline versions
├── monitoring/                  # Old monitoring scripts
├── utilities/                   # One-time utilities
└── logs/                        # Old processing logs
```

### Documentation
```
acid/
├── E3_ARCHITECTURE.md           # E3 pattern explanation
├── PERFORMANCE_MILESTONE.md    # Performance achievements
├── FIXES_SUMMARY.md            # Bug fixes documentation
└── IMPLEMENTATION_COMPLETE.md   # Implementation notes
```