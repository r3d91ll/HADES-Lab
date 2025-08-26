# ArXiv Tools - ACID-Compliant High-Performance Pipeline

Advanced pipeline achieving **11.3 papers/minute** with 100% ACID compliance and zero failures across 1000+ papers.

## 🚀 Quick Start

```bash
# Set environment variables
export PGPASSWORD="your_password"
export ARANGO_PASSWORD="your_password"
export CUDA_VISIBLE_DEVICES="0,1"  # Use both GPUs

# Run the ACID pipeline (11.3 papers/minute)
cd pipelines/
python arxiv_pipeline.py --config ../configs/acid_pipeline_phased.yaml

# Monitor progress
cd ../monitoring/
python acid_monitoring.py
```

## 📊 Performance Metrics

- **Processing Rate**: 11.3 papers/minute (validated on 1000 papers)
- **Success Rate**: 100% (zero failures)
- **Extraction Phase**: 11.3 papers/minute (36 CPU workers)
- **Embedding Phase**: Queued separately (8 GPU workers)
- **ACID Compliant**: Full atomicity, consistency, isolation, durability
- **Test Results**: 1000 documents in 88.5 minutes

## 📁 Directory Structure (Consolidated 2025-08-25)

```
tools/arxiv/
├── pipelines/           # Main processing pipelines
│   ├── arxiv_pipeline.py              # ACID pipeline (11.3 papers/min)
│   ├── arango_acid_processor.py       # ACID transaction handler
│   ├── worker_pool.py                 # Parallel worker management
│   └── hybrid_search.py               # Cross-database search
├── monitoring/          # Real-time monitoring tools
│   ├── monitor_overnight.py                  # Production monitor with checkpoint awareness
│   ├── monitor_unified_embeddings.py         # Database status monitor
│   ├── monitor_pipeline_live.py              # Live queue monitoring
│   ├── pipeline_status_reporter.py           # JSON status reporter
│   └── MONITOR_README.md                     # Monitoring documentation
├── scripts/             # Utility and management scripts
│   ├── reset_databases.py                    # Clean slate for new runs
│   ├── run_overnight_unified.sh              # Overnight batch runner
│   ├── run_unified_hysteresis.sh             # Standard runner
│   └── conveyance_measurement.py             # Theory validation
├── database/            # Database setup and schemas
├── utils/               # Verification and checking tools
│   ├── check_db_status.py                    # Database health checks
│   ├── verify_database_sync.py               # Sync verification
│   └── daily_arxiv_update.py                 # Daily metadata updates
├── logs/                # Processing logs
├── checkpoints/         # Resume points
├── README.md           # This file
├── CLAUDE.md           # Developer guidance
└── __init__.py         # Package initialization
```

## 🏗️ Architecture Overview

### ACID Pipeline Architecture  

The ACID pipeline (`arxiv_pipeline.py`) implements a sophisticated multi-stage processing system with full transactional guarantees:

```
PostgreSQL (Avernus DB)          ArangoDB (academy_store)
    │                                    │
    ├─[Metadata Query]                   │
    │                                    │
    ▼                                    │
Extraction Workers (36 CPU)             │
    │                                    │
    ├─[PDF + LaTeX Processing]          │
    │  (Docling v2)                     │
    ▼                                    │
/dev/shm/acid_staging/                  │
    │  (JSON intermediates)             │
    ▼                                    │
Embedding Workers (8 GPU)                │
    │                                    │
    ├─[Jina v4 Embeddings]              │
    │  (Late Chunking)                  │
    ▼                                    ▼
ACID Transactions ───────────────► Store Atomically
```

### Key Features

1. **ACID Compliance**: Full atomicity, consistency, isolation, durability
2. **Phase Separation**: Extract (CPU) and Embed (GPU) phases run independently
3. **RamFS Staging**: Uses `/dev/shm/acid_staging/` for fast inter-phase communication
4. **Late Chunking**: Preserves 32k token context for superior semantic understanding
5. **Stream Transactions**: ArangoDB's `begin_transaction()` for multi-collection atomicity
6. **Zero Failures**: 100% success rate on 1000+ paper tests

## 📊 Performance Metrics

### Current Performance (ACID Pipeline)

- **Processing Rate**: 11.3 papers/minute (678/hour)
- **Extraction Phase**: 36 CPU workers processing PDFs
- **Embedding Phase**: 8 GPU workers (4 per A6000)
- **Memory Usage**: ~7-8GB VRAM per worker with fp16
- **Success Rate**: 100% (zero failures in 1000 paper test)
- **Source Coverage**: ~84% papers have both PDF+LaTeX

### Projections

| Papers | Time Estimate | 
|--------|--------------|
| 100    | ~9 minutes   |
| 1,000  | ~1.5 hours   |
| 10,000 | ~15 hours    |
| 100,000| ~6 days      |
| 375,000| ~23 days     |

## 🎯 Worker Architecture

### Worker Isolation Model

Each GPU worker operates in complete isolation:

1. **Pulls complete paper** from embedding queue
2. **Processes all chunks** with late chunking (maintains context)
3. **Generates all embeddings** for that paper
4. **Sends to write queue** as atomic unit
5. **Repeats** with next paper

**No inter-worker communication**: Workers don't share state, memory, or embeddings. This ensures:
- Deterministic results (same paper always produces same embeddings)
- Simple failure recovery (failed worker doesn't affect others)
- Easy scaling (add/remove workers without coordination)

## Essential Commands

### Prerequisites

```bash
# Set environment variables in ~/.bashrc
export PGPASSWORD='your_postgres_password'
export ARANGO_PASSWORD='your_arango_password'
export CUDA_VISIBLE_DEVICES=1  # or 0,1 for dual GPU

# Source the environment
source ~/.bashrc
```

### Essential Commands

#### 🔥 **Current Production Pipeline** (ACID)

```bash
# Quick test run - Process 100 papers
cd pipelines/
python arxiv_pipeline.py \
    --config ../configs/acid_pipeline_phased.yaml \
    --count 100 \
    --arango-password "$ARANGO_PASSWORD"

# Full run with script - Process 1000 papers with monitoring
cd ../scripts/
./run_acid_pipeline.sh

# Monitor running pipeline (separate terminal)
cd ../monitoring/
python acid_monitoring.py

# Resume embedding phase if extraction completed
cd ../scripts/
python run_embedding_phase_only.py \
    --config ../configs/acid_pipeline_phased.yaml \
    --arango-password "$ARANGO_PASSWORD"
```

#### ⚙️ **Configuration Overrides**

```bash
# Use specific config file
python3 arxiv_pipeline.py \
    --config ../../configs/processors/arxiv_hybrid_v2.yaml \
    --count 100

# Override worker counts for testing
python3 arxiv_pipeline.py \
    --count 10 \
    --extraction-workers 4 \
    --gpu-workers 1 \
    --batch-size 10
```

#### 📊 **Monitoring and Status**

```bash
# Check database status
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 utils/check_db_status.py --detailed

# Verify database integrity/sync
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 utils/verify_database_sync.py
# Search across dual embeddings
python3 hybrid_search.py "transformer attention mechanism"
```

## Processing Pipeline Features

## 🔧 Configuration

### Primary Configuration (`configs/processors/arxiv_unified.yaml`)

```yaml
multiprocessing:
  extraction_workers: 24      # CPU workers for text extraction
  gpu_workers: 8              # Total GPU workers (4 per GPU)
  gpu_devices: [0,1]          # Available GPUs
  write_workers: 2            # Database write workers
  
  # Queue sizes with hysteresis control
  extraction_queue_size: 1000
  embedding_queue_size: 1000  
  write_queue_size: 1000

processor:
  embedder_config: ../embedder.yaml  # Jina v4 configuration
  process_latex_with_pdf: true       # Combine sources
  use_docling: true                   # Docling v2 for PDFs
```

### Embedder Configuration (`configs/embedder.yaml`)

```yaml
embedder:
  model_name: jinaai/jina-embeddings-v4
  device: cuda
  use_fp16: true
  max_context_window: 32768
  chunk_size_tokens: 256
  chunk_overlap_tokens: 64
  late_chunking_enabled: true
  embedding_dimensions: 2048
```

### 🔧 **Advanced Features**

- **Dual Embedding Architecture**: Separate PDF and LaTeX embeddings with context enhancement
- **Multiprocessing Pipeline**: Configurable extraction workers (CPU) and GPU workers
- **Count Enforcement**: Processes exactly the specified number (fixed in recent update)
- **Batch Processing**: Stable 1000-paper batches for GPU efficiency
- **Graceful Shutdown**: Clean worker termination with Ctrl+C
- **Error Recovery**: Comprehensive error handling with detailed logging

### 📈 **Performance Optimization**

ACID pipeline achieves **678 papers/hour** (11.3 papers/minute):

```bash
# High-performance configuration
--extraction-workers 36    # CPU workers for PDF extraction
--gpu-workers 8           # 4 per GPU (RTX A6000)
--batch-size 1000         # Large batches for GPU efficiency
```

**Projected Performance**:
- 678 papers/hour × 72 hours = **48,800 papers** in 3-day run
- Target 76k papers would require ~5.5 days at current rate

## Dual Embedding Architecture

### 🔄 **Processing Flow**

1. **PostgreSQL Query**: Get unprocessed papers from experiment window
2. **Parallel Extraction**: 16 workers extract from PDFs and LaTeX sources
3. **Context Enhancement**: LaTeX equations/structures injected into PDF embeddings
4. **Dual GPU Processing**: Generate embeddings for both sources
5. **Storage**: Route to correct ArangoDB collections
6. **Tracking**: Update PostgreSQL with processing status

### 🗄️ **Database Collections**

```javascript
// ArangoDB Collections (Unified Design)
arxiv_unified_embeddings {  // Combined PDF+LaTeX embeddings
  _key: "sanitized_arxiv_id",
  source: "unified",        // Combined from both sources
  pdf_path: "path/to/pdf",
  latex_source: "raw_latex_text",
  chunk_embeddings: [...],  // Unified chunks with context
  model: "jinaai/jina-embeddings-v4",
  processing_date: "2025-08-22T10:00:00Z"
}

arxiv_structures {          // Extracted structures
  _key: "sanitized_arxiv_id",
  equations: [...],         // LaTeX equations
  tables: [...],           // Structured table data  
  citations: [...],        // Citation extraction
  images: [...],           // Image metadata
  extraction_source: "pdf" // or "latex"
}
```

### 🔍 **Hybrid Search**

```bash
# Search across both embedding types
python3 hybrid_search.py "attention mechanism transformer"

# Results include:
# - PDF-based matches (with LaTeX context enhancement)
# - LaTeX-based matches (pure mathematical content)
# - Relevance scores for both sources
```

## Database Setup and Import

### 📋 **PostgreSQL Schema Setup**

```bash
# Create database and schema
export PGPASSWORD='your_password'
createdb -h localhost -U postgres Avernus

# Apply schema
psql -h localhost -U postgres -d Avernus < database/setup_arxiv_datalake.sql

# Add dual embedding tracking columns
psql -h localhost -U postgres -d Avernus < database/add_dual_embedding_tracking.sql
```

### 📥 **ArXiv Metadata Import**

```bash
# Import from ArXiv snapshot (production method)
python3 database/build_arxiv_datalake_v2.py \
    --password "$PGPASSWORD" \
    --batch-size 5000 \
    --metadata-file /bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json

# Check import progress
python3 database/database_tools.py \
    --password "$PGPASSWORD" \
    --stats-only

# Verify import integrity  
python3 utils/verify_database_sync.py \
    --password "$PGPASSWORD"
```

### 🔄 **Daily Updates**

Keep metadata current with automated daily updates:

```bash
# Manual update from yesterday
python3 utils/daily_arxiv_update.py \
    --password "$PGPASSWORD"

# Update from specific date
python3 utils/daily_arxiv_update.py \
    --password "$PGPASSWORD" \
    --start-date 2024-01-01

# Setup automated cron job
echo "0 2 * * * cd /home/todd/olympus/HADES/tools/arxiv && PGPASSWORD=$PGPASSWORD python3 utils/daily_arxiv_update.py" | crontab -
```

## Configuration

### 📝 **Primary Config**: `configs/processors/arxiv_hybrid_v2.yaml`

```yaml
# PostgreSQL Configuration (Source of Truth)
postgresql:
  host: localhost
  database: Avernus  # Production database
  user: postgres

# ArangoDB Configuration (Expensive Computations Only)  
arangodb:
  host: 192.168.1.69
  database: academy_store
  collections:
    pdf_embeddings: arxiv_pdf_embeddings
    latex_embeddings: arxiv_latex_embeddings
    structures: arxiv_structures

# Multiprocessing Configuration
multiprocessing:
  extraction_workers: 16      # CPU workers
  gpu_workers: 2             # GPU workers
  extraction_queue_size: 2000
  embedding_queue_size: 500

# Experiment Window (Dec 2012 - Aug 2016)
experiment:
  start_date: "2012-12-01"
  end_date: "2016-08-31"
```

### 🎛️ **CLI Overrides**

All config values can be overridden via command line:

```bash
--extraction-workers 16      # Override worker count
--gpu-workers 2             # Override GPU worker count  
--batch-size 1000           # Override batch size
--config /path/to/config    # Use different config file
```

## Monitoring & Debugging

### 📊 **Real-time Monitoring**

```bash
# Database status with detailed metrics
ARANGO_PASSWORD=$ARANGO_PASSWORD python3 utils/check_db_status.py --detailed --json

# GPU utilization monitoring
nvidia-smi -l 1
# System resource monitoring (during processing)
htop
```

### 📝 **Log Analysis**

```bash
# Pipeline logs (auto-created during processing)
tail -f logs/hybrid_pipeline_v2.log

# Worker-specific logs
tail -f logs/extraction_workers.log
tail -f logs/embedding_workers.log
tail -f logs/resource_monitor.log

# Daily update logs
tail -f logs/daily_arxiv_update.log
```

### 🔧 **Common Issues**

**Count Limit Not Enforced** ✅ FIXED
- Issue: Processing more papers than requested
- Solution: Updated `arxiv_pipeline.py` with proper count enforcement

**Queue Bottlenecks** ✅ FIXED  
- Issue: Limited to ~300 papers due to small queues
- Solution: Increased queue sizes to 2000/500 in config

**GPU Memory Fragmentation** ✅ IMPROVED
- Issue: 16+ processes on single GPU causing inefficiency
- Solution: Optimized worker allocation and batch processing

**Worker Shutdown Issues** ✅ FIXED
- Issue: Workers not terminating cleanly
- Solution: Improved poison pill handling and graceful shutdown

## Performance Metrics

### 🚀 **Current Performance** (ACID Pipeline)

- **Processing Rate**: 678 papers/hour (11.3 papers/minute)
- **Success Rate**: 100% (0 failures in 1000-paper test)
- **Worker Configuration**: 36 extraction + 8 GPU workers
- **Batch Size**: 1000 papers for optimal GPU utilization

### 📈 **Scaling Projections**

```
3-day run (72 hours):     ~48,800 papers
1-week run (168 hours):   ~113,900 papers  
Full experiment window:   ~23 days for 375,000 papers
```

### 🎯 **Target Performance**

- **Experiment Window**: 375,000 papers available
- **Current Timeline**: ~23 days for complete processing
- **Achievement**: 11.3 papers/minute with 100% success rate

## Mathematical Framework Implementation

This toolkit implements **C = (W·R·H)/T · Ctx^α**:

- **W (WHAT)**: Dual Jina v4 embeddings (2048-dim) with late chunking
- **R (WHERE)**: PostgreSQL relations + ArangoDB graph proximity
- **H (WHO)**: Multiprocessing pipeline with GPU acceleration
- **T (Time)**: Minimized through batching, checkpointing, worker optimization
- **Ctx**: Enhanced through LaTeX context injection, extracted structures
- **α ≈ 1.5-2.0**: Super-linear context amplification measured empirically

### 🧪 **Theory Validation**

```bash
# Measure conveyance improvement
python3 conveyance_measurement.py \
    --sample-size 100 \
    --compare-latex-enhancement

# Results show LaTeX context injection improves semantic search relevance
```

## Development & Testing

### 🧪 **Testing Pipeline**

```bash
# Small test run
python3 arxiv_pipeline.py \
    --count 10 \
    --batch-size 10 \
    --extraction-workers 2 \
    --gpu-workers 1

# Medium test run  
python3 arxiv_pipeline.py \
    --count 100 \
    --batch-size 100

# Large test run (current benchmark)
python3 arxiv_pipeline.py \
    --count 1000 \
    --batch-size 1000 \
    --extraction-workers 16 \
    --gpu-workers 2
```

### 🔍 **Pre-flight Checks**

```bash
# Verify system components
python3 utils/check_db_status.py --detailed

# Test database connections
python3 database/database_tools.py --test-connections

# Verify GPU availability
nvidia-smi
```

## Files Reference

### 🗂️ **Production Files** (Active)

| File | Purpose | Usage |
|------|---------|--------|
| `pipelines/arxiv_pipeline.py` | ACID pipeline | Primary processing tool (11.3 papers/min) |
| `pipelines/arango_acid_processor.py` | ACID transaction handler | Ensures atomic database operations |
| `monitoring/acid_monitoring.py` | Real-time monitoring | Track processing progress |
| `scripts/run_acid_pipeline.sh` | Launch script | Automated pipeline execution |
| `scripts/run_embedding_phase_only.py` | Embedding resume | Process staged JSON files |

### 📚 **Documentation**

| File | Content |
|------|---------|
| `CLAUDE.md` | Developer guidance and technical details |
| `README.md` | This user guide |
| `ACHERON_ARCHIVE_LOG.md` | Archive log of cleaned up files |

### 🏛️ **Archived Files** (Historical)

12 pipeline variants moved to `../../Acheron/HADES/tools/arxiv/` with timestamps:
- `arxiv_pipeline_dataparallel.py` - Early GPU parallelization attempt
- `arxiv_pipeline_hysteresis.py` - First hysteresis implementation
- `arxiv_pipeline_memory_fix.py` - Memory optimization experiments
- `arxiv_pipeline_nvlink.py` - NVLink-specific optimizations
- `arxiv_pipeline_unified.py` - First unified approach
- `arxiv_pipeline_unified_flow.py` - Flow control experiments
- Plus database cleanup scripts and test runners

## See Also

- [CLAUDE.md](CLAUDE.md) - Detailed developer guidance
- [ACHERON_ARCHIVE_LOG.md](ACHERON_ARCHIVE_LOG.md) - Archive documentation  
- [../../CLAUDE.md](../../CLAUDE.md) - HADES project overview
- [../../configs/processors/arxiv_hybrid_v2.yaml](../../configs/processors/arxiv_hybrid_v2.yaml) - Configuration reference