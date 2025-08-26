# HADES ArXiv Tool Migration Package

This directory contains the complete ArXiv processing pipeline and dependencies for migration to a private repository.

## Contents

### Core Components
- `arxiv/` - Complete ArXiv processing tools
  - `pipelines/` - ACID-compliant processing pipelines
  - `configs/` - Configuration files
  - `monitoring/` - Real-time monitoring tools
  - `utils/` - Utility scripts
  - `tests/` - ACID compliance and integration tests
  - `scripts/` - Operational scripts
  - `database/` - PostgreSQL schemas

### Dependencies
- `core_framework/` - Core framework (embedders, extractors, storage)
- `configs/` - Configuration files for processors
- `test_scripts/` - Test and validation scripts

### Documentation
- `CLAUDE.md` - AI assistant guidelines
- `pyproject.toml` - Poetry dependencies
- `poetry.lock` - Locked dependency versions

## Migration Steps

1. **Create Private Repository**
   ```bash
   # On GitHub, create new private repository
   # Name suggestion: HADES-experiments or HADES-private
   ```

2. **Initialize New Repository**
   ```bash
   cd migration/
   git init
   git add .
   git commit -m "Initial commit: ACID-compliant ArXiv pipeline achieving 6.8 papers/min"
   ```

3. **Connect to Private Repository**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/HADES-private.git
   git branch -M main
   git push -u origin main
   ```

4. **Set Up Issue Templates**
   Create `.github/ISSUE_TEMPLATE/experiment.md`:
   ```markdown
   ---
   name: Experiment Report
   about: Document an experimental run
   title: '[EXPERIMENT] '
   labels: experiment
   ---
   
   ## Hypothesis
   <!-- What are we testing? Reference H1-H4 if applicable -->
   
   ## Configuration
   - Pipeline: 
   - Batch size: 
   - Workers: 
   - GPU config: 
   
   ## Method
   <!-- How will we test this? -->
   
   ## Results
   - Papers processed: 
   - Success rate: 
   - Processing rate: 
   - Conveyance (C): 
   
   ## Analysis
   <!-- What did we learn? -->
   
   ## Next Steps
   <!-- What should we test next? -->
   ```

5. **Configure Secrets**
   In repository settings, add secrets:
   - `ARANGO_PASSWORD`
   - `PGPASSWORD`

## Current Performance Baseline

As of 2025-08-25:
- **End-to-end rate**: 6.8 papers/minute
- **Extraction rate**: 13.9 papers/minute
- **Embedding rate**: 14.0 papers/minute
- **Success rate**: 100%
- **ACID compliance**: Full (all tests passing)

## Mathematical Framework

The system implements: **C = (W·R·H)/T · Ctx^α**

Current optimizations:
- **W**: Jina v4 embeddings (2048-dim) with late chunking
- **R**: Dual database (PostgreSQL + ArangoDB)
- **H**: Dual GPU processing (A6000s)
- **T**: Minimized through parallel processing
- **Ctx**: Enhanced through LaTeX/PDF extraction
- **α**: Empirically measuring 1.5-2.0

## Experimental Priorities

1. **H1**: Measure α for different document types
2. **H2**: Validate zero-propagation gate
3. **H3**: Discover theory-practice bridges
4. **H4**: Implement self-optimization cycles

## Environment Setup

```bash
# Install dependencies
poetry install

# Set environment variables
export PGPASSWORD="your_password"
export ARANGO_PASSWORD="your_password"
export CUDA_VISIBLE_DEVICES=0,1

# Run pipeline
cd arxiv/pipelines/
python arxiv_pipeline.py --config ../configs/acid_pipeline_phased.yaml --count 100
```

## Contact

Project Lead: Todd
Mathematical Framework: Information Reconstructionism
Target: 375k papers in experiment window (Dec 2012 - Aug 2016)