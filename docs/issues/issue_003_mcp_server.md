# Issue #3: Add MCP Server Functionality for Pipeline Automation

## Overview

Implement Model Context Protocol (MCP) server to enable automated pipeline control through Claude. This will allow Claude to directly orchestrate paper processing, monitor progress, and manage the HADES infrastructure.

## Background

Currently, the pipeline must be manually executed via command line. An MCP server would enable:
- Automated paper processing based on queries
- Real-time pipeline monitoring
- Intelligent retry of failed papers
- Dynamic configuration adjustments

## Requirements

### Core Functionality

1. **Pipeline Control**
   - Start/stop/pause pipeline processing
   - Configure processing parameters
   - Select paper sources (local, specific lists, date ranges)

2. **Monitoring**
   - Real-time progress tracking
   - Error reporting and diagnostics
   - Performance metrics (papers/minute, GPU utilization)

3. **Database Operations**
   - Query ArangoDB collections
   - Check processing status
   - Retrieve embeddings and structures

4. **Resource Management**
   - GPU memory monitoring
   - Worker pool management
   - Staging directory cleanup

### MCP Tools to Implement

```python
# Pipeline control
async def process_papers(arxiv_ids: List[str], config: Dict)
async def process_date_range(start: str, end: str, config: Dict)
async def pause_processing()
async def resume_processing()

# Monitoring
async def get_pipeline_status()
async def get_failed_papers()
async def get_processing_metrics()

# Database
async def query_embeddings(query: str, limit: int)
async def get_paper_status(arxiv_id: str)
async def check_collection_stats()

# Resource management
async def clear_gpu_memory()
async def adjust_worker_count(workers: int)
async def cleanup_staging()
```

## Implementation Plan

### Phase 1: Basic Infrastructure
- Create `core/mcp_server/server.py` with MCP framework
- Implement basic pipeline control tools
- Add authentication and connection management

### Phase 2: Pipeline Integration
- Connect to existing `arxiv_pipeline.py`
- Implement checkpoint management
- Add progress tracking

### Phase 3: Advanced Features
- Intelligent retry mechanisms
- Dynamic configuration based on performance
- Multi-pipeline orchestration

## Technical Details

### Dependencies
- `mcp` package for server implementation
- AsyncIO for concurrent operations
- Integration with existing pipeline classes

### Configuration
```yaml
mcp_server:
  host: 192.168.1.69
  port: 8765
  auth_token: ${MCP_AUTH_TOKEN}
  pipeline_config: tools/arxiv/configs/acid_pipeline_phased.yaml
```

### Security Considerations
- Authentication token required
- Rate limiting on tool calls
- Sandboxed execution environment

## Benefits

1. **Automation**: Claude can autonomously process papers based on research needs
2. **Efficiency**: Intelligent retry and resource management
3. **Observability**: Real-time monitoring without manual log checking
4. **Scalability**: Easy to add new processing capabilities

## Related Files

- Previous attempt: `Acheron/postgresql_migration_2025-08-26/mcp_tools_2025-08-26_00-29-46.py`
- Pipeline: `tools/arxiv/pipelines/arxiv_pipeline.py`
- Config: `tools/arxiv/configs/acid_pipeline_phased.yaml`

## Priority

**Medium** - Useful for automation but not blocking current functionality

## Estimated Effort

- Initial implementation: 2-3 days
- Full integration: 1 week
- Testing and refinement: 3-4 days

## Labels

`enhancement`, `mcp`, `automation`, `infrastructure`