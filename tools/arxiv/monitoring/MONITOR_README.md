# Pipeline Live Monitoring Dashboard

A real-time monitoring system for the unified ArXiv processing pipeline with live CLI interface, queue statistics, database counts, and resource utilization.

## Features

- 🔄 **Real-time Queue Monitoring**: Live queue sizes for extraction, embedding, and write queues
- 💾 **Database Statistics**: ArangoDB collection counts and PostgreSQL processing progress  
- 📊 **Processing Rates**: Papers per minute and trend indicators
- 🖥️ **System Resources**: CPU, RAM, and GPU utilization with progress bars
- 👥 **Worker Status**: Active worker counts by type
- 📈 **Historical Trends**: Rate changes and processing patterns
- ⚡ **Live Updates**: Refreshes every 10 seconds with clean terminal interface

## Quick Start

### Prerequisites

```bash
# Set environment variables
export PGPASSWORD="your_postgres_password"
export ARANGO_PASSWORD="your_arango_password"

# Install Python dependencies (if needed)
pip3 install psycopg2-binary python-arango psutil gputil
```

### Launch Monitor

```bash
# Easy way (with environment variables)
cd /home/todd/olympus/HADES/tools/arxiv
./monitor_live.sh

# Or directly with Python
python3 monitor_pipeline_live.py --pg-password "$PGPASSWORD" --arango-password "$ARANGO_PASSWORD"
```

## Dashboard Interface

```
🔍 UNIFIED PIPELINE LIVE MONITOR
Last Updated: 2025-08-22 23:45:12
================================================

📊 PROCESSING PROGRESS
┌─ Papers Processed: 1,234 / 375,000
│  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.3%
│  Rate: 45.2 papers/min ↗
└─ Remaining: 373,766 papers

💾 DATABASE STATUS
┌─ ArangoDB Collections:
│  ├─ Unified Embeddings: 1,234
│  └─ Structures: 1,198
└─ PostgreSQL: 1,234 processed

🔄 QUEUE STATUS
┌─ Extraction:  156/2000 │███░░░░░░░░░░░░│
├─ Embedding:   89/1000 │██░░░░░░░░░░░░░│
└─ Write:       23/4000 │░░░░░░░░░░░░░░░│

👥 WORKER STATUS
┌─ Extraction Workers: 8 active
├─ GPU Workers: 4 active
├─ Write Workers: 2 active
└─ Total Processes: 15

🖥️  SYSTEM RESOURCES
┌─ CPU: 68.5% │█████████████░░░░░░│
├─ RAM: 45.2GB/256GB │████░░░░░░░░░░░░░░░│
├─ GPU Memory: 18.3GB/24GB │███████████████░░░░│
└─ GPU Util: 87.2%

📈 RECENT TRENDS (last 10 samples)
┌─ Processing Rate: ↗
├─ ArangoDB Growth: ↗
├─ CPU Usage: ━
└─ GPU Memory: ↘

================================================
Press Ctrl+C to exit • Update every 10 seconds
================================================
```

## Real-time Queue Monitoring

The monitoring system reads real-time queue statistics from a status file that the pipeline creates. To enable this:

### Integration with Pipeline

Add this to your pipeline code:

```python
from pipeline_status_reporter import start_status_reporting, report_queue_sizes

# At pipeline startup
reporter = start_status_reporting("pipeline_status.json")

# In your worker loops (periodically)
report_queue_sizes(
    extraction=extraction_queue.qsize(),
    embedding=embedding_queue.qsize(), 
    write=write_queue.qsize()
)
```

### Status File Structure

The pipeline creates `pipeline_status.json`:

```json
{
  "timestamp": "2025-08-22T23:45:12.123456",
  "queues": {
    "extraction_queue": 156,
    "embedding_queue": 89,
    "write_queue": 23
  },
  "workers": {
    "extraction_active": 8,
    "embedding_active": 4,
    "write_active": 2
  },
  "processing": {
    "papers_processed": 1234,
    "current_batch_size": 32,
    "last_paper_id": "2012.15432",
    "errors_count": 3
  },
  "rates": {
    "extraction_rate": 120.5,
    "embedding_rate": 45.2,
    "write_rate": 78.3
  }
}
```

## Controls

- **Ctrl+C**: Exit monitoring
- **Terminal Resize**: Automatically adjusts display
- **10-second Updates**: Automatic refresh cycle

## Troubleshooting

### Database Connection Issues

```bash
# Test PostgreSQL
PGPASSWORD=$PGPASSWORD psql -h localhost -U postgres -d Avernus -c "SELECT 1;"

# Test ArangoDB  
curl -u root:$ARANGO_PASSWORD http://192.168.1.69:8529/_api/version
```

### Missing Dependencies

```bash
pip3 install psycopg2-binary python-arango psutil gputil
```

### Queue Monitoring Not Working

Make sure the pipeline is using the status reporter:

```python
# Add to pipeline initialization
from pipeline_status_reporter import start_status_reporting
start_status_reporting()
```

### Performance Impact

The monitoring system is designed to be lightweight:
- Database queries run every 10 seconds
- Status file reads are fast (JSON parsing)
- Process scanning uses efficient psutil calls
- Minimal CPU/memory overhead

## Architecture

### Components

1. **DatabaseMonitor**: Queries PostgreSQL and ArangoDB for counts
2. **ProcessMonitor**: Scans running processes and reads status files
3. **ResourceMonitor**: System CPU/RAM/GPU statistics  
4. **LiveDashboard**: Terminal UI with progress bars and trends

### Data Flow

```
Pipeline → Status File → Process Monitor → Live Dashboard → Terminal
             ↑
Database Monitor → Live Dashboard
             ↑  
Resource Monitor → Live Dashboard
```

### Update Cycle

1. Collect all statistics (1-2 seconds)
2. Render dashboard interface
3. Wait 10 seconds
4. Repeat

This provides real-time insight into pipeline performance without impacting processing speed.