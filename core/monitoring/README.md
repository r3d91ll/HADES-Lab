# Monitoring - Progress Tracking & Performance Monitoring

The monitoring module provides comprehensive tracking of pipeline execution, performance metrics, and resource utilization, implementing the TIME (T) dimension of the Conveyance Framework while ensuring system observability.

## Overview

Monitoring components track the temporal aspects of information processing, measuring efficiency (C = (W·R·H/T)·Ctx^α) and providing real-time feedback on system performance. They enable optimization of the TIME dimension to maximize Conveyance.

## Architecture

```
monitoring/
├── progress_tracker.py    # Real-time progress tracking
├── compat_monitor.py     # Compatibility monitoring
├── migration_guide.py    # Migration utilities
└── __init__.py          # Public API exports
```

## Theoretical Foundation

### Conveyance Framework Integration

```python
C = (W·R·H/T)·Ctx^α
```

- **TIME (T)**: Direct measurement of processing latency and throughput
- **Efficiency View**: Monitors C/T ratio for optimization
- **Capability View**: Tracks C at fixed T for performance comparison
- **Context Monitoring**: Ensures Ctx components maintain coherence

### Temporal Optimization

Monitoring enables T minimization while maintaining W, R, H quality, directly increasing Conveyance through the efficiency equation.

## Core Components

### ProgressTracker

Real-time tracking of multi-phase pipeline execution:

```python
from core.monitoring import ProgressTracker

# Initialize tracker
tracker = ProgressTracker(
    total_items=1000,
    phases=["extraction", "embedding", "storage"],
    log_interval=10  # Log every 10 items
)

# Track progress
with tracker.phase("extraction"):
    for item in items:
        process_item(item)
        tracker.update(1)

        # Log current stats
        stats = tracker.get_stats()
        print(f"Rate: {stats['items_per_second']:.2f} items/sec")
        print(f"ETA: {stats['eta_seconds']:.0f} seconds")

# Get phase metrics
extraction_stats = tracker.get_phase_stats("extraction")
print(f"Extraction took {extraction_stats['duration']:.2f} seconds")
print(f"Rate: {extraction_stats['rate']:.2f} items/sec")
```

### Resource Monitor

System resource utilization tracking:

```python
from core.monitoring import ResourceMonitor

monitor = ResourceMonitor(
    track_gpu=True,
    track_memory=True,
    track_disk=True
)

# Start monitoring
monitor.start()

# Perform operations
process_documents()

# Get resource usage
stats = monitor.stop()
print(f"Peak GPU Memory: {stats['gpu_memory_peak_mb']} MB")
print(f"Peak RAM: {stats['ram_peak_gb']} GB")
print(f"Disk I/O: {stats['disk_read_mb']} MB read")
print(f"Duration: {stats['duration_seconds']} seconds")
```

### Performance Metrics

Comprehensive performance measurement:

```python
from core.monitoring import PerformanceMetrics

metrics = PerformanceMetrics()

# Track operation
with metrics.timer("extraction"):
    result = extract_document("paper.pdf")

# Track throughput
metrics.record_throughput("documents", count=10, duration=30.5)

# Track error rates
metrics.record_error("extraction", exception_type="PDFError")

# Get summary
summary = metrics.get_summary()
print(f"Extraction: {summary['extraction']['mean_time']:.3f}s")
print(f"Throughput: {summary['throughput']['documents_per_second']:.2f}")
print(f"Error rate: {summary['errors']['rate']:.2%}")
```

### Pipeline Monitor

End-to-end pipeline monitoring:

```python
from core.monitoring import PipelineMonitor

monitor = PipelineMonitor(
    pipeline_name="ACID Pipeline",
    phases=["extract", "chunk", "embed", "store"],
    metrics_file="pipeline_metrics.json"
)

# Monitor full pipeline
with monitor.run():
    # Extraction phase
    with monitor.phase("extract"):
        documents = extract_documents(files)
        monitor.record_metric("documents", len(documents))

    # Embedding phase
    with monitor.phase("embed"):
        embeddings = generate_embeddings(documents)
        monitor.record_metric("embeddings", len(embeddings))

    # Storage phase
    with monitor.phase("store"):
        stored = store_in_database(embeddings)
        monitor.record_metric("stored", stored)

# Get pipeline report
report = monitor.get_report()
print(f"Total time: {report['total_duration']:.2f}s")
print(f"Bottleneck: {report['slowest_phase']}")
print(f"Efficiency: {report['efficiency_score']:.2f}")
```

## Monitoring Patterns

### Basic Progress Tracking

```python
from core.monitoring import ProgressTracker
from tqdm import tqdm

# Simple progress bar
tracker = ProgressTracker(total=len(files))

for file in tqdm(files, desc="Processing"):
    process_file(file)
    tracker.update(1)

print(f"Processed {tracker.n} files in {tracker.elapsed:.2f}s")
```

### Multi-phase Tracking

```python
# Track complex pipeline
tracker = ProgressTracker(
    total_items=1000,
    phases=["download", "extract", "process", "upload"]
)

# Phase 1: Download
with tracker.phase("download") as phase:
    for url in urls:
        download(url)
        phase.update(1)

# Phase 2: Extract
with tracker.phase("extract") as phase:
    for file in files:
        extract(file)
        phase.update(1)

# Get comprehensive metrics
metrics = tracker.get_metrics()
for phase_name, phase_metrics in metrics.items():
    print(f"{phase_name}:")
    print(f"  Duration: {phase_metrics['duration']:.2f}s")
    print(f"  Rate: {phase_metrics['rate']:.2f} items/s")
    print(f"  Percentage: {phase_metrics['percentage']:.1f}%")
```

### Real-time Monitoring Dashboard

```python
from core.monitoring import MonitoringDashboard

# Create dashboard
dashboard = MonitoringDashboard(
    port=8080,
    refresh_interval=1.0  # Update every second
)

# Start dashboard
dashboard.start()

# Register metrics
dashboard.register_metric("throughput", unit="docs/sec")
dashboard.register_metric("gpu_memory", unit="MB")
dashboard.register_metric("error_rate", unit="%")

# Update metrics during processing
for batch in batches:
    results = process_batch(batch)

    dashboard.update_metric("throughput", results.throughput)
    dashboard.update_metric("gpu_memory", get_gpu_memory())
    dashboard.update_metric("error_rate", results.error_rate)

# Access at http://localhost:8080
```

### Distributed Monitoring

```python
from core.monitoring import DistributedMonitor

# Central monitor
monitor = DistributedMonitor(
    role="coordinator",
    redis_host="localhost",
    redis_port=6379
)

# Worker monitors
worker_monitor = DistributedMonitor(
    role="worker",
    worker_id="worker_1",
    redis_host="localhost"
)

# Workers report progress
worker_monitor.report_progress("extraction", completed=50, total=100)

# Coordinator aggregates
stats = monitor.get_aggregate_stats()
print(f"Total progress: {stats['overall_progress']:.1%}")
print(f"Active workers: {stats['active_workers']}")
print(f"Combined throughput: {stats['total_throughput']:.2f}/s")
```

## Performance Analysis

### Bottleneck Detection

```python
from core.monitoring import BottleneckAnalyzer

analyzer = BottleneckAnalyzer()

# Profile pipeline
with analyzer.profile("pipeline"):
    with analyzer.stage("extraction"):
        extract_documents()  # 10s

    with analyzer.stage("embedding"):
        generate_embeddings()  # 30s

    with analyzer.stage("storage"):
        store_results()  # 5s

# Identify bottlenecks
bottlenecks = analyzer.find_bottlenecks()
print(f"Primary bottleneck: {bottlenecks[0]['stage']}")
print(f"Time spent: {bottlenecks[0]['duration']:.2f}s")
print(f"Percentage: {bottlenecks[0]['percentage']:.1f}%")

# Get optimization suggestions
suggestions = analyzer.suggest_optimizations()
for suggestion in suggestions:
    print(f"- {suggestion}")
```

### Resource Profiling

```python
from core.monitoring import ResourceProfiler

profiler = ResourceProfiler(
    profile_cpu=True,
    profile_memory=True,
    profile_gpu=True,
    profile_disk=True
)

# Profile operation
with profiler.profile("document_processing"):
    process_large_dataset()

# Get resource report
report = profiler.get_report()

print("Resource Usage:")
print(f"  CPU: {report['cpu']['mean_usage']:.1f}% (peak: {report['cpu']['peak']:.1f}%)")
print(f"  Memory: {report['memory']['mean_gb']:.2f} GB (peak: {report['memory']['peak_gb']:.2f} GB)")
print(f"  GPU: {report['gpu']['mean_usage']:.1f}% (memory: {report['gpu']['memory_gb']:.2f} GB)")
print(f"  Disk I/O: {report['disk']['read_mb_per_sec']:.2f} MB/s read, {report['disk']['write_mb_per_sec']:.2f} MB/s write")
```

### Efficiency Metrics

```python
from core.monitoring import EfficiencyCalculator

calculator = EfficiencyCalculator()

# Record pipeline execution
calculator.record_execution(
    phase="extraction",
    items_processed=100,
    duration=50.0,
    resources_used={
        "cpu_hours": 0.014,
        "gpu_hours": 0.0,
        "memory_gb_hours": 0.28
    }
)

calculator.record_execution(
    phase="embedding",
    items_processed=100,
    duration=150.0,
    resources_used={
        "cpu_hours": 0.021,
        "gpu_hours": 0.042,
        "memory_gb_hours": 0.35
    }
)

# Calculate efficiency metrics
efficiency = calculator.calculate()

print(f"Overall efficiency: {efficiency['overall_score']:.2f}/100")
print(f"Throughput: {efficiency['items_per_hour']:.0f} items/hour")
print(f"Cost efficiency: ${efficiency['cost_per_1000_items']:.2f}/1000 items")
print(f"Resource utilization: {efficiency['resource_utilization']:.1%}")
```

## Alerting & Notifications

### Threshold Alerts

```python
from core.monitoring import AlertManager

alerts = AlertManager()

# Define alert rules
alerts.add_rule(
    name="high_error_rate",
    condition=lambda metrics: metrics['error_rate'] > 0.05,
    message="Error rate exceeds 5%",
    severity="warning"
)

alerts.add_rule(
    name="low_throughput",
    condition=lambda metrics: metrics['throughput'] < 10,
    message="Throughput below 10 items/sec",
    severity="critical"
)

# Monitor and check alerts
metrics = get_current_metrics()
triggered = alerts.check(metrics)

for alert in triggered:
    print(f"[{alert.severity}] {alert.name}: {alert.message}")
    send_notification(alert)
```

### Slack Integration

```python
from core.monitoring import SlackNotifier

notifier = SlackNotifier(
    webhook_url="https://hooks.slack.com/services/...",
    channel="#monitoring"
)

# Send progress updates
notifier.send_progress(
    phase="Extraction",
    progress=0.75,
    eta_minutes=15,
    rate="25 docs/min"
)

# Send alerts
notifier.send_alert(
    severity="warning",
    message="GPU memory usage exceeds 90%",
    details={"gpu_memory_gb": 22.5, "threshold_gb": 20.0}
)

# Send completion
notifier.send_completion(
    pipeline="ACID Pipeline",
    duration_minutes=45,
    items_processed=1000,
    success_rate=0.98
)
```

## Logging Integration

### Structured Logging

```python
from core.monitoring import StructuredLogger

logger = StructuredLogger(
    name="pipeline",
    log_file="pipeline.log",
    json_format=True
)

# Log with context
logger.info(
    "Processing batch",
    batch_id=123,
    size=100,
    phase="extraction"
)

# Log metrics
logger.metric(
    "performance",
    throughput=25.5,
    latency_ms=40,
    error_rate=0.01
)

# Log errors with context
try:
    process_document(doc)
except Exception as e:
    logger.error(
        "Document processing failed",
        document_id=doc.id,
        error_type=type(e).__name__,
        error_message=str(e),
        traceback=True
    )
```

### Log Aggregation

```python
from core.monitoring import LogAggregator

aggregator = LogAggregator(
    log_files=["worker1.log", "worker2.log", "worker3.log"],
    output_file="aggregated.log"
)

# Aggregate and analyze
stats = aggregator.aggregate()

print(f"Total log entries: {stats['total_entries']}")
print(f"Error entries: {stats['error_count']}")
print(f"Warning entries: {stats['warning_count']}")
print(f"Average throughput: {stats['avg_throughput']:.2f}")

# Get error summary
errors = aggregator.get_errors()
for error_type, count in errors.items():
    print(f"{error_type}: {count} occurrences")
```

## Configuration

### YAML Configuration

```yaml
# monitoring_config.yaml
monitoring:
  progress:
    log_interval: 10
    show_eta: true
    show_rate: true

  resources:
    track_cpu: true
    track_memory: true
    track_gpu: true
    sample_interval: 1.0

  alerts:
    error_rate_threshold: 0.05
    throughput_min: 10
    gpu_memory_max_gb: 20

  logging:
    level: INFO
    format: json
    file: pipeline.log
    rotation: daily
    retention_days: 30
```

### Environment Variables

```bash
# Monitoring configuration
export MONITORING_ENABLED=true
export MONITORING_LOG_LEVEL=INFO
export MONITORING_INTERVAL=5

# Alert thresholds
export ALERT_ERROR_RATE=0.05
export ALERT_THROUGHPUT_MIN=10
export ALERT_GPU_MEMORY_MAX=20

# Notification settings
export SLACK_WEBHOOK_URL="https://..."
export EMAIL_ALERTS=true
```

## Visualization

### Metrics Dashboard

```python
from core.monitoring import MetricsDashboard
import matplotlib.pyplot as plt

dashboard = MetricsDashboard()

# Collect metrics over time
metrics_history = []
for batch in batches:
    metrics = process_batch(batch)
    metrics_history.append(metrics)
    dashboard.update(metrics)

# Generate visualizations
dashboard.plot_throughput(metrics_history)
dashboard.plot_resource_usage(metrics_history)
dashboard.plot_error_rates(metrics_history)

# Save report
dashboard.save_report("pipeline_report.html")
```

### Real-time Plotting

```python
from core.monitoring import RealtimePlotter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plotter = RealtimePlotter(
    metrics=["throughput", "gpu_memory", "error_rate"],
    window_size=100  # Last 100 data points
)

# Start real-time plotting
fig, axes = plotter.create_figure()

def update(frame):
    metrics = get_current_metrics()
    plotter.update_data(metrics)
    plotter.refresh_plots()

ani = FuncAnimation(fig, update, interval=1000)  # Update every second
plt.show()
```

## Best Practices

### 1. Use Context Managers

```python
# Good - automatic cleanup
with ProgressTracker(total=100) as tracker:
    for item in items:
        process(item)
        tracker.update(1)

# Ensures proper resource cleanup and final metrics
```

### 2. Log at Appropriate Intervals

```python
# Avoid excessive logging
tracker = ProgressTracker(
    total=1000000,
    log_interval=10000  # Log every 10k items, not every item
)
```

### 3. Monitor Resource Usage

```python
# Track resource usage for optimization
with ResourceMonitor() as monitor:
    process_batch()

    if monitor.gpu_memory_mb > 20000:  # 20GB
        reduce_batch_size()
```

### 4. Set Meaningful Alerts

```python
# Define actionable alerts
alerts.add_rule(
    name="processing_stalled",
    condition=lambda m: m['items_processed_last_minute'] == 0,
    action=restart_worker,
    auto_resolve=True
)
```

## Testing

```python
import pytest
from core.monitoring import ProgressTracker, ResourceMonitor

def test_progress_tracking():
    """Test progress tracker."""
    tracker = ProgressTracker(total=100)

    for i in range(100):
        tracker.update(1)

    assert tracker.n == 100
    assert tracker.finished

def test_resource_monitoring():
    """Test resource monitor."""
    monitor = ResourceMonitor()

    with monitor:
        # Simulate work
        data = [0] * 1000000

    stats = monitor.get_stats()
    assert stats['duration_seconds'] > 0
    assert stats['ram_peak_mb'] > 0

def test_phase_tracking():
    """Test multi-phase tracking."""
    tracker = ProgressTracker(
        total=100,
        phases=["extract", "process"]
    )

    with tracker.phase("extract"):
        for i in range(50):
            tracker.update(1)

    phase_stats = tracker.get_phase_stats("extract")
    assert phase_stats['items'] == 50
```

## Performance Impact

| Monitoring Level | Overhead | Use Case |
|-----------------|----------|----------|
| Disabled | 0% | Production (max performance) |
| Basic | 0.1-0.5% | Production (standard) |
| Detailed | 1-2% | Development/Testing |
| Profiling | 5-10% | Debugging/Optimization |

## Migration Guide

### From tqdm

```python
# Old approach
from tqdm import tqdm
for item in tqdm(items):
    process(item)

# New approach
from core.monitoring import ProgressTracker
tracker = ProgressTracker(total=len(items))
for item in items:
    process(item)
    tracker.update(1)
# Plus: metrics, phases, resource tracking
```

### From print statements

```python
# Old approach
print(f"Processing {i}/{total}")
print(f"Rate: {i/elapsed:.2f}/s")

# New approach
from core.monitoring import StructuredLogger
logger = StructuredLogger("pipeline")
logger.info("Processing", current=i, total=total, rate=i/elapsed)
# Plus: structured output, aggregation, analysis
```

## Related Components

- [Workflows](../workflows/README.md) - Pipeline orchestration with monitoring
- [Config](../config/README.md) - Monitoring configuration
- [Database](../database/README.md) - Metrics storage
- [Processors](../processors/README.md) - Processing monitoring