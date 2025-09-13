"""
Core Monitoring Module
======================

Theory Connection - Information Reconstructionism:
The monitoring system implements real-time measurement and feedback across
all dimensions of the Conveyance Framework C = (W·R·H/T)·Ctx^α. It serves
as the empirical foundation for system optimization, providing continuous
assessment of Context amplification and enabling data-driven improvements.

From Actor-Network Theory: Monitoring components act as "immutable mobiles"
that transport performance information across network boundaries, enabling
distributed components to maintain coherent understanding of system behavior
and optimization opportunities.

Key Components:
- MetricsCollector: Abstract base for metrics collection with Conveyance Framework integration
- PerformanceMonitor: System resource monitoring with GPU support and alerting
- ProgressTracker: Multi-step progress tracking with ETA calculation and Context scoring
- Built-in metrics: Counter, gauge, histogram, timer for comprehensive measurement

The module consolidates monitoring functionality from tools/arxiv/monitoring/
into reusable core infrastructure while maintaining backward compatibility.
"""

from .metrics_base import (
    MetricsCollector,
    MetricType,
    MetricLevel,
    MetricValue,
    MetricSeries
)

from .performance_monitor import (
    PerformanceMonitor,
    SystemResources,
    GPUResources,
    ProcessingPhase
)

from .progress_tracker import (
    ProgressTracker,
    ProgressStep,
    ProgressState
)

from .compat_monitor import (
    LegacyMonitorInterface,
    create_legacy_monitor
)

# Version information
__version__ = "1.0.0"

# Export main interfaces
__all__ = [
    # Base metrics classes
    'MetricsCollector',
    'MetricType',
    'MetricLevel',
    'MetricValue',
    'MetricSeries',

    # Performance monitoring
    'PerformanceMonitor',
    'SystemResources',
    'GPUResources',
    'ProcessingPhase',

    # Progress tracking
    'ProgressTracker',
    'ProgressStep',
    'ProgressState',

    # Backward compatibility
    'LegacyMonitorInterface',
    'create_legacy_monitor',

    # Factory functions
    'create_performance_monitor',
    'create_progress_tracker',
]


def create_performance_monitor(component_name: str, log_dir: str = None,
                             monitor_interval: float = 5.0,
                             enable_gpu_monitoring: bool = True) -> PerformanceMonitor:
    """
    Factory function to create a performance monitor with standard configuration.

    Theory Connection: Implements efficient WHO dimension monitoring through
    standardized resource tracking patterns that minimize TIME overhead
    while maximizing Context coherence.

    Args:
        component_name: Name of the component being monitored
        log_dir: Directory for performance logs
        monitor_interval: Monitoring interval in seconds
        enable_gpu_monitoring: Enable GPU resource monitoring

    Returns:
        Configured performance monitor

    Example:
        >>> monitor = create_performance_monitor('arxiv_pipeline')
        >>> monitor.start_monitoring()
        >>> monitor.start_phase('extraction', total_items=1000)
    """
    from pathlib import Path

    monitor = PerformanceMonitor(
        component_name=component_name,
        log_dir=Path(log_dir) if log_dir else None
    )

    if enable_gpu_monitoring:
        # Set reasonable GPU memory thresholds
        monitor.gpu_memory_warning_threshold = 85.0
        monitor.gpu_memory_critical_threshold = 95.0

    # Start monitoring automatically if requested
    if monitor_interval > 0:
        monitor.start_monitoring(interval=monitor_interval)

    return monitor


def create_progress_tracker(name: str, description: str = "",
                          auto_save: bool = False,
                          save_path: str = None) -> ProgressTracker:
    """
    Factory function to create a progress tracker with optional auto-save.

    Theory Connection: Provides temporal WHERE → WHAT transformation
    tracking with Context coherence measurement and automatic persistence.

    Args:
        name: Tracker name
        description: Human-readable description
        auto_save: Enable automatic progress saving
        save_path: Path for automatic saves

    Returns:
        Configured progress tracker

    Example:
        >>> tracker = create_progress_tracker('paper_processing', 'ArXiv paper pipeline')
        >>> tracker.add_step('extraction', 'Extract text and metadata', 1000)
        >>> tracker.start_step('extraction')
    """
    tracker = ProgressTracker(name=name, description=description)

    if auto_save and save_path:
        from pathlib import Path
        import threading

        save_file = Path(save_path)

        def auto_save_callback(tracker_instance, step):
            """Auto-save callback for progress updates."""
            try:
                tracker_instance.save_progress(save_file)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Auto-save failed: {e}")

        # Add auto-save callback for all significant state changes
        for state in ProgressState:
            if state in [ProgressState.COMPLETED, ProgressState.FAILED, ProgressState.CANCELLED]:
                tracker.add_step_callback(state, auto_save_callback)

    return tracker


# Default monitoring configurations for common use cases
DEFAULT_PERFORMANCE_CONFIG = {
    'monitor_interval': 5.0,
    'cpu_warning_threshold': 80.0,
    'cpu_critical_threshold': 95.0,
    'memory_warning_threshold': 85.0,
    'memory_critical_threshold': 95.0,
    'gpu_memory_warning_threshold': 85.0,
    'gpu_memory_critical_threshold': 95.0
}

DEFAULT_METRICS_CONFIG = {
    'flush_interval': 60.0,
    'history_size': 10000,
    'enable_conveyance_scoring': True
}

# Common metric dimension mappings for Conveyance Framework
DIMENSION_MAPPINGS = {
    # WHERE dimension - spatial and positional metrics
    'where': [
        'file_paths_processed',
        'component_transitions',
        'storage_locations',
        'network_hops',
        'processing_stages'
    ],

    # WHAT dimension - content and quality metrics
    'what': [
        'documents_processed',
        'text_extracted',
        'embeddings_generated',
        'validation_success_rate',
        'content_quality_score'
    ],

    # WHO dimension - agent and capability metrics
    'who': [
        'cpu_utilization',
        'memory_usage',
        'gpu_utilization',
        'worker_efficiency',
        'resource_availability'
    ],

    # TIME dimension - temporal and efficiency metrics
    'time': [
        'processing_latency',
        'throughput_rate',
        'phase_duration',
        'queue_wait_time',
        'response_time'
    ]
}


def get_dimension_for_metric(metric_name: str) -> str:
    """
    Get Conveyance Framework dimension for a metric name.

    Args:
        metric_name: Name of the metric

    Returns:
        Dimension name ('where', 'what', 'who', 'time', or 'unknown')
    """
    for dimension, metric_patterns in DIMENSION_MAPPINGS.items():
        if any(pattern in metric_name.lower() for pattern in metric_patterns):
            return dimension
    return 'unknown'


def calculate_context_score(local_coherence: float, instruction_fit: float,
                          actionability: float, grounding: float) -> float:
    """
    Calculate Context score for Conveyance Framework.

    Theory Connection: Implements Ctx = wL·L + wI·I + wA·A + wG·G
    with equal weights (0.25 each) as specified in the framework.

    Args:
        local_coherence: Local coherence component (0.0-1.0)
        instruction_fit: Instruction fit component (0.0-1.0)
        actionability: Actionability component (0.0-1.0)
        grounding: Grounding component (0.0-1.0)

    Returns:
        Context score (0.0-1.0)
    """
    # Equal weights as specified in framework
    weights = {'L': 0.25, 'I': 0.25, 'A': 0.25, 'G': 0.25}

    context_score = (
        weights['L'] * local_coherence +
        weights['I'] * instruction_fit +
        weights['A'] * actionability +
        weights['G'] * grounding
    )

    return max(0.0, min(1.0, context_score))


def calculate_conveyance(where: float, what: float, who: float, time: float,
                        context: float, alpha: float = 1.8) -> float:
    """
    Calculate Conveyance score using efficiency view.

    Theory Connection: Implements C = (W·R·H/T)·Ctx^α where:
    - W = WHAT (content quality)
    - R = WHERE (relational positioning)
    - H = WHO (agent capability)
    - T = TIME (convergence time)
    - Ctx = Context (amplification factor)
    - α = Super-linear amplification exponent

    Args:
        where: WHERE dimension score (0.0-1.0)
        what: WHAT dimension score (0.0-1.0)
        who: WHO dimension score (0.0-1.0)
        time: TIME dimension score (0.0-1.0)
        context: Context score (0.0-1.0)
        alpha: Amplification exponent (1.5-2.0)

    Returns:
        Conveyance score
    """
    # Zero-propagation gate
    if any(dim == 0 for dim in [where, what, who]) or time == 0:
        return 0.0

    # Efficiency view: C = (W·R·H/T)·Ctx^α
    base_conveyance = (what * where * who) / time
    amplified_conveyance = base_conveyance * (context ** alpha)

    return amplified_conveyance