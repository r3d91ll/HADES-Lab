"""
Core Monitoring Usage Examples
==============================

Theory Connection - Information Reconstructionism:
These examples demonstrate how to implement real-time measurement of the
Conveyance Framework C = (W·R·H/T)·Ctx^α using the core monitoring infrastructure.
Each example shows optimal patterns for Context coherence measurement and
system optimization through empirical feedback.

From Actor-Network Theory: Monitoring examples serve as "inscription devices"
that enable consistent measurement practices across distributed components,
maintaining network coherence through standardized observation patterns.
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Import core monitoring components
from core.monitoring import (
    create_performance_monitor,
    create_progress_tracker,
    create_legacy_monitor,
    PerformanceMonitor,
    ProgressTracker,
    MetricType,
    calculate_conveyance
)


def example_basic_performance_monitoring():
    """
    Example 1: Basic Performance Monitoring

    Theory Connection: Implements WHO dimension monitoring through
    system resource tracking, enabling real-time assessment of
    agent capability utilization.
    """
    print("Example 1: Basic Performance Monitoring")
    print("=" * 40)

    # Create performance monitor with automatic GPU detection
    monitor = create_performance_monitor(
        component_name='example_processor',
        monitor_interval=2.0,
        enable_gpu_monitoring=True
    )

    print("Starting performance monitoring...")
    monitor.start_monitoring()

    # Simulate some processing work
    print("Simulating processing work...")
    time.sleep(5)

    # Get current system status
    status = monitor.get_system_status()
    print(f"CPU Usage: {status['system']['cpu_percent']:.1f}%")
    print(f"Memory Usage: {status['system']['memory_percent']:.1f}%")

    if status['gpus']:
        for gpu in status['gpus']:
            print(f"GPU {gpu['gpu_id']}: {gpu['utilization_percent']:.1f}% util, "
                  f"{gpu['memory_used_gb']:.1f}GB used")

    # Stop monitoring and cleanup
    monitor.stop_monitoring()
    monitor.cleanup()
    print("Performance monitoring example completed.\n")


def example_progress_tracking():
    """
    Example 2: Multi-Step Progress Tracking

    Theory Connection: Implements WHERE → WHAT transformation tracking
    across processing pipeline phases, enabling Context coherence
    measurement and Conveyance optimization.
    """
    print("Example 2: Multi-Step Progress Tracking")
    print("=" * 40)

    # Create progress tracker for a multi-phase pipeline
    tracker = create_progress_tracker(
        name='example_pipeline',
        description='Example document processing pipeline'
    )

    # Define processing phases
    phases = [
        ('extraction', 'Extract text and metadata', 100),
        ('embedding', 'Generate embeddings', 100),
        ('storage', 'Store in database', 100)
    ]

    # Add all steps
    for phase_id, phase_name, item_count in phases:
        tracker.add_step(phase_id, phase_name, item_count)

    # Simulate processing phases
    for phase_id, phase_name, item_count in phases:
        print(f"\nStarting phase: {phase_name}")
        tracker.start_step(phase_id)

        # Simulate processing with occasional errors
        for i in range(1, item_count + 1):
            # Simulate processing time
            time.sleep(0.05)

            # Simulate occasional failures (5% failure rate)
            failed_count = max(0, i // 20)
            successful_count = i - failed_count

            tracker.update_step(phase_id, completed=successful_count, failed=failed_count)

            # Report progress every 25 items
            if i % 25 == 0:
                step_status = tracker.get_step_status(phase_id)
                print(f"  Progress: {step_status['completion_percent']:.1f}% "
                      f"({step_status['processing_rate']:.2f} items/sec)")

        tracker.complete_step(phase_id)
        print(f"Completed phase: {phase_name}")

    # Get final results
    overall_progress = tracker.get_overall_progress()
    conveyance_metrics = tracker.calculate_conveyance_metrics()

    print(f"\nPipeline completed!")
    print(f"Total processing time: {overall_progress['duration_seconds']:.1f} seconds")
    print(f"Overall success rate: {overall_progress['success_rate']:.1f}%")
    print(f"Conveyance score: {conveyance_metrics.get('conveyance', 0):.3f}")
    print("Progress tracking example completed.\n")


def example_custom_metrics_collection():
    """
    Example 3: Custom Metrics Collection

    Theory Connection: Demonstrates how to extend monitoring for specific
    Conveyance Framework dimensions, enabling targeted optimization of
    Context components (L, I, A, G).
    """
    print("Example 3: Custom Metrics Collection")
    print("=" * 40)

    # Create monitor with custom metrics
    monitor = PerformanceMonitor('custom_example')

    # Register custom metrics for different Conveyance dimensions
    monitor.register_metric('documents_processed', MetricType.COUNTER,
                          'Total documents processed', 'documents',
                          {'what': 'content_processing'})

    monitor.register_metric('extraction_quality', MetricType.GAUGE,
                          'Text extraction quality score', 'score',
                          {'what': 'content_quality'})

    monitor.register_metric('file_access_time', MetricType.HISTOGRAM,
                          'File access latency', 'seconds',
                          {'where': 'file_system_access'})

    monitor.register_metric('worker_efficiency', MetricType.GAUGE,
                          'Worker processing efficiency', 'percent',
                          {'who': 'worker_performance'})

    print("Simulating custom metric collection...")

    # Simulate processing with custom metrics
    for i in range(1, 21):
        # Simulate document processing
        monitor.increment('documents_processed')

        # Simulate varying quality scores
        quality_score = 0.8 + (i % 10) * 0.02  # 0.8 to 0.98
        monitor.gauge('extraction_quality', quality_score)

        # Simulate file access times
        access_time = 0.1 + (i % 5) * 0.05  # 0.1 to 0.3 seconds
        monitor.histogram('file_access_time', access_time)

        # Simulate worker efficiency
        efficiency = 85 + (i % 15)  # 85% to 99%
        monitor.gauge('worker_efficiency', efficiency)

        time.sleep(0.1)

    # Calculate Conveyance metrics from collected data
    conveyance_scores = monitor.calculate_conveyance_score()
    print("Conveyance Framework Scores:")
    for dimension, score in conveyance_scores.items():
        print(f"  {dimension.upper()}: {score:.3f}")

    # Get metrics summary
    summary = monitor.get_summary()
    print(f"\nCollected {summary['metrics_count']} different metrics")
    print("Custom metrics collection example completed.\n")


def example_legacy_integration():
    """
    Example 4: Legacy Monitoring Integration

    Theory Connection: Demonstrates backward compatibility while enabling
    enhanced Context measurement, allowing gradual migration to unified
    Conveyance Framework optimization.
    """
    print("Example 4: Legacy Monitoring Integration")
    print("=" * 40)

    # Create legacy-compatible monitor
    legacy_monitor = create_legacy_monitor(
        component_name='legacy_example',
        log_dir='logs'
    )

    print("Starting legacy-compatible monitoring...")
    legacy_monitor.start_monitoring()

    # Use legacy interface methods
    gpu_status = legacy_monitor.get_gpu_status()
    print(f"GPU available: {gpu_status['available']}")

    system_status = legacy_monitor.get_system_status()
    print(f"System CPU: {system_status['cpu_percent']:.1f}%")

    # Simulate processing phases (legacy style)
    legacy_monitor.progress_tracker.add_step('main_processing', 'Process items', 50)
    legacy_monitor.progress_tracker.start_step('main_processing')

    for i in range(1, 51):
        # Simulate processing
        time.sleep(0.02)
        legacy_monitor.progress_tracker.update_step('main_processing', completed=i)

        if i % 10 == 0:
            print(f"  Processed {i}/50 items")

    legacy_monitor.progress_tracker.complete_step('main_processing')

    # Enhanced capabilities still available
    conveyance_metrics = legacy_monitor.progress_tracker.calculate_conveyance_metrics()
    print(f"Enhanced Conveyance Score: {conveyance_metrics.get('conveyance', 0):.3f}")

    # Cleanup
    legacy_monitor.cleanup()
    print("Legacy integration example completed.\n")


def example_real_time_dashboard():
    """
    Example 5: Real-Time Dashboard Integration

    Theory Connection: Demonstrates continuous Context monitoring with
    real-time feedback for system optimization, enabling dynamic adjustment
    of processing parameters based on Conveyance measurements.
    """
    print("Example 5: Real-Time Dashboard (Simulated)")
    print("=" * 40)

    # Create monitor with dashboard-style reporting
    monitor = create_performance_monitor(
        component_name='dashboard_example',
        monitor_interval=1.0  # High frequency for demo
    )

    tracker = create_progress_tracker('dashboard_processing', 'Real-time processing demo')
    tracker.add_step('main', 'Processing documents', 100)
    tracker.start_step('main')

    print("Running real-time dashboard simulation...")
    print("(In real implementation, this would display updating dashboard)")

    # Simulate dashboard updates
    for i in range(1, 101):
        # Update progress
        tracker.update_step('main', completed=i)

        # Get current metrics
        if i % 20 == 0:  # Update every 20 items
            overall = tracker.get_overall_progress()
            conveyance = tracker.calculate_conveyance_metrics()

            print(f"\nDashboard Update - Item {i}/100:")
            print(f"  Progress: {overall['completion_percent']:.1f}%")
            print(f"  Rate: {overall.get('processing_rate', 0):.2f} items/sec")
            print(f"  Conveyance: {conveyance.get('conveyance', 0):.3f}")
            print(f"  ETA: {overall.get('estimated_completion', 'Calculating...')}")

        time.sleep(0.05)  # Simulate processing time

    tracker.complete_step('main')
    monitor.cleanup()
    print("\nReal-time dashboard example completed.\n")


def example_alert_system():
    """
    Example 6: Performance Alert System

    Theory Connection: Implements Context degradation detection through
    threshold monitoring, enabling proactive system optimization before
    zero-propagation conditions occur (C = 0).
    """
    print("Example 6: Performance Alert System")
    print("=" * 40)

    # Create monitor with custom alert thresholds
    monitor = create_performance_monitor('alert_example')

    # Lower thresholds for demo purposes
    monitor.cpu_warning_threshold = 50.0
    monitor.cpu_critical_threshold = 80.0
    monitor.memory_warning_threshold = 60.0

    # Add custom alert callback
    def custom_alert_handler(alert_type: str, message: str):
        print(f"🚨 ALERT [{alert_type}]: {message}")

    monitor.add_alert_callback(custom_alert_handler)

    print("Starting alert monitoring...")
    monitor.start_monitoring()

    # Simulate processing that triggers alerts
    print("Simulating high resource usage...")

    # Create some CPU load (this is just simulation)
    start_time = time.time()
    while time.time() - start_time < 10:  # Run for 10 seconds
        # Light processing to potentially trigger alerts
        _ = sum(i * i for i in range(1000))
        time.sleep(0.1)

    monitor.stop_monitoring()
    monitor.cleanup()
    print("Alert system example completed.\n")


def run_all_examples():
    """Run all monitoring examples in sequence."""
    print("HADES Core Monitoring Examples")
    print("=" * 50)
    print("Demonstrating Conveyance Framework (C = (W·R·H/T)·Ctx^α) measurement")
    print("through practical monitoring implementations.\n")

    examples = [
        example_basic_performance_monitoring,
        example_progress_tracking,
        example_custom_metrics_collection,
        example_legacy_integration,
        example_real_time_dashboard,
        example_alert_system
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example failed: {e}\n")

    print("All monitoring examples completed!")
    print("\nKey Benefits Demonstrated:")
    print("• Unified monitoring across system components")
    print("• Automatic Conveyance Framework scoring")
    print("• Real-time Context coherence measurement")
    print("• Legacy compatibility with enhanced capabilities")
    print("• Proactive alerting and optimization feedback")


if __name__ == "__main__":
    run_all_examples()