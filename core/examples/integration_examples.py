"""
Core Framework Integration Examples
===================================

Theory Connection - Information Reconstructionism:
These integration examples demonstrate how configuration and monitoring systems
work together to optimize the Conveyance Framework C = (W·R·H/T)·Ctx^α across
complete processing pipelines. The examples show how Context coherence is
maintained through unified measurement and configuration management.

From Actor-Network Theory: Integration examples serve as "boundary objects"
that demonstrate stable relationships between configuration, monitoring, and
processing components, enabling coherent system behavior across distributed
network participants.
"""

import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Import core framework components
from core.config import (
    BaseConfig,
    get_config,
    register_config,
    ConfigScope,
    ProcessingConfig,
    StorageConfig
)

from core.monitoring import (
    create_performance_monitor,
    create_progress_tracker,
    PerformanceMonitor,
    ProgressTracker,
    MetricType,
    calculate_conveyance
)


class PipelineConfig(BaseConfig):
    """
    Complete pipeline configuration combining processing and storage.

    Theory Connection: Represents unified Context configuration that
    coordinates all Conveyance Framework dimensions across pipeline phases.
    """

    # Processing configuration
    workers: int = 8
    batch_size: int = 32
    chunk_size: int = 1024
    chunk_overlap: int = 200
    use_gpu: bool = True
    gpu_devices: List[int] = [0, 1]

    # Storage configuration
    host: str = "localhost"
    port: int = 8529
    database: str = "academy_store"
    staging_directory: str = "/tmp/pipeline_staging"

    # Quality thresholds
    min_success_rate: float = 0.95
    max_error_rate: float = 0.05

    # Performance targets
    target_throughput: float = 10.0  # items per second

    def validate_semantics(self) -> List[str]:
        """Validate pipeline configuration coherence."""
        errors = []

        # Validate worker/GPU relationship
        if self.use_gpu and len(self.gpu_devices) > 0:
            workers_per_gpu = self.workers / len(self.gpu_devices)
            if workers_per_gpu > 8:
                errors.append(f"Too many workers per GPU: {workers_per_gpu:.1f}")

        # Validate chunk relationships
        if self.chunk_overlap >= self.chunk_size:
            errors.append("Chunk overlap must be smaller than chunk size")

        # Validate quality thresholds
        if self.min_success_rate + self.max_error_rate > 1.0:
            errors.append("Success rate + error rate cannot exceed 100%")

        return errors


class IntegratedPipeline:
    """
    Example processing pipeline with integrated configuration and monitoring.

    Theory Connection: Demonstrates complete Conveyance Framework implementation
    with real-time Context measurement and adaptive optimization based on
    empirical performance feedback.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize integrated pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.name = "integrated_pipeline"

        # Initialize monitoring
        self.performance_monitor = create_performance_monitor(
            component_name=self.name,
            monitor_interval=1.0,
            enable_gpu_monitoring=config.use_gpu
        )

        self.progress_tracker = create_progress_tracker(
            name=f"{self.name}_progress",
            description="Integrated pipeline with configuration and monitoring"
        )

        # Initialize metrics
        self._setup_metrics()

        # Performance tracking
        self.start_time = None
        self.total_processed = 0
        self.total_errors = 0

    def _setup_metrics(self) -> None:
        """Setup pipeline-specific metrics."""
        # Register custom metrics for Conveyance Framework dimensions
        self.performance_monitor.register_metric(
            'config_coherence', MetricType.GAUGE,
            'Configuration context score', 'score',
            {'context': 'configuration'}
        )

        self.performance_monitor.register_metric(
            'pipeline_throughput', MetricType.GAUGE,
            'Pipeline throughput', 'items/sec',
            {'time': 'processing_rate'}
        )

        self.performance_monitor.register_metric(
            'quality_score', MetricType.GAUGE,
            'Processing quality score', 'score',
            {'what': 'processing_quality'}
        )

    async def run_pipeline(self, total_items: int = 100) -> Dict[str, Any]:
        """
        Run complete pipeline with monitoring and configuration integration.

        Theory Connection: Implements full Conveyance Framework optimization
        through real-time measurement and adaptive parameter adjustment.

        Args:
            total_items: Number of items to process

        Returns:
            Pipeline execution results with Conveyance metrics
        """
        print(f"Starting integrated pipeline with {total_items} items...")
        print(f"Configuration context score: {self.config.get_context_score():.3f}")

        # Start monitoring
        self.performance_monitor.start_monitoring()
        self.start_time = datetime.utcnow()

        # Record configuration coherence
        config_score = self.config.get_context_score()
        self.performance_monitor.gauge('config_coherence', config_score)

        # Setup pipeline phases based on configuration
        phases = [
            ('extraction', 'Extract content', total_items),
            ('processing', 'Process and chunk', total_items),
            ('embedding', 'Generate embeddings', total_items),
            ('storage', 'Store results', total_items)
        ]

        # Add progress steps
        for phase_id, phase_name, item_count in phases:
            self.progress_tracker.add_step(phase_id, phase_name, item_count)

        # Execute pipeline phases
        results = {}
        for phase_id, phase_name, item_count in phases:
            print(f"\nExecuting phase: {phase_name}")

            phase_result = await self._execute_phase(
                phase_id, phase_name, item_count
            )

            results[phase_id] = phase_result

            # Adaptive configuration adjustment based on performance
            await self._adapt_configuration(phase_result)

        # Calculate final results
        final_results = await self._finalize_results(results)

        # Stop monitoring
        self.performance_monitor.stop_monitoring()

        return final_results

    async def _execute_phase(self, phase_id: str, phase_name: str,
                           item_count: int) -> Dict[str, Any]:
        """Execute individual pipeline phase."""
        # Start phase monitoring
        self.performance_monitor.start_phase(phase_id, item_count)
        self.progress_tracker.start_step(phase_id)

        phase_start = time.time()
        processed = 0
        errors = 0

        # Simulate processing based on configuration
        batch_size = min(self.config.batch_size, item_count)
        processing_delay = 0.1 / self.config.workers  # Simulate parallel processing

        for i in range(0, item_count, batch_size):
            batch_end = min(i + batch_size, item_count)
            batch_items = batch_end - i

            # Simulate batch processing
            await asyncio.sleep(processing_delay)

            # Simulate occasional errors based on configuration targets
            batch_errors = 0
            if processed / max(1, item_count) > 0.5:  # Later items more likely to error
                if time.time() % 10 < 1:  # Roughly 10% chance
                    batch_errors = min(1, batch_items)

            processed += batch_items - batch_errors
            errors += batch_errors
            self.total_processed += batch_items - batch_errors
            self.total_errors += batch_errors

            # Update progress
            self.progress_tracker.update_step(
                phase_id,
                completed=processed,
                failed=errors
            )

            self.performance_monitor.update_phase_progress(
                phase_id,
                processed,
                errors
            )

            # Update metrics
            current_time = time.time()
            phase_duration = current_time - phase_start
            if phase_duration > 0:
                throughput = processed / phase_duration
                self.performance_monitor.gauge('pipeline_throughput', throughput)

                # Quality score based on error rate
                quality = 1.0 - (errors / max(1, processed + errors))
                self.performance_monitor.gauge('quality_score', quality)

        # Complete phase
        phase_duration = time.time() - phase_start
        self.performance_monitor.end_phase(phase_id)
        self.progress_tracker.complete_step(phase_id)

        return {
            'processed': processed,
            'errors': errors,
            'duration': phase_duration,
            'throughput': processed / phase_duration if phase_duration > 0 else 0
        }

    async def _adapt_configuration(self, phase_result: Dict[str, Any]) -> None:
        """
        Adapt configuration based on phase performance.

        Theory Connection: Implements feedback-driven optimization of
        Context parameters to maximize Conveyance score.
        """
        throughput = phase_result['throughput']
        error_rate = phase_result['errors'] / max(1, phase_result['processed'])

        # Adaptive adjustments
        if throughput < self.config.target_throughput * 0.8:
            # Throughput too low - reduce batch size for better parallelization
            if self.config.batch_size > 8:
                self.config.batch_size = max(8, self.config.batch_size - 4)
                print(f"  Adapted: Reduced batch size to {self.config.batch_size}")

        elif error_rate > self.config.max_error_rate:
            # Too many errors - reduce batch size and workers for stability
            if self.config.batch_size > 4:
                self.config.batch_size = max(4, self.config.batch_size - 2)
                print(f"  Adapted: Reduced batch size to {self.config.batch_size} for stability")

        elif throughput > self.config.target_throughput * 1.2 and error_rate < 0.01:
            # Performing well - can increase batch size
            if self.config.batch_size < 64:
                self.config.batch_size = min(64, self.config.batch_size + 4)
                print(f"  Adapted: Increased batch size to {self.config.batch_size}")

        # Update configuration coherence metric
        updated_score = self.config.get_context_score()
        self.performance_monitor.gauge('config_coherence', updated_score)

    async def _finalize_results(self, phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final pipeline results and Conveyance metrics."""
        total_duration = (datetime.utcnow() - self.start_time).total_seconds()

        # Calculate overall metrics
        overall_throughput = self.total_processed / total_duration
        overall_error_rate = self.total_errors / max(1, self.total_processed + self.total_errors)
        overall_quality = 1.0 - overall_error_rate

        # Get monitoring results
        performance_summary = self.performance_monitor.get_summary()
        progress_summary = self.progress_tracker.get_overall_progress()
        conveyance_metrics = self.progress_tracker.calculate_conveyance_metrics()

        return {
            'execution': {
                'total_processed': self.total_processed,
                'total_errors': self.total_errors,
                'duration_seconds': total_duration,
                'throughput': overall_throughput,
                'error_rate': overall_error_rate,
                'quality_score': overall_quality
            },
            'configuration': {
                'final_config': self.config.to_dict(),
                'context_score': self.config.get_context_score(),
                'adaptations_made': True
            },
            'monitoring': {
                'performance_metrics': performance_summary,
                'progress_metrics': progress_summary,
                'conveyance_framework': conveyance_metrics
            },
            'success': overall_quality >= self.config.min_success_rate
        }

    def cleanup(self):
        """Cleanup pipeline resources."""
        if self.performance_monitor:
            self.performance_monitor.cleanup()


async def example_integrated_pipeline():
    """
    Example: Complete integrated pipeline with configuration and monitoring.

    Theory Connection: Demonstrates full Conveyance Framework implementation
    with real-time Context measurement and adaptive optimization.
    """
    print("Integrated Pipeline Example")
    print("=" * 50)

    # Register pipeline configuration
    register_config('pipeline', PipelineConfig, ConfigScope.MODULE)

    # Get configuration with some overrides
    config = get_config('pipeline',
                       workers=4,
                       batch_size=16,
                       target_throughput=8.0)

    print(f"Pipeline configuration loaded:")
    print(f"  Workers: {config.workers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Use GPU: {config.use_gpu}")
    print(f"  Context score: {config.get_context_score():.3f}")

    # Create and run pipeline
    pipeline = IntegratedPipeline(config)

    try:
        results = await pipeline.run_pipeline(total_items=80)

        print("\nPipeline Results:")
        print("=" * 30)
        execution = results['execution']
        print(f"Processed: {execution['total_processed']} items")
        print(f"Errors: {execution['total_errors']}")
        print(f"Duration: {execution['duration_seconds']:.1f} seconds")
        print(f"Throughput: {execution['throughput']:.2f} items/sec")
        print(f"Quality: {execution['quality_score']:.3f}")

        conveyance = results['monitoring']['conveyance_framework']
        print(f"\nConveyance Framework Scores:")
        print(f"  WHERE: {conveyance.get('where', 0):.3f}")
        print(f"  WHAT:  {conveyance.get('what', 0):.3f}")
        print(f"  WHO:   {conveyance.get('who', 0):.3f}")
        print(f"  TIME:  {conveyance.get('time', 0):.3f}")
        print(f"  Context: {conveyance.get('context', 0):.3f}")
        print(f"  Overall: {conveyance.get('conveyance', 0):.3f}")

        success = results['success']
        print(f"\nPipeline Success: {'✓' if success else '✗'}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")

    finally:
        pipeline.cleanup()

    print("\nIntegrated pipeline example completed.\n")


def example_configuration_monitoring_sync():
    """
    Example: Configuration and monitoring synchronization.

    Theory Connection: Shows how configuration changes trigger
    monitoring adjustments to maintain Context coherence.
    """
    print("Configuration-Monitoring Synchronization Example")
    print("=" * 55)

    # Create base configuration
    config = ProcessingConfig(workers=4, batch_size=16)
    monitor = create_performance_monitor('sync_example')

    print(f"Initial configuration:")
    print(f"  Workers: {config.workers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Context score: {config.get_context_score():.3f}")

    # Start monitoring
    monitor.start_monitoring()

    # Record initial configuration metrics
    monitor.gauge('config_workers', config.workers)
    monitor.gauge('config_batch_size', config.batch_size)
    monitor.gauge('config_context_score', config.get_context_score())

    # Simulate configuration changes
    configurations = [
        ProcessingConfig(workers=8, batch_size=32),
        ProcessingConfig(workers=16, batch_size=8),
        ProcessingConfig(workers=6, batch_size=24)
    ]

    for i, new_config in enumerate(configurations, 1):
        print(f"\nConfiguration update {i}:")
        print(f"  Workers: {config.workers} → {new_config.workers}")
        print(f"  Batch size: {config.batch_size} → {new_config.batch_size}")

        # Update configuration
        config = new_config

        # Update monitoring metrics
        monitor.gauge('config_workers', config.workers)
        monitor.gauge('config_batch_size', config.batch_size)
        monitor.gauge('config_context_score', config.get_context_score())

        print(f"  Context score: {config.get_context_score():.3f}")

        time.sleep(1)  # Simulate processing time

    # Get final monitoring summary
    summary = monitor.get_summary()
    print(f"\nMonitoring captured {summary['metrics_count']} configuration changes")

    monitor.cleanup()
    print("Configuration-monitoring sync example completed.\n")


async def run_all_integration_examples():
    """Run all integration examples."""
    print("HADES Core Framework Integration Examples")
    print("=" * 60)
    print("Demonstrating unified configuration and monitoring for")
    print("complete Conveyance Framework optimization.\n")

    examples = [
        example_integrated_pipeline,
        lambda: example_configuration_monitoring_sync()
    ]

    for example_func in examples:
        try:
            if asyncio.iscoroutinefunction(example_func):
                await example_func()
            else:
                example_func()
        except Exception as e:
            print(f"Integration example failed: {e}\n")

    print("All integration examples completed!")
    print("\nKey Integration Benefits Demonstrated:")
    print("• Unified configuration and monitoring framework")
    print("• Real-time Conveyance Framework measurement")
    print("• Adaptive configuration based on performance feedback")
    print("• Context coherence maintenance across pipeline phases")
    print("• Theory-connected empirical optimization")


if __name__ == "__main__":
    asyncio.run(run_all_integration_examples())