"""
Migration Guide for Core Monitoring Integration
===============================================

Theory Connection - Information Reconstructionism:
This migration guide facilitates the transition from distributed monitoring
approaches to unified Conveyance Framework measurement. It preserves existing
WHERE positioning while enhancing Context coherence through standardized
measurement patterns across all system components.

From Actor-Network Theory: Acts as a "translation guide" that enables
existing monitoring components to maintain their network relationships
while adopting enhanced measurement capabilities for Context amplification.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

# Example migrations for common patterns
logger = logging.getLogger(__name__)


class MonitoringMigrationGuide:
    """
    Guide for migrating existing monitoring code to core framework.

    Theory Connection: Provides systematic approach to maintaining
    Context coherence while upgrading monitoring infrastructure.
    """

    @staticmethod
    def migrate_gpu_status_function():
        """
        Example migration for GPU status collection.

        OLD PATTERN (tools/arxiv/monitoring/monitor_phased.py):
        ```python
        def get_gpu_status():
            try:
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"
                ], capture_output=True, text=True, check=True, timeout=5)

                gpus = []
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpus.append({
                            'id': int(parts[0]),
                            'util': int(parts[1]),
                            'mem_used': int(parts[2]) / 1024,
                            'mem_total': int(parts[3]) / 1024
                        })
                return gpus
            except Exception:
                return []
        ```

        NEW PATTERN (using core monitoring):
        ```python
        from core.monitoring import create_performance_monitor

        monitor = create_performance_monitor('my_component')
        monitor.start_monitoring()

        # GPU status is automatically collected
        system_status = monitor.get_system_status()
        gpus = system_status['gpus']  # Already parsed and structured
        ```
        """
        return {
            'old_pattern': 'Manual subprocess calls with custom parsing',
            'new_pattern': 'Automatic collection with structured data',
            'benefits': [
                'Automatic error handling and retries',
                'Standardized data format across components',
                'Integration with Conveyance Framework metrics',
                'Built-in alerting and threshold monitoring'
            ]
        }

    @staticmethod
    def migrate_progress_tracking():
        """
        Example migration for progress tracking.

        OLD PATTERN (manual progress variables):
        ```python
        processed_count = 0
        total_count = 1000
        start_time = datetime.now()

        for item in items:
            # Process item
            processed_count += 1

            if processed_count % 10 == 0:
                elapsed = datetime.now() - start_time
                rate = processed_count / elapsed.total_seconds()
                print(f"Progress: {processed_count}/{total_count} ({rate:.1f}/sec)")
        ```

        NEW PATTERN (using ProgressTracker):
        ```python
        from core.monitoring import create_progress_tracker

        tracker = create_progress_tracker('processing', 'Process items')
        tracker.add_step('main', 'Processing items', total_count)
        tracker.start_step('main')

        for item in items:
            # Process item
            tracker.update_step('main', processed_count)

            # Automatic rate calculation, ETA, and reporting

        tracker.complete_step('main')
        conveyance_metrics = tracker.calculate_conveyance_metrics()
        ```
        """
        return {
            'old_pattern': 'Manual counters and rate calculations',
            'new_pattern': 'Structured progress tracking with metrics',
            'benefits': [
                'Automatic ETA calculation',
                'Built-in error tracking',
                'Conveyance Framework scoring',
                'Persistent progress state',
                'Multi-step pipeline support'
            ]
        }

    @staticmethod
    def migrate_configuration_loading():
        """
        Example migration for configuration loading.

        OLD PATTERN (manual YAML loading):
        ```python
        import yaml
        import os

        def load_config():
            config_path = "config.yaml"
            if not os.path.exists(config_path):
                config_path = "default_config.yaml"

            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Override with environment variables
            if os.getenv("WORKERS"):
                config['workers'] = int(os.getenv("WORKERS"))

            return config
        ```

        NEW PATTERN (using ConfigManager):
        ```python
        from core.config import get_config, ProcessingConfig

        # Automatic hierarchical loading with validation
        config = get_config('processing', workers=8)  # Override example

        # Or for custom configuration classes
        from core.config import config_manager
        config_manager.register('my_config', MyConfigClass)
        config = get_config('my_config')
        ```
        """
        return {
            'old_pattern': 'Manual file loading with ad-hoc env var handling',
            'new_pattern': 'Hierarchical configuration with validation',
            'benefits': [
                'Automatic source priority resolution',
                'Schema validation and error reporting',
                'Caching and performance optimization',
                'Environment variable integration',
                'Configuration versioning and tracking'
            ]
        }

    @staticmethod
    def create_compatibility_wrapper(existing_monitor_class):
        """
        Create compatibility wrapper for existing monitor classes.

        Args:
            existing_monitor_class: Existing monitor class to wrap

        Returns:
            Enhanced monitor class with core framework integration
        """
        from core.monitoring import LegacyMonitorInterface

        class EnhancedMonitor(existing_monitor_class, LegacyMonitorInterface):
            """Enhanced monitor with core framework integration."""

            def __init__(self, *args, **kwargs):
                # Initialize both parent classes
                existing_monitor_class.__init__(self, *args, **kwargs)
                LegacyMonitorInterface.__init__(
                    self,
                    component_name=getattr(self, 'component_name', 'enhanced_monitor')
                )

            def run(self, *args, **kwargs):
                """Override run method to start enhanced monitoring."""
                # Start enhanced monitoring
                self.start_monitoring()

                try:
                    # Call original run method
                    return super().run(*args, **kwargs)
                finally:
                    # Cleanup enhanced monitoring
                    self.cleanup()

        return EnhancedMonitor

    @staticmethod
    def generate_migration_report(source_dir: Path) -> Dict[str, Any]:
        """
        Generate migration report for existing monitoring code.

        Theory Connection: Analyzes existing WHERE positioning and
        Context measurement patterns to recommend optimal migration path.

        Args:
            source_dir: Directory containing existing monitoring code

        Returns:
            Migration report with recommendations
        """
        report = {
            'source_directory': str(source_dir),
            'analysis_timestamp': str(datetime.utcnow()),
            'files_analyzed': [],
            'patterns_found': {
                'gpu_monitoring': [],
                'progress_tracking': [],
                'configuration_loading': [],
                'custom_metrics': []
            },
            'recommendations': [],
            'migration_complexity': 'unknown'
        }

        if not source_dir.exists():
            report['error'] = f"Source directory does not exist: {source_dir}"
            return report

        # Analyze Python files
        for py_file in source_dir.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                report['files_analyzed'].append(str(py_file))

                # Check for common patterns
                if 'nvidia-smi' in content:
                    report['patterns_found']['gpu_monitoring'].append(str(py_file))

                if any(pattern in content for pattern in ['progress', 'processed', 'rate']):
                    report['patterns_found']['progress_tracking'].append(str(py_file))

                if any(pattern in content for pattern in ['yaml.load', 'json.load', 'config']):
                    report['patterns_found']['configuration_loading'].append(str(py_file))

                if any(pattern in content for pattern in ['metric', 'counter', 'gauge', 'timer']):
                    report['patterns_found']['custom_metrics'].append(str(py_file))

            except Exception as e:
                logger.warning(f"Failed to analyze {py_file}: {e}")

        # Generate recommendations
        total_patterns = sum(len(patterns) for patterns in report['patterns_found'].values())

        if total_patterns == 0:
            report['migration_complexity'] = 'minimal'
            report['recommendations'].append(
                "No major monitoring patterns found. "
                "Consider adding core monitoring for enhanced capabilities."
            )
        elif total_patterns < 5:
            report['migration_complexity'] = 'low'
            report['recommendations'].append(
                "Few monitoring patterns found. "
                "Gradual migration recommended using compatibility wrappers."
            )
        elif total_patterns < 15:
            report['migration_complexity'] = 'medium'
            report['recommendations'].append(
                "Multiple monitoring patterns found. "
                "Plan phased migration starting with most critical components."
            )
        else:
            report['migration_complexity'] = 'high'
            report['recommendations'].append(
                "Extensive monitoring code found. "
                "Consider comprehensive migration plan with thorough testing."
            )

        # Specific recommendations based on patterns
        if report['patterns_found']['gpu_monitoring']:
            report['recommendations'].append(
                "Replace manual GPU monitoring with PerformanceMonitor for "
                "better error handling and Conveyance Framework integration."
            )

        if report['patterns_found']['progress_tracking']:
            report['recommendations'].append(
                "Migrate progress tracking to ProgressTracker for automatic "
                "ETA calculation and Context coherence measurement."
            )

        if report['patterns_found']['configuration_loading']:
            report['recommendations'].append(
                "Adopt ConfigManager for hierarchical configuration loading "
                "with validation and environment variable integration."
            )

        return report


def print_migration_examples():
    """Print common migration examples for reference."""
    print("HADES Core Monitoring Migration Examples")
    print("=" * 50)

    guide = MonitoringMigrationGuide()

    examples = [
        ('GPU Status Collection', guide.migrate_gpu_status_function()),
        ('Progress Tracking', guide.migrate_progress_tracking()),
        ('Configuration Loading', guide.migrate_configuration_loading())
    ]

    for title, example in examples:
        print(f"\n{title}")
        print("-" * len(title))
        print(f"Old Pattern: {example['old_pattern']}")
        print(f"New Pattern: {example['new_pattern']}")
        print("Benefits:")
        for benefit in example['benefits']:
            print(f"  • {benefit}")


if __name__ == "__main__":
    # Print examples when run directly
    print_migration_examples()

    # Example migration report
    from datetime import datetime

    print("\n\nExample Migration Analysis")
    print("=" * 30)

    guide = MonitoringMigrationGuide()
    report = guide.generate_migration_report(Path("tools/arxiv/monitoring"))

    print(f"Files analyzed: {len(report['files_analyzed'])}")
    print(f"Migration complexity: {report['migration_complexity']}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")