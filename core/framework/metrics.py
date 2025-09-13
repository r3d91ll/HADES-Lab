"""
Metrics Collection
==================

File-based metrics collection system.
Future: Can export to Prometheus, StatsD, etc.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


class MetricsCollector:
    """
    Collect metrics to files, not database.

    Features:
    - Counter metrics (incremental)
    - Timer metrics (duration tracking)
    - Gauge metrics (point-in-time values)
    - File-based storage (no DB pollution)
    - Thread-safe operations
    """

    def __init__(self, processor_name: str):
        """
        Initialize metrics collector.

        Args:
            processor_name: Name of the processor
        """
        self.processor_name = processor_name
        self.metrics = {
            'processor': processor_name,
            'start_time': time.time(),
            'counters': {},
            'timers': {},
            'gauges': {},
            'errors': []
        }
        self.timers_start = {}
        self.lock = Lock()

        # Ensure metrics directory exists
        self.metrics_dir = Path(__file__).parent.parent / "logs"
        self.metrics_dir.mkdir(exist_ok=True)

    def increment(self, metric: str, value: int = 1):
        """
        Increment a counter metric.

        Args:
            metric: Metric name
            value: Amount to increment
        """
        with self.lock:
            if metric not in self.metrics['counters']:
                self.metrics['counters'][metric] = 0
            self.metrics['counters'][metric] += value

    def decrement(self, metric: str, value: int = 1):
        """
        Decrement a counter metric.

        Args:
            metric: Metric name
            value: Amount to decrement
        """
        self.increment(metric, -value)

    def timer_start(self, metric: str):
        """
        Start a timer.

        Args:
            metric: Timer name
        """
        with self.lock:
            self.timers_start[metric] = time.time()

    def timer_end(self, metric: str):
        """
        End a timer and record duration.

        Args:
            metric: Timer name
        """
        with self.lock:
            if metric in self.timers_start:
                duration = time.time() - self.timers_start[metric]
                if metric not in self.metrics['timers']:
                    self.metrics['timers'][metric] = []
                self.metrics['timers'][metric].append(duration)
                del self.timers_start[metric]

    def gauge(self, metric: str, value: float):
        """
        Set a gauge metric.

        Args:
            metric: Metric name
            value: Gauge value
        """
        with self.lock:
            self.metrics['gauges'][metric] = value

    def record_error(self, error: str):
        """
        Record an error.

        Args:
            error: Error message
        """
        with self.lock:
            self.metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(error)
            })

    def get_summary(self) -> dict[str, Any]:
        """
        Get metrics summary.

        Returns:
            Dictionary of metrics
        """
        with self.lock:
            summary = self.metrics.copy()

            # Calculate timer statistics
            timer_stats = {}
            for timer_name, durations in self.metrics['timers'].items():
                if durations:
                    timer_stats[timer_name] = {
                        'count': len(durations),
                        'total': sum(durations),
                        'average': sum(durations) / len(durations),
                        'min': min(durations),
                        'max': max(durations)
                    }
            summary['timer_stats'] = timer_stats

            # Add current runtime
            summary['runtime'] = time.time() - self.metrics['start_time']

            return summary

    def flush(self):
        """Write metrics to file (not database!)."""
        with self.lock:
            self.metrics['end_time'] = time.time()
            self.metrics['duration'] = self.metrics['end_time'] - self.metrics['start_time']

            # Calculate timer statistics
            timer_stats = {}
            for timer_name, durations in self.metrics['timers'].items():
                if durations:
                    timer_stats[timer_name] = {
                        'count': len(durations),
                        'total': sum(durations),
                        'average': sum(durations) / len(durations),
                        'min': min(durations),
                        'max': max(durations)
                    }

            # Write to metrics file (append mode, JSONL format)
            try:
                metrics_file = self.metrics_dir / "metrics.jsonl"
                # Ensure directory exists
                metrics_file.parent.mkdir(parents=True, exist_ok=True)

                with metrics_file.open('a') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'processor': self.processor_name,
                        'duration': self.metrics['duration'],
                        'counters': self.metrics['counters'],
                        'timer_stats': timer_stats,
                        'gauges': self.metrics['gauges'],
                        'errors': self.metrics['errors']
                    }, f)
                    f.write('\n')
            except OSError as e:
                # Log but don't crash - metrics are important but not critical
                import sys
                print(f"Warning: Failed to save metrics for {self.processor_name}: {e}", file=sys.stderr)
            except json.JSONEncodeError as e:
                # Handle JSON encoding errors separately
                import sys
                print(f"Warning: Failed to encode metrics for {self.processor_name}: {e}", file=sys.stderr)
