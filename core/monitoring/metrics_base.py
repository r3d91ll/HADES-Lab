"""
Base Metrics Collection Interface
=================================

Theory Connection - Information Reconstructionism:
The metrics system implements real-time measurement of the Conveyance Framework
C = (W·R·H/T)·Ctx^α across all system dimensions. Metrics collection represents
the feedback mechanism that enables system optimization through empirical
measurement of Context amplification effects.

From Actor-Network Theory: Metrics act as "immutable mobiles" that transport
performance information across network boundaries, enabling distributed
components to maintain coherent understanding of system state.

Key Theoretical Mappings:
- WHERE (R): Spatial metrics (file paths, network topology, component positioning)
- WHAT (W): Content quality metrics (extraction success, embedding quality)
- WHO (H): Agent capability metrics (GPU utilization, worker efficiency)
- TIME (T): Temporal metrics (latency, throughput, processing rates)
- Context (Ctx): Coherence metrics (validation success, error rates, semantic consistency)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Union[int, float])


class MetricType(Enum):
    """
    Types of metrics collected.

    Theory Connection: Each type corresponds to different dimensions
    of the Conveyance Framework, enabling targeted optimization.
    """
    COUNTER = "counter"         # Cumulative values (papers processed)
    GAUGE = "gauge"             # Point-in-time values (GPU memory usage)
    HISTOGRAM = "histogram"     # Value distributions (processing times)
    TIMER = "timer"             # Duration measurements (extraction time)
    RATE = "rate"               # Rates over time (papers per minute)


class MetricLevel(Enum):
    """Metric importance levels for filtering and alerting."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class MetricValue:
    """
    Individual metric value with metadata.

    Theory Connection: Encapsulates measurement of specific Conveyance
    Framework dimensions with temporal and spatial context.
    """
    value: Union[int, float, str, bool]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    level: MetricLevel = MetricLevel.INFO

    def __post_init__(self):
        """Ensure timestamp is set."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'metadata': self.metadata,
            'level': self.level.name
        }


@dataclass
class MetricSeries:
    """
    Time series of metric values.

    Theory Connection: Represents temporal evolution of Conveyance
    dimensions, enabling analysis of Context amplification over time.
    """
    name: str
    metric_type: MetricType
    values: List[MetricValue] = field(default_factory=list)
    description: str = ""
    unit: str = ""
    dimensions: Dict[str, str] = field(default_factory=dict)

    def add_value(self, value: Union[int, float, str, bool],
                  timestamp: Optional[datetime] = None,
                  labels: Optional[Dict[str, str]] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  level: MetricLevel = MetricLevel.INFO) -> None:
        """
        Add value to metric series.

        Args:
            value: Metric value
            timestamp: Value timestamp (defaults to now)
            labels: Metric labels
            metadata: Additional metadata
            level: Metric importance level
        """
        metric_value = MetricValue(
            value=value,
            timestamp=timestamp or datetime.utcnow(),
            labels=labels or {},
            metadata=metadata or {},
            level=level
        )
        self.values.append(metric_value)

    def get_latest(self) -> Optional[MetricValue]:
        """Get the most recent value."""
        return self.values[-1] if self.values else None

    def get_window(self, window_size: timedelta) -> List[MetricValue]:
        """
        Get values within time window.

        Args:
            window_size: Time window duration

        Returns:
            Values within the window
        """
        if not self.values:
            return []

        cutoff = datetime.utcnow() - window_size
        return [v for v in self.values if v.timestamp >= cutoff]

    def calculate_rate(self, window_size: timedelta = timedelta(minutes=1)) -> Optional[float]:
        """
        Calculate rate over time window.

        Args:
            window_size: Time window for rate calculation

        Returns:
            Rate per second or None if insufficient data
        """
        window_values = self.get_window(window_size)
        if len(window_values) < 2:
            return None

        # Calculate total change and time span
        first_value = window_values[0]
        last_value = window_values[-1]

        if not isinstance(first_value.value, (int, float)) or not isinstance(last_value.value, (int, float)):
            return None

        value_change = last_value.value - first_value.value
        time_span = (last_value.timestamp - first_value.timestamp).total_seconds()

        return value_change / time_span if time_span > 0 else None

    def get_statistics(self, window_size: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Calculate statistics for metric series.

        Args:
            window_size: Time window (all values if None)

        Returns:
            Statistics dictionary
        """
        values = self.get_window(window_size) if window_size else self.values
        if not values:
            return {}

        numeric_values = [v.value for v in values if isinstance(v.value, (int, float))]
        if not numeric_values:
            return {
                'count': len(values),
                'latest': values[-1].value if values else None
            }

        return {
            'count': len(values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': sum(numeric_values) / len(numeric_values),
            'latest': values[-1].value,
            'total': sum(numeric_values)
        }


class MetricsCollector(ABC):
    """
    Abstract base class for metrics collection.

    Theory Connection - Conveyance Framework Implementation:
    The collector optimizes C = (W·R·H/T)·Ctx^α by providing efficient
    measurement infrastructure that minimizes TIME overhead while
    maximizing Context coherence through structured data collection.

    Subclasses implement specific storage backends (file, database, network)
    while maintaining consistent interface for Context measurement.
    """

    def __init__(self, component_name: str):
        """
        Initialize metrics collector.

        Args:
            component_name: Name of the component being monitored
        """
        self.component_name = component_name
        self._series: Dict[str, MetricSeries] = {}
        self._lock = threading.RLock()
        self._active_timers: Dict[str, datetime] = {}
        self.start_time = datetime.utcnow()

    @abstractmethod
    def flush(self) -> None:
        """
        Flush metrics to storage backend.

        Theory Connection: Ensures Context coherence by persisting
        measurements across system boundaries and time periods.
        """
        pass

    def register_metric(self, name: str, metric_type: MetricType,
                       description: str = "", unit: str = "",
                       dimensions: Optional[Dict[str, str]] = None) -> None:
        """
        Register a new metric series.

        Theory Connection: Establishes WHERE positioning in metric
        namespace and defines semantic relationships for Context coherence.

        Args:
            name: Metric name
            metric_type: Type of metric
            description: Human-readable description
            unit: Measurement unit
            dimensions: Conveyance Framework dimensions this metric measures
        """
        with self._lock:
            if name in self._series:
                logger.warning(f"Metric '{name}' already registered, skipping")
                return

            self._series[name] = MetricSeries(
                name=name,
                metric_type=metric_type,
                description=description,
                unit=unit,
                dimensions=dimensions or {}
            )

        logger.debug(f"Registered metric: {name} ({metric_type.value})")

    def increment(self, name: str, value: Union[int, float] = 1,
                  labels: Optional[Dict[str, str]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Increment a counter metric.

        Theory Connection: Measures cumulative progress in WHAT dimension
        (papers processed, chunks embedded, validations passed).

        Args:
            name: Metric name
            value: Amount to increment
            labels: Metric labels
            metadata: Additional metadata
        """
        with self._lock:
            if name not in self._series:
                self.register_metric(name, MetricType.COUNTER)

            series = self._series[name]
            current_value = 0
            if series.values:
                latest = series.values[-1]
                if isinstance(latest.value, (int, float)):
                    current_value = latest.value

            series.add_value(
                value=current_value + value,
                labels=labels,
                metadata=metadata
            )

    def gauge(self, name: str, value: Union[int, float],
              labels: Optional[Dict[str, str]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Set a gauge metric value.

        Theory Connection: Measures instantaneous system state across
        WHO (GPU utilization) and TIME (current latency) dimensions.

        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
            metadata: Additional metadata
        """
        with self._lock:
            if name not in self._series:
                self.register_metric(name, MetricType.GAUGE)

            self._series[name].add_value(
                value=value,
                labels=labels,
                metadata=metadata
            )

    def histogram(self, name: str, value: Union[int, float],
                  labels: Optional[Dict[str, str]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add value to histogram metric.

        Theory Connection: Measures distribution of Conveyance dimensions,
        enabling analysis of Context amplification variability.

        Args:
            name: Metric name
            value: Value to add to histogram
            labels: Metric labels
            metadata: Additional metadata
        """
        with self._lock:
            if name not in self._series:
                self.register_metric(name, MetricType.HISTOGRAM)

            self._series[name].add_value(
                value=value,
                labels=labels,
                metadata=metadata
            )

    def timer_start(self, name: str) -> None:
        """
        Start a timer.

        Theory Connection: Begins measurement of TIME dimension
        for specific processing operations.

        Args:
            name: Timer name
        """
        with self._lock:
            self._active_timers[name] = datetime.utcnow()

    def timer_end(self, name: str,
                  labels: Optional[Dict[str, str]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        End a timer and record duration.

        Theory Connection: Completes TIME dimension measurement,
        contributing to overall Conveyance calculation.

        Args:
            name: Timer name
            labels: Metric labels
            metadata: Additional metadata

        Returns:
            Duration in seconds or None if timer wasn't started
        """
        with self._lock:
            if name not in self._active_timers:
                logger.warning(f"Timer '{name}' not started")
                return None

            start_time = self._active_timers.pop(name)
            duration = (datetime.utcnow() - start_time).total_seconds()

            if name not in self._series:
                self.register_metric(name, MetricType.TIMER, unit="seconds")

            self._series[name].add_value(
                value=duration,
                labels=labels,
                metadata=metadata
            )

            return duration

    def record_error(self, error: str, error_type: str = "general",
                     labels: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an error occurrence.

        Theory Connection: Measures Context degradation events that
        can trigger zero-propagation (C = 0) in Conveyance Framework.

        Args:
            error: Error message
            error_type: Error category
            labels: Metric labels
            metadata: Additional metadata
        """
        error_labels = labels or {}
        error_labels['error_type'] = error_type

        error_metadata = metadata or {}
        error_metadata['error_message'] = error

        self.increment(
            name='errors_total',
            labels=error_labels,
            metadata=error_metadata
        )

        # Record as gauge for latest error
        self.gauge(
            name='last_error',
            value=1,
            labels=error_labels,
            metadata=error_metadata
        )

    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """
        Get metric series by name.

        Args:
            name: Metric name

        Returns:
            Metric series or None if not found
        """
        with self._lock:
            return self._series.get(name)

    def list_metrics(self) -> List[str]:
        """
        List all registered metric names.

        Returns:
            List of metric names
        """
        with self._lock:
            return list(self._series.keys())

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.

        Theory Connection: Provides aggregate view of Context coherence
        across all Conveyance Framework dimensions.

        Returns:
            Metrics summary dictionary
        """
        with self._lock:
            summary = {
                'component': self.component_name,
                'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
                'metrics_count': len(self._series),
                'active_timers': len(self._active_timers),
                'metrics': {}
            }

            for name, series in self._series.items():
                summary['metrics'][name] = {
                    'type': series.metric_type.value,
                    'description': series.description,
                    'unit': series.unit,
                    'dimensions': series.dimensions,
                    'values_count': len(series.values),
                    'latest': series.get_latest().value if series.get_latest() else None,
                    'statistics': series.get_statistics(timedelta(minutes=5))
                }

            return summary

    def calculate_conveyance_score(self) -> Dict[str, float]:
        """
        Calculate Conveyance Framework scores from collected metrics.

        Theory Connection: Implements C = (W·R·H/T)·Ctx^α calculation
        using empirical measurements from system operation.

        Returns:
            Dictionary with dimension scores and overall conveyance
        """
        # Extract dimension-specific metrics
        where_metrics = [s for s in self._series.values() if 'where' in s.dimensions]
        what_metrics = [s for s in self._series.values() if 'what' in s.dimensions]
        who_metrics = [s for s in self._series.values() if 'who' in s.dimensions]
        time_metrics = [s for s in self._series.values() if 'time' in s.dimensions]

        # Calculate dimension scores (0.0 to 1.0)
        def calculate_dimension_score(metrics: List[MetricSeries]) -> float:
            if not metrics:
                return 0.0

            scores = []
            for metric in metrics:
                latest = metric.get_latest()
                if latest and isinstance(latest.value, (int, float)):
                    # Normalize based on metric type and expected ranges
                    normalized_score = min(1.0, max(0.0, latest.value / 100))  # Basic normalization
                    scores.append(normalized_score)

            return sum(scores) / len(scores) if scores else 0.0

        # Calculate individual dimensions
        where_score = calculate_dimension_score(where_metrics)
        what_score = calculate_dimension_score(what_metrics)
        who_score = calculate_dimension_score(who_metrics)
        time_score = calculate_dimension_score(time_metrics)

        # Context score (equal weights: wL=wI=wA=wG=0.25)
        context_score = 0.25 * (where_score + what_score + who_score + time_score)

        # Alpha (super-linear amplification exponent)
        alpha = 1.8  # Default value in [1.5, 2.0] range

        # Calculate conveyance (efficiency view)
        if time_score > 0:
            conveyance_score = (where_score * what_score * who_score / time_score) * (context_score ** alpha)
        else:
            conveyance_score = 0.0  # Zero-propagation

        return {
            'where': where_score,
            'what': what_score,
            'who': who_score,
            'time': time_score,
            'context': context_score,
            'alpha': alpha,
            'conveyance': conveyance_score
        }