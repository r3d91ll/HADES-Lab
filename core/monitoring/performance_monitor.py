"""
Performance Monitoring System
============================

Theory Connection - Information Reconstructionism:
Performance monitoring implements real-time measurement of system efficiency
across all Conveyance Framework dimensions. The monitor acts as a feedback
mechanism that enables continuous optimization of C = (W·R·H/T)·Ctx^α through
empirical measurement of processing rates, resource utilization, and quality metrics.

From Actor-Network Theory: The monitor serves as an "inscription device" that
translates distributed system performance into coherent representations,
enabling stakeholders to maintain alignment on system behavior and optimization targets.

This consolidates functionality from tools/arxiv/monitoring/ into the core framework.
"""

import time
import threading
import subprocess
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import deque

from .metrics_base import MetricsCollector, MetricType, MetricLevel, MetricValue

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """
    System resource snapshot.

    Theory Connection: Represents WHO dimension (system capability)
    measurements at a specific point in time.
    """
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GPUResources:
    """
    GPU resource snapshot.

    Theory Connection: Represents WHO dimension for GPU-accelerated
    processing components, crucial for embedding and extraction phases.
    """
    gpu_id: int
    utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProcessingPhase:
    """
    Processing phase tracking.

    Theory Connection: Represents temporal segmentation of WHERE
    (processing location) and WHAT (content transformation) dimensions.
    """
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    items_processed: int = 0
    items_total: Optional[int] = None
    errors_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[timedelta]:
        """Get phase duration."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def is_active(self) -> bool:
        """Check if phase is currently active."""
        return self.end_time is None

    @property
    def processing_rate(self) -> Optional[float]:
        """Calculate processing rate (items per second)."""
        if self.duration is None or self.duration.total_seconds() == 0:
            return None
        return self.items_processed / self.duration.total_seconds()

    @property
    def completion_percent(self) -> Optional[float]:
        """Calculate completion percentage."""
        if self.items_total is None or self.items_total == 0:
            return None
        return (self.items_processed / self.items_total) * 100


class PerformanceMonitor(MetricsCollector):
    """
    Comprehensive performance monitoring system.

    Theory Connection - Conveyance Framework Optimization:
    Implements continuous measurement and optimization of C = (W·R·H/T)·Ctx^α
    by tracking all framework dimensions:

    - WHERE (R): File system paths, processing locations, component positioning
    - WHAT (W): Content quality, extraction success rates, validation metrics
    - WHO (H): System resources, GPU utilization, worker efficiency
    - TIME (T): Processing latency, throughput rates, phase durations
    - Context (Ctx): Configuration coherence, error rates, system stability

    The monitor enables feedback-driven optimization by providing real-time
    visibility into system performance across all dimensions.
    """

    def __init__(self, component_name: str, log_dir: Optional[Path] = None):
        """
        Initialize performance monitor.

        Args:
            component_name: Name of the component being monitored
            log_dir: Directory for performance logs
        """
        super().__init__(component_name)

        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self._phases: Dict[str, ProcessingPhase] = {}
        self._resource_history: deque = deque(maxlen=1000)  # Keep last 1000 measurements
        self._gpu_history: Dict[int, deque] = {}  # Per-GPU history

        # Monitoring configuration
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 5.0  # seconds
        self._alert_callbacks: List[Callable] = []

        # Performance thresholds for alerts
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 85.0
        self.memory_critical_threshold = 95.0
        self.gpu_memory_warning_threshold = 90.0
        self.gpu_memory_critical_threshold = 95.0

        # Initialize built-in metrics
        self._register_builtin_metrics()

        logger.info(f"Performance monitor initialized for {component_name}")

    def _register_builtin_metrics(self) -> None:
        """Register built-in performance metrics."""
        # System resource metrics (WHO dimension)
        self.register_metric("cpu_percent", MetricType.GAUGE, "CPU utilization", "%",
                           {"who": "system_cpu"})
        self.register_metric("memory_percent", MetricType.GAUGE, "Memory utilization", "%",
                           {"who": "system_memory"})
        self.register_metric("memory_used_gb", MetricType.GAUGE, "Memory used", "GB",
                           {"who": "system_memory"})

        # GPU resource metrics (WHO dimension)
        self.register_metric("gpu_utilization", MetricType.GAUGE, "GPU utilization", "%",
                           {"who": "gpu_compute"})
        self.register_metric("gpu_memory_used", MetricType.GAUGE, "GPU memory used", "GB",
                           {"who": "gpu_memory"})
        self.register_metric("gpu_memory_percent", MetricType.GAUGE, "GPU memory utilization", "%",
                           {"who": "gpu_memory"})

        # Processing metrics (WHAT dimension)
        self.register_metric("items_processed", MetricType.COUNTER, "Total items processed", "items",
                           {"what": "processing_output"})
        self.register_metric("processing_errors", MetricType.COUNTER, "Processing errors", "errors",
                           {"what": "processing_quality"})
        self.register_metric("processing_rate", MetricType.GAUGE, "Processing rate", "items/sec",
                           {"what": "processing_efficiency"})

        # Time metrics (TIME dimension)
        self.register_metric("processing_latency", MetricType.HISTOGRAM, "Processing latency", "seconds",
                           {"time": "processing_duration"})
        self.register_metric("phase_duration", MetricType.TIMER, "Phase duration", "seconds",
                           {"time": "phase_execution"})

    def start_monitoring(self, interval: float = 5.0) -> None:
        """
        Start continuous performance monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self._monitor_interval = interval
        self._monitoring_active = True

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"{self.component_name}-monitor",
            daemon=True
        )
        self._monitor_thread.start()

        logger.info(f"Started performance monitoring with {interval}s interval")

    def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10.0)

        logger.info("Stopped performance monitoring")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.debug("Performance monitoring loop started")

        while self._monitoring_active:
            try:
                # Collect system resources
                system_resources = self._collect_system_resources()
                if system_resources:
                    self._record_system_resources(system_resources)

                # Collect GPU resources
                gpu_resources = self._collect_gpu_resources()
                for gpu in gpu_resources:
                    self._record_gpu_resources(gpu)

                # Check for alerts
                self._check_alerts(system_resources, gpu_resources)

                # Update processing rates for active phases
                self._update_processing_rates()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            # Sleep until next monitoring cycle
            time.sleep(self._monitor_interval)

        logger.debug("Performance monitoring loop stopped")

    def _collect_system_resources(self) -> Optional[SystemResources]:
        """Collect current system resource usage."""
        try:
            # CPU and memory information
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Load average (Unix-like systems)
            load_avg = []
            try:
                load_avg = list(psutil.getloadavg())
            except (AttributeError, OSError):
                # Windows or other systems without getloadavg
                pass

            return SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                load_average=load_avg
            )

        except Exception as e:
            logger.warning(f"Failed to collect system resources: {e}")
            return None

    def _collect_gpu_resources(self) -> List[GPUResources]:
        """Collect GPU resource usage using nvidia-smi."""
        gpu_resources = []

        try:
            # Query GPU information
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10, check=True)

            if not result.stdout:
                return gpu_resources

            # Parse GPU information
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    try:
                        gpu_id = int(parts[0])
                        utilization = int(parts[1])
                        memory_used = float(parts[2]) / 1024  # Convert MB to GB
                        memory_total = float(parts[3]) / 1024

                        # Optional fields (may be 'N/A')
                        temperature = None
                        power_draw = None
                        try:
                            if len(parts) > 4 and parts[4] != 'N/A':
                                temperature = float(parts[4])
                            if len(parts) > 5 and parts[5] != 'N/A':
                                power_draw = float(parts[5])
                        except (ValueError, IndexError):
                            pass

                        gpu_resources.append(GPUResources(
                            gpu_id=gpu_id,
                            utilization_percent=utilization,
                            memory_used_gb=memory_used,
                            memory_total_gb=memory_total,
                            temperature_c=temperature,
                            power_draw_w=power_draw
                        ))

                    except (ValueError, IndexError) as e:
                        logger.debug(f"Failed to parse GPU line '{line}': {e}")
                        continue

        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi query timed out")
        except subprocess.CalledProcessError as e:
            logger.debug(f"nvidia-smi failed: {e}")
        except FileNotFoundError:
            logger.debug("nvidia-smi not found (no NVIDIA GPUs or drivers)")

        return gpu_resources

    def _record_system_resources(self, resources: SystemResources) -> None:
        """Record system resource measurements."""
        self._resource_history.append(resources)

        # Update metrics
        self.gauge("cpu_percent", resources.cpu_percent)
        self.gauge("memory_percent", resources.memory_percent)
        self.gauge("memory_used_gb", resources.memory_used_gb)

    def _record_gpu_resources(self, resources: GPUResources) -> None:
        """Record GPU resource measurements."""
        # Initialize history for this GPU if needed
        if resources.gpu_id not in self._gpu_history:
            self._gpu_history[resources.gpu_id] = deque(maxlen=1000)

        self._gpu_history[resources.gpu_id].append(resources)

        # Update metrics with GPU labels
        labels = {"gpu_id": str(resources.gpu_id)}
        self.gauge("gpu_utilization", resources.utilization_percent, labels=labels)
        self.gauge("gpu_memory_used", resources.memory_used_gb, labels=labels)

        memory_percent = (resources.memory_used_gb / resources.memory_total_gb) * 100
        self.gauge("gpu_memory_percent", memory_percent, labels=labels)

    def _check_alerts(self, system_resources: Optional[SystemResources],
                     gpu_resources: List[GPUResources]) -> None:
        """Check for alert conditions and trigger callbacks."""
        alerts = []

        # System resource alerts
        if system_resources:
            if system_resources.cpu_percent >= self.cpu_critical_threshold:
                alerts.append(("cpu_critical", f"CPU usage {system_resources.cpu_percent:.1f}% >= {self.cpu_critical_threshold}%"))
            elif system_resources.cpu_percent >= self.cpu_warning_threshold:
                alerts.append(("cpu_warning", f"CPU usage {system_resources.cpu_percent:.1f}% >= {self.cpu_warning_threshold}%"))

            if system_resources.memory_percent >= self.memory_critical_threshold:
                alerts.append(("memory_critical", f"Memory usage {system_resources.memory_percent:.1f}% >= {self.memory_critical_threshold}%"))
            elif system_resources.memory_percent >= self.memory_warning_threshold:
                alerts.append(("memory_warning", f"Memory usage {system_resources.memory_percent:.1f}% >= {self.memory_warning_threshold}%"))

        # GPU resource alerts
        for gpu in gpu_resources:
            memory_percent = (gpu.memory_used_gb / gpu.memory_total_gb) * 100

            if memory_percent >= self.gpu_memory_critical_threshold:
                alerts.append(("gpu_memory_critical", f"GPU {gpu.gpu_id} memory {memory_percent:.1f}% >= {self.gpu_memory_critical_threshold}%"))
            elif memory_percent >= self.gpu_memory_warning_threshold:
                alerts.append(("gpu_memory_warning", f"GPU {gpu.gpu_id} memory {memory_percent:.1f}% >= {self.gpu_memory_warning_threshold}%"))

        # Trigger alert callbacks
        for alert_type, message in alerts:
            logger.warning(f"Performance alert: {message}")
            for callback in self._alert_callbacks:
                try:
                    callback(alert_type, message)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

    def _update_processing_rates(self) -> None:
        """Update processing rates for active phases."""
        for phase in self._phases.values():
            if phase.is_active and phase.processing_rate:
                self.gauge("processing_rate", phase.processing_rate,
                         labels={"phase": phase.name})

    def start_phase(self, name: str, total_items: Optional[int] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start a new processing phase.

        Theory Connection: Establishes temporal boundaries for WHERE
        and WHAT dimension measurements, enabling phase-specific optimization.

        Args:
            name: Phase name
            total_items: Expected total number of items to process
            metadata: Additional phase metadata
        """
        if name in self._phases and self._phases[name].is_active:
            logger.warning(f"Phase '{name}' already active")
            return

        phase = ProcessingPhase(
            name=name,
            start_time=datetime.utcnow(),
            items_total=total_items,
            metadata=metadata or {}
        )

        self._phases[name] = phase
        logger.info(f"Started processing phase: {name}")

    def update_phase_progress(self, name: str, items_processed: int,
                            errors_count: int = 0) -> None:
        """
        Update phase progress.

        Args:
            name: Phase name
            items_processed: Number of items processed
            errors_count: Number of errors encountered
        """
        if name not in self._phases:
            logger.warning(f"Unknown phase: {name}")
            return

        phase = self._phases[name]
        if not phase.is_active:
            logger.warning(f"Phase '{name}' is not active")
            return

        phase.items_processed = items_processed
        phase.errors_count = errors_count

        # Update metrics
        self.gauge("items_processed", items_processed, labels={"phase": name})
        if phase.processing_rate:
            self.gauge("processing_rate", phase.processing_rate, labels={"phase": name})

    def end_phase(self, name: str) -> Optional[ProcessingPhase]:
        """
        End a processing phase.

        Args:
            name: Phase name

        Returns:
            Completed phase information
        """
        if name not in self._phases:
            logger.warning(f"Unknown phase: {name}")
            return None

        phase = self._phases[name]
        if not phase.is_active:
            logger.warning(f"Phase '{name}' already ended")
            return phase

        phase.end_time = datetime.utcnow()

        # Record phase duration
        if phase.duration:
            self.histogram("phase_duration", phase.duration.total_seconds(),
                         labels={"phase": name})

        logger.info(f"Ended processing phase: {name} "
                   f"({phase.items_processed} items in {phase.duration})")

        return phase

    def get_phase_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get current phase status.

        Args:
            name: Phase name

        Returns:
            Phase status dictionary
        """
        if name not in self._phases:
            return None

        phase = self._phases[name]
        return {
            'name': phase.name,
            'is_active': phase.is_active,
            'start_time': phase.start_time.isoformat(),
            'end_time': phase.end_time.isoformat() if phase.end_time else None,
            'duration_seconds': phase.duration.total_seconds() if phase.duration else None,
            'items_processed': phase.items_processed,
            'items_total': phase.items_total,
            'completion_percent': phase.completion_percent,
            'processing_rate': phase.processing_rate,
            'errors_count': phase.errors_count,
            'metadata': phase.metadata
        }

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system resource status.

        Returns:
            System status dictionary
        """
        status = {
            'system': None,
            'gpus': [],
            'monitoring_active': self._monitoring_active
        }

        # Latest system resources
        if self._resource_history:
            latest = self._resource_history[-1]
            status['system'] = {
                'cpu_percent': latest.cpu_percent,
                'memory_percent': latest.memory_percent,
                'memory_used_gb': latest.memory_used_gb,
                'memory_total_gb': latest.memory_total_gb,
                'disk_usage_percent': latest.disk_usage_percent,
                'disk_free_gb': latest.disk_free_gb,
                'load_average': latest.load_average,
                'timestamp': latest.timestamp.isoformat()
            }

        # Latest GPU resources
        for gpu_id, history in self._gpu_history.items():
            if history:
                latest = history[-1]
                status['gpus'].append({
                    'gpu_id': latest.gpu_id,
                    'utilization_percent': latest.utilization_percent,
                    'memory_used_gb': latest.memory_used_gb,
                    'memory_total_gb': latest.memory_total_gb,
                    'memory_percent': (latest.memory_used_gb / latest.memory_total_gb) * 100,
                    'temperature_c': latest.temperature_c,
                    'power_draw_w': latest.power_draw_w,
                    'timestamp': latest.timestamp.isoformat()
                })

        return status

    def add_alert_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Add alert callback function.

        Args:
            callback: Function to call on alerts (alert_type, message)
        """
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable) -> bool:
        """
        Remove alert callback function.

        Args:
            callback: Callback function to remove

        Returns:
            True if callback was removed
        """
        try:
            self._alert_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def flush(self) -> None:
        """
        Flush metrics to log files.

        Theory Connection: Ensures Context persistence by writing
        performance measurements to durable storage.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Write summary to log file
        summary_file = self.log_dir / f"performance_summary_{self.component_name}_{timestamp}.json"
        try:
            summary = self.get_summary()
            summary['system_status'] = self.get_system_status()
            summary['phases'] = {name: self.get_phase_status(name)
                               for name in self._phases.keys()}

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)

            logger.debug(f"Performance summary written to {summary_file}")

        except Exception as e:
            logger.error(f"Failed to write performance summary: {e}")

        # Call parent flush
        super().flush()

    def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        self.stop_monitoring()
        self.flush()