"""
Backward Compatibility Monitor
==============================

Theory Connection - Information Reconstructionism:
This module provides backward compatibility with existing monitoring tools
while leveraging the new core monitoring infrastructure. It acts as a
translation layer that preserves existing interfaces while enabling
gradual migration to the unified monitoring system.

From Actor-Network Theory: Serves as a "boundary object" that maintains
stable relationships with existing tools while introducing new capabilities
for Context coherence measurement and Conveyance Framework optimization.
"""

import os
import sys
import json
import time
import sqlite3
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import threading

from .performance_monitor import PerformanceMonitor
from .progress_tracker import ProgressTracker, ProgressState

logger = logging.getLogger(__name__)


class LegacyMonitorInterface:
    """
    Compatibility interface for existing monitoring tools.

    Theory Connection: Maintains existing WHERE positioning (file locations,
    log paths) while adding new Context measurement capabilities that enable
    Conveyance Framework optimization without breaking existing workflows.
    """

    def __init__(self, component_name: str, log_dir: Optional[Path] = None):
        """
        Initialize legacy monitor interface.

        Args:
            component_name: Name of the component being monitored
            log_dir: Directory for log files
        """
        self.component_name = component_name
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Core monitoring components
        self.performance_monitor = PerformanceMonitor(component_name, self.log_dir)
        self.progress_tracker = ProgressTracker(f"{component_name}_progress")

        # Legacy compatibility
        self.start_time = datetime.utcnow()
        self._monitoring_active = False
        self._dashboard_thread: Optional[threading.Thread] = None

    def get_gpu_status(self) -> Dict[str, Any]:
        """
        Get GPU status in legacy format.

        Theory Connection: Preserves existing WHO dimension measurements
        while enabling integration with new Conveyance Framework scoring.

        Returns:
            GPU status dictionary in legacy format
        """
        system_status = self.performance_monitor.get_system_status()

        if not system_status['gpus']:
            return {'gpus': [], 'available': False}

        legacy_gpus = []
        for gpu in system_status['gpus']:
            legacy_gpus.append({
                'index': gpu['gpu_id'],
                'name': f"GPU {gpu['gpu_id']}",  # Simplified name
                'memory_used': int(gpu['memory_used_gb'] * 1024),  # Convert to MB
                'memory_total': int(gpu['memory_total_gb'] * 1024),
                'utilization': int(gpu['utilization_percent']),
                'temperature': int(gpu.get('temperature_c', 0) or 0)
            })

        return {'gpus': legacy_gpus, 'available': True}

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status in legacy format.

        Returns:
            System status dictionary in legacy format
        """
        system_status = self.performance_monitor.get_system_status()

        if not system_status['system']:
            return {
                'cpu_percent': 0.0,
                'memory': {'used_gb': 0.0, 'total_gb': 0.0, 'percent': 0.0},
                'disk': {'staging_used_gb': 0.0, 'staging_files': 0}
            }

        sys_info = system_status['system']
        return {
            'cpu_percent': sys_info['cpu_percent'],
            'memory': {
                'used_gb': sys_info['memory_used_gb'],
                'total_gb': sys_info['memory_total_gb'],
                'percent': sys_info['memory_percent']
            },
            'disk': {
                'staging_used_gb': sys_info.get('disk_free_gb', 0.0),  # Approximation
                'staging_files': 0  # Would need specific directory scan
            }
        }

    def get_directory_size(self, path: Path) -> int:
        """
        Get total size of directory in bytes.

        Args:
            path: Directory path

        Returns:
            Total size in bytes
        """
        total = 0
        if path.exists():
            for p in path.rglob('*'):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except (OSError, IOError):
                        continue
        return total

    def load_checkpoint(self, checkpoint_file: Path) -> Dict[str, Any]:
        """
        Load checkpoint file in legacy format.

        Args:
            checkpoint_file: Path to checkpoint file

        Returns:
            Checkpoint data dictionary
        """
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
        return {}

    def get_database_stats(self, sqlite_db: Path) -> Dict[str, Any]:
        """
        Get statistics from SQLite database.

        Theory Connection: Provides WHAT dimension measurements through
        database content analysis, enabling Context coherence assessment.

        Args:
            sqlite_db: Path to SQLite database

        Returns:
            Database statistics dictionary
        """
        stats = {
            'total_processed': 0,
            'extraction_complete': 0,
            'embedding_complete': 0,
            'failed': 0,
            'current_phase': 'unknown'
        }

        if not sqlite_db.exists():
            return stats

        try:
            conn = sqlite3.connect(str(sqlite_db))
            cursor = conn.cursor()

            # Get table info to understand schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            if 'papers' in tables:
                # Get column info
                cursor.execute("PRAGMA table_info(papers);")
                columns = [row[1] for row in cursor.fetchall()]

                # Adapt queries based on available columns
                if 'processing_status' in columns:
                    cursor.execute("SELECT COUNT(*) FROM papers WHERE processing_status = 'PROCESSED'")
                    stats['total_processed'] = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM papers WHERE processing_status = 'FAILED'")
                    stats['failed'] = cursor.fetchone()[0]

                if 'phase_completed' in columns:
                    cursor.execute("SELECT COUNT(*) FROM papers WHERE phase_completed >= 1")
                    stats['extraction_complete'] = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM papers WHERE phase_completed = 2")
                    stats['embedding_complete'] = cursor.fetchone()[0]

                    # Determine current phase
                    if stats['embedding_complete'] > 0 and stats['embedding_complete'] < stats['total_processed']:
                        stats['current_phase'] = 'embedding'
                    elif stats['extraction_complete'] > 0:
                        stats['current_phase'] = 'extraction'
                    elif stats['embedding_complete'] >= stats['total_processed']:
                        stats['current_phase'] = 'complete'

            conn.close()

        except Exception as e:
            logger.error(f"Error reading database {sqlite_db}: {e}")

        return stats

    def parse_recent_logs(self, log_file: Path, lines: int = 100) -> Dict[str, Any]:
        """
        Parse recent log entries for errors and progress.

        Theory Connection: Extracts Context coherence indicators from
        processing logs, enabling real-time assessment of system stability.

        Args:
            log_file: Path to log file
            lines: Number of recent lines to analyze

        Returns:
            Log analysis dictionary
        """
        info = {
            'recent_errors': [],
            'last_paper': None,
            'current_rate': 0.0,
            'phase_info': {}
        }

        if not log_file.exists():
            return info

        try:
            # Get last N lines efficiently
            if hasattr(subprocess, 'run'):
                # Use tail command if available (Unix-like systems)
                try:
                    result = subprocess.run(
                        ['tail', '-n', str(lines), str(log_file)],
                        capture_output=True, text=True, timeout=10
                    )
                    log_lines = result.stdout.split('\n')
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                    # Fallback to reading file directly
                    with open(log_file, 'r', encoding='utf-8') as f:
                        all_lines = f.readlines()
                        log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            else:
                # Direct file reading
                with open(log_file, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                    log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            for line in log_lines:
                line = line.strip()
                if not line:
                    continue

                # Check for errors
                if any(level in line.upper() for level in ['ERROR', 'CRITICAL', 'EXCEPTION']):
                    info['recent_errors'].append(line)

                # Check for paper processing
                if 'processing paper' in line.lower() or 'processed:' in line.lower():
                    # Extract paper ID if possible
                    parts = line.split()
                    for part in parts:
                        if '.' in part and part[0].isdigit():
                            info['last_paper'] = part
                            break

                # Check for rate information
                if 'papers/minute' in line.lower():
                    try:
                        # Look for numeric value before "papers/minute"
                        words = line.split()
                        for i, word in enumerate(words):
                            if 'papers/minute' in word.lower() and i > 0:
                                rate_str = words[i-1].replace(',', '')
                                info['current_rate'] = float(rate_str)
                                break
                    except (ValueError, IndexError):
                        pass

                # Check for phase changes
                if 'phase' in line.lower():
                    if 'extraction' in line.lower():
                        info['phase_info']['current'] = 'extraction'
                    elif 'embedding' in line.lower():
                        info['phase_info']['current'] = 'embedding'

        except Exception as e:
            logger.error(f"Error parsing logs from {log_file}: {e}")

        return info

    def calculate_eta(self, processed: int, total: int, rate: float) -> str:
        """
        Calculate estimated time of arrival.

        Theory Connection: Projects TIME dimension completion based on
        current processing velocity and remaining work.

        Args:
            processed: Number of items processed
            total: Total number of items
            rate: Processing rate (items per minute)

        Returns:
            Human-readable ETA string
        """
        if rate <= 0 or processed >= total:
            return "N/A"

        remaining = total - processed
        minutes_remaining = remaining / rate
        eta = datetime.utcnow() + timedelta(minutes=minutes_remaining)

        # Format nicely
        if minutes_remaining < 60:
            return f"{int(minutes_remaining)} minutes"
        elif minutes_remaining < 1440:  # Less than a day
            hours = minutes_remaining / 60
            return f"{hours:.1f} hours ({eta.strftime('%H:%M')})"
        else:
            days = minutes_remaining / 1440
            return f"{days:.1f} days ({eta.strftime('%a %H:%M')})"

    def print_dashboard(self, **kwargs) -> None:
        """
        Print monitoring dashboard with enhanced metrics.

        Theory Connection: Provides comprehensive view of all Conveyance
        Framework dimensions while maintaining familiar interface format.
        """
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        # Get enhanced stats
        system_stats = self.get_system_status()
        gpu_stats = self.get_gpu_status()
        overall_progress = self.progress_tracker.get_overall_progress()
        conveyance_metrics = self.progress_tracker.calculate_conveyance_metrics()

        # Calculate runtime
        runtime = datetime.utcnow() - self.start_time
        runtime_str = str(runtime).split('.')[0]  # Remove microseconds

        # Print header with enhanced information
        print("=" * 80)
        print(f"HADES MONITORING DASHBOARD - {self.component_name}".center(80))
        print("=" * 80)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runtime: {runtime_str}")
        print(f"Current: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Progress section with Conveyance metrics
        print("PROGRESS & CONVEYANCE METRICS")
        print("-" * 50)
        print(f"Overall Progress: {overall_progress.get('completion_percent', 0):.1f}%")
        print(f"Processing Rate: {overall_progress.get('processing_rate', 0):.1f} items/sec")
        print(f"Success Rate: {overall_progress.get('success_rate', 0):.1f}%")

        if conveyance_metrics:
            print()
            print("Conveyance Framework Scores:")
            print(f"  WHERE (R): {conveyance_metrics.get('where', 0):.3f}")
            print(f"  WHAT (W):  {conveyance_metrics.get('what', 0):.3f}")
            print(f"  WHO (H):   {conveyance_metrics.get('who', 0):.3f}")
            print(f"  TIME (T):  {conveyance_metrics.get('time', 0):.3f}")
            print(f"  Context:   {conveyance_metrics.get('context', 0):.3f}")
            print(f"  Overall C: {conveyance_metrics.get('conveyance', 0):.3f}")
        print()

        # System Resources
        print("SYSTEM RESOURCES")
        print("-" * 40)
        print(f"CPU Usage: {system_stats['cpu_percent']:.1f}%")
        memory = system_stats['memory']
        print(f"Memory: {memory['used_gb']:.1f}GB / {memory['total_gb']:.1f}GB ({memory['percent']:.1f}%)")
        print()

        # GPU Status
        if gpu_stats['available'] and gpu_stats['gpus']:
            print("GPU STATUS")
            print("-" * 40)
            for gpu in gpu_stats['gpus']:
                mem_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
                print(f"GPU {gpu['index']}: {gpu.get('name', 'Unknown')}")
                print(f"  Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({mem_pct:.1f}%)")
                print(f"  Utilization: {gpu['utilization']}%")
                if gpu.get('temperature', 0) > 0:
                    print(f"  Temperature: {gpu['temperature']}°C")
            print()

        # Footer
        print("=" * 80)
        print("Enhanced with HADES Core Monitoring Framework")
        print("Theory-Connected Performance Analysis")

    def start_monitoring(self, **kwargs) -> None:
        """
        Start enhanced monitoring.

        Args:
            **kwargs: Additional monitoring parameters
        """
        # Start core performance monitoring
        self.performance_monitor.start_monitoring()

        # Set monitoring flag
        self._monitoring_active = True

        logger.info(f"Started enhanced monitoring for {self.component_name}")

    def stop_monitoring(self) -> None:
        """Stop enhanced monitoring."""
        self._monitoring_active = False

        # Stop core performance monitoring
        self.performance_monitor.stop_monitoring()

        # Stop dashboard if running
        if self._dashboard_thread and self._dashboard_thread.is_alive():
            self._dashboard_thread.join(timeout=5.0)

        logger.info(f"Stopped enhanced monitoring for {self.component_name}")

    def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        self.stop_monitoring()
        self.performance_monitor.cleanup()

        # Flush final metrics
        try:
            self.performance_monitor.flush()
        except Exception as e:
            logger.warning(f"Failed to flush final metrics: {e}")


def create_legacy_monitor(component_name: str, log_dir: str = None) -> LegacyMonitorInterface:
    """
    Factory function to create legacy-compatible monitor.

    Theory Connection: Provides transition mechanism from existing
    monitoring patterns to enhanced Conveyance Framework measurement
    while preserving backward compatibility.

    Args:
        component_name: Name of component to monitor
        log_dir: Directory for log files

    Returns:
        Legacy monitor interface with enhanced capabilities
    """
    return LegacyMonitorInterface(
        component_name=component_name,
        log_dir=Path(log_dir) if log_dir else None
    )