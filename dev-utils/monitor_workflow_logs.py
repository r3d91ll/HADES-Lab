#!/usr/bin/env python3
"""
Workflow Log Monitor
====================

Monitors the actual workflow progress by reading the JSON logs.
Shows real throughput based on what's being processed, not database counts.

Usage:
    python monitor_workflow_logs.py
    python monitor_workflow_logs.py --log-dir /tmp
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional

class WorkflowLogMonitor:
    """
    Monitor workflow progress from structured logs.
    """

    def __init__(self, log_dir: str = "/tmp", interval: int = 5):
        """
        Initialize monitor.

        Args:
            log_dir: Directory containing log files
            interval: Update interval in seconds
        """
        self.log_dir = Path(log_dir)
        self.interval = interval

        # Track metrics
        self.total_batches_queued = 0
        self.total_batches_processed = 0
        self.total_records_stored = 0
        self.worker_stats = {}
        self.start_time = None
        self.last_update = None

        # Rate calculation
        self.recent_batches = deque(maxlen=20)  # Last 20 batch timestamps
        self.recent_stored = deque(maxlen=20)   # Last 20 storage timestamps

    def find_latest_log(self) -> Optional[Path]:
        """Find the most recent workflow log file."""
        # Look for structured log files
        log_patterns = [
            "arxiv_parallel_workflow*.log",
            "arxiv_memory_workflow*.log",
            "worker_*.log"
        ]

        latest_file = None
        latest_time = 0

        for pattern in log_patterns:
            for log_file in self.log_dir.glob(pattern):
                if log_file.stat().st_mtime > latest_time:
                    latest_time = log_file.stat().st_mtime
                    latest_file = log_file

        return latest_file

    def parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse a JSON log line."""
        try:
            # Handle both JSON and text logs
            if line.startswith('{'):
                return json.loads(line)
            return None
        except json.JSONDecodeError:
            return None

    def update_metrics(self, log_entry: Dict):
        """Update metrics from log entry."""
        event = log_entry.get('event', '')

        # Track batch queueing
        if event == 'batch_queued':
            self.total_batches_queued += 1
            batch_size = log_entry.get('batch_size', 0)
            timestamp = log_entry.get('timestamp', '')
            self.recent_batches.append((timestamp, batch_size))

        # Track batch processing
        elif event == 'batch_processed':
            worker_id = log_entry.get('worker_id', 0)
            results_count = log_entry.get('results_count', 0)

            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = {
                    'batches': 0,
                    'records': 0
                }

            self.worker_stats[worker_id]['batches'] += 1
            self.worker_stats[worker_id]['records'] += results_count
            self.total_batches_processed += 1

        # Track storage
        elif 'stored' in event or 'storage' in event:
            count = log_entry.get('count', 0) or log_entry.get('stored', 0)
            if count > 0:
                self.total_records_stored += count
                timestamp = log_entry.get('timestamp', '')
                self.recent_stored.append((timestamp, count))

        # Track start time
        elif event == 'workflow_started' or event == 'initializing_components':
            if not self.start_time:
                self.start_time = log_entry.get('timestamp', '')

    def calculate_rates(self) -> Dict[str, float]:
        """Calculate processing rates."""
        rates = {
            'queue_rate': 0,
            'process_rate': 0,
            'store_rate': 0,
            'overall_rate': 0
        }

        # Calculate queue rate from recent batches
        if len(self.recent_batches) > 1:
            total_records = sum(b[1] for b in self.recent_batches)
            time_span = self.interval * len(self.recent_batches)
            if time_span > 0:
                rates['queue_rate'] = total_records / time_span

        # Calculate processing rate from workers
        total_worker_records = sum(w['records'] for w in self.worker_stats.values())
        if self.start_time and total_worker_records > 0:
            # Rough elapsed time calculation
            elapsed = time.time() - (time.time() - 300)  # Assume started within last 5 min
            rates['process_rate'] = total_worker_records / max(elapsed, 1)

        # Calculate storage rate
        if len(self.recent_stored) > 1:
            total_stored = sum(s[1] for s in self.recent_stored)
            time_span = self.interval * len(self.recent_stored)
            if time_span > 0:
                rates['store_rate'] = total_stored / time_span

        # Overall rate is minimum of the pipeline stages
        active_rates = [r for r in [rates['queue_rate'], rates['process_rate'], rates['store_rate']] if r > 0]
        if active_rates:
            rates['overall_rate'] = min(active_rates)

        return rates

    def display_status(self):
        """Display current status."""
        os.system('clear' if os.name == 'posix' else 'cls')

        print("=" * 60)
        print("WORKFLOW LOG MONITOR")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Queue status
        print("📥 QUEUE STATUS")
        print(f"   Batches Queued: {self.total_batches_queued}")
        print(f"   Batches Processed: {self.total_batches_processed}")
        pending = max(0, self.total_batches_queued - self.total_batches_processed)
        print(f"   Pending: {pending}")
        print()

        # Worker status
        if self.worker_stats:
            print("👷 WORKER STATUS")
            for worker_id, stats in sorted(self.worker_stats.items()):
                print(f"   Worker {worker_id}: {stats['batches']} batches, {stats['records']} records")
            print()

        # Storage status
        print("💾 STORAGE STATUS")
        print(f"   Records Stored: {self.total_records_stored}")
        print()

        # Rates
        rates = self.calculate_rates()
        print("⚡ PROCESSING RATES")
        if rates['queue_rate'] > 0:
            print(f"   Queue Rate: {rates['queue_rate']:.1f} records/sec")
        if rates['process_rate'] > 0:
            print(f"   Process Rate: {rates['process_rate']:.1f} records/sec")
        if rates['store_rate'] > 0:
            print(f"   Storage Rate: {rates['store_rate']:.1f} records/sec")
        if rates['overall_rate'] > 0:
            print(f"   Overall Rate: {rates['overall_rate']:.1f} records/sec")
        print()

        print("-" * 60)
        print("Reading from workflow logs...")
        print("Press Ctrl+C to stop")

    def monitor_logs(self):
        """Monitor log files continuously."""
        # Track file position
        file_positions = {}

        print("Starting log monitor...")
        print(f"Monitoring directory: {self.log_dir}")

        try:
            while True:
                # Find all log files
                log_files = list(self.log_dir.glob("*.log"))

                for log_file in log_files:
                    # Skip if not a workflow log
                    if not any(x in log_file.name for x in ['workflow', 'worker']):
                        continue

                    # Get last position or start from beginning
                    last_pos = file_positions.get(str(log_file), 0)

                    try:
                        with open(log_file, 'r') as f:
                            # Seek to last position
                            f.seek(last_pos)

                            # Read new lines
                            for line in f:
                                log_entry = self.parse_log_line(line.strip())
                                if log_entry:
                                    self.update_metrics(log_entry)

                            # Update position
                            file_positions[str(log_file)] = f.tell()

                    except Exception as e:
                        # File might be rotating or locked
                        continue

                # Display status
                self.display_status()

                # Wait before next update
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped")
            print(f"Final stats:")
            print(f"  Batches processed: {self.total_batches_processed}")
            print(f"  Records stored: {self.total_records_stored}")

    def run(self):
        """Run the monitor."""
        self.monitor_logs()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor workflow progress from logs"
    )

    parser.add_argument(
        '--log-dir',
        default='/tmp',
        help='Directory containing log files'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Update interval in seconds'
    )

    args = parser.parse_args()

    monitor = WorkflowLogMonitor(
        log_dir=args.log_dir,
        interval=args.interval
    )
    monitor.run()


if __name__ == "__main__":
    main()