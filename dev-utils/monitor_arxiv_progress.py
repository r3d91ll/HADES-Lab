#!/usr/bin/env python3
"""
Real-time ArXiv Processing Monitor
===================================

Monitors the ArXiv ingestion workflow in real-time, displaying:
- Current processing rate (papers/second)
- Estimated time remaining
- GPU utilization
- Database growth rate

Usage:
    python monitor_arxiv_progress.py
    python monitor_arxiv_progress.py --interval 5
    python monitor_arxiv_progress.py --target 2828998
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database.database_factory import DatabaseFactory


class ArxivProgressMonitor:
    """
    Real-time monitoring of ArXiv processing workflow.

    Tracks database growth and calculates throughput metrics.
    """

    def __init__(self, target_records: int = 2828998, interval: int = 10):
        """
        Initialize monitor.

        Args:
            target_records: Total records to process
            interval: Update interval in seconds
        """
        self.target_records = target_records
        self.interval = interval
        self.db = None

        # Tracking state
        self.start_time = None
        self.start_count = None
        self.last_count = None
        self.last_time = None

        # Collection names
        self.metadata_collection = "arxiv_metadata"

    def connect(self) -> bool:
        """Connect to database."""
        try:
            self.db = DatabaseFactory.get_arango(
                database="academy_store",
                username="root",
                use_unix=True
            )
            return True
        except Exception as e:
            print(f"❌ Failed to connect to ArangoDB: {e}")
            return False

    def get_current_count(self) -> int:
        """Get current document count."""
        if not self.db.has_collection(self.metadata_collection):
            return 0

        collection = self.db.collection(self.metadata_collection)
        return collection.count()

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU utilization stats."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode != 0:
                return {}

            stats = {}
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_id = int(parts[0])
                    stats[f"gpu_{gpu_id}"] = {
                        'name': parts[1],
                        'util': float(parts[2]),
                        'mem_used': float(parts[3]),
                        'mem_total': float(parts[4]),
                        'mem_percent': (float(parts[3]) / float(parts[4]) * 100) if float(parts[4]) > 0 else 0
                    }

            return stats
        except Exception:
            return {}

    def format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 0:
            return "N/A"

        td = timedelta(seconds=int(seconds))
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        else:
            return f"{minutes}m {seconds}s"

    def display_stats(self, current_count: int, elapsed: float, rate: float, instant_rate: float):
        """Display formatted statistics."""
        # Clear screen for clean display
        os.system('clear' if os.name == 'posix' else 'cls')

        print("=" * 60)
        print("ARXIV PROCESSING MONITOR")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Progress
        progress_percent = (current_count / self.target_records * 100) if self.target_records > 0 else 0
        remaining = self.target_records - current_count

        print(f"📊 PROGRESS")
        print(f"   Processed: {current_count:,} / {self.target_records:,} ({progress_percent:.2f}%)")
        print(f"   Remaining: {remaining:,}")

        # Progress bar
        bar_width = 40
        filled = int(bar_width * progress_percent / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"   [{bar}]")
        print()

        # Throughput
        print(f"⚡ THROUGHPUT")
        print(f"   Overall Rate: {rate:.2f} papers/second")
        print(f"   Current Rate: {instant_rate:.2f} papers/second (last {self.interval}s)")

        # Time estimates
        if rate > 0:
            eta_seconds = remaining / rate
            total_seconds = self.target_records / rate

            print()
            print(f"⏱️  TIME ESTIMATES")
            print(f"   Elapsed: {self.format_time(elapsed)}")
            print(f"   Remaining: {self.format_time(eta_seconds)}")
            print(f"   Total: {self.format_time(total_seconds)}")

            # ETA
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            print(f"   ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

        # GPU Stats
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            print()
            print(f"🖥️  GPU STATUS")
            for gpu_id, stats in sorted(gpu_stats.items()):
                print(f"   {stats['name']} (GPU {gpu_id[-1]}):")
                print(f"     Utilization: {stats['util']:.0f}%")
                print(f"     Memory: {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MB ({stats['mem_percent']:.1f}%)")

        print()
        print("-" * 60)
        print(f"Press Ctrl+C to stop monitoring")

    def run(self):
        """Run the monitoring loop."""
        if not self.connect():
            return

        print("Starting monitoring...")
        print(f"Update interval: {self.interval} seconds")
        print(f"Target records: {self.target_records:,}")
        print()

        # Initialize tracking
        self.start_count = self.get_current_count()
        self.start_time = time.time()
        self.last_count = self.start_count
        self.last_time = self.start_time

        try:
            while True:
                time.sleep(self.interval)

                # Get current state
                current_count = self.get_current_count()
                current_time = time.time()

                # Calculate rates
                total_processed = current_count - self.start_count
                total_elapsed = current_time - self.start_time
                overall_rate = total_processed / total_elapsed if total_elapsed > 0 else 0

                # Instant rate (since last check)
                instant_processed = current_count - self.last_count
                instant_elapsed = current_time - self.last_time
                instant_rate = instant_processed / instant_elapsed if instant_elapsed > 0 else 0

                # Display
                self.display_stats(current_count, total_elapsed, overall_rate, instant_rate)

                # Update last values
                self.last_count = current_count
                self.last_time = current_time

                # Check if complete
                if current_count >= self.target_records:
                    print()
                    print("🎉 PROCESSING COMPLETE!")
                    print(f"   Total time: {self.format_time(total_elapsed)}")
                    print(f"   Average rate: {overall_rate:.2f} papers/second")
                    break

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")

            # Final stats
            if self.start_count is not None:
                current_count = self.get_current_count()
                total_processed = current_count - self.start_count
                total_elapsed = time.time() - self.start_time
                overall_rate = total_processed / total_elapsed if total_elapsed > 0 else 0

                print(f"\nFinal Statistics:")
                print(f"  Processed: {total_processed:,} papers")
                print(f"  Time: {self.format_time(total_elapsed)}")
                print(f"  Rate: {overall_rate:.2f} papers/second")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor ArXiv processing progress"
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Update interval in seconds (default: 10)'
    )

    parser.add_argument(
        '--target',
        type=int,
        default=2828998,
        help='Target number of records (default: 2828998)'
    )

    args = parser.parse_args()

    # Check for ARANGO_PASSWORD
    if not os.environ.get('ARANGO_PASSWORD'):
        print("❌ ERROR: ARANGO_PASSWORD environment variable not set")
        print("Please set: export ARANGO_PASSWORD='your-password'")
        sys.exit(1)

    # Run monitor
    monitor = ArxivProgressMonitor(
        target_records=args.target,
        interval=args.interval
    )
    monitor.run()


if __name__ == "__main__":
    main()