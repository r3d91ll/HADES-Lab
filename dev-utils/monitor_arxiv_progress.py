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
        Create a new ArxivProgressMonitor.
        
        Parameters:
            target_records (int): Target number of documents to monitor until completion (default 2,828,998).
            interval (int): Polling interval in seconds between updates (default 10).
        
        Sets initial internal tracking state and the metadata collection name ("arxiv_metadata").
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
        """
        Establish a connection to ArangoDB and store the client on self.db.
        
        Attempts to obtain an Arango client for the "academy_store" database and assigns it to self.db. Returns True when the connection is successful; on failure the exception is caught, no exception is propagated, and the method returns False.
        
        Returns:
            bool: True if connected and self.db is set, False otherwise.
        """
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
        """
        Return the number of documents in the configured metadata collection.
        
        If the collection does not exist in the connected database, returns 0.
        
        Returns:
            int: Current document count in `self.metadata_collection`, or 0 if the collection is missing.
        """
        if not self.db.has_collection(self.metadata_collection):
            return 0

        collection = self.db.collection(self.metadata_collection)
        return collection.count()

    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Return per-GPU utilization and memory statistics by invoking `nvidia-smi`.
        
        Queries `nvidia-smi` for GPU index, name, GPU utilization percent, memory used, and memory total. On success returns a dictionary keyed by "gpu_<index>" where each value is a dict with:
        - name (str): GPU model name,
        - util (float): GPU utilization percent,
        - mem_used (float): memory used (same units as reported by `nvidia-smi`),
        - mem_total (float): total memory,
        - mem_percent (float): memory usage as a percentage of total.
        
        If `nvidia-smi` is not available, the command fails, times out, or any parsing error occurs, an empty dict is returned.
        """
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
        """
        Convert a duration in seconds to a concise, human-readable string.
        
        Returns "N/A" for negative inputs. Outputs use the largest relevant units:
        - Days present: "Xd Xh Xm" (days, hours, minutes)
        - Hours present (no days): "Xh Xm Ys" (hours, minutes, seconds)
        - Otherwise: "Xm Ys" (minutes, seconds)
        
        Parameters:
            seconds (float): Duration in seconds.
        
        Returns:
            str: Formatted time string as described above (or "N/A" for negative input).
        """
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
        """
        Render the live monitoring UI for ArXiv ingestion to the terminal.
        
        Clears the terminal and prints a formatted status panel showing progress toward the target,
        throughput metrics, time estimates (elapsed, remaining, total, and ETA when computable),
        and per-GPU utilization/memory if available. This function has the side effect of writing
        to standard output and clearing the screen.
        
        Parameters:
            current_count (int): Number of documents currently in the metadata collection.
            elapsed (float): Seconds elapsed since monitoring started.
            rate (float): Average processing rate in papers per second since start.
            instant_rate (float): Processing rate in papers per second measured over the last interval.
        
        Returns:
            None
        """
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
        """
        Start the real-time monitoring loop that tracks ArXiv ingestion progress.
        
        This method establishes a database connection, initializes counters and timestamps, then enters a periodic loop that:
        - sleeps for the configured interval,
        - reads the current document count from the metadata collection,
        - computes overall and instantaneous processing rates,
        - updates a terminal display with progress, ETA, and GPU stats,
        and exits when the configured target record count is reached.
        
        On KeyboardInterrupt the loop stops gracefully and prints final aggregated statistics.
        
        Returns:
            None
        """
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
    """
    Parse CLI options and run the ArxivProgressMonitor.
    
    This is the module's entry point: it parses the command-line options --interval
    (update interval in seconds) and --target (target number of records), verifies
    the ARANGO_PASSWORD environment variable is set (exits with code 1 if missing),
    creates an ArxivProgressMonitor configured with the parsed values, and starts
    monitoring by calling monitor.run().
    """
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