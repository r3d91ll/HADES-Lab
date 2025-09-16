#!/usr/bin/env python3
"""
Simple Processing Monitor
=========================

Dead simple monitor that just counts records in database every N seconds
and calculates the rate based on actual growth.
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database.database_factory import DatabaseFactory


def main():
    parser = argparse.ArgumentParser(description="Simple progress monitor")

    parser.add_argument('--interval', type=int, default=30,
                       help='Check interval in seconds (default: 30)')
    parser.add_argument('--collection', default='arxiv_metadata',
                       help='Collection to monitor')
    parser.add_argument('--database', default='arxiv_repository',
                       help='Database name (default: arxiv_repository)')
    parser.add_argument('--username', default='root',
                       help='Database username (default: root)')

    args = parser.parse_args()

    # Check for ARANGO_PASSWORD
    if not os.environ.get('ARANGO_PASSWORD'):
        print("ERROR: ARANGO_PASSWORD not set")
        sys.exit(1)

    # Connect to database
    try:
        db = DatabaseFactory.get_arango(
            database=args.database,
            username=args.username,
            use_unix=True
        )
        print(f"✅ Connected to ArangoDB database: {args.database}")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)

    # Check collection exists
    if not db.has_collection(args.collection):
        print(f"❌ Collection {args.collection} not found")
        sys.exit(1)

    collection = db.collection(args.collection)

    print(f"\nMonitoring {args.collection} every {args.interval} seconds")
    print("Press Ctrl+C to stop\n")
    print("-" * 60)

    # Initial count
    last_count = collection.count()
    last_time = time.time()
    start_count = last_count
    start_time = last_time

    print(f"{datetime.now().strftime('%H:%M:%S')} | Start count: {last_count:,}")

    try:
        while True:
            time.sleep(args.interval)

            # Get current count
            current_count = collection.count()
            current_time = time.time()

            # Calculate rates
            interval_docs = current_count - last_count
            interval_time = current_time - last_time
            interval_rate = interval_docs / interval_time if interval_time > 0 else 0

            total_docs = current_count - start_count
            total_time = current_time - start_time
            overall_rate = total_docs / total_time if total_time > 0 else 0

            # Display
            timestamp = datetime.now().strftime('%H:%M:%S')

            if interval_docs > 0:
                print(f"{timestamp} | Total: {current_count:,} | "
                      f"+{interval_docs:,} in {interval_time:.0f}s | "
                      f"Rate: {interval_rate:.1f}/s (current), {overall_rate:.1f}/s (avg)")
            else:
                print(f"{timestamp} | Total: {current_count:,} | "
                      f"No change | "
                      f"Overall avg: {overall_rate:.1f}/s")

            # Update for next iteration
            last_count = current_count
            last_time = current_time

    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("Final Statistics:")
        print(f"  Total processed: {current_count - start_count:,}")
        print(f"  Total time: {total_time:.1f} seconds")
        print(f"  Average rate: {overall_rate:.1f} docs/second")

        if overall_rate > 0:
            remaining = 2828998 - current_count
            eta_seconds = remaining / overall_rate
            eta_hours = eta_seconds / 3600
            print(f"  Estimated time remaining: {eta_hours:.1f} hours")


if __name__ == "__main__":
    main()