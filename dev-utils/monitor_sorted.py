#!/usr/bin/env python3
"""
Sorted Workflow Monitor
========================

Monitors progress of the sorted workflow by tracking:
- Metadata records loaded
- Embeddings created
- Processing rate
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


def format_time(seconds):
    """Format seconds into human readable time."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    if hours > 0:
        return f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"
    elif minutes > 0:
        return f"{minutes:.0f}m {seconds:.0f}s"
    else:
        return f"{seconds:.0f}s"


def main():
    parser = argparse.ArgumentParser(description="Monitor sorted workflow progress")

    parser.add_argument('--interval', type=int, default=10,
                       help='Check interval in seconds (default: 10)')
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
        print(f"✅ Connected to {args.database}")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)

    # Collections to monitor
    metadata_coll = 'arxiv_metadata'
    embeddings_coll = 'arxiv_abstract_embeddings'
    chunks_coll = 'arxiv_abstract_chunks'

    # Check collections exist
    for coll_name in [metadata_coll, embeddings_coll, chunks_coll]:
        if not db.has_collection(coll_name):
            print(f"⚠️  Collection {coll_name} not found")

    print(f"\nMonitoring every {args.interval} seconds")
    print("Press Ctrl+C to stop\n")
    print("=" * 80)

    # Initial counts
    last_metadata = db.collection(metadata_coll).count() if db.has_collection(metadata_coll) else 0
    last_embeddings = db.collection(embeddings_coll).count() if db.has_collection(embeddings_coll) else 0
    last_chunks = db.collection(chunks_coll).count() if db.has_collection(chunks_coll) else 0
    last_time = time.time()

    start_metadata = last_metadata
    start_embeddings = last_embeddings
    start_time = last_time

    print(f"{datetime.now().strftime('%H:%M:%S')} | Initial counts:")
    print(f"  Metadata: {last_metadata:,}")
    print(f"  Embeddings: {last_embeddings:,}")
    print(f"  Chunks: {last_chunks:,}")
    print("-" * 80)

    try:
        while True:
            time.sleep(args.interval)

            # Get current counts
            curr_metadata = db.collection(metadata_coll).count() if db.has_collection(metadata_coll) else 0
            curr_embeddings = db.collection(embeddings_coll).count() if db.has_collection(embeddings_coll) else 0
            curr_chunks = db.collection(chunks_coll).count() if db.has_collection(chunks_coll) else 0
            curr_time = time.time()

            # Calculate deltas
            delta_metadata = curr_metadata - last_metadata
            delta_embeddings = curr_embeddings - last_embeddings
            delta_chunks = curr_chunks - last_chunks
            delta_time = curr_time - last_time

            # Calculate rates
            metadata_rate = delta_metadata / delta_time if delta_time > 0 else 0
            embeddings_rate = delta_embeddings / delta_time if delta_time > 0 else 0

            # Overall stats
            total_time = curr_time - start_time
            overall_processed = curr_embeddings - start_embeddings
            overall_rate = overall_processed / total_time if total_time > 0 else 0

            # Calculate unprocessed
            unprocessed = curr_metadata - curr_embeddings

            # Display
            timestamp = datetime.now().strftime('%H:%M:%S')

            # Show different output based on what's happening
            if delta_metadata > 0 and delta_embeddings == 0:
                # Still loading metadata
                print(f"{timestamp} | Loading metadata: {curr_metadata:,} (+{delta_metadata:,}) | "
                      f"Rate: {metadata_rate:.0f}/s")
            elif delta_embeddings > 0:
                # Processing embeddings
                print(f"{timestamp} | Embeddings: {curr_embeddings:,}/{curr_metadata:,} | "
                      f"+{delta_embeddings:,} @ {embeddings_rate:.1f}/s | "
                      f"Unprocessed: {unprocessed:,}")

                # ETA calculation
                if embeddings_rate > 0:
                    eta_seconds = unprocessed / embeddings_rate
                    print(f"         | Overall: {overall_rate:.1f}/s avg | "
                          f"ETA: {format_time(eta_seconds)}")
            else:
                # No changes
                print(f"{timestamp} | No changes | "
                      f"Embeddings: {curr_embeddings:,}/{curr_metadata:,}")

            # Check for completion
            if curr_metadata > 0 and curr_embeddings == curr_metadata:
                print("\n✅ All records processed!")
                break

            # Update for next iteration
            last_metadata = curr_metadata
            last_embeddings = curr_embeddings
            last_chunks = curr_chunks
            last_time = curr_time

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Final Statistics:")
        print(f"  Metadata loaded: {curr_metadata:,}")
        print(f"  Embeddings created: {curr_embeddings:,}")
        print(f"  Chunks created: {curr_chunks:,}")
        print(f"  Unprocessed: {curr_metadata - curr_embeddings:,}")
        print(f"  Total time: {format_time(total_time)}")
        print(f"  Average rate: {overall_rate:.1f} embeddings/second")

        if overall_rate > 0 and curr_metadata > curr_embeddings:
            remaining = curr_metadata - curr_embeddings
            eta_seconds = remaining / overall_rate
            print(f"  Estimated time remaining: {format_time(eta_seconds)}")


if __name__ == "__main__":
    main()
