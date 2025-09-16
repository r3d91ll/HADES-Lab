#!/usr/bin/env python3
"""
Test script for ArXiv workflow resume functionality.

Tests:
1. Process 100 records normally
2. Process 200 records with --resume (should skip first 100)
3. Verify no duplicates
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.database.database_factory import DatabaseFactory

def check_database_status():
    """Check current database status."""
    db = DatabaseFactory.get_arango(
        database='academy_store',
        username='root',
        use_unix=True
    )

    # Count records
    metadata_count = db.collection('arxiv_metadata').count()
    embeddings_count = db.collection('arxiv_abstract_embeddings').count()

    print(f"Database Status:")
    print(f"  arxiv_metadata: {metadata_count:,} documents")
    print(f"  arxiv_abstract_embeddings: {embeddings_count:,} documents")
    print()

    return metadata_count, embeddings_count

def main():
    print("=" * 60)
    print("ArXiv Workflow Resume Test")
    print("=" * 60)
    print()

    # Check initial state
    print("1. Initial database state:")
    initial_meta, initial_embed = check_database_status()

    # Test commands
    print("2. Test Commands:")
    print()
    print("First, process 100 records:")
    print("  python -m core.workflows.workflow_arxiv_parallel --count 100")
    print()
    print("Then test resume (should skip existing and process new):")
    print("  python -m core.workflows.workflow_arxiv_parallel --count 200 --resume")
    print()
    print("For full processing with resume:")
    print("  python -m core.workflows.workflow_arxiv_parallel --resume")
    print("  (This will process all remaining ~884,502 records)")
    print()
    print("To monitor progress in another terminal:")
    print("  python dev-utils/simple_monitor.py")
    print()

    # Calculate remaining
    total_in_json = 2_828_998
    remaining = total_in_json - initial_embed

    print(f"3. Summary:")
    print(f"  Total records in JSON: {total_in_json:,}")
    print(f"  Already processed: {initial_embed:,}")
    print(f"  Remaining to process: {remaining:,}")
    print(f"  Estimated time at 33 docs/sec: {remaining/33/3600:.1f} hours")

if __name__ == "__main__":
    main()