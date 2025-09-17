#!/usr/bin/env python3
"""
Test ArXiv Metadata Processing with 1000 Records
==================================================

Quick test script to validate the metadata processing workflow
with a small subset of records before running the full dataset.
"""

import os
import sys
from pathlib import Path

# Add project root to path
# From tests/arxiv/file.py: parent = tests/arxiv, parent.parent = tests, parent.parent.parent = HADES-Lab
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.workflows.workflow_arxiv_metadata import ArxivMetadataWorkflow
from core.tools.arxiv.arxiv_metadata_config import ArxivMetadataConfig


def main():
    """Run test with 1000 records."""

    print("=" * 60)
    print("ArXiv Metadata Processing Test")
    print("Testing with 1000 records")
    print("=" * 60)
    print()

    # Check for database password
    if not os.environ.get('ARANGO_PASSWORD'):
        print("❌ ERROR: ARANGO_PASSWORD environment variable not set")
        print("Please run: export ARANGO_PASSWORD='your-password'")
        sys.exit(1)

    # Configure for test
    config = ArxivMetadataConfig(
        max_records=1000,
        batch_size=100,
        embedding_batch_size=32,  # Smaller for testing
        checkpoint_interval=500,
        monitor_interval=10,
        drop_collections=True,  # Clean start for test
        checkpoint_file=Path("/tmp/test_metadata_checkpoint.json"),
        state_file=Path("/tmp/test_metadata_state.json")
    )

    print("Configuration:")
    print(f"  Records to process: {config.max_records}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Drop collections: {config.drop_collections}")
    print(f"  Embedder: {config.embedder_model}")
    print(f"  GPU: {config.use_gpu}")
    print()

    try:
        # Validate configuration
        config.validate_full()
        print("✅ Configuration validated")
        print()

        # Create workflow
        print("Initializing workflow...")
        workflow = ArxivMetadataWorkflow(config)

        # Execute
        print("Starting processing...")
        print("-" * 40)
        result = workflow.execute()

        # Print results
        print()
        print("=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"  Success: {'✅' if result.success else '❌'} {result.success}")
        print(f"  Processed: {result.items_processed}")
        print(f"  Failed: {result.items_failed}")
        print(f"  Duration: {result.duration_seconds:.2f} seconds")
        print(f"  Success Rate: {result.success_rate:.2f}%")

        if result.metadata:
            throughput = result.metadata.get('throughput', 0)
            print(f"  Throughput: {throughput:.2f} records/second")

            # Check performance
            if throughput >= 10:  # Lower target for test
                print(f"  ✅ Good throughput for test")
            else:
                print(f"  ⚠️  Low throughput (expected >10 rec/s for test)")

        # Verify database
        print()
        print("Database Verification:")
        print("-" * 40)

        # Check collections were created
        from core.database.database_factory import DatabaseFactory

        db = DatabaseFactory.get_arango(
            database=config.arango_database,
            username=config.arango_username
        )

        for collection in [config.metadata_collection,
                          config.chunks_collection,
                          config.embeddings_collection]:
            if db.has_collection(collection):
                count = db.collection(collection).count()
                print(f"  ✅ {collection}: {count} documents")
            else:
                print(f"  ❌ {collection}: NOT FOUND")

        print()
        if result.success:
            print("✅ TEST PASSED - Ready for full dataset processing!")
            print()
            print("To process the full dataset (2.8M records), run:")
            print("  python metadata_processor.py --mode production")
        else:
            print("❌ TEST FAILED - Please check errors above")

        return result.success

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)