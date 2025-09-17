#!/usr/bin/env python3
"""
Test suite for ArXiv Metadata Workflow
"""

import sys
from core.workflows.workflow_arxiv_metadata import ArxivMetadataWorkflow, ArxivMetadataConfig


def test_workflow():
    """Test the workflow with a small subset of records."""

    # Configure for testing
    config = ArxivMetadataConfig(
        #max_records=1000,  # Test with 1000 records
        batch_size=1000,
        checkpoint_interval=500,
        drop_collections=True  # Clean start for testing
    )

    # Create and execute workflow
    workflow = ArxivMetadataWorkflow(config)
    result = workflow.execute()

    # Print results
    print(f"\nTest Results:")
    print(f"  Success: {result.success}")
    print(f"  Processed: {result.items_processed}")
    print(f"  Failed: {result.items_failed}")
    print(f"  Duration: {result.duration_seconds:.2f} seconds")
    print(f"  Success Rate: {result.success_rate:.2f}%")

    if result.metadata:
        print(f"  Throughput: {result.metadata.get('throughput', 0):.2f} records/second")

    # Print progress tracker summary
    print("\nProgress Summary:")
    for step in workflow.progress_tracker.steps.values():
        print(f"  {step.name}: {step.completed_items}/{step.total_items} ({step.completion_percent:.1f}%)")

    return result.success


if __name__ == "__main__":
    # Run test
    success = test_workflow()
    sys.exit(0 if success else 1)