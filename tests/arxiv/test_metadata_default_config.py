#!/usr/bin/env python3
"""
Test ArXiv Metadata Processing with Default Configuration
==========================================================

Tests the workflow using the default configuration to ensure
all default values work correctly before testing overrides.
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
# From tests/arxiv/file.py: parent = tests/arxiv, parent.parent = tests, parent.parent.parent = HADES-Lab
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.workflows.workflow_arxiv_metadata import ArxivMetadataWorkflow
from core.tools.arxiv.arxiv_metadata_config import ArxivMetadataConfig


def test_default_config():
    """
    Validate the default ArXiv metadata workflow configuration with lightweight test overrides.
    
    Loads the default YAML config at core/config/workflows/arxiv_metadata_default.yaml, applies test-specific overrides (reduced max_records, batch sizes, checkpoint/monitor intervals, and target throughput), instantiates ArxivMetadataConfig, runs full validation and semantic validation, and prints a compact report of selected configuration fields and validation results.
    
    Returns:
        bool: True if full validation succeeded and no blocking semantic errors were found; False if full validation raises an exception or other fatal error occurs.
    """

    print("=" * 60)
    print("Testing Default Configuration")
    print("=" * 60)
    print()

    # Load default config
    default_config_path = Path(project_root) / "core/config/workflows/arxiv_metadata_default.yaml"

    print(f"Loading default config from: {default_config_path}")
    with open(default_config_path, 'r') as f:
        default_dict = yaml.safe_load(f)

    # Override for testing (just process 100 records)
    default_dict['max_records'] = 100
    default_dict['batch_size'] = 50  # Reasonable for testing
    default_dict['embedding_batch_size'] = 32  # Must be less than batch_size
    default_dict['checkpoint_interval'] = 100  # Minimum is 100
    default_dict['monitor_interval'] = 10
    default_dict['target_throughput'] = 10.0  # Lower target for test

    # Create config instance
    config = ArxivMetadataConfig(**default_dict)

    print("\nDefault Configuration Values:")
    print("-" * 40)
    print(f"  metadata_file: {config.metadata_file}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  embedding_batch_size: {config.embedding_batch_size}")
    print(f"  embedder_model: {config.embedder_model}")
    print(f"  use_gpu: {config.use_gpu}")
    print(f"  use_fp16: {config.use_fp16}")
    print(f"  chunk_size_tokens: {config.chunk_size_tokens}")
    print(f"  checkpoint_interval: {config.checkpoint_interval}")
    print(f"  target_throughput: {config.target_throughput}")
    print()

    # Validate configuration
    print("Validating configuration...")
    try:
        config.validate_full()
        print("✅ Configuration validation passed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

    # Check semantic validation
    semantic_errors = config.validate_semantics()
    if semantic_errors:
        print("⚠️  Semantic validation warnings:")
        for error in semantic_errors:
            print(f"  - {error}")
    else:
        print("✅ Semantic validation passed")

    print()
    return True


def test_default_workflow():
    """
    Execute the ArXiv metadata workflow using the default YAML configuration with test overrides.
    
    Loads core/config/workflows/arxiv_metadata_default.yaml, applies lightweight overrides (e.g., max_records, batch_size, embedding_batch_size, checkpoint_interval, monitor_interval, target_throughput, drop_collections), constructs ArxivMetadataConfig and ArxivMetadataWorkflow, then runs workflow.execute().
    
    Important behavior:
    - Requires the ARANGO_PASSWORD environment variable to be set; otherwise the function returns False without running.
    - The override drop_collections=True will request a clean start for the test run.
    - Prints a brief execution summary (success flag, processed/failed counts, duration, and throughput when available).
    
    Returns:
        bool: True if the workflow completed successfully (result.success is True); False on configuration/environment issues, execution failure, or exceptions.
    """

    print("=" * 60)
    print("Testing Workflow with Default Configuration")
    print("=" * 60)
    print()

    # Check for database password
    if not os.environ.get('ARANGO_PASSWORD'):
        print("❌ ERROR: ARANGO_PASSWORD environment variable not set")
        print("Please run: export ARANGO_PASSWORD='your-password'")
        return False

    # Load default config
    default_config_path = Path(project_root) / "core/config/workflows/arxiv_metadata_default.yaml"
    with open(default_config_path, 'r') as f:
        default_dict = yaml.safe_load(f)

    # Override for quick testing
    default_dict['max_records'] = 100
    default_dict['batch_size'] = 50  # Reasonable for testing
    default_dict['embedding_batch_size'] = 32  # Must be less than batch_size
    default_dict['checkpoint_interval'] = 100  # Minimum is 100
    default_dict['monitor_interval'] = 10
    default_dict['target_throughput'] = 10.0  # Lower target for test
    default_dict['drop_collections'] = True  # Clean start for test

    # Create config
    config = ArxivMetadataConfig(**default_dict)

    try:
        # Create workflow
        print("Creating workflow with default config...")
        workflow = ArxivMetadataWorkflow(config)

        # Execute
        print("Executing workflow...")
        print("-" * 40)
        result = workflow.execute()

        # Print results
        print()
        print("Results:")
        print("-" * 40)
        print(f"  Success: {'✅' if result.success else '❌'}")
        print(f"  Processed: {result.items_processed}")
        print(f"  Failed: {result.items_failed}")
        print(f"  Duration: {result.duration_seconds:.2f} seconds")

        if result.metadata:
            throughput = result.metadata.get('throughput', 0)
            print(f"  Throughput: {throughput:.2f} records/second")

        return result.success

    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Run the default-configuration test suite.
    
    Performs two checks in sequence:
    1. Validates the default ArXiv metadata configuration using test overrides.
    2. If validation succeeds, attempts to execute the workflow with the default configuration (and test overrides).
    
    Prints progress, warnings, and a final summary to stdout.
    
    Returns:
        bool: True if all tests pass; False if any test fails.
    """

    print("=" * 70)
    print("DEFAULT CONFIGURATION TEST SUITE")
    print("=" * 70)
    print()

    # Test 1: Configuration validation
    if not test_default_config():
        print("\n❌ Default configuration validation failed")
        return False

    print()

    # Test 2: Workflow execution
    if not test_default_workflow():
        print("\n❌ Default workflow execution failed")
        return False

    print()
    print("=" * 70)
    print("✅ ALL DEFAULT CONFIGURATION TESTS PASSED")
    print("=" * 70)
    print()
    print("The default configuration is working correctly!")
    print("You can now create workflow-specific overrides in:")
    print("  core/config/workflows/arxiv_metadata_custom.yaml")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)