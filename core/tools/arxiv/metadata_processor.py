#!/usr/bin/env python3
"""
ArXiv Metadata Processor CLI
=============================

Command-line interface for processing ArXiv metadata dataset.
Supports test, development, and production modes with full monitoring.

Usage:
    python metadata_processor.py --mode test
    python metadata_processor.py --mode production --confirm
    python metadata_processor.py --config custom_config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.workflows.workflow_arxiv_metadata import ArxivMetadataWorkflow
from core.workflows.workflow_arxiv_parallel import ArxivParallelWorkflow
from core.tools.arxiv.arxiv_metadata_config import ArxivMetadataConfig
from core.monitoring.progress_tracker import ProgressTracker


def load_config(config_path: str, mode: str = "default") -> ArxivMetadataConfig:
    """
    Load an ArxivMetadataConfig from a YAML file and apply mode-specific overrides.
    
    Loads YAML from config_path, promotes any keys defined under the chosen mode (e.g., "test", "development", "production")
    into the top-level config (overwriting top-level values), and removes the mode sections before constructing
    and returning an ArxivMetadataConfig.
    
    Parameters:
        config_path (str): Path to the YAML configuration file.
        mode (str): Mode whose section should be applied as overrides (commonly "test", "development", or "production").
    
    Returns:
        ArxivMetadataConfig: Config instance built from the merged YAML data.
    """
    # Load YAML file
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Apply mode-specific overrides
    if mode in config_dict and isinstance(config_dict[mode], dict):
        for key, value in config_dict[mode].items():
            config_dict[key] = value

    # Remove mode-specific sections from config dict
    # These are not actual config fields, just override sections
    for mode_key in ['test', 'development', 'production']:
        config_dict.pop(mode_key, None)

    # Create config instance
    config = ArxivMetadataConfig(**config_dict)

    return config


def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("ArXiv Metadata Processor")
    print("Processing 2.8M records with Jina v4 embeddings")
    print("=" * 60)
    print()


def print_config_summary(config: ArxivMetadataConfig, mode: str):
    """
    Print a concise, human-readable summary of the pipeline configuration to stdout.
    
    Displays key settings from the provided ArxivMetadataConfig (metadata file, record limits, batch sizes,
    database and embedder settings, GPU usage, checkpoint interval, and target throughput) under a header
    showing the selected mode.
    
    Parameters:
        config (ArxivMetadataConfig): Configuration object whose fields will be summarized.
        mode (str): Mode name (e.g., "test", "development", "production") shown in the header.
    """
    print(f"Configuration Mode: {mode.upper()}")
    print("-" * 40)
    print(f"  Metadata file: {config.metadata_file}")
    print(f"  Max records: {config.max_records or 'ALL'}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Embedding batch: {config.embedding_batch_size}")
    print(f"  Drop collections: {config.drop_collections}")
    print(f"  Database: {config.arango_database}")
    print(f"  Embedder: {config.embedder_model}")
    print(f"  GPU: {config.use_gpu} ({config.gpu_device})")
    print(f"  Checkpoint interval: {config.checkpoint_interval}")
    print(f"  Target throughput: {config.target_throughput} papers/sec")
    print("-" * 40)
    print()


def confirm_production():
    """
    Prompt the user to confirm running in production mode.
    
    Displays a prominent warning about processing ~2.8 million records and the estimated runtime, then prompts the user for confirmation. Blocks waiting for stdin and returns True only if the user types "yes" (case-insensitive); otherwise returns False.
    """
    print("⚠️  WARNING: Production Mode")
    print("This will process approximately 2.8 MILLION records!")
    print("Estimated time: 16+ hours")
    print()
    response = input("Are you sure you want to continue? (type 'yes' to confirm): ")
    return response.lower() == 'yes'


def main():
    """
    Entry point for the ArXiv metadata processing CLI.
    
    Parses command-line arguments, loads and applies a YAML configuration (with per-mode overrides),
    applies CLI overrides, validates the final configuration, selects and runs the appropriate workflow
    (single-GPU or parallel multi-GPU), and prints run results and progress.
    
    Behavior notes:
    - Reads ARANGO_PASSWORD from the environment and aborts early if missing.
    - Prompts for explicit confirmation when run in production mode unless --confirm is provided.
    - Supports CLI overrides: drop-collections, --max-records / --count, --resume, --workers, --batch-size, --no-gpu, and log level.
    - Auto-adjusts batch size when increasing workers if batch size is not explicitly provided.
    - On successful completion, prints summary statistics (processed, failed, duration, throughput).
    - If interrupted by KeyboardInterrupt, saves progress to checkpoints and exits with code 130.
    - On normal completion, exits with code 0 on success or 1 on failure; other failures also result in exit code 1.
    
    Side effects:
    - Writes to stdout/stderr, configures logging, may prompt the user, and may call sys.exit().
    """
    parser = argparse.ArgumentParser(
        description="Process ArXiv metadata with embeddings"
    )

    parser.add_argument(
        '--mode',
        choices=['test', 'development', 'production'],
        default='test',
        help='Processing mode (default: test)'
    )

    parser.add_argument(
        '--config',
        default='configs/metadata_pipeline.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Skip confirmation for production mode'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    parser.add_argument(
        '--drop-collections',
        action='store_true',
        help='Drop existing collections before processing'
    )

    parser.add_argument(
        '--max-records',
        type=int,
        help='Override maximum records to process'
    )

    parser.add_argument(
        '--count',
        type=int,
        help='Number of records to process (alias for --max-records)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        choices=[1, 2],
        help='Number of GPU workers (1 or 2, default: 1)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Print banner
    print_banner()

    # Check for ARANGO_PASSWORD
    if not os.environ.get('ARANGO_PASSWORD'):
        print("❌ ERROR: ARANGO_PASSWORD environment variable not set")
        print("Please set: export ARANGO_PASSWORD='your-password'")
        sys.exit(1)

    try:
        # Load configuration
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent / config_path

        if not config_path.exists():
            print(f"❌ ERROR: Configuration file not found: {config_path}")
            sys.exit(1)

        config = load_config(str(config_path), args.mode)

        # Apply command-line overrides
        if args.drop_collections:
            config.drop_collections = True

        # Handle --count as alias for --max-records
        if args.count is not None:
            config.max_records = args.count
        elif args.max_records is not None:
            config.max_records = args.max_records

        # Handle --resume flag
        if args.resume:
            config.resume_from_checkpoint = True
            config.drop_collections = False  # Don't drop if resuming
            print("📌 Resuming from checkpoint...")

        # Handle --workers for multi-GPU
        if args.workers > 1:
            config.num_workers = args.workers
            # Adjust batch size for multi-GPU if not explicitly set
            if args.batch_size is None:
                config.batch_size = 200  # 100 per worker for 2 workers
                print(f"📊 Auto-adjusted batch size to {config.batch_size} for {args.workers} workers")

        if args.batch_size is not None:
            config.batch_size = args.batch_size

        if args.no_gpu:
            config.use_gpu = False

        # Print configuration summary
        print_config_summary(config, args.mode)

        # Confirm production mode
        if args.mode == 'production' and not args.confirm:
            if not confirm_production():
                print("Aborted by user")
                sys.exit(0)

        # Validate configuration
        config.validate_full()

        # Create and execute workflow
        print("Starting workflow...")

        # Choose workflow based on number of workers
        if hasattr(config, 'num_workers') and config.num_workers > 1:
            print(f"🚀 Using parallel workflow with {config.num_workers} GPU workers")
            workflow = ArxivParallelWorkflow(config)
        else:
            print("🔧 Using single-GPU workflow")
            workflow = ArxivMetadataWorkflow(config)

        print()
        result = workflow.execute()

        # Print results
        print()
        print("=" * 60)
        print("WORKFLOW COMPLETED")
        print("=" * 60)
        print(f"  Success: {result.success}")
        print(f"  Processed: {result.items_processed:,}")
        print(f"  Failed: {result.items_failed:,}")
        print(f"  Duration: {result.duration_seconds:.2f} seconds")
        print(f"  Success Rate: {result.success_rate:.2f}%")

        if result.metadata:
            throughput = result.metadata.get('throughput', 0)
            print(f"  Throughput: {throughput:.2f} records/second")

            # Check if target throughput was met
            if throughput >= config.target_throughput:
                print(f"  ✅ Target throughput achieved! ({throughput:.2f} >= {config.target_throughput})")
            else:
                print(f"  ⚠️  Below target throughput ({throughput:.2f} < {config.target_throughput})")

        # Print progress summary
        if hasattr(workflow, 'progress_tracker'):
            print()
            print("Progress Summary:")
            for step in workflow.progress_tracker.steps.values():
                print(f"  {step.name}: {step.completed_items:,}/{step.total_items:,} ({step.completion_percent:.1f}%)")

        # Exit with appropriate code
        sys.exit(0 if result.success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Progress has been saved to checkpoint files")
        print("Run again with same configuration to resume")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()