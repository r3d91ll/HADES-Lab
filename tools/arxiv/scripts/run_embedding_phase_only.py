#!/usr/bin/env python3
"""
Run only the embedding phase of the ACID pipeline.
Used when extraction is complete but embedding failed.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.arxiv_pipeline import PhaseManager, ArangoDBManager
from pipelines.worker_pool import ProcessingWorkerPool
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_staged_files(staging_dir: str) -> List[str]:
    """Get list of staged JSON files."""
    staging_path = Path(staging_dir)
    if not staging_path.exists():
        raise FileNotFoundError(f"Staging directory not found: {staging_dir}")
    
    json_files = sorted(staging_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} staged JSON files")
    return [str(f) for f in json_files]


def main():
    """Main entry point for embedding-only processing."""
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "acid_pipeline_phased.yaml"
    config = load_config(config_path)
    
    # Override with environment password
    arango_password = os.environ.get('ARANGO_PASSWORD')
    if not arango_password:
        logger.error("ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
    
    config['arango']['password'] = arango_password
    
    # Get staged files
    staging_dir = config['staging']['directory']
    json_files = get_staged_files(staging_dir)
    
    if not json_files:
        logger.error("No staged JSON files found")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info(f"EMBEDDING PHASE - Processing {len(json_files)} staged documents")
    logger.info("=" * 80)
    
    # Initialize phase manager
    phase_manager = PhaseManager(config)
    
    # Start embedding phase directly
    try:
        embedding_results = phase_manager.start_embedding_phase(
            json_files=json_files,
            gpu_devices=config['phases']['embedding']['gpu_devices'],
            workers_per_gpu=config['phases']['embedding']['workers_per_gpu']
        )
        
        # Print results
        logger.info("=" * 80)
        logger.info("EMBEDDING PHASE COMPLETE")
        logger.info(f"  Processed: {embedding_results['processed']} papers")
        logger.info(f"  Failed: {embedding_results['failed']} papers")
        logger.info(f"  Time: {embedding_results['time']:.1f} seconds")
        logger.info(f"  Rate: {embedding_results['rate']:.1f} papers/minute")
        logger.info("=" * 80)
        
        # Clean up staging if configured
        if config['staging'].get('cleanup_on_complete', False):
            logger.info("Cleaning up staging directory...")
            for json_file in json_files:
                try:
                    os.remove(json_file)
                except Exception as e:
                    logger.warning(f"Failed to remove {json_file}: {e}")
        
    except Exception as e:
        logger.error(f"Embedding phase failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()