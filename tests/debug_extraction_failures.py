#!/usr/bin/env python3
"""
Debug Extraction Failures
========================

Test script to debug why specific documents are failing extraction.
Tests with 10 known failed documents from the 1608 batch.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.arxiv.arxiv_document_manager import ArXivDocumentManager
from core.processors.generic_document_processor import GenericDocumentProcessor
import yaml

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level for more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Failed documents from the last run
    failed_ids = [
        '1608.02383', '1608.02337', '1608.02303', '1608.02302', '1608.02296',
        '1608.02295', '1608.02285', '1608.02282', '1608.02281', '1608.02280'
    ]
    
    logger.info(f"Testing extraction with {len(failed_ids)} previously failed documents")
    
    # Load configuration
    config_path = 'tools/arxiv/configs/acid_pipeline_phased.yaml'
    config = load_config(config_path)
    
    # Override to use fewer workers for debugging
    config['phases']['extraction']['workers'] = 2  # Less parallelism for clearer logs
    config['phases']['embedding']['workers'] = 2
    
    # Disable checkpointing for this test
    config['checkpoint']['enabled'] = False
    
    # Initialize components
    arxiv_manager = ArXivDocumentManager(pdf_base_dir='/bulk-store/arxiv-data/pdf')
    processor = GenericDocumentProcessor(config=config, collection_prefix="test")
    
    # Prepare documents
    tasks = arxiv_manager.prepare_documents_from_ids(failed_ids)
    
    if not tasks:
        logger.error("No tasks prepared!")
        return
    
    logger.info(f"Prepared {len(tasks)} tasks")
    
    # Check if PDFs exist
    for task in tasks:
        pdf_path = Path(task.pdf_path)
        if pdf_path.exists():
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {task.document_id}: PDF exists ({size_mb:.2f} MB)")
        else:
            logger.error(f"✗ {task.document_id}: PDF NOT FOUND at {task.pdf_path}")
    
    # Process documents
    logger.info("="*80)
    logger.info("Starting extraction test...")
    logger.info("="*80)
    
    try:
        results = processor.process_documents(tasks)
        
        # Analyze results
        logger.info("="*80)
        logger.info("RESULTS:")
        logger.info(f"Extraction success: {len(results['extraction']['success'])}")
        logger.info(f"Extraction failed: {len(results['extraction']['failed'])}")
        
        if results['extraction']['success']:
            logger.info(f"Successfully extracted: {results['extraction']['success']}")
        
        if results['extraction']['failed']:
            logger.error(f"Failed to extract: {results['extraction']['failed']}")
        
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)

if __name__ == "__main__":
    main()