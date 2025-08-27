#!/usr/bin/env python3
"""
ArXiv Pipeline v2 - Using Separated Architecture
================================================

This pipeline uses the new separated architecture:
1. ArXivDocumentManager - Handles ArXiv-specific logic
2. GenericDocumentProcessor - Handles actual processing

This allows other sources to reuse the same processing pipeline.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.arxiv.arxiv_document_manager import ArXivDocumentManager
from core.processors.generic_document_processor import GenericDocumentProcessor
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ArXivPipelineV2:
    """
    ArXiv processing pipeline using separated architecture.
    """
    
    def __init__(self, config_path: str):
        """Initialize pipeline."""
        self.config = load_config(config_path)
        
        # Initialize ArXiv-specific manager
        pdf_dir = self.config.get('processing', {}).get('local', {}).get('pdf_dir', '/bulk-store/arxiv-data/pdf')
        self.arxiv_manager = ArXivDocumentManager(pdf_base_dir=pdf_dir)
        
        # Initialize generic processor with "arxiv" prefix for collections
        self.processor = GenericDocumentProcessor(
            config=self.config,
            collection_prefix="arxiv"
        )
        
        # Initialize checkpointing
        self._init_checkpoint()
        
        logger.info("Initialized ArXiv Pipeline v2 with separated architecture")
    
    def _init_checkpoint(self):
        """Initialize checkpoint system."""
        checkpoint_config = self.config.get('checkpoint', {})
        self.checkpoint_enabled = checkpoint_config.get('enabled', 'auto')
        self.checkpoint_auto_threshold = checkpoint_config.get('auto_threshold', 500)
        self.checkpoint_save_interval = checkpoint_config.get('save_interval', 100)
        self.checkpoint_file = Path(checkpoint_config.get('file', 'arxiv_pipeline_v2_checkpoint.json'))
        
        # Load existing checkpoint if it exists
        self.checkpoint_data = self._load_checkpoint()
        self.processed_ids = set(self.checkpoint_data.get('processed_ids', []))
        
        if self.processed_ids:
            logger.info(f"Loaded checkpoint with {len(self.processed_ids)} previously processed documents")
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return {}
    
    def _save_checkpoint(self, extraction_results=None, embedding_results=None):
        """Save checkpoint data."""
        self.checkpoint_data['processed_ids'] = list(self.processed_ids)
        self.checkpoint_data['last_saved'] = datetime.now().isoformat()
        
        # Save intermediate phase results if available
        if extraction_results:
            self.checkpoint_data['last_extraction'] = {
                'success': extraction_results.get('success', []),
                'failed': extraction_results.get('failed', []),
                'staged_files': extraction_results.get('staged_files', [])
            }
        
        if embedding_results:
            self.checkpoint_data['last_embedding'] = {
                'success': embedding_results.get('success', []),
                'failed': embedding_results.get('failed', [])
            }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=2)
            logger.debug(f"Checkpoint saved with {len(self.processed_ids)} processed documents")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _should_use_checkpoint(self, count: int) -> bool:
        """Determine if checkpointing should be enabled."""
        if self.checkpoint_enabled == 'true':
            return True
        elif self.checkpoint_enabled == 'false':
            return False
        else:  # auto mode
            return count >= self.checkpoint_auto_threshold
    
    def run(self, source: str = 'recent', count: int = 100, arxiv_ids: List[str] = None):
        """
        Run the pipeline.
        
        Args:
            source: Source of documents ('recent', 'specific', 'directory')
            count: Number of documents to process
            arxiv_ids: Specific ArXiv IDs to process (if source='specific')
        """
        start_time = datetime.now()
        
        # Determine if we should use checkpointing
        use_checkpoint = self._should_use_checkpoint(count)
        
        logger.info(f"{'='*80}")
        logger.info(f"ArXiv Pipeline v2 - Starting")
        logger.info(f"  Source: {source}")
        logger.info(f"  Count: {count}")
        logger.info(f"  Checkpointing: {'ENABLED' if use_checkpoint else 'DISABLED'}")
        if use_checkpoint and self.processed_ids:
            logger.info(f"  Previously processed: {len(self.processed_ids)} documents")
        logger.info(f"{'='*80}")
        
        # Step 1: Prepare documents using ArXiv manager
        if source == 'recent':
            tasks = self.arxiv_manager.prepare_recent_documents(count=count)
        elif source == 'specific' and arxiv_ids:
            tasks = self.arxiv_manager.prepare_documents_from_ids(arxiv_ids)
        elif source == 'directory':
            # Example: process from a specific YYMM directory
            year_month = self.config.get('processing', {}).get('directory', {}).get('year_month', '2301')
            tasks = self.arxiv_manager.prepare_documents_from_directory(year_month, limit=count)
        else:
            logger.error(f"Unknown source: {source}")
            return
        
        if not tasks:
            logger.warning("No documents prepared for processing")
            return
        
        # Filter out already processed documents if checkpointing is enabled
        if use_checkpoint and self.processed_ids:
            original_count = len(tasks)
            tasks = [t for t in tasks if t.document_id not in self.processed_ids]
            if original_count != len(tasks):
                logger.info(f"Filtered out {original_count - len(tasks)} already processed documents")
        
        if not tasks:
            logger.info("All documents already processed according to checkpoint")
            return
        
        logger.info(f"Processing {len(tasks)} ArXiv documents")
        
        # Step 2: Process documents using generic processor
        results = self.processor.process_documents(tasks)
        
        # Step 3: Update checkpoint if enabled
        if use_checkpoint and results.get('success'):
            # Add successfully processed documents to checkpoint
            if 'embedding' in results and 'success' in results['embedding']:
                for doc_id in results['embedding']['success']:
                    self.processed_ids.add(doc_id)
                
                # Save checkpoint periodically or at the end
                self._save_checkpoint(
                    extraction_results=results.get('extraction'),
                    embedding_results=results.get('embedding')
                )
                logger.info(f"Checkpoint updated: {len(self.processed_ids)} total documents processed")
        
        # Step 3: Generate report
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.info(f"{'='*80}")
        logger.info(f"PIPELINE COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        if results['success']:
            extraction = results['extraction']
            embedding = results['embedding']
            
            logger.info(f"Extraction: {len(extraction['success'])} success, {len(extraction['failed'])} failed")
            logger.info(f"Embedding: {len(embedding['success'])} success, {len(embedding['failed'])} failed")
            logger.info(f"Total processed: {results['total_processed']} documents")
            
            if results['total_processed'] > 0:
                rate = results['total_processed'] / elapsed * 60
                logger.info(f"Processing rate: {rate:.1f} documents/minute")
                
                if 'chunks_created' in embedding:
                    avg_chunks = embedding['chunks_created'] / results['total_processed']
                    logger.info(f"Average chunks per document: {avg_chunks:.1f}")
        
        logger.info(f"{'='*80}")
        
        # Save results
        results_file = Path(f"arxiv_pipeline_v2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump({
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'elapsed_seconds': elapsed,
                'source': source,
                'count_requested': count,
                'results': results
            }, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='ArXiv Pipeline v2 - Separated Architecture')
    parser.add_argument('--config', type=str, 
                       default='tools/arxiv/configs/acid_pipeline_phased.yaml',
                       help='Configuration file path')
    parser.add_argument('--source', type=str, default='recent',
                       choices=['recent', 'specific', 'directory'],
                       help='Source of documents')
    parser.add_argument('--count', type=int, default=100,
                       help='Number of documents to process')
    parser.add_argument('--arxiv-ids', type=str, nargs='+',
                       help='Specific ArXiv IDs to process')
    parser.add_argument('--arango-password', type=str,
                       help='ArangoDB password (overrides env)')
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.arango_password:
        os.environ['ARANGO_PASSWORD'] = args.arango_password
    
    # Create and run pipeline
    pipeline = ArXivPipelineV2(args.config)
    pipeline.run(
        source=args.source,
        count=args.count,
        arxiv_ids=args.arxiv_ids
    )


if __name__ == '__main__':
    main()