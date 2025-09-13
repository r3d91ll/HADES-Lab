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
from core.workflows.workflow_pdf_batch import GenericDocumentProcessor
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.
    
    Parameters:
        config_path (str): Path to a YAML file containing the configuration.
    
    Returns:
        Dict[str, Any]: Parsed configuration as a dictionary (the YAML root should be a mapping).
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ArXivPipelineV2:
    """
    ArXiv processing pipeline using separated architecture.
    """
    
    def __init__(self, config_path: str):
        """
        Create and initialize an ArXivPipelineV2 instance.
        
        Loads configuration from the provided YAML path, determines the PDF base directory (uses config value at `processing.local.pdf_dir` or the default '/bulk-store/arxiv-data/pdf'), instantiates the ArXivDocumentManager and GenericDocumentProcessor (with collection prefix "arxiv"), and initializes checkpointing state.
        
        Parameters:
            config_path (str): Path to the YAML configuration file used to configure the pipeline.
        """
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
        """
        Initialize checkpointing state from configuration and load any existing checkpoint file.
        
        Reads the 'checkpoint' section from self.config and sets:
        - self.checkpoint_enabled: 'true', 'false', or 'auto' (default 'auto')
        - self.checkpoint_auto_threshold: number used when mode is 'auto' (default 500)
        - self.checkpoint_save_interval: how often to persist (default 100)
        - self.checkpoint_file: Path to the checkpoint JSON file (default 'arxiv_pipeline_v2_checkpoint.json')
        
        Loads checkpoint contents via self._load_checkpoint() into self.checkpoint_data and initializes
        self.processed_ids as a set of previously processed document IDs (empty set if no checkpoint).
        Logs the count of loaded processed IDs when any are present.
        """
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
        """
        Load and return checkpoint data from the configured checkpoint file.
        
        If the checkpoint file exists and contains valid JSON, its contents are returned as a dict.
        If the file does not exist or cannot be read/parsed, an empty dict is returned.
        """
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return {}
    
    def _save_checkpoint(self, extraction_results=None, embedding_results=None):
        """
        Persist current checkpoint state to the configured checkpoint file.
        
        Updates internal checkpoint_data with the current set of processed IDs and a last_saved ISO timestamp.
        If provided, records the most recent extraction and embedding phase summaries under `last_extraction` and
        `last_embedding` (each uses keys `success` and `failed`; extraction also includes `staged_files`).
        
        Parameters:
            extraction_results (dict, optional): Extraction phase summary with optional keys
                `success` (list), `failed` (list), and `staged_files` (list). Missing keys default to empty lists.
            embedding_results (dict, optional): Embedding phase summary with optional keys
                `success` (list) and `failed` (list). Missing keys default to empty lists.
        
        Notes:
            - The method writes JSON to `self.checkpoint_file`. I/O errors are logged but not propagated.
            - Does not return a value.
        """
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
        """
        Decide whether checkpointing should be enabled for this run.
        
        If the pipeline config explicitly sets checkpointing to 'true' or 'false', that value is respected.
        Otherwise (auto mode), checkpointing is enabled when the requested document count is greater than
        or equal to the configured auto-threshold.
        
        Parameters:
            count (int): Number of documents requested for this run; used when checkpointing is in auto mode.
        
        Returns:
            bool: True if checkpointing should be used, False otherwise.
        """
        if self.checkpoint_enabled == 'true':
            return True
        elif self.checkpoint_enabled == 'false':
            return False
        else:  # auto mode
            return count >= self.checkpoint_auto_threshold
    
    def run(self, source: str = 'recent', count: int = 100, arxiv_ids: List[str] = None):
        """
        Run the ArXiv processing pipeline: prepare tasks, process them, update checkpoints, and write a results file.
        
        This orchestrates end-to-end pipeline steps:
        - Prepares document tasks from one of three sources: 'recent' (most recent submissions), 'specific' (requires arxiv_ids), or 'directory' (uses configured year_month).
        - Optionally filters out previously processed documents when checkpointing is enabled.
        - Processes documents via the configured GenericDocumentProcessor.
        - Updates and persists checkpoint data with extraction/embedding outcomes when available.
        - Logs a summary report (timing, success/failure counts, processing rate) and writes a JSON results file named arxiv_pipeline_v2_results_<timestamp>.json containing timing, source, requested count, and the full results payload.
        
        Parameters:
            source (str): Source of documents; one of 'recent', 'specific', or 'directory'. If 'specific', provide arxiv_ids.
            count (int): Number of documents to request/limit from the source.
            arxiv_ids (List[str] | None): List of ArXiv IDs to process when source == 'specific'. Omitted otherwise.
        
        Returns:
            None
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
    """
    Command-line entry point for the ArXiv Pipeline v2.
    
    Parses CLI arguments (config path, source type, count, optional ArXiv IDs, and optional ArangoDB password),
    optionally sets the ARANGO_PASSWORD environment variable when provided, constructs an ArXivPipelineV2 using
    the resolved configuration file, and runs the pipeline with the requested source, count, and IDs.
    
    Defaults:
    - config: 'tools/arxiv/configs/acid_pipeline_phased.yaml'
    - source: 'recent' (choices: 'recent', 'specific', 'directory')
    - count: 100
    
    Side effects:
    - May set the ARANGO_PASSWORD environment variable.
    - Instantiates and runs the ArXivPipelineV2, which performs file I/O and network/DB operations as configured.
    """
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