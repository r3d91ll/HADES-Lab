#!/usr/bin/env python3
"""
MCP Tool Endpoints for ArXiv Pipeline
======================================

Provides MCP-compatible tool functions for the hybrid pipeline.
These can be integrated into the MCP server for Claude integration.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Union

# Add HADES root to path
hades_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(hades_root))

from tools.arxiv.hybrid_pipeline_v3 import HybridPipelineV3

logger = logging.getLogger(__name__)


class ArxivMCPTools:
    """
    MCP tool endpoints for ArXiv processing.
    
    These methods are designed to be called from the MCP server
    and provide a clean interface to the hybrid pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MCP tools.
        
        Args:
            config_path: Path to config file (defaults to standard location)
        """
        if config_path is None:
            config_path = str(hades_root / 'configs/processors/arxiv_hybrid.yaml')
        
        self.config_path = config_path
        self.pipeline = None
    
    def _init_pipeline(self) -> HybridPipelineV3:
        """Initialize pipeline if not already initialized."""
        if self.pipeline is None:
            # Get passwords from environment
            pg_password = os.environ.get('PGPASSWORD', '')
            arango_password = os.environ.get('ARANGO_PASSWORD', '')
            
            if not pg_password or not arango_password:
                raise ValueError("Database passwords not set in environment")
            
            self.pipeline = HybridPipelineV3(
                config_path=self.config_path,
                pg_password=pg_password,
                arango_password=arango_password
            )
        
        return self.pipeline
    
    async def process_date_range(
        self,
        start_date: str,
        end_date: str,
        categories: Optional[List[str]] = None,
        authors: Optional[List[str]] = None,
        max_papers: Optional[int] = None
    ) -> Dict:
        """
        Process papers within a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            categories: Optional list of categories (e.g., ['cs.AI', 'cs.LG'])
            authors: Optional list of author names
            max_papers: Maximum number of papers to process
            
        Returns:
            Processing statistics
            
        Example:
            >>> await process_date_range('2013-01-01', '2013-06-30', categories=['cs.AI'])
        """
        pipeline = self._init_pipeline()
        
        # Get papers in date range
        papers = pipeline.get_papers_by_date_range(
            start_date=start_date,
            end_date=end_date,
            categories=categories,
            authors=authors,
            limit=max_papers
        )
        
        # Process papers
        stats = pipeline.run(papers)
        
        return {
            'processed': stats['processed'],
            'failed': stats['failed'],
            'duration_seconds': stats.get('duration', 0),
            'papers_per_minute': stats.get('papers_per_minute', 0),
            'date_range': f"{start_date} to {end_date}",
            'filters': {
                'categories': categories,
                'authors': authors
            }
        }
    
    async def process_specific_papers(
        self,
        arxiv_ids: List[str],
        force: bool = False
    ) -> Dict:
        """
        Process specific papers by their arXiv IDs.
        
        Args:
            arxiv_ids: List of arXiv IDs (e.g., ['1301.0001', 'cond-mat/0001001'])
            force: If True, reprocess even if already in database
            
        Returns:
            Processing statistics
            
        Example:
            >>> await process_specific_papers(['1301.0001', '1301.0002'])
        """
        pipeline = self._init_pipeline()
        
        # Clear processed IDs if force flag is set
        if force:
            pipeline.checkpoint['processed_ids'] = set()
        
        # Get papers by IDs
        papers = pipeline.get_papers_by_ids(arxiv_ids)
        
        # Process papers
        stats = pipeline.run(papers)
        
        return {
            'processed': stats['processed'],
            'failed': stats['failed'],
            'requested': len(arxiv_ids),
            'found': len(papers),
            'duration_seconds': stats.get('duration', 0),
            'force_reprocess': force
        }
    
    async def process_paper_list_file(
        self,
        file_path: str,
        force: bool = False
    ) -> Dict:
        """
        Process papers from a file containing arXiv IDs.
        
        Args:
            file_path: Path to file with arXiv IDs (one per line)
            force: If True, reprocess even if already in database
            
        Returns:
            Processing statistics
            
        Example:
            >>> await process_paper_list_file('/tmp/papers_to_process.txt')
        """
        pipeline = self._init_pipeline()
        
        # Clear processed IDs if force flag is set
        if force:
            pipeline.checkpoint['processed_ids'] = set()
        
        # Get papers from file
        papers = pipeline.get_papers_from_file(file_path)
        
        # Process papers
        stats = pipeline.run(papers)
        
        return {
            'processed': stats['processed'],
            'failed': stats['failed'],
            'found': len(papers),
            'duration_seconds': stats.get('duration', 0),
            'file_path': file_path,
            'force_reprocess': force
        }
    
    async def process_experiment_window(
        self,
        categories: Optional[List[str]] = None,
        authors: Optional[List[str]] = None,
        max_papers: Optional[int] = None
    ) -> Dict:
        """
        Process all papers in the configured experiment window.
        
        Args:
            categories: Optional list of categories to filter
            authors: Optional list of author names to filter
            max_papers: Maximum number of papers to process
            
        Returns:
            Processing statistics
            
        Example:
            >>> await process_experiment_window(categories=['cs.AI'], max_papers=100)
        """
        pipeline = self._init_pipeline()
        
        # Get experiment window from config
        experiment_config = pipeline.config['experiment']
        
        # Get papers in experiment window
        papers = pipeline.get_papers_by_date_range(
            start_date=experiment_config['start_date'],
            end_date=experiment_config['end_date'],
            categories=categories,
            authors=authors,
            limit=max_papers
        )
        
        # Process papers
        stats = pipeline.run(papers)
        
        return {
            'processed': stats['processed'],
            'failed': stats['failed'],
            'duration_seconds': stats.get('duration', 0),
            'papers_per_minute': stats.get('papers_per_minute', 0),
            'experiment_window': f"{experiment_config['start_date']} to {experiment_config['end_date']}",
            'filters': {
                'categories': categories,
                'authors': authors,
                'max_papers': max_papers
            }
        }
    
    async def get_processing_status(self) -> Dict:
        """
        Get current processing status and checkpoint information.
        
        Returns:
            Status information including checkpoint stats
            
        Example:
            >>> await get_processing_status()
        """
        pipeline = self._init_pipeline()
        
        # Load latest checkpoint
        checkpoint = pipeline.load_checkpoint()
        
        # Get counts from ArangoDB
        processed_in_db = 0
        try:
            query = f"""
                FOR doc IN {pipeline.embeddings_collection}
                COLLECT WITH COUNT INTO count
                RETURN count
            """
            cursor = pipeline.arango_db.aql.execute(query)
            processed_in_db = next(cursor, 0)
        except Exception as e:
            logger.warning(f"Could not check ArangoDB: {e}")
        
        return {
            'checkpoint': {
                'total_processed': checkpoint.get('total_processed', 0),
                'total_failed': checkpoint.get('total_failed', 0),
                'last_save': checkpoint.get('last_save', 'Never'),
                'failed_papers': len(checkpoint.get('failed_ids', {}))
            },
            'database': {
                'papers_with_embeddings': processed_in_db,
                'embeddings_collection': pipeline.embeddings_collection,
                'structures_collection': pipeline.structures_collection
            },
            'config': {
                'experiment_window': f"{pipeline.config['experiment']['start_date']} to {pipeline.config['experiment']['end_date']}",
                'batch_size': pipeline.config['processor']['batch_size'],
                'pdf_base_path': pipeline.config['processor']['pdf_base_path']
            }
        }
    
    async def clear_checkpoint(self) -> Dict:
        """
        Clear the checkpoint to start fresh.
        
        Returns:
            Confirmation message
            
        Example:
            >>> await clear_checkpoint()
        """
        pipeline = self._init_pipeline()
        
        old_checkpoint = pipeline.checkpoint.copy()
        
        pipeline.checkpoint = {
            'processed_ids': set(),
            'failed_ids': {},
            'total_processed': 0,
            'total_failed': 0,
            'last_batch_time': None
        }
        
        pipeline.save_checkpoint()
        
        return {
            'status': 'cleared',
            'old_stats': {
                'total_processed': old_checkpoint.get('total_processed', 0),
                'total_failed': old_checkpoint.get('total_failed', 0)
            }
        }
    
    async def get_failed_papers(self, limit: int = 10) -> List[Dict]:
        """
        Get list of papers that failed processing.
        
        Args:
            limit: Maximum number of failed papers to return
            
        Returns:
            List of failed paper information
            
        Example:
            >>> await get_failed_papers(limit=5)
        """
        pipeline = self._init_pipeline()
        
        checkpoint = pipeline.load_checkpoint()
        failed_ids = checkpoint.get('failed_ids', {})
        
        failed_papers = []
        for arxiv_id, info in list(failed_ids.items())[:limit]:
            failed_papers.append({
                'arxiv_id': arxiv_id,
                'attempts': info.get('attempts', 0),
                'last_attempt': info.get('last_attempt', 'Unknown')
            })
        
        return failed_papers


# For testing the MCP tools directly
async def test_mcp_tools():
    """Test the MCP tool endpoints."""
    tools = ArxivMCPTools()
    
    # Test getting status
    print("Testing get_processing_status...")
    status = await tools.get_processing_status()
    print(json.dumps(status, indent=2))
    
    # Test processing specific papers
    print("\nTesting process_specific_papers...")
    result = await tools.process_specific_papers(
        arxiv_ids=['1301.0001'],
        force=False
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import asyncio
    
    # Set passwords for testing
    if not os.environ.get('PGPASSWORD'):
        print("Please set PGPASSWORD environment variable")
        sys.exit(1)
    if not os.environ.get('ARANGO_PASSWORD'):
        print("Please set ARANGO_PASSWORD environment variable")
        sys.exit(1)
    
    asyncio.run(test_mcp_tools())