#!/usr/bin/env python3
"""
GitHub Pipeline
===============

Processes GitHub repositories using the generic document processor.
Similar architecture to ArXiv pipeline but for code repositories.
"""

import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.github.github_document_manager import GitHubDocumentManager
from core.processors.generic_document_processor import GenericDocumentProcessor

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


class GitHubPipeline:
    """
    GitHub processing pipeline using generic document processor.
    """
    
    def __init__(self, config_path: str):
        """Initialize pipeline."""
        self.config = load_config(config_path)
        
        # Initialize GitHub manager
        github_config = self.config.get('processing', {}).get('github', {})
        self.github_manager = GitHubDocumentManager(
            clone_dir=github_config.get('clone_dir', '/tmp/github_repos'),
            cleanup=github_config.get('cleanup_after_processing', True)
        )
        
        # Initialize generic processor with "github" prefix
        self.processor = GenericDocumentProcessor(
            config=self.config,
            collection_prefix="github"
        )
        
        logger.info("Initialized GitHub Pipeline")
    
    def process_repository(self, repo_url: str) -> Dict[str, Any]:
        """
        Process a single repository.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            Processing results
        """
        start_time = datetime.now()
        
        logger.info(f"{'='*60}")
        logger.info(f"Processing repository: {repo_url}")
        logger.info(f"{'='*60}")
        
        # Step 1: Prepare repository files
        tasks = self.github_manager.prepare_repository(repo_url)
        
        if not tasks:
            logger.warning(f"No files to process in {repo_url}")
            return {'success': False, 'error': 'No files found'}
        
        logger.info(f"Found {len(tasks)} files to process")
        
        # Step 2: Process files through generic processor
        results = self.processor.process_documents(tasks)
        
        # Step 3: Report results
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        logger.info(f"{'='*60}")
        logger.info(f"PROCESSING COMPLETE")
        logger.info(f"Repository: {repo_url}")
        logger.info(f"Time: {elapsed:.1f} seconds")
        
        if results['success']:
            extraction = results['extraction']
            embedding = results['embedding']
            
            logger.info(f"Files extracted: {len(extraction['success'])}")
            logger.info(f"Files embedded: {len(embedding['success'])}")
            
            if extraction['failed']:
                logger.warning(f"Extraction failures: {len(extraction['failed'])}")
            if embedding['failed']:
                logger.warning(f"Embedding failures: {len(embedding['failed'])}")
        
        logger.info(f"{'='*60}")
        
        return results
    
    def process_repositories(self, repo_urls: List[str]) -> Dict[str, Any]:
        """
        Process multiple repositories.
        
        Args:
            repo_urls: List of repository URLs
            
        Returns:
            Combined processing results
        """
        all_results = {
            'repositories': [],
            'total_files': 0,
            'total_success': 0,
            'total_failed': 0
        }
        
        for repo_url in repo_urls:
            try:
                results = self.process_repository(repo_url)
                all_results['repositories'].append({
                    'url': repo_url,
                    'results': results
                })
                
                if results.get('success'):
                    all_results['total_files'] += results.get('total_processed', 0)
                    all_results['total_success'] += len(results['embedding']['success'])
                    all_results['total_failed'] += len(results['embedding']['failed'])
                    
            except Exception as e:
                logger.error(f"Failed to process {repo_url}: {e}")
                all_results['repositories'].append({
                    'url': repo_url,
                    'error': str(e)
                })
        
        return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GitHub Repository Processing Pipeline')
    parser.add_argument('--config', type=str,
                       default='configs/github_simple.yaml',
                       help='Configuration file path')
    parser.add_argument('--repo', type=str,
                       help='Single repository URL to process')
    parser.add_argument('--repos', type=str, nargs='+',
                       help='Multiple repository URLs to process')
    parser.add_argument('--file', type=str,
                       help='File containing repository URLs (one per line)')
    parser.add_argument('--arango-password', type=str,
                       help='ArangoDB password (overrides env)')
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.arango_password:
        os.environ['ARANGO_PASSWORD'] = args.arango_password
    
    # Get list of repositories to process
    repo_urls = []
    
    if args.repo:
        repo_urls = [args.repo]
    elif args.repos:
        repo_urls = args.repos
    elif args.file:
        with open(args.file, 'r') as f:
            repo_urls = [line.strip() for line in f if line.strip()]
    else:
        # Default test repository
        repo_urls = ['https://github.com/kennethreitz/setup.py']
    
    # Create and run pipeline
    pipeline = GitHubPipeline(args.config)
    
    if len(repo_urls) == 1:
        results = pipeline.process_repository(repo_urls[0])
    else:
        results = pipeline.process_repositories(repo_urls)
    
    # Save results
    results_file = Path(f"github_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()