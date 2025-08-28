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
    """
    Load configuration from a YAML file.
    
    Parameters:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        Dict[str, Any]: Parsed configuration as a dictionary (result of `yaml.safe_load`).
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class GitHubPipeline:
    """
    GitHub processing pipeline using generic document processor.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the GitHubPipeline.
        
        Loads configuration from the given YAML path, creates a GitHubDocumentManager (using
        processing.github.clone_dir and processing.github.cleanup_after_processing with
        defaults '/tmp/github_repos' and True), and constructs a GenericDocumentProcessor
        configured with the loaded config and a "github" collection prefix.
        
        Parameters:
            config_path (str): Path to the YAML configuration file used to initialize the pipeline.
        """
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
        Process a single GitHub repository: prepare its files, run the document processor, and return the processor's results.
        
        Parameters:
            repo_url (str): HTTPS GitHub repository URL or repo identifier to process.
        
        Returns:
            Dict[str, Any]: Result dictionary returned by the GenericDocumentProcessor. On failure to find files returns {'success': False, 'error': 'No files found'}. When successful, the dictionary typically contains:
                - 'success' (bool): overall success flag
                - 'extraction' (dict): with keys 'success' (list of extracted items) and 'failed' (list of extraction failures)
                - 'embedding' (dict): with keys 'success' (list of embedded items) and 'failed' (list of embedding failures)
                - other processor-specific metadata (e.g., counts, per-file details)
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
        Process multiple GitHub repository URLs and aggregate per-repository results.
        
        This calls self.process_repository(repo_url) for each URL and builds an aggregate
        summary containing:
        - "repositories": list of per-repo entries; each entry contains "url" and either
          "results" (the processor's result dict) or "error" (stringified exception).
        - "total_files": sum of each repo's results.get('total_processed', 0) when processing succeeded.
        - "total_success": sum of counts of successful embeddings (len(results['embedding']['success'])).
        - "total_failed": sum of counts of failed embeddings (len(results['embedding']['failed'])).
        
        Parameters:
            repo_urls (List[str]): Iterable of repository URLs to process.
        
        Returns:
            Dict[str, Any]: Aggregated results dictionary as described above.
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
    """
    Entry point for the CLI that runs the GitHub repository processing pipeline.
    
    Parses command-line arguments to determine configuration and which repository URLs to process, initializes a GitHubPipeline, runs processing for a single repository or multiple repositories, and writes the results to a timestamped JSON file.
    
    Behavior details:
    - CLI options:
      - --config: path to a YAML config file (default: 'configs/github_simple.yaml').
      - --repo: a single repository URL to process.
      - --repos: one or more repository URLs to process.
      - --file: path to a file containing repository URLs (one per line).
      - --arango-password: optional ArangoDB password; if provided, sets the ARANGO_PASSWORD environment variable for the process.
    - Repository selection priority: --repo, then --repos, then --file. If none provided, defaults to ['https://github.com/kennethreitz/setup.py'].
    - Processing:
      - If a single URL is supplied, calls GitHubPipeline.process_repository; otherwise calls GitHubPipeline.process_repositories.
    - Output:
      - Results are written as pretty-printed JSON to a file named github_results_<YYYYMMDD_HHMMSS>.json in the current working directory.
    - Side effects: may set ARANGO_PASSWORD in the environment and creates a JSON results file.
    
    Returns:
        None
    """
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