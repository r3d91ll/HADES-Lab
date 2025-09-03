#!/usr/bin/env python3
"""
Process the three GitHub repositories for word2vec evolution experiment.
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import GitHub pipeline manager for direct use
from tools.github.github_pipeline_manager import GitHubPipelineManager

def process_repository(repo_url: str, name: str, manager: GitHubPipelineManager):
    """
    Process a single GitHub repository using the GitHub pipeline.
    
    Args:
        repo_url: GitHub URL (owner/repo format)
        name: Friendly name for logging
        manager: GitHubPipelineManager instance
    """
    logger.info(f"Processing {name} ({repo_url})...")
    
    try:
        # Process the repository directly using the manager
        results = manager.process_repository(repo_url)
        
        # Check results
        if results and 'stored' in results:
            stored_count = results.get('stored', 0)
            logger.info(f"  ✓ {name} processed successfully")
            logger.info(f"    Stored {stored_count} embeddings")
            
            # Log additional stats if available
            if 'repository' in results:
                repo_info = results['repository']
                logger.info(f"    Repository: {repo_info.get('full_name', repo_url)}")
                if 'stats' in repo_info:
                    stats = repo_info['stats']
                    logger.info(f"    Files: {stats.get('file_count', 'N/A')}")
                    logger.info(f"    Chunks: {stats.get('chunk_count', 'N/A')}")
            
            return True
        else:
            logger.error(f"  ✗ {name} processing returned no results")
            return False
            
    except Exception as e:
        logger.error(f"  ✗ {name} failed with error: {e}")
        logger.error(f"    Error details: {str(e)}")
        return False

def main():
    # Our three repositories
    repositories = [
        {
            'repo': 'dav/word2vec',
            'name': 'word2vec',
            'paper': '1301.3781',
            'year': 2013
        },
        {
            'repo': 'bnosac/doc2vec',
            'name': 'doc2vec', 
            'paper': '1405.4053',
            'year': 2014
        },
        {
            'repo': 'tech-srl/code2vec',
            'name': 'code2vec',
            'paper': '1803.09473',
            'year': 2018
        }
    ]
    
    logger.info("=" * 80)
    logger.info("PROCESSING GITHUB REPOSITORIES FOR WORD2VEC EVOLUTION")
    logger.info("=" * 80)
    logger.info(f"Processing {len(repositories)} repositories")
    for repo_info in repositories:
        logger.info(f"  • {repo_info['name']} ({repo_info['year']}): github.com/{repo_info['repo']}")
        logger.info(f"    Paper: arxiv.org/abs/{repo_info['paper']}")
    logger.info("=" * 80)
    
    # First, ensure the GitHub graph is set up
    setup_script = Path(__file__).parent.parent.parent / 'tools' / 'github' / 'setup_github_graph.py'
    if setup_script.exists():
        logger.info("\nEnsuring GitHub graph collections exist...")
        result = subprocess.run(
            [sys.executable, str(setup_script)],
            capture_output=True,
            text=True,
            cwd=str(setup_script.parent)
        )
        
        # Check return code first
        if result.returncode != 0:
            logger.error("Failed to setup GitHub graph collections!")
            logger.error(f"Return code: {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            if result.stdout:
                logger.error(f"Standard output: {result.stdout}")
            sys.exit(1)
        
        # Only check for success messages after confirming return code is 0
        if "created" in result.stdout.lower() or "already exists" in result.stdout.lower():
            logger.info("  ✓ GitHub graph collections ready")
        else:
            logger.warning("  GitHub graph setup completed but status unclear")
            logger.debug(f"  Output: {result.stdout[:500]}")
    
    # Load config and initialize manager
    logger.info("\nInitializing GitHub pipeline manager...")
    config_path = Path(__file__).parent.parent.parent / 'tools' / 'github' / 'configs' / 'github_simple.yaml'
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set ArangoDB password from environment
    arango_password = os.getenv('ARANGO_PASSWORD')
    if not arango_password:
        logger.error("ARANGO_PASSWORD environment variable is required but not set")
        sys.exit(1)
    config['arango']['password'] = arango_password
    
    # Create manager instance
    try:
        manager = GitHubPipelineManager(config)
        logger.info("  ✓ Pipeline manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize GitHub pipeline manager: {e}")
        sys.exit(1)
    
    # Process each repository
    logger.info("\n" + "-" * 80)
    logger.info("Starting repository processing...")
    logger.info("-" * 80)
    
    results = []
    start_time = datetime.now()
    
    for repo_info in repositories:
        logger.info(f"\n[{len(results)+1}/{len(repositories)}] {repo_info['name'].upper()}")
        success = process_repository(repo_info['repo'], repo_info['name'], manager)
        results.append({
            'repo': repo_info['repo'],
            'name': repo_info['name'],
            'paper': repo_info['paper'],
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    logger.info(f"Successful: {successful}/{len(results)}")
    logger.info(f"Total time: {elapsed:.1f} seconds")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        logger.info(f"  {status} {result['name']}: {result['repo']}")
    
    # Save results
    results_file = Path(__file__).parent / 'github_processing_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'repositories': results,
            'successful': successful,
            'total': len(results),
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    if successful == len(results):
        logger.info("\n✓ All repositories processed successfully!")
        return 0
    else:
        logger.info(f"\n⚠ {len(results) - successful} repositories failed")
        return 1

if __name__ == "__main__":
    exit(main())