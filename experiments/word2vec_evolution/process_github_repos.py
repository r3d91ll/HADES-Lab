#!/usr/bin/env python3
"""
Process the three GitHub repositories for word2vec evolution experiment.
"""

import os
import sys
import subprocess
import json
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

def process_repository(repo_url: str, name: str):
    """
    Process a single GitHub repository using the GitHub pipeline.
    
    Args:
        repo_url: GitHub URL (owner/repo format)
        name: Friendly name for logging
    """
    logger.info(f"Processing {name} ({repo_url})...")
    
    # Path to GitHub pipeline
    pipeline_script = Path(__file__).parent.parent.parent / 'tools' / 'github' / 'github_pipeline_manager.py'
    
    # Command to run
    cmd = [
        sys.executable,
        str(pipeline_script),
        '--repo', repo_url
    ]
    
    logger.info(f"  Command: {' '.join(cmd)}")
    
    try:
        # Run the pipeline
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(pipeline_script.parent),
            timeout=300  # 5 minute timeout per repo
        )
        
        if result.returncode == 0:
            logger.info(f"  ✓ {name} processed successfully")
            
            # Log any important output
            if "stored" in result.stdout.lower():
                for line in result.stdout.split('\n'):
                    if 'stored' in line.lower() or 'success' in line.lower():
                        logger.info(f"    {line.strip()}")
            
            return True
        else:
            logger.error(f"  ✗ {name} failed with return code: {result.returncode}")
            if result.stderr:
                logger.error(f"    Error: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ {name} timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"  ✗ {name} failed with error: {e}")
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
        if "created" in result.stdout.lower() or "already exists" in result.stdout.lower():
            logger.info("  ✓ GitHub graph collections ready")
    
    # Process each repository
    logger.info("\n" + "-" * 80)
    logger.info("Starting repository processing...")
    logger.info("-" * 80)
    
    results = []
    start_time = datetime.now()
    
    for repo_info in repositories:
        logger.info(f"\n[{len(results)+1}/{len(repositories)}] {repo_info['name'].upper()}")
        success = process_repository(repo_info['repo'], repo_info['name'])
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