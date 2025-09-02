#!/usr/bin/env python3
"""
Test GitHub Processing - Simple
================================

Simple test to verify GitHub repository processing works.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.github.github_document_manager import GitHubDocumentManager
from core.framework.extractors.code_extractor import CodeExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_github_clone_and_extract():
    """Test cloning and extracting a small repository."""
    
    # Use a small test repository
    test_repo = "kennethreitz/setup.py"  # Small, simple repo
    
    logger.info(f"Testing with repository: {test_repo}")
    
    # Initialize GitHub manager
    github_manager = GitHubDocumentManager(
        clone_dir="/tmp/github_test",
        cleanup=False  # Keep for inspection
    )
    
    # Prepare repository
    tasks = github_manager.prepare_repository(f"https://github.com/{test_repo}")
    
    if not tasks:
        logger.error("No tasks prepared!")
        return False
    
    logger.info(f"Prepared {len(tasks)} files for processing")
    
    # Test code extraction on first few files
    code_extractor = CodeExtractor()
    
    for i, task in enumerate(tasks[:5]):  # Test first 5 files
        logger.info(f"\nProcessing file {i+1}: {task.document_id}")
        
        # Extract content
        result = code_extractor.extract(task.pdf_path)  # pdf_path holds the file path
        
        if result:
            logger.info(f"  ✓ Extracted {result['num_lines']} lines")
            logger.info(f"  File size: {result['file_size']} bytes")
            logger.info(f"  Metadata: {result.get('metadata', {})}")
        else:
            logger.error(f"  ✗ Extraction failed")
    
    # Cleanup test directory
    import shutil
    test_dir = Path("/tmp/github_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        logger.info(f"\nCleaned up {test_dir}")
    
    return True


def test_multiple_repos():
    """Test processing multiple repositories."""
    
    test_repos = [
        "kennethreitz/setup.py",
        "psf/requests-html",  # Another small repo
    ]
    
    logger.info(f"Testing with {len(test_repos)} repositories")
    
    github_manager = GitHubDocumentManager(cleanup=True)
    
    all_tasks = github_manager.prepare_repositories_from_list(
        [f"https://github.com/{repo}" for repo in test_repos]
    )
    
    logger.info(f"Total files prepared: {len(all_tasks)}")
    
    # Group by repository
    repos = {}
    for task in all_tasks:
        repo_name = task.metadata.get('repository', 'unknown')
        if repo_name not in repos:
            repos[repo_name] = []
        repos[repo_name].append(task)
    
    for repo_name, repo_tasks in repos.items():
        logger.info(f"  {repo_name}: {len(repo_tasks)} files")
    
    return True


def main():
    """Run tests."""
    logger.info("=" * 60)
    logger.info("GitHub Processing Test")
    logger.info("=" * 60)
    
    # Test 1: Clone and extract single repo
    logger.info("\nTest 1: Clone and Extract Single Repository")
    success1 = test_github_clone_and_extract()
    
    # Test 2: Process multiple repos
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Process Multiple Repositories")
    success2 = test_multiple_repos()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info(f"Test 1 (Single repo): {'✓ PASSED' if success1 else '✗ FAILED'}")
    logger.info(f"Test 2 (Multiple repos): {'✓ PASSED' if success2 else '✗ FAILED'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()