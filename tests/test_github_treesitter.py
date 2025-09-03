#!/usr/bin/env python3
"""
Test GitHub Pipeline with Tree-sitter Integration
=================================================

Tests the enhanced GitHub document processing with Tree-sitter symbol extraction.
"""

import sys
import logging
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.github.github_document_manager import GitHubDocumentManager
from core.processors.generic_document_processor import GenericDocumentProcessor
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_tree_sitter_extraction():
    """Test Tree-sitter symbol extraction on a repository."""
    
    # Test repository
    repo_url = "https://github.com/dav/word2vec"
    
    # Initialize GitHub manager
    github_manager = GitHubDocumentManager(
        clone_dir="/bulk-store/git",
        cleanup=False  # Keep repos for analysis
    )
    
    # Clone and prepare documents
    logger.info(f"Cloning repository: {repo_url}")
    tasks = github_manager.prepare_repository(repo_url)
    
    if not tasks:
        logger.error("No tasks generated from repository")
        return
    
    logger.info(f"Generated {len(tasks)} document tasks")
    
    # Test Tree-sitter extraction on a few files
    from core.framework.extractors.code_extractor import CodeExtractor
    
    code_extractor = CodeExtractor(use_tree_sitter=True)
    
    # Process first 3 code files
    code_count = 0
    for task in tasks[:10]:
        if task.metadata.get('type') == 'code':
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {task.document_id}")
            logger.info(f"File: {task.pdf_path}")
            
            # Extract with Tree-sitter
            result = code_extractor.extract(task.pdf_path)
            
            if result and result.get('symbols'):
                logger.info(f"Language: {result.get('language')}")
                
                # Display symbol counts
                symbols = result['symbols']
                logger.info("Symbol Table:")
                for category, items in symbols.items():
                    if items:
                        logger.info(f"  {category}: {len(items)} items")
                        # Show first 3 items
                        for item in items[:3]:
                            if isinstance(item, dict):
                                name = item.get('name', item.get('statement', str(item)))
                                line = item.get('line', '?')
                                logger.info(f"    - {name} (line {line})")
                
                # Display metrics
                if result.get('code_metrics'):
                    metrics = result['code_metrics']
                    logger.info("Code Metrics:")
                    logger.info(f"  Lines of code: {metrics.get('lines_of_code', 0)}")
                    logger.info(f"  Complexity: {metrics.get('complexity', 0)}")
                    logger.info(f"  Max depth: {metrics.get('max_depth', 0)}")
                    logger.info(f"  Node count: {metrics.get('node_count', 0)}")
                
                # Display symbol hash
                if result.get('symbol_hash'):
                    logger.info(f"Symbol hash: {result['symbol_hash'][:16]}...")
                
                code_count += 1
                if code_count >= 3:
                    break
            else:
                logger.info("No Tree-sitter symbols extracted (might not be a supported language)")
    
    logger.info(f"\n{'='*60}")
    logger.info("Tree-sitter extraction test complete!")
    
    # Now test full pipeline with a small subset
    logger.info("\nTesting full pipeline with Tree-sitter metadata...")
    
    # Configure for GitHub processing
    config = {
        'arango': {
            'host': os.getenv('ARANGO_HOST', 'http://192.168.1.69:8529'),
            'database': 'academy_store',
            'username': 'root',
            'password': os.getenv('ARANGO_PASSWORD')
        },
        'phases': {
            'extraction': {
                'type': 'code',  # Use CodeExtractor for code files
                'workers': 2,
                'timeout_seconds': 30
            },
            'embedding': {
                'workers': 1,
                'gpu_devices': [0]
            }
        },
        'staging': {
            'directory': '/dev/shm/github_test_staging'
        }
    }
    
    # Create staging directory
    Path(config['staging']['directory']).mkdir(parents=True, exist_ok=True)
    
    # Initialize processor with GitHub collections
    processor = GenericDocumentProcessor(config=config)
    # Override collections for GitHub
    processor.collections = {
        'papers': 'github_papers',
        'chunks': 'github_chunks',
        'embeddings': 'github_embeddings',
        'structures': 'github_structures'
    }
    
    # Process just 2 files to test
    test_tasks = tasks[:2]
    logger.info(f"\nProcessing {len(test_tasks)} files through full pipeline...")
    
    results = processor.process_documents(test_tasks)
    
    # Display results
    logger.info("\nPipeline Results:")
    logger.info(f"Total processed: {results.get('total_processed', 0)}")
    logger.info(f"Successful: {results.get('extraction_success', 0) + results.get('embedding_success', 0)}")
    logger.info(f"Failed: {results.get('extraction_failed', 0) + results.get('embedding_failed', 0)}")
    
    if results.get('processing_rate'):
        logger.info(f"Processing rate: {results['processing_rate']:.2f} docs/min")
    
    # Check if Tree-sitter metadata was stored
    if results.get('embedding_success', 0) > 0:
        logger.info("\nChecking stored metadata in ArangoDB...")
        from core.database.arango_db_manager import ArangoDBManager
        
        db_manager = ArangoDBManager(config['arango'])
        
        # Query for documents with Tree-sitter data
        query = """
        FOR doc IN github_papers
            FILTER doc.has_tree_sitter == true
            LIMIT 5
            RETURN {
                id: doc.document_id,
                language: doc.language,
                has_symbols: doc.has_tree_sitter,
                function_count: LENGTH(doc.symbols.functions),
                class_count: LENGTH(doc.symbols.classes),
                complexity: doc.code_metrics.complexity
            }
        """
        
        cursor = db_manager.db.aql.execute(query)
        docs_with_symbols = list(cursor)
        
        if docs_with_symbols:
            logger.info(f"Found {len(docs_with_symbols)} documents with Tree-sitter symbols:")
            for doc in docs_with_symbols:
                logger.info(f"  - {doc['id']}: {doc['language']} "
                          f"({doc['function_count']} functions, "
                          f"{doc['class_count']} classes, "
                          f"complexity: {doc['complexity']})")
        else:
            logger.info("No documents with Tree-sitter symbols found in database")
    
    # Clean up staging
    import shutil
    shutil.rmtree(config['staging']['directory'], ignore_errors=True)
    
    logger.info("\nTest complete!")


if __name__ == "__main__":
    test_tree_sitter_extraction()