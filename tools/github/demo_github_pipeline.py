#!/usr/bin/env python3
"""
Demo GitHub Pipeline with Tree-sitter
======================================

Demonstrates the full GitHub pipeline processing code and config files.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.github.github_document_manager import GitHubDocumentManager
from core.processors.generic_document_processor import GenericDocumentProcessor
import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_pipeline():
    """Demo the GitHub pipeline on a repository."""
    
    # Load configuration
    config_path = Path(__file__).parent / "configs" / "github_simple.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Add ArangoDB password
    arango_password = os.getenv('ARANGO_PASSWORD')
    if not arango_password:
        raise ValueError("ARANGO_PASSWORD environment variable is required but not set")
    config['arango']['password'] = arango_password
    
    # Repository to process
    repo_url = "https://github.com/dav/word2vec"
    
    logger.info("="*60)
    logger.info("GitHub Pipeline Demo with Tree-sitter")
    logger.info("="*60)
    
    # Step 1: Clone and prepare repository
    logger.info(f"\n1. Cloning repository: {repo_url}")
    github_manager = GitHubDocumentManager(
        clone_dir=config['processing']['github']['clone_dir'],
        cleanup=config['processing']['github']['cleanup_after_processing']
    )
    
    tasks = github_manager.prepare_repository(repo_url)
    logger.info(f"   Found {len(tasks)} files to process")
    
    # Show file type distribution
    file_types = {}
    for task in tasks:
        ext = Path(task.pdf_path).suffix
        file_types[ext] = file_types.get(ext, 0) + 1
    
    logger.info("\n   File types found:")
    for ext, count in sorted(file_types.items()):
        logger.info(f"     {ext}: {count} files")
    
    # Step 2: Process through pipeline
    logger.info("\n2. Processing files through pipeline...")
    logger.info(f"   - Extraction: Using CodeExtractor with Tree-sitter")
    logger.info(f"   - Embedding: Using Jina v4 with coding LoRA")
    logger.info(f"   - Storage: ArangoDB with symbol metadata")
    
    # Configure processor
    processor = GenericDocumentProcessor(config=config)
    # Override collections for GitHub
    processor.collections = {
        'papers': 'github_papers',
        'chunks': 'github_chunks', 
        'embeddings': 'github_embeddings',
        'structures': 'github_structures'
    }
    
    # Process first 5 files as demo
    demo_tasks = tasks[:5]
    logger.info(f"\n   Processing {len(demo_tasks)} files as demo...")
    
    results = processor.process_documents(demo_tasks)
    
    # Step 3: Show results
    logger.info("\n3. Processing Results:")
    logger.info(f"   - Total processed: {results.get('total_processed', 0)}")
    logger.info(f"   - Extraction phase: {results.get('extraction_success', 0)} success, {results.get('extraction_failed', 0)} failed")
    logger.info(f"   - Embedding phase: {results.get('embedding_success', 0)} success, {results.get('embedding_failed', 0)} failed")
    
    if results.get('processing_rate'):
        logger.info(f"   - Processing rate: {results['processing_rate']:.2f} docs/min")
    
    # Step 4: Query stored data
    logger.info("\n4. Querying stored data...")
    
    from core.database import ArangoDBManager
    db_manager = ArangoDBManager(config['arango'])
    
    # Query for different file types
    query = """
    FOR doc IN @@collection
        COLLECT language = doc.language WITH COUNT INTO count
        SORT count DESC
        RETURN {
            language: language,
            count: count
        }
    """
    bind_vars = {'@collection': 'github_papers'}
    
    cursor = db_manager.db.aql.execute(query, bind_vars=bind_vars)
    language_stats = list(cursor)
    
    if language_stats:
        logger.info("\n   Languages in database:")
        for stat in language_stats:
            logger.info(f"     {stat['language'] or 'unknown'}: {stat['count']} files")
    
    # Query for files with Tree-sitter data
    query = """
    FOR doc IN @@collection
        FILTER doc.has_tree_sitter == true
        LIMIT 3
        RETURN {
            id: doc.document_id,
            language: doc.language,
            symbols: doc.symbols ? {
                functions: LENGTH(doc.symbols.functions || []),
                classes: LENGTH(doc.symbols.classes || []),
                config_type: doc.symbols.config_type
            } : null,
            metrics: doc.code_metrics
        }
    """
    bind_vars = {'@collection': 'github_papers'}
    
    cursor = db_manager.db.aql.execute(query, bind_vars=bind_vars)
    sample_docs = list(cursor)
    
    if sample_docs:
        logger.info("\n   Sample documents with Tree-sitter data:")
        for doc in sample_docs:
            logger.info(f"\n     {doc['id']}:")
            logger.info(f"       Language: {doc['language']}")
            if doc.get('symbols'):
                if doc['symbols'].get('config_type'):
                    logger.info(f"       Config type: {doc['symbols']['config_type']}")
                else:
                    logger.info(f"       Functions: {doc['symbols'].get('functions', 0)}")
                    logger.info(f"       Classes: {doc['symbols'].get('classes', 0)}")
            if doc.get('metrics'):
                logger.info(f"       Complexity: {doc['metrics'].get('complexity', 0)}")
    
    logger.info("\n" + "="*60)
    logger.info("Demo complete!")
    logger.info("\nKey features demonstrated:")
    logger.info("✓ Tree-sitter symbol extraction for code files")
    logger.info("✓ Minimal metadata for config files (JSON, YAML, TOML)")
    logger.info("✓ Jina v4 coding LoRA handles semantic understanding")
    logger.info("✓ Symbol metadata stored in ArangoDB for enhanced retrieval")


if __name__ == "__main__":
    demo_pipeline()