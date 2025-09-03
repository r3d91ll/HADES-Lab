#!/usr/bin/env python3
"""
Test C Files with Tree-sitter
==============================

Process C source files from the word2vec repository.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.github.github_document_manager import GitHubDocumentManager
from core.processors.generic_document_processor import GenericDocumentProcessor
from core.framework.extractors.code_extractor import CodeExtractor
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_c_files():
    """Test Tree-sitter on C source files."""
    
    # Get C files from word2vec repo
    repo_path = Path("/bulk-store/git/dav_word2vec")
    c_files = list(repo_path.glob("src/*.c"))[:3]  # Test first 3 C files
    
    logger.info(f"Found {len(c_files)} C files to test")
    
    # Test Tree-sitter extraction
    extractor = CodeExtractor(use_tree_sitter=True)
    
    for c_file in c_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {c_file.name}")
        
        result = extractor.extract(str(c_file))
        
        if result and result.get('symbols'):
            logger.info(f"Language: {result.get('language')}")
            
            symbols = result['symbols']
            logger.info("Symbol Table:")
            for category, items in symbols.items():
                if items:
                    logger.info(f"  {category}: {len(items)} items")
                    for item in items[:3]:
                        if isinstance(item, dict):
                            name = item.get('name', item.get('statement', str(item)))
                            line = item.get('line', '?')
                            logger.info(f"    - {name} (line {line})")
            
            if result.get('code_metrics'):
                metrics = result['code_metrics']
                logger.info("Code Metrics:")
                logger.info(f"  Lines: {metrics.get('lines_of_code', 0)}")
                logger.info(f"  Complexity: {metrics.get('complexity', 0)}")
        else:
            logger.warning(f"No Tree-sitter data for {c_file.name}")
    
    # Now process through pipeline
    logger.info(f"\n{'='*60}")
    logger.info("Processing C files through pipeline...")
    
    # Create tasks for C files
    from core.processors.generic_document_processor import DocumentTask
    
    tasks = []
    for c_file in c_files:
        rel_path = c_file.relative_to(repo_path.parent)
        task = DocumentTask(
            document_id=str(rel_path).replace('/', '_'),
            pdf_path=str(c_file),  # Using pdf_path for file path
            latex_path=None,
            metadata={
                'type': 'code',
                'language': 'c',
                'repo': 'dav/word2vec'
            }
        )
        tasks.append(task)
    
    # Configure pipeline
    config = {
        'arango': {
            'host': os.getenv('ARANGO_HOST', 'http://192.168.1.69:8529'),
            'database': 'academy_store',
            'username': 'root',
            'password': os.getenv('ARANGO_PASSWORD')
        },
        'phases': {
            'extraction': {
                'type': 'code',
                'workers': 2,
                'timeout_seconds': 30
            },
            'embedding': {
                'workers': 1,
                'gpu_devices': [0]
            }
        },
        'staging': {
            'directory': '/dev/shm/c_test_staging'
        }
    }
    
    # Create staging
    Path(config['staging']['directory']).mkdir(parents=True, exist_ok=True)
    
    # Process
    processor = GenericDocumentProcessor(config=config)
    processor.collections = {
        'papers': 'github_papers',
        'chunks': 'github_chunks',
        'embeddings': 'github_embeddings',
        'structures': 'github_structures'
    }
    
    results = processor.process_documents(tasks)
    
    logger.info("\nResults:")
    logger.info(f"Processed: {results.get('total_processed', 0)}")
    logger.info(f"Extraction success: {results.get('extraction_success', 0)}")
    logger.info(f"Embedding success: {results.get('embedding_success', 0)}")
    
    # Query database for C files with symbols
    from core.database.arango_db_manager import ArangoDBManager
    
    db_manager = ArangoDBManager(config['arango'])
    
    query = """
    FOR doc IN github_papers
        FILTER doc.language == "c"
        LIMIT 5
        RETURN {
            id: doc.document_id,
            language: doc.language,
            functions: LENGTH(doc.symbols.functions || []),
            structs: LENGTH(doc.symbols.structs || []),
            includes: LENGTH(doc.symbols.includes || []),
            complexity: doc.code_metrics.complexity
        }
    """
    
    cursor = db_manager.db.aql.execute(query)
    c_docs = list(cursor)
    
    logger.info(f"\nC files in database: {len(c_docs)}")
    for doc in c_docs:
        logger.info(f"  {doc['id']}: {doc['functions']} functions, "
                   f"{doc['structs']} structs, {doc['includes']} includes")
    
    # Clean up
    import shutil
    shutil.rmtree(config['staging']['directory'], ignore_errors=True)
    
    logger.info("\nTest complete!")


if __name__ == "__main__":
    test_c_files()