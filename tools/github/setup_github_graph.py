#!/usr/bin/env python3
"""
Setup GitHub Graph Collections
===============================

Creates the graph structure for GitHub repository storage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.arxiv.pipelines.arango_db_manager import ArangoDBManager
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_github_graph():
    """Create GitHub graph collections and indexes."""
    
    # Connect to ArangoDB
    arango_password = os.getenv('ARANGO_PASSWORD')
    if not arango_password:
        raise RuntimeError("ARANGO_PASSWORD environment variable is required but not set")
    
    config = {
        'host': os.getenv('ARANGO_HOST', 'http://192.168.1.69:8529'),
        'database': 'academy_store',
        'username': 'root',
        'password': arango_password
    }
    
    db_manager = ArangoDBManager(config)
    db = db_manager.db
    
    logger.info("Setting up GitHub graph collections...")
    
    # 1. Create vertex collections
    vertex_collections = [
        'github_repositories',  # Repository metadata
        'github_papers',        # Files (keep existing)
        'github_chunks',        # Text chunks (keep existing)
        'github_embeddings'     # Embeddings (keep existing)
    ]
    
    for collection_name in vertex_collections:
        if not db.has_collection(collection_name):
            collection = db.create_collection(collection_name)
            logger.info(f"Created vertex collection: {collection_name}")
        else:
            logger.info(f"Collection exists: {collection_name}")
    
    # 2. Create edge collections
    edge_collections = [
        ('github_repo_files', 'github_repositories', 'github_papers'),      # repo -> files
        ('github_has_chunks', 'github_papers', 'github_chunks'),           # file -> chunks
        ('github_has_embeddings', 'github_chunks', 'github_embeddings'),   # chunk -> embeddings
    ]
    
    for edge_name, from_vertex, to_vertex in edge_collections:
        if not db.has_collection(edge_name):
            collection = db.create_collection(edge_name, edge=True)
            logger.info(f"Created edge collection: {edge_name} ({from_vertex} -> {to_vertex})")
        else:
            logger.info(f"Edge collection exists: {edge_name}")
    
    # 3. Create indexes for efficient queries
    
    # Repository indexes
    repo_collection = db.collection('github_repositories')
    try:
        repo_collection.add_hash_index(fields=['full_name'], unique=True)
        logger.info("Created index on github_repositories.full_name")
    except Exception as e:
        logger.warning(f"Index on full_name may already exist: {e}")
    
    try:
        repo_collection.add_hash_index(fields=['owner'])
        logger.info("Created index on github_repositories.owner")
    except Exception as e:
        logger.warning(f"Index on owner may already exist: {e}")
    
    try:
        repo_collection.add_hash_index(fields=['language'])
        logger.info("Created index on github_repositories.language")
    except Exception as e:
        logger.warning(f"Index on language may already exist: {e}")
    
    # Papers indexes (files)
    papers_collection = db.collection('github_papers')
    try:
        papers_collection.add_hash_index(fields=['repository'])  # Quick filter by repo
        logger.info("Created index on github_papers.repository")
    except Exception as e:
        logger.warning(f"Index on repository may already exist: {e}")
    
    try:
        papers_collection.add_hash_index(fields=['language'])    # Filter by language
        logger.info("Created index on github_papers.language")
    except Exception as e:
        logger.warning(f"Index on language may already exist: {e}")
    
    try:
        papers_collection.add_hash_index(fields=['has_tree_sitter'])
        logger.info("Created index on github_papers.has_tree_sitter")
    except Exception as e:
        logger.warning(f"Index on has_tree_sitter may already exist: {e}")
    
    # Chunks indexes
    chunks_collection = db.collection('github_chunks')
    try:
        chunks_collection.add_hash_index(fields=['document_id'])
        logger.info("Created index on github_chunks.document_id")
    except Exception as e:
        logger.warning(f"Index on document_id may already exist: {e}")
    
    # Embeddings indexes
    embeddings_collection = db.collection('github_embeddings')
    try:
        embeddings_collection.add_hash_index(fields=['chunk_id'])
        logger.info("Created index on github_embeddings.chunk_id")
    except Exception as e:
        logger.warning(f"Index on chunk_id may already exist: {e}")
    
    try:
        embeddings_collection.add_hash_index(fields=['has_symbols'])
        logger.info("Created index on github_embeddings.has_symbols")
    except Exception as e:
        logger.warning(f"Index on has_symbols may already exist: {e}")
    
    # Edge indexes for traversal
    try:
        db.collection('github_repo_files').add_hash_index(fields=['_from'])
        db.collection('github_repo_files').add_hash_index(fields=['_to'])
        logger.info("Created indexes on github_repo_files")
    except Exception as e:
        logger.warning(f"Edge indexes on github_repo_files may already exist: {e}")
    
    try:
        db.collection('github_has_chunks').add_hash_index(fields=['_from'])
        db.collection('github_has_chunks').add_hash_index(fields=['_to'])
        logger.info("Created indexes on github_has_chunks")
    except Exception as e:
        logger.warning(f"Edge indexes on github_has_chunks may already exist: {e}")
    
    try:
        db.collection('github_has_embeddings').add_hash_index(fields=['_from'])
        db.collection('github_has_embeddings').add_hash_index(fields=['_to'])
        logger.info("Created indexes on github_has_embeddings")
    except Exception as e:
        logger.warning(f"Edge indexes on github_has_embeddings may already exist: {e}")
    
    # 4. Create a graph if needed
    if not db.has_graph('github_graph'):
        from arango.graph import Graph
        
        github_graph = db.create_graph('github_graph')
        
        # Add vertex collections
        for collection in vertex_collections:
            try:
                github_graph.create_vertex_collection(collection)
            except Exception as e:
                logger.debug(f"Vertex collection {collection} may already exist in graph: {e}")
        
        # Add edge definitions
        github_graph.create_edge_definition(
            edge_collection='github_repo_files',
            from_vertex_collections=['github_repositories'],
            to_vertex_collections=['github_papers']
        )
        
        github_graph.create_edge_definition(
            edge_collection='github_has_chunks',
            from_vertex_collections=['github_papers'],
            to_vertex_collections=['github_chunks']
        )
        
        github_graph.create_edge_definition(
            edge_collection='github_has_embeddings',
            from_vertex_collections=['github_chunks'],
            to_vertex_collections=['github_embeddings']
        )
        
        logger.info("Created github_graph with edge definitions")
    else:
        logger.info("Graph 'github_graph' already exists")
    
    logger.info("\n" + "="*60)
    logger.info("GitHub graph structure ready!")
    logger.info("\nVertex collections:")
    for v in vertex_collections:
        logger.info(f"  - {v}")
    logger.info("\nEdge collections:")
    for e, f, t in edge_collections:
        logger.info(f"  - {e}: {f} -> {t}")
    logger.info("\nYou can now:")
    logger.info("  1. Store repositories with relationships")
    logger.info("  2. Query across all repositories")
    logger.info("  3. Find theory-practice bridges")
    logger.info("  4. Compare repository implementations")


if __name__ == "__main__":
    setup_github_graph()