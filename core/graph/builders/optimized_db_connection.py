#!/usr/bin/env python3
"""
Optimized database connection with connection pooling and batch optimizations.
"""

import os
from arango import ArangoClient
from arango.http import HTTPClient
import logging

logger = logging.getLogger(__name__)


class OptimizedArangoConnection:
    """Optimized ArangoDB connection with pooling and performance tweaks."""
    
    def __init__(self, pool_size: int = 10):
        """Initialize optimized connection.
        
        Args:
            pool_size: Number of connections in the pool
        """
        # Create client with custom HTTP settings
        self.client = ArangoClient(
            hosts='http://127.0.0.1:8529',
            http_client=HTTPClient(
                pool_connections=pool_size,  # Connection pool size
                pool_maxsize=pool_size * 2,  # Max pool size
                max_retries=3,
                pool_block=False  # Non-blocking pool
            )
        )
        
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        logger.info(f"Initialized optimized connection with pool size {pool_size}")
    
    def bulk_insert(self, collection_name: str, documents: list, chunk_size: int = 1000000):
        """Bulk insert documents with optimal batching.
        
        Args:
            collection_name: Name of the collection
            documents: List of documents to insert
            chunk_size: Size of each insertion batch
        """
        collection = self.db.collection(collection_name)
        
        # Process in chunks
        for i in range(0, len(documents), chunk_size):
            chunk = documents[i:i+chunk_size]
            
            # Use import_bulk for maximum performance
            result = collection.import_bulk(
                chunk,
                on_duplicate='replace',  # Replace on duplicate
                batch_size=10000,  # Internal batch size
                details=False  # Don't return details for speed
            )
            
            if result['errors'] > 0:
                logger.warning(f"Bulk insert had {result['errors']} errors")
    
    def fast_insert_many(self, collection_name: str, documents: list):
        """Fast insert using insert_many with optimizations.
        
        Args:
            collection_name: Name of the collection
            documents: List of documents to insert
        """
        collection = self.db.collection(collection_name)
        
        # Insert with optimizations
        collection.insert_many(
            documents,
            overwrite=True,  # Overwrite existing
            silent=True,  # Don't return created documents
            sync=False  # Don't wait for sync to disk
        )
    
    def get_optimized_db(self):
        """Get the optimized database connection."""
        return self.db


# Singleton instance
_connection = None

def get_optimized_connection(pool_size: int = 10):
    """Get or create the optimized connection singleton.
    
    Args:
        pool_size: Connection pool size
        
    Returns:
        OptimizedArangoConnection instance
    """
    global _connection
    if _connection is None:
        _connection = OptimizedArangoConnection(pool_size=pool_size)
    return _connection