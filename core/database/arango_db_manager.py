"""
ArangoDB Manager - Shared Database Management
==============================================

Shared database management module for all HADES-Lab processing pipelines.
Provides connection pooling, transaction support, and distributed locking.
Used by ArXiv, GitHub, and other data source processors.
"""

import os
import time
import random
import logging
from typing import Dict, Any
from functools import wraps
from queue import Queue, Empty
from datetime import datetime, timedelta

from arango import ArangoClient
from arango.exceptions import DocumentInsertError, ArangoServerError

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0, exponential_base=2):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ArangoServerError, ConnectionError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Add jitter to prevent thundering herd
                        jittered_delay = delay * (0.5 + random.random())
                        logger.warning(
                            f"Database operation '{func.__name__}' failed (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}. "
                            f"Retrying in {jittered_delay:.2f}s..."
                        )
                        time.sleep(jittered_delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        raise
                except Exception as e:
                    # Non-retryable exceptions
                    raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class ArangoDBManager:
    """
    ArangoDB connection manager with transaction support and connection pooling.
    
    Implements connection pooling for better performance under concurrent load,
    with automatic retry logic for transient failures.
    """
    
    def __init__(self, config: Dict[str, Any], pool_size: int = 10):
        """
        Initialize connection to ArangoDB with connection pooling.
        
        Args:
            config: Database configuration
            pool_size: Number of connections in the pool (default: 10)
        """
        self.config = config
        self.pool_size = pool_size
        
        # Create connection pool
        self.connections = []
        self.available_connections = Queue(maxsize=pool_size)
        
        # Initialize connections
        for _ in range(pool_size):
            client = ArangoClient(
                hosts=config['host']
            )
            db = client.db(
                config['database'],
                username=config['username'],
                password=os.environ.get('ARANGO_PASSWORD', config.get('password', ''))
            )
            self.connections.append((client, db))
            self.available_connections.put(db)
        
        # Use first connection for setup
        self.db = self.connections[0][1]
        
        # Ensure collections exist
        self._ensure_collections()
    
    def get_connection(self, timeout: float = 5.0):
        """
        Get a connection from the pool.
        
        Args:
            timeout: Maximum time to wait for a connection
            
        Returns:
            Database connection from the pool
        """
        try:
            return self.available_connections.get(timeout=timeout)
        except Empty:
            raise TimeoutError(f"Could not get database connection within {timeout} seconds")
    
    def return_connection(self, db):
        """Return a connection to the pool."""
        self.available_connections.put(db)
    
    def with_connection(self, func):
        """
        Decorator to automatically manage connection checkout/return.
        
        Usage:
            @db_manager.with_connection
            def my_function(db, ...):
                # Use db connection
                pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            db = self.get_connection()
            try:
                return func(db, *args, **kwargs)
            finally:
                self.return_connection(db)
        return wrapper
    
    def _ensure_collections(self):
        """Ensure required collections exist according to ACID schema."""
        collections = [
            'arxiv_papers',     # Metadata collection
            'arxiv_chunks',     # Text chunks
            'arxiv_embeddings', # Vector embeddings
            'arxiv_equations',  # LaTeX equations
            'arxiv_tables',     # Structured tables
            'arxiv_images',     # Image metadata
            'arxiv_locks'       # Distributed locking
        ]
        for coll_name in collections:
            if not self.db.has_collection(coll_name):
                self.db.create_collection(coll_name)
                logger.info(f"Created collection: {coll_name}")
        
        # Create TTL index on locks collection for automatic cleanup
        if self.db.has_collection('arxiv_locks'):
            locks_coll = self.db.collection('arxiv_locks')
            # Check if TTL index already exists
            existing_indexes = locks_coll.indexes()
            has_ttl = any(idx.get('type') == 'ttl' for idx in existing_indexes)
            if not has_ttl:
                locks_coll.add_ttl_index(fields=['expiresAt'], expiry_time=0)
                logger.info("Created TTL index on arxiv_locks collection")
    
    def insert_document(self, collection: str, document: Dict[str, Any]):
        """Insert a document into a collection."""
        coll = self.db.collection(collection)
        try:
            result = coll.insert(document, overwrite=True)
            return result
        except DocumentInsertError as e:
            logger.error(f"Failed to insert document into {collection}: {type(e).__name__}: {e}")
            raise
    
    @retry_with_backoff(max_retries=3, base_delay=0.5)
    def begin_transaction(self, write_collections, read_collections=None, lock_timeout=5):
        """Begin a real ArangoDB stream transaction with retry logic."""
        return self.db.begin_transaction(
            write=write_collections,
            read=read_collections or [],
            exclusive=[],  # No exclusive locks needed
            sync=True,     # Ensure durability
            allow_implicit=False,
            lock_timeout=lock_timeout
        )
    
    @retry_with_backoff(max_retries=2, base_delay=0.1, max_delay=1.0)
    def acquire_lock(self, paper_id: str, timeout_minutes: int = 10) -> bool:
        """
        Acquire distributed lock for paper processing with retry logic.
        
        Uses exponential backoff for transient failures, but doesn't retry
        if lock is already held (DocumentInsertError).
        """
        try:
            sanitized_id = paper_id.replace('.', '_').replace('/', '_')
            self.db.collection('arxiv_locks').insert({
                '_key': sanitized_id,
                'paper_id': paper_id,
                'worker_id': os.getpid(),
                'acquired_at': datetime.now().isoformat(),
                'expiresAt': int((datetime.now() + timedelta(minutes=timeout_minutes)).timestamp())
            })
            logger.debug(f"Acquired lock for {paper_id}")
            return True
        except DocumentInsertError:
            # Don't retry for this - lock is already held
            logger.debug(f"Lock already held for {paper_id}")
            return False
    
    def release_lock(self, paper_id: str):
        """Release distributed lock for paper."""
        try:
            sanitized_id = paper_id.replace('.', '_').replace('/', '_')
            self.db.collection('arxiv_locks').delete(sanitized_id)
            logger.debug(f"Released lock for {paper_id}")
        except Exception as e:
            logger.warning(f"Failed to release processing lock for paper {paper_id} (non-critical): {type(e).__name__}: {e}")