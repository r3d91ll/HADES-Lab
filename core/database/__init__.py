"""
Core Database Module
====================

Shared database utilities for all HADES-Lab processing pipelines.
Provides factory pattern for database connections and optimized clients.
"""

from .database_factory import DatabaseFactory

# Import from subdirectories
from .arango import ArangoDBManager, retry_with_backoff

__all__ = [
    'DatabaseFactory',
    'ArangoDBManager',
    'retry_with_backoff'
]