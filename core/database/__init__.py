"""
Core Database Module
====================

Shared database utilities for all HADES-Lab processing pipelines.
"""

from .arango_db_manager import ArangoDBManager, retry_with_backoff

__all__ = ['ArangoDBManager', 'retry_with_backoff']