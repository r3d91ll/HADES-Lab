"""
ArangoDB Database Interface

Provides optimized ArangoDB connections including Unix socket support
for improved performance and atomic transaction handling.
"""

from .arango_client import ArangoDBManager, retry_with_backoff

# Try to import Unix socket client if available
try:
    from .arango_unix_client import ArangoUnixClient
except ImportError:
    pass

__all__ = [
    'ArangoDBManager',
    'retry_with_backoff',
    'ArangoUnixClient',
]