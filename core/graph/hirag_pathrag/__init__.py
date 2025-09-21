"""HiRAG × PathRAG graph toolkit built on ArangoDB.

This package bundles schema management, configuration helpers, and
service scaffolding for constructing the hierarchical graph backed by
ArangoDB collections sourced from the ``arxiv_repository`` database.
"""

from .schema_manager import GraphSchemaManager
from .service import HiragPathragConfig, HiragPathragService

__all__ = [
    "GraphSchemaManager",
    "HiragPathragConfig",
    "HiragPathragService",
]
