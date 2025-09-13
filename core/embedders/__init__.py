"""
Embedders Module

Provides embedding models for transforming text into vector representations.
All embedders follow the Conveyance Framework, preserving the WHAT dimension
while enabling efficient WHERE computations in vector space.

This module replaces core.framework.embedders with a cleaner structure.
"""

from .embedders_base import EmbedderBase, EmbeddingConfig
from .embedders_factory import EmbedderFactory

# Auto-register available embedders
try:
    from .embedders_jina import JinaV4Embedder
    EmbedderFactory.register("jina", JinaV4Embedder)
except ImportError:
    pass

# Backward compatibility exports
__all__ = [
    'EmbedderBase',
    'EmbeddingConfig',
    'EmbedderFactory',
    'JinaV4Embedder',  # May not be available
]

# Convenience function for backward compatibility
def create_embedder(model_name: str = "jinaai/jina-embeddings-v4", **kwargs):
    """
    Create an embedder instance by delegating to EmbedderFactory.
    
    Parameters:
        model_name (str): Model identifier or factory key (defaults to "jinaai/jina-embeddings-v4").
        **kwargs: Additional keyword arguments forwarded to EmbedderFactory.create.
    
    Returns:
        EmbedderBase: A new embedder instance for the requested model.
    """
    return EmbedderFactory.create(model_name, **kwargs)