"""
Data Processing Module

Provides text and structured data processing capabilities.
Processing pipelines have been moved to core.workflows.
"""

# Import from text subdirectory
from .text.chunking_strategies import (
    ChunkingStrategy,
    TokenBasedChunking,
    SemanticChunking,
    SlidingWindowChunking,
    ChunkingStrategyFactory
)

__all__ = [
    "ChunkingStrategy",
    "TokenBasedChunking",
    "SemanticChunking",
    "SlidingWindowChunking",
    "ChunkingStrategyFactory",
    # Deprecated - kept for backward compatibility
    "DocumentProcessor",
    "ProcessingConfig",
    "ProcessingResult",
]

# Lazy import for backward compatibility using __getattr__
def __getattr__(name):
    """
    Lazy import deprecated classes from workflows.

    This provides backward compatibility while encouraging migration
    to the new module structure.
    """
    import warnings

    deprecated_imports = {
        "DocumentProcessor": "core.workflows.workflow_pdf",
        "ProcessingConfig": "core.workflows.workflow_pdf",
        "ProcessingResult": "core.workflows.workflow_pdf",
    }

    if name in deprecated_imports:
        module_path = deprecated_imports[name]
        warnings.warn(
            f"Importing {name} from core.processors is deprecated. "
            f"Please use {module_path} instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Dynamically import from the new location
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")