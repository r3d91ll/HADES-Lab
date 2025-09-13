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

# Backward compatibility - redirect to workflows
import warnings

def _deprecated_import(name):
    warnings.warn(
        f"Importing {name} from core.processors is deprecated. "
        f"Please use core.workflows instead.",
        DeprecationWarning,
        stacklevel=2
    )

# These will be imported from workflows for backward compatibility
try:
    from core.workflows.workflow_pdf import DocumentProcessor, ProcessingConfig, ProcessingResult
    _deprecated_import("DocumentProcessor")
except ImportError:
    pass

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