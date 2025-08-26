"""
Data Infrastructure Module

Provides data loading, PDF processing, and chunk management utilities
for processing academic papers from arXiv and other sources.
"""

from .document_processor import DocumentProcessor, ProcessingConfig, ProcessingResult
from .chunking_strategies import (
    ChunkingStrategy,
    TokenBasedChunking,
    SemanticChunking,
    SlidingWindowChunking,
    ChunkingStrategyFactory
)

__all__ = [
    "DocumentProcessor", 
    "ProcessingConfig", 
    "ProcessingResult",
    "ChunkingStrategy",
    "TokenBasedChunking",
    "SemanticChunking",
    "SlidingWindowChunking",
    "ChunkingStrategyFactory"
]