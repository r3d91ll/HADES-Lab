"""
Core Processors Module
======================

This module provides text processing utilities including chunking strategies.
"""

from .text.chunking_strategies import (
    ChunkingStrategy,
    ChunkingStrategyFactory,
    FixedSizeChunking,
    SemanticChunking
)

__all__ = [
    'ChunkingStrategy',
    'ChunkingStrategyFactory',
    'FixedSizeChunking',
    'SemanticChunking',
]