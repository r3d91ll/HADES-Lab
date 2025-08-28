#!/usr/bin/env python3
"""
Chunking Strategies Module
===========================

Implements various chunking strategies for document processing.
Each strategy represents a different approach to maintaining the WHERE
dimension of information while transforming continuous text into discrete chunks.

In Actor-Network Theory terms, chunking creates new boundaries and obligatory
passage points within documents, fundamentally changing how information flows
through our processing pipeline.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """
    Represents a chunk of text before embedding.
    
    This is a boundary object that maintains coherence between
    the continuous document and discrete processing units.
    """
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    metadata: Dict[str, Any]
    
    @property
    def char_count(self) -> int:
        """Character count in chunk."""
        return len(self.text)
    
    @property
    def token_count_estimate(self) -> int:
        """Rough estimate of token count (words)."""
        return len(self.text.split())


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.
    
    Each strategy implements a different philosophy about how to
    preserve information across chunk boundaries, reflecting different
    theoretical approaches to the continuity/discreteness problem.
    """
    
    @abstractmethod
    def create_chunks(self, text: str, **kwargs) -> List[TextChunk]:
        """Create chunks from text using the strategy."""
        pass
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text before chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove null characters
        text = text.replace('\x00', '')
        return text.strip()


class TokenBasedChunking(ChunkingStrategy):
    """
    Token-based chunking with configurable overlap.
    
    This strategy prioritizes consistent chunk sizes over semantic boundaries,
    treating the document as a continuous stream of tokens. It represents
    a mechanical approach to information division.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize token-based chunking.
        
        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            tokenizer: Optional tokenizer (uses simple split if None)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        
        if chunk_overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")
    
    def create_chunks(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Create token-based chunks with overlap.
        
        The overlap ensures that information at chunk boundaries isn't lost,
        preventing zero-propagation at the edges of our processing units.
        """
        text = self._clean_text(text)
        
        # Tokenize text
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(text)
        else:
            # Simple word tokenization
            tokens = text.split()
        
        chunks = []
        stride = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(tokens), stride):
            # Get chunk tokens
            chunk_tokens = tokens[i:i + self.chunk_size]
            
            if not chunk_tokens:
                break
            
            # Reconstruct text from tokens
            if self.tokenizer:
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            else:
                chunk_text = ' '.join(chunk_tokens)
            
            # Calculate character positions (approximate)
            start_char = len(' '.join(tokens[:i]))
            end_char = start_char + len(chunk_text)
            
            chunk = TextChunk(
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                chunk_index=len(chunks),
                metadata={
                    'strategy': 'token_based',
                    'chunk_size': self.chunk_size,
                    'overlap': self.chunk_overlap,
                    'token_count': len(chunk_tokens),
                    'first_token_index': i,
                    'last_token_index': min(i + self.chunk_size, len(tokens))
                }
            )
            chunks.append(chunk)
            
            # Stop if we've processed all tokens
            if i + self.chunk_size >= len(tokens):
                break
        
        logger.info(f"Created {len(chunks)} token-based chunks from {len(tokens)} tokens")
        return chunks


class SemanticChunking(ChunkingStrategy):
    """
    Semantic chunking based on document structure.
    
    This strategy respects natural boundaries in the document (paragraphs,
    sections, sentences), treating the document as a hierarchical structure
    rather than a flat stream. It represents an anthropological approach
    to information division, respecting the document's inherent organization.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        respect_sentences: bool = True,
        respect_paragraphs: bool = True
    ):
        """
        Initialize semantic chunking.
        
        Args:
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk
            respect_sentences: Keep sentences intact
            respect_paragraphs: Try to keep paragraphs intact
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.respect_sentences = respect_sentences
        self.respect_paragraphs = respect_paragraphs
    
    def create_chunks(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Create semantically meaningful chunks.
        
        Attempts to preserve natural document boundaries, treating
        paragraphs and sentences as atomic units when possible.
        """
        text = self._clean_text(text)
        
        # Split into paragraphs
        paragraphs = self._split_paragraphs(text)
        
        chunks = []
        current_chunk_text = ""
        current_chunk_start = 0
        
        for para in paragraphs:
            para_tokens = len(para.split())
            
            # If paragraph is too large, split it
            if para_tokens > self.max_chunk_size:
                # Flush current chunk if any
                if current_chunk_text:
                    chunks.append(self._create_chunk(
                        current_chunk_text,
                        current_chunk_start,
                        len(chunks)
                    ))
                    current_chunk_start += len(current_chunk_text) + 1
                    current_chunk_text = ""
                
                # Split large paragraph into sentences
                if self.respect_sentences:
                    sentences = self._split_sentences(para)
                    for sent in sentences:
                        sent_tokens = len(sent.split())
                        
                        if len(current_chunk_text.split()) + sent_tokens > self.max_chunk_size:
                            # Save current chunk
                            if current_chunk_text:
                                chunks.append(self._create_chunk(
                                    current_chunk_text,
                                    current_chunk_start,
                                    len(chunks)
                                ))
                                current_chunk_start += len(current_chunk_text) + 1
                            current_chunk_text = sent
                        else:
                            current_chunk_text = (current_chunk_text + " " + sent).strip()
                else:
                    # Force split at token boundary
                    chunks.extend(self._force_split_text(para, current_chunk_start))
                    current_chunk_start += len(para) + 1
            
            # If adding paragraph doesn't exceed limit, add it
            elif len(current_chunk_text.split()) + para_tokens <= self.max_chunk_size:
                current_chunk_text = (current_chunk_text + "\n\n" + para).strip()
            
            # Otherwise, save current chunk and start new one
            else:
                if current_chunk_text:
                    chunks.append(self._create_chunk(
                        current_chunk_text,
                        current_chunk_start,
                        len(chunks)
                    ))
                    current_chunk_start += len(current_chunk_text) + 1
                current_chunk_text = para
        
        # Don't forget last chunk
        if current_chunk_text:
            chunks.append(self._create_chunk(
                current_chunk_text,
                current_chunk_start,
                len(chunks)
            ))
        
        logger.info(f"Created {len(chunks)} semantic chunks from {len(paragraphs)} paragraphs")
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or typical paragraph markers
        paragraphs = re.split(r'\n\n+', text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could use more sophisticated methods)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _force_split_text(self, text: str, start_char: int) -> List[TextChunk]:
        """Force split text that's too large for a single chunk."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.max_chunk_size):
            chunk_words = words[i:i + self.max_chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                chunk_index=len(chunks),
                metadata={
                    'strategy': 'semantic_forced_split',
                    'token_count': len(chunk_words)
                }
            ))
            start_char += len(chunk_text) + 1
        
        return chunks
    
    def _create_chunk(self, text: str, start_char: int, index: int) -> TextChunk:
        """Create a TextChunk object."""
        return TextChunk(
            text=text,
            start_char=start_char,
            end_char=start_char + len(text),
            chunk_index=index,
            metadata={
                'strategy': 'semantic',
                'max_size': self.max_chunk_size,
                'min_size': self.min_chunk_size,
                'token_count': len(text.split()),
                'paragraph_count': len(self._split_paragraphs(text)),
                'sentence_count': len(self._split_sentences(text))
            }
        )


class SlidingWindowChunking(ChunkingStrategy):
    """
    Sliding window chunking with maximal overlap.
    
    This strategy creates highly overlapping chunks to ensure no information
    is lost at boundaries. It represents a conservative approach that
    prioritizes information preservation over efficiency.
    """
    
    def __init__(
        self,
        window_size: int = 512,
        step_size: int = 256
    ):
        """
        Initialize sliding window chunking.
        
        Args:
            window_size: Size of the sliding window in tokens
            step_size: Number of tokens to slide the window
        """
        self.window_size = window_size
        self.step_size = step_size
        
        if step_size > window_size:
            raise ValueError("Step size cannot be larger than window size")
    
    def create_chunks(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Create chunks using sliding window approach.
        
        High overlap ensures continuity of information across chunks,
        at the cost of increased redundancy and processing overhead.
        """
        text = self._clean_text(text)
        tokens = text.split()  # Simple tokenization
        
        chunks = []
        for i in range(0, len(tokens) - self.window_size + 1, self.step_size):
            window_tokens = tokens[i:i + self.window_size]
            chunk_text = ' '.join(window_tokens)
            
            start_char = len(' '.join(tokens[:i]))
            
            chunk = TextChunk(
                text=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                chunk_index=len(chunks),
                metadata={
                    'strategy': 'sliding_window',
                    'window_size': self.window_size,
                    'step_size': self.step_size,
                    'overlap_ratio': 1 - (self.step_size / self.window_size),
                    'token_count': len(window_tokens)
                }
            )
            chunks.append(chunk)
        
        # Handle remainder if any
        if len(tokens) > self.window_size:
            last_index = len(chunks) * self.step_size
            if last_index < len(tokens):
                remaining_tokens = tokens[last_index:]
                if len(remaining_tokens) >= self.step_size:  # Only if substantial
                    chunk_text = ' '.join(remaining_tokens)
                    start_char = len(' '.join(tokens[:last_index]))
                    
                    chunk = TextChunk(
                        text=chunk_text,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        chunk_index=len(chunks),
                        metadata={
                            'strategy': 'sliding_window_remainder',
                            'token_count': len(remaining_tokens)
                        }
                    )
                    chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} sliding window chunks with {self.step_size}/{self.window_size} step/window")
        return chunks


class ChunkingStrategyFactory:
    """
    Factory for creating chunking strategies.
    
    Provides a centralized way to instantiate different chunking strategies
    based on configuration, allowing for easy switching between approaches.
    """
    
    @staticmethod
    def create_strategy(
        strategy_type: str,
        **kwargs
    ) -> ChunkingStrategy:
        """
        Create a chunking strategy based on type.
        
        Args:
            strategy_type: Type of strategy ('token', 'semantic', 'sliding')
            **kwargs: Strategy-specific parameters
            
        Returns:
            ChunkingStrategy instance
        """
        strategies = {
            'token': TokenBasedChunking,
            'semantic': SemanticChunking,
            'sliding': SlidingWindowChunking,
            'sliding_window': SlidingWindowChunking
        }
        
        strategy_class = strategies.get(strategy_type.lower())
        if not strategy_class:
            raise ValueError(f"Unknown chunking strategy: {strategy_type}")
        
        return strategy_class(**kwargs)