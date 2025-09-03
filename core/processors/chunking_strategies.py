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
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
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
        """
        Return the number of characters in this chunk's text.
        
        Returns:
            int: Character count (equivalent to len(self.text)).
        """
        return len(self.text)
    
    @property
    def token_count_estimate(self) -> int:
        """
        Return a rough estimate of the token count for this chunk.
        
        This uses a simple whitespace split on the chunk text and should be treated as an approximate token count (use a tokenizer for exact tokenization).
        Returns:
            int: Estimated number of tokens (words) in the chunk.
        """
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
        """
        Create a sequence of TextChunk objects representing segments of the input text.
        
        This is the abstract interface for chunking implementations. Implementations should accept the raw document string in `text` and return a list of TextChunk instances that together cover meaningful portions of the input (possibly with overlap depending on the strategy). The exact behavior, chunk size semantics, and accepted `**kwargs` are strategy-specific; callers should consult the concrete strategy's documentation for those details.
        
        Parameters:
            text (str): The input document to split into chunks.
        
        Returns:
            List[TextChunk]: A list of TextChunk objects (may be empty).
        """
        pass
    
    def _clean_text(self, text: str) -> str:
        """
        Normalize input text for chunking by collapsing all whitespace to single spaces, removing null characters, and trimming leading/trailing spaces.
        
        This produces a compact, single-line-safe string suitable for downstream tokenization and chunk boundary calculations.
        """
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
        tokenizer: Optional[Union[Callable[[str], List[str]], Any]] = None
    ):
        """
        Initialize a token-based chunking strategy.
        
        Parameters:
            chunk_size (int): Target number of tokens per chunk.
            chunk_overlap (int): Number of tokens to overlap between consecutive chunks; must be less than chunk_size.
            tokenizer: Optional tokenizer that can be either:
                - A callable function that takes a string and returns a list of tokens: Callable[[str], List[str]]
                - A tokenizer-like object with .tokenize(text) and .convert_tokens_to_string(tokens) methods
                - None (uses simple whitespace split)
        
        Raises:
            ValueError: If `chunk_overlap` is greater than or equal to `chunk_size`.
            TypeError: If tokenizer is provided but doesn't support either expected interface.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Validate tokenizer interface if provided
        if tokenizer is not None:
            if not (callable(tokenizer) or 
                    (hasattr(tokenizer, 'tokenize') and hasattr(tokenizer, 'convert_tokens_to_string'))):
                raise TypeError(
                    "Tokenizer must be either a callable that returns List[str], "
                    "or an object with .tokenize() and .convert_tokens_to_string() methods"
                )
        
        self.tokenizer = tokenizer
        
        if chunk_overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk size")
    
    def create_chunks(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Create token-based text chunks with configurable overlap.
        
        Generates a sequence of TextChunk objects by sliding a token window of size
        `self.chunk_size` across the cleaned text in steps of `self.chunk_size - self.chunk_overlap`.
        If a tokenizer was provided to the strategy it will be used for tokenization and
        reconstruction; otherwise simple whitespace splitting and joining are used.
        Character offsets (start_char, end_char) are approximate and derived from token
        boundaries, and metadata includes token indices and chunking parameters.
        
        Returns:
            List[TextChunk]: Ordered list of created chunks covering the input text (may
            overlap). 
        """
        text = self._clean_text(text)
        
        # Tokenize text using appropriate interface
        if self.tokenizer:
            if callable(self.tokenizer):
                # Tokenizer is a callable function
                tokens = self.tokenizer(text)
            else:
                # Tokenizer is an object with .tokenize() method
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
                if callable(self.tokenizer):
                    # For callable tokenizers, join tokens with space
                    chunk_text = ' '.join(chunk_tokens)
                else:
                    # For tokenizer objects, use convert_tokens_to_string method
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
        Configure a semantic chunking strategy that preserves paragraph and sentence boundaries.
        
        Parameters:
            max_chunk_size (int): Maximum number of tokens allowed in a chunk; larger paragraphs/sentences will be split.
            min_chunk_size (int): Minimum number of tokens preferred for a chunk; used to avoid producing overly small chunks.
            respect_sentences (bool): If True, attempts to keep sentence boundaries intact when splitting oversized paragraphs.
            respect_paragraphs (bool): If True, prefers to keep paragraphs together when forming chunks.
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.respect_sentences = respect_sentences
        self.respect_paragraphs = respect_paragraphs
    
    def create_chunks(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Create semantically meaningful TextChunk objects from the input text.
        
        This method preserves natural document boundaries by grouping content into chunks that align with paragraphs and, when enabled, sentences. It:
        - Splits the cleaned input into paragraphs.
        - Accumulates paragraphs (and, if a paragraph exceeds max_chunk_size and sentence-respecting is enabled, sentences) into chunks not exceeding self.max_chunk_size tokens.
        - For paragraphs larger than max_chunk_size, either splits them into sentences (if respect_sentences is True) or force-splits at token boundaries.
        - Produces a list of TextChunk instances with metadata (e.g., token counts, paragraph/sentence counts). Character offsets are estimated from the segmentation and may be approximate.
        
        Parameters:
            text (str): Raw document text to chunk. It will be normalized by the strategy before splitting.
        
        Returns:
            List[TextChunk]: Ordered list of chunks covering the input text, each with start/end character positions, chunk_index, and metadata.
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
        """
        Split text into non-empty paragraphs.
        
        Paragraphs are defined by two or more consecutive newline characters; each returned paragraph is stripped of leading/trailing whitespace and empty segments are omitted. Preserves original order of paragraphs.
        """
        # Split on double newlines or typical paragraph markers
        paragraphs = re.split(r'\n\n+', text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using a simple end-punctuation heuristic.
        
        This performs a lightweight split on whitespace that follows '.', '!' or '?', then strips surrounding whitespace and filters out empty results. It is a simple heuristic and may not correctly handle edge cases such as abbreviations, ellipses, or quoted text.
        
        Returns:
            List[str]: Non-empty sentence strings in original order.
        """
        # Simple sentence splitting (could use more sophisticated methods)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _force_split_text(self, text: str, start_char: int) -> List[TextChunk]:
        """
        Force-split an oversized text into consecutive TextChunk objects of at most `self.max_chunk_size` tokens.
        
        The text is split on whitespace into words and rejoined into chunks of up to `max_chunk_size` words. `start_char` is used as the initial character offset for the first chunk; subsequent chunks' `start_char`/`end_char` are computed by advancing the offset by the length of the produced chunk text plus a single separating space. Character positions are therefore approximate and do not preserve original spacing or punctuation.
        
        Parameters:
            text (str): The input text to split.
            start_char (int): Character index in the original document corresponding to the start of `text`.
        
        Returns:
            List[TextChunk]: A list of TextChunk objects with metadata containing `strategy='semantic_forced_split'` and `token_count`. Chunk indices are zero-based and reflect the order of the produced chunks.
        """
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
        """
        Create a TextChunk representing a semantic chunk of the original document.
        
        Parameters:
            text (str): Chunk text content.
            start_char (int): Character index in the original text where this chunk begins.
            index (int): Sequential chunk index.
        
        Returns:
            TextChunk: A chunk with start/end character offsets, index, and metadata including:
                - strategy: 'semantic'
                - max_size, min_size: configured semantic chunk size limits
                - token_count: approximate token count (whitespace split)
                - paragraph_count: number of paragraphs in the chunk
                - sentence_count: number of sentences in the chunk
        """
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
        Create a SlidingWindowChunking instance.
        
        Parameters:
            window_size (int): Number of tokens contained in each sliding window (default 512).
            step_size (int): Number of tokens to advance the window on each step (default 256).
        
        Raises:
            ValueError: If `step_size` is greater than `window_size`.
        """
        self.window_size = window_size
        self.step_size = step_size
        
        if step_size > window_size:
            raise ValueError("Step size cannot be larger than window size")
    
    def create_chunks(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Create a sequence of overlapping text chunks using a sliding-window token approach.
        
        This produces successive windows of tokens of size `window_size`, advancing by `step_size` tokens each step to preserve continuity between adjacent chunks (higher overlap increases redundancy). Chunks are returned as TextChunk objects and include metadata fields such as `strategy`, `window_size`, `step_size`, `overlap_ratio`, and `token_count`. If a non-trivial trailing segment of tokens remains after the last full window, a final "sliding_window_remainder" chunk is emitted.
        
        Parameters:
            text (str): Input document to chunk. Tokenization is performed by simple whitespace splitting.
        
        Returns:
            List[TextChunk]: Ordered list of chunks covering the input text. `start_char`/`end_char` are estimated from token boundaries (approximate character offsets).
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
        Return a ChunkingStrategy instance corresponding to strategy_type.
        
        Creates one of: 'token' -> TokenBasedChunking, 'semantic' -> SemanticChunking,
        'sliding' or 'sliding_window' -> SlidingWindowChunking. The lookup is case-insensitive.
        
        Parameters:
            strategy_type (str): Name of the strategy to create.
            **kwargs: Passed to the selected strategy class constructor.
        
        Returns:
            ChunkingStrategy: An instance of the requested chunking strategy.
        
        Raises:
            ValueError: If strategy_type is not one of the supported names.
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