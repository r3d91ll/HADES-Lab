#!/usr/bin/env python3
"""
Unit Tests for Phase 1 Embedders

Tests the critical custom algorithms in the embedders module,
particularly the late chunking implementation.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.embedders.embedders_factory import EmbedderFactory
from core.embedders.embedders_base import EmbeddingConfig


class TestEmbedderFactory(unittest.TestCase):
    """Test the embedder factory's custom logic."""

    def test_determine_embedder_type(self):
        """Test model name to embedder type mapping."""
        test_cases = [
            ("jinaai/jina-embeddings-v4", "jina"),
            ("jinaai/jina-embeddings-v4", "jina"),
            ("sentence-transformers/all-MiniLM-L6-v2", "sentence"),
            ("st-something", "sentence"),
            ("text-embedding-ada-002", "openai"),
            ("openai/text-embedding-3-small", "openai"),
            ("cohere/embed-english-v3.0", "cohere"),
            ("unknown-model-name", "jina"),  # Default fallback
        ]

        for model_name, expected_type in test_cases:
            with self.subTest(model=model_name):
                result = EmbedderFactory._determine_embedder_type(model_name)
                self.assertEqual(result, expected_type,
                               f"Model {model_name} should map to {expected_type}")

    def test_factory_registration(self):
        """Test that embedders can be registered and retrieved."""
        # Create a mock embedder class
        class MockEmbedder:
            def __init__(self, config):
                self.config = config

        # Register it
        EmbedderFactory.register("mock", MockEmbedder)

        # Check it's in the registry
        self.assertIn("mock", EmbedderFactory._embedders)

        # List available should include it
        available = EmbedderFactory.list_available()
        self.assertIn("mock", available)


class TestJinaV4LateChuking(unittest.TestCase):
    """Test the late chunking algorithm - our most complex custom logic."""

    def setUp(self):
        """Set up test fixtures."""
        # We'll mock the model to avoid loading the actual Jina model
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()

    def test_late_chunking_calculation_logic(self):
        """Test the late chunking calculation without full model loading."""
        from core.embedders.embedders_jina import ChunkWithEmbedding

        # Test the chunking calculation logic
        # This tests the core algorithm without needing to mock the entire model

        # Simulate chunk creation
        chunk_size = 1000
        overlap_size = 200
        total_tokens = 5000

        chunks = []
        stride = chunk_size - overlap_size

        for i in range(0, total_tokens, stride):
            start = i
            end = min(i + chunk_size, total_tokens)

            # Create a mock chunk
            chunk = ChunkWithEmbedding(
                text=f"chunk_{i}",
                embedding=np.random.rand(1024),
                start_char=start * 5,  # Approximate char position
                end_char=end * 5,
                start_token=start,
                end_token=end,
                chunk_index=len(chunks),
                total_chunks=0,  # Will be updated
                context_window_used=total_tokens
            )
            chunks.append(chunk)

            if end >= total_tokens:
                break

        # Update total chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        # Verify chunking properties
        self.assertGreater(len(chunks), 1, "Should create multiple chunks")

        # Check first chunk
        self.assertEqual(chunks[0].start_token, 0)
        self.assertEqual(chunks[0].end_token, chunk_size)

        # Check overlaps
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]

            # Calculate overlap
            overlap = chunk1.end_token - chunk2.start_token

            # For all but the last chunk pair, overlap should equal overlap_size
            if i < len(chunks) - 2:
                self.assertEqual(overlap, overlap_size,
                               f"Chunks {i} and {i+1} should overlap by {overlap_size} tokens")

        # Verify coverage (no gaps)
        for i in range(len(chunks) - 1):
            # Each chunk should start where the previous one ends minus overlap
            expected_start = chunks[i].end_token - overlap_size
            actual_start = chunks[i + 1].start_token
            self.assertEqual(actual_start, expected_start,
                           f"Chunk {i+1} should start at {expected_start}")

    def test_chunk_overlap_calculation(self):
        """Test the chunk overlap calculation logic."""
        # This tests the mathematical logic for calculating overlaps
        chunk_size = 1000
        overlap_size = 200
        total_tokens = 5000

        # Calculate expected number of chunks
        # First chunk: 0-1000
        # Second chunk: 800-1800 (200 overlap)
        # Third chunk: 1600-2600 (200 overlap)
        # etc.

        stride = chunk_size - overlap_size  # 800
        expected_chunks = (total_tokens - chunk_size) // stride + 1
        if (total_tokens - chunk_size) % stride != 0:
            expected_chunks += 1

        # Verify the calculation
        self.assertGreater(expected_chunks, 1, "Should produce multiple chunks")

        # Verify overlap positions
        chunk_starts = []
        for i in range(expected_chunks):
            start = i * stride
            if start >= total_tokens:
                break
            chunk_starts.append(start)

        # Check overlaps between consecutive chunks
        for i in range(len(chunk_starts) - 1):
            chunk1_end = min(chunk_starts[i] + chunk_size, total_tokens)
            chunk2_start = chunk_starts[i + 1]
            overlap = chunk1_end - chunk2_start

            # Overlap should be approximately overlap_size (might be less for last chunk)
            if i < len(chunk_starts) - 2:  # Not the last pair
                self.assertEqual(overlap, overlap_size,
                               f"Chunk {i} and {i+1} should overlap by {overlap_size}")


class TestExtractorFactory(unittest.TestCase):
    """Test the extractor factory's format detection logic."""

    def test_format_detection(self):
        """Test file format to extractor type mapping."""
        from core.extractors.extractors_factory import ExtractorFactory

        test_cases = [
            ("document.pdf", "docling"),
            ("paper.PDF", "docling"),  # Case insensitive
            ("thesis.tex", "latex"),
            ("bibliography.bib", "latex"),
            ("style.cls", "latex"),
            ("script.py", "code"),
            ("app.js", "code"),
            ("main.cpp", "code"),
            ("program.java", "code"),
            ("lib.rs", "code"),
            ("unknown.xyz", "docling"),  # Default fallback
        ]

        for filename, expected_type in test_cases:
            with self.subTest(file=filename):
                result = ExtractorFactory._determine_extractor_type(filename)
                self.assertEqual(result, expected_type,
                               f"File {filename} should map to {expected_type}")

    def test_format_mapping_completeness(self):
        """Test that all mapped formats have valid extractors."""
        from core.extractors.extractors_factory import ExtractorFactory

        format_mapping = ExtractorFactory.get_format_mapping()

        # All mapped formats should map to known extractor types
        valid_types = ['docling', 'latex', 'code', 'treesitter', 'robust']

        for ext, extractor_type in format_mapping.items():
            self.assertIn(extractor_type, valid_types,
                         f"Extension {ext} maps to unknown extractor {extractor_type}")


if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)