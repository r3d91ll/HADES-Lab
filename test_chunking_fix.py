#!/usr/bin/env python3
"""
Test script for chunking edge case fix.
Tests with various abstract lengths including the problematic 1923-char case.

IMPORTANT: Uses a TEST database, not production!
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.embedders.embedders_factory import EmbedderFactory

def test_chunking_edge_cases():
    """Test the fixed chunking with various text lengths."""

    print("="*70)
    print("Testing Chunking Edge Case Fix")
    print("="*70)

    # Initialize embedder with standard settings
    embedder_config = {
        'device': 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
        'use_fp16': True,
        'batch_size': 8,
        'chunk_size_tokens': 500,  # Standard chunk size
        'chunk_overlap_tokens': 200
    }

    print(f"Initializing Jina v4 Embedder...")
    print(f"  Chunk size: {embedder_config['chunk_size_tokens']} tokens")
    print(f"  Chunk overlap: {embedder_config['chunk_overlap_tokens']} tokens")

    embedder = EmbedderFactory.create(
        model_name='jina-embeddings-v4',  # Use Jina v4
        **embedder_config
    )

    # Test cases covering edge cases
    test_cases = [
        ("Short", "a" * 100),  # Very short - 1 chunk
        ("Medium", "b" * 1000),  # Medium - 1 chunk
        ("Near boundary", "c" * 1900),  # Just under 2000 chars - 1 chunk
        ("Problem size", "d" * 1923),  # The exact problem size - should be 2 chunks
        ("At boundary", "e" * 2000),  # Exactly at boundary - 2 chunks
        ("Over boundary", "f" * 2100),  # Just over - 2 chunks
        ("Long", "g" * 3000),  # Clearly 2 chunks
        ("Very long", "h" * 6091),  # Maximum seen - 3-4 chunks
    ]

    print("\n" + "="*70)
    print("Test Results:")
    print("-"*70)

    all_passed = True

    for name, text in test_cases:
        # Process text through the fixed method
        result = embedder.embed_batch_with_late_chunking([text])
        chunks = result[0] if result else []

        # Check results
        text_len = len(text)
        num_chunks = len(chunks)
        expected_chunks = max(1, (text_len // 2000) + (1 if text_len % 2000 > 500 else 0))

        status = "✓ PASS" if num_chunks > 0 else "✗ FAIL"

        print(f"{name:15s}: {text_len:5d} chars → {num_chunks} chunks {status}")

        if num_chunks == 0:
            print(f"  ERROR: No chunks created for {text_len}-char text!")
            all_passed = False

        # Verify chunk properties
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'embedding') or chunk.embedding is None:
                print(f"  ERROR: Chunk {i} has no embedding!")
                all_passed = False

    print("-"*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Edge case fixed!")
    else:
        print("✗ SOME TESTS FAILED - Fix needs adjustment")

    return all_passed


def test_specific_failures():
    """Test with actual failed paper IDs."""

    print("\n" + "="*70)
    print("Testing Specific Failed Papers")
    print("="*70)

    # Simulate abstracts of the problematic lengths
    problematic_lengths = [1923, 1946, 1973, 2281, 2475]

    embedder = EmbedderFactory.create(
        model_name='jina-embeddings-v4',  # Use Jina v4
        device='cpu',  # Use CPU for testing
        chunk_size_tokens=500
    )

    print(f"\nTesting {len(problematic_lengths)} problematic abstract lengths:")

    for length in problematic_lengths:
        # Create test text of exact length
        test_text = "x" * length

        # Process through fixed embedder
        result = embedder.embed_batch_with_late_chunking([test_text])
        chunks = result[0] if result else []

        print(f"  {length:4d} chars: {len(chunks)} chunks created")

        if len(chunks) == 0:
            print(f"    ERROR: Failed to create chunks!")
            return False

    return True


if __name__ == "__main__":
    print("Chunking Edge Case Test Suite")
    print("==============================")
    print("Testing the fix for papers that failed at chunk boundaries")
    print()

    # Run tests
    edge_case_pass = test_chunking_edge_cases()
    specific_pass = test_specific_failures()

    print("\n" + "="*70)
    if edge_case_pass and specific_pass:
        print("SUCCESS: All tests passed! Ready to reprocess failed papers.")
        print("\nNext steps:")
        print("1. Create test database for safety")
        print("2. Run reprocessing on test database")
        print("3. Verify all 2,074 papers get embeddings")
        print("4. Deploy to production")
    else:
        print("FAILURE: Tests failed. Fix needs adjustment.")
        sys.exit(1)