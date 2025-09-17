#!/usr/bin/env python3
"""
Simple test of the chunking fix.
Tests just the problematic 1923-character case.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.embedders.embedders_factory import EmbedderFactory

def test_problematic_size():
    """Test the exact problematic size that was failing."""

    print("Testing 1923-character text (the problem size)")
    print("="*50)

    # Create text of exactly 1923 characters
    test_text = "x" * 1923

    # Initialize embedder - use sentence transformer for CPU testing
    embedder = EmbedderFactory.create(
        model_name='jinaai/jina-embeddings-v3',  # v3 works on CPU
        device='cpu',  # Use CPU for testing
        chunk_size_tokens=500,
        chunk_overlap_tokens=200
    )

    print(f"Text length: {len(test_text)} chars")
    print(f"Chunk size: 500 tokens (~2000 chars)")
    print(f"Expected chunks: 1 or 2")

    # Process the text
    try:
        result = embedder.embed_batch_with_late_chunking([test_text])
        chunks = result[0] if result else []

        print(f"\nResult: {len(chunks)} chunks created")

        if chunks:
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i}: {chunk.start_char}-{chunk.end_char} ({chunk.end_char - chunk.start_char} chars)")
            print("✓ SUCCESS: Text was chunked and embedded")
            return True
        else:
            print("✗ FAILURE: No chunks created")
            return False

    except Exception as e:
        print(f"✗ FAILURE: {e}")
        return False

if __name__ == "__main__":
    success = test_problematic_size()
    sys.exit(0 if success else 1)