#!/usr/bin/env python3
"""
Test suite for Jina v4 Embedder
"""

import torch
import numpy as np
from core.embedders.embedders_jina import JinaV4Embedder


def test_jina_v4() -> bool:
    """
    Run a basic integration test of the JinaV4Embedder.
    
    Performs three checks:
    1. Embeds two text samples with embed_texts and asserts the result has shape (2, 2048).
    2. Embeds one code snippet with embed_code and asserts the result has shape (1, 2048).
    3. Processes a long document with process_long_document to verify late-chunking produces chunk metadata and embeddings (prints details of the first chunk if any).
    
    The embedder is instantiated on "cuda" when available, otherwise "cpu".
    
    Returns:
        True on success.
    
    Raises:
        AssertionError: if any embedding shape assertion fails.
    """
    print("Testing Jina v4 Embedder...")

    # Initialize embedder
    embedder = JinaV4Embedder(device="cuda" if torch.cuda.is_available() else "cpu")

    # Test text embedding
    texts = [
        "Information Reconstructionism demonstrates multiplicative dependencies.",
        "When any dimension equals zero, information ceases to exist."
    ]

    embeddings = embedder.embed_texts(texts)
    print(f"✓ Text embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (2, 2048), f"Expected (2, 2048), got {embeddings.shape}"

    # Test code embedding
    code = [
        "def calculate_information(where, what, conveyance):\n    return where * what * conveyance"
    ]

    code_embeddings = embedder.embed_code(code)
    print(f"✓ Code embeddings shape: {code_embeddings.shape}")
    assert code_embeddings.shape == (1, 2048), f"Expected (1, 2048), got {code_embeddings.shape}"

    # Test PROPER late chunking
    long_text = "Information theory " * 1000  # Long text
    chunk_embeddings = embedder.process_long_document(long_text)
    print(f"✓ Late chunking produced {len(chunk_embeddings)} chunks")
    if chunk_embeddings:
        first_chunk = chunk_embeddings[0]
        print(f"  First chunk: {first_chunk.start_char}-{first_chunk.end_char} chars, "
              f"{first_chunk.start_token}-{first_chunk.end_token} tokens")
        print(f"  Context window used: {first_chunk.context_window_used} tokens")
        print(f"  Embedding shape: {first_chunk.embedding.shape}")

    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    test_jina_v4()