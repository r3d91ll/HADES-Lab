#!/usr/bin/env python3
"""
Simple Batch Processing Test
=============================

Lightweight test of document processing without heavy embedding models.
Tests batching, error handling, and chunking strategies.
"""

import os
import sys
import time
import tempfile
import random
from pathlib import Path
from typing import List, Tuple

# Set up path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock the embedder to avoid loading heavy models
class MockEmbedder:
    """Mock embedder for testing without loading models."""
    
    def __init__(self, *args, **kwargs):
        self.embedding_dim = 2048
    
    def embed_texts(self, texts: List[str], batch_size: int = 1):
        """Return random embeddings for testing."""
        import numpy as np
        return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
    
    def embed_with_late_chunking(self, text: str):
        """Return mock chunks with embeddings."""
        import numpy as np
        from core.framework.embedders import ChunkWithEmbedding
        
        # Simple chunking for testing
        words = text.split()
        chunk_size = 100
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_text = ' '.join(words[i:i+chunk_size])
            chunk = ChunkWithEmbedding(
                text=chunk_text,
                embedding=np.random.randn(self.embedding_dim).astype(np.float32),
                start_char=i,
                end_char=i + len(chunk_text),
                start_token=i,
                end_token=min(i + chunk_size, len(words)),
                chunk_index=len(chunks),
                total_chunks=0,  # Will update later
                context_window_used=chunk_size
            )
            chunks.append(chunk)
        
        # Update total chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks

# Monkey patch the embedder
import core.framework.embedders
core.framework.embedders.JinaV4Embedder = MockEmbedder

from core.processors.document_processor import (
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult
)

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_test_documents(num_docs: int, temp_dir: Path) -> List[Path]:
    """Create test text documents."""
    docs = []
    
    for i in range(num_docs):
        # Mix of document sizes
        if i % 20 == 0:
            # Empty document (5%)
            content = ""
        elif i % 10 == 0:
            # Small document (10%)
            content = f"Document {i}: Small content with just a few words."
        elif i % 5 == 0:
            # Large document (20%)
            content = f"Document {i}: " + " ".join([f"word{j}" for j in range(500)])
        else:
            # Medium document (65%)
            content = f"Document {i}: " + " ".join([f"word{j}" for j in range(200)])
        
        doc_path = temp_dir / f"doc_{i:04d}.txt"
        doc_path.write_text(content)
        docs.append(doc_path)
    
    return docs


def test_batch_processing():
    """Test batch processing with 100 documents."""
    
    print("\n" + "="*60)
    print("BATCH PROCESSING TEST - 100 Documents")
    print("="*60 + "\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test documents
        print("Creating test documents...")
        docs = create_test_documents(100, temp_path)
        print(f"✓ Created {len(docs)} test documents")
        
        # Test different chunking strategies
        strategies = ['traditional', 'late']
        
        for strategy in strategies:
            print(f"\n{'='*40}")
            print(f"Testing {strategy.upper()} chunking")
            print(f"{'='*40}")
            
            config = ProcessingConfig(
                chunking_strategy=strategy,
                chunk_size_tokens=100,
                chunk_overlap_tokens=20,
                batch_size=10
            )
            
            processor = DocumentProcessor(config)
            
            # Process documents
            print("Processing documents...")
            start_time = time.time()
            
            results = []
            batch_size = 10
            
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                batch_paths = [(p, None) for p in batch]
                batch_ids = [f"doc_{j}" for j in range(i, min(i+batch_size, len(docs)))]
                
                batch_results = processor.process_batch(batch_paths, batch_ids)
                results.extend(batch_results)
                
                if (i + batch_size) % 20 == 0 or i + batch_size >= len(docs):
                    print(f"  Processed {min(i + batch_size, len(docs))}/{len(docs)}")
            
            elapsed = time.time() - start_time
            
            # Analyze results
            successful = sum(1 for r in results if r.success)
            failed = sum(1 for r in results if not r.success)
            empty_results = sum(1 for r in results if not r.chunks)
            total_chunks = sum(len(r.chunks) for r in results)
            
            print(f"\n📊 Results:")
            print(f"  Time: {elapsed:.2f}s ({len(docs)/elapsed:.1f} docs/sec)")
            print(f"  Success: {successful}/{len(docs)}")
            print(f"  Failed: {failed}")
            print(f"  Empty: {empty_results}")
            print(f"  Total chunks: {total_chunks}")
            
            if failed > 0:
                # Show first few errors
                errors = [(i, r.errors) for i, r in enumerate(results) if not r.success][:3]
                for i, err in errors:
                    print(f"    Error {i}: {err}")


def test_error_handling():
    """Test error handling cases."""
    
    print("\n" + "="*40)
    print("ERROR HANDLING TESTS")
    print("="*40)
    
    # Test invalid chunking parameters
    print("\nTesting invalid chunking parameters...")
    try:
        config = ProcessingConfig(
            chunk_size_tokens=100,
            chunk_overlap_tokens=150  # Overlap > size
        )
        processor = DocumentProcessor(config)
        
        # Try processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
            f.write("Test content")
            f.flush()
            result = processor.process_document(f.name)
        
        print("  ✗ Should have failed with invalid parameters")
    except ValueError as e:
        print(f"  ✓ Caught invalid parameters: {e}")
    
    # Test non-existent file
    print("\nTesting non-existent file...")
    config = ProcessingConfig()
    processor = DocumentProcessor(config)
    
    result = processor.process_document("/non/existent/file.txt")
    if not result.success:
        print("  ✓ Non-existent file handled correctly")
    else:
        print("  ✗ Non-existent file should have failed")
    
    # Test empty file
    print("\nTesting empty file...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        empty_file = f.name
    
    result = processor.process_document(empty_file)
    if not result.success or not result.chunks:
        print("  ✓ Empty file handled correctly")
    else:
        print(f"  ✗ Empty file produced {len(result.chunks)} chunks")
    
    os.unlink(empty_file)
    
    # Test batch with mixed valid/invalid
    print("\nTesting mixed batch...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        valid_doc = temp_path / "valid.txt"
        valid_doc.write_text("Valid content here")
        
        empty_doc = temp_path / "empty.txt"
        empty_doc.write_text("")
        
        batch = [
            (valid_doc, None),
            (empty_doc, None),
            (Path("/invalid.txt"), None)
        ]
        
        results = processor.process_batch(batch, ["doc1", "doc2", "doc3"])
        
        successful = sum(1 for r in results if r.success)
        print(f"  ✓ Mixed batch: {successful}/3 successful")


def test_chunking_strategies():
    """Test different chunking strategies."""
    
    print("\n" + "="*40)
    print("CHUNKING STRATEGY TESTS")
    print("="*40)
    
    # Create a test document
    test_text = """
    This is the first paragraph of the test document. It contains multiple sentences
    that should be processed correctly. The chunking strategy should handle this well.
    
    This is the second paragraph. It's shorter but still meaningful.
    
    The third paragraph contains more content to ensure we have enough text for testing
    various chunking approaches. We want to see how different strategies handle
    paragraph boundaries and overlapping content.
    """ * 3  # Repeat to get more content
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_text)
        f.flush()
        test_file = f.name
    
    strategies = [
        ('traditional', {'chunk_size_tokens': 50, 'chunk_overlap_tokens': 10}),
        ('late', {'chunk_size_tokens': 50, 'chunk_overlap_tokens': 10})
    ]
    
    for strategy, params in strategies:
        print(f"\nTesting {strategy} chunking:")
        
        config = ProcessingConfig(chunking_strategy=strategy, **params)
        processor = DocumentProcessor(config)
        
        result = processor.process_document(test_file)
        
        if result.success:
            print(f"  ✓ Chunks created: {len(result.chunks)}")
            print(f"    Processing time: {result.total_processing_time:.3f}s")
            if result.chunks:
                print(f"    First chunk size: {len(result.chunks[0].text)} chars")
        else:
            print(f"  ✗ Failed: {result.errors}")
    
    os.unlink(test_file)


def main():
    """Run all tests."""
    
    print("="*60)
    print("SIMPLE BATCH PROCESSING TEST SUITE")
    print("="*60)
    
    # Test batch processing
    test_batch_processing()
    
    # Test error handling
    test_error_handling()
    
    # Test chunking strategies
    test_chunking_strategies()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()