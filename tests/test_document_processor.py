#!/usr/bin/env python3
"""
Test Document Processor
========================

Tests for the generic document processor to ensure it maintains
performance and correctness while being source-agnostic.
"""

import time
import logging
from pathlib import Path
from typing import List
import sys

from core.workflows.workflow_pdf import (
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult,
    ExtractionResult
)
# Note: chunking strategies may need to be updated based on new module structure
# from core.processors.chunking_strategies import (
#     ChunkingStrategyFactory,
#     TokenBasedChunking,
#     SemanticChunking
# )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_document_processor_initialization():
    """Test that DocumentProcessor initializes correctly."""
    print("\n=== Testing DocumentProcessor Initialization ===")
    
    # Test with default config
    processor = DocumentProcessor()
    assert processor.config is not None
    assert processor.config.embedding_model == 'jina-v4'
    assert processor.config.embedding_dim == 2048
    print("✓ Default initialization successful")
    
    # Test with custom config
    custom_config = ProcessingConfig(
        use_gpu=False,
        chunk_size_tokens=500,
        chunk_overlap_tokens=100,
        chunking_strategy='semantic'
    )
    processor = DocumentProcessor(custom_config)
    assert processor.config.use_gpu == False
    assert processor.config.chunk_size_tokens == 500
    print("✓ Custom configuration successful")
    
    return True


def test_chunking_strategies():
    """
    Run unit-style checks for token-based, semantic, and sliding-window chunking strategies.
    
    Creates a multi-paragraph sample text, instantiates each chunking strategy via ChunkingStrategyFactory with representative parameters, and verifies that each strategy produces at least one chunk. For sliding-window chunking, also checks that adjacent chunks share tokens (overlap). Intended for use in the test suite to validate chunk generation behavior; raises AssertionError on failure.
    
    Returns:
        bool: True if all chunking strategy checks pass.
    """
    print("\n=== Testing Chunking Strategies ===")
    
    sample_text = """
    This is the first paragraph of our test document. It contains multiple sentences
    that should be kept together when using semantic chunking. The paragraph has
    enough content to test our chunking logic properly.
    
    This is the second paragraph. It's shorter but still meaningful. We want to see
    how different strategies handle paragraph boundaries.
    
    The third paragraph is here to ensure we have enough content for testing sliding
    windows and overlapping chunks. It needs to be long enough to trigger multiple
    chunks when using smaller chunk sizes.
    """ * 5  # Repeat to ensure we have enough text
    
    # Test token-based chunking
    token_chunker = ChunkingStrategyFactory.create_strategy(
        'token',
        chunk_size=50,
        chunk_overlap=10
    )
    token_chunks = token_chunker.create_chunks(sample_text)
    assert len(token_chunks) > 0
    print(f"✓ Token-based chunking: {len(token_chunks)} chunks created")
    
    # Test semantic chunking
    semantic_chunker = ChunkingStrategyFactory.create_strategy(
        'semantic',
        max_chunk_size=100,
        min_chunk_size=20
    )
    semantic_chunks = semantic_chunker.create_chunks(sample_text)
    assert len(semantic_chunks) > 0
    print(f"✓ Semantic chunking: {len(semantic_chunks)} chunks created")
    
    # Test sliding window chunking
    sliding_chunker = ChunkingStrategyFactory.create_strategy(
        'sliding',
        window_size=40,
        step_size=20
    )
    sliding_chunks = sliding_chunker.create_chunks(sample_text)
    assert len(sliding_chunks) > 0
    print(f"✓ Sliding window chunking: {len(sliding_chunks)} chunks created")
    
    # Verify overlap in sliding window
    if len(sliding_chunks) > 1:
        overlap = set(sliding_chunks[0].text.split()) & set(sliding_chunks[1].text.split())
        assert len(overlap) > 0
        print(f"✓ Sliding window overlap verified: {len(overlap)} common tokens")
    
    return True


def test_processing_result_serialization():
    """
    Validate ProcessingResult.to_dict() produces a complete, correctly structured serialization.
    
    Creates a mock ExtractionResult and two ChunkWithEmbedding instances (2048-dim embeddings), assembles a ProcessingResult, serializes it with to_dict(), and asserts:
    - overall success flag is preserved,
    - chunks are serialized with text and full-length embeddings,
    - top-level sections 'extraction', 'processing_metadata', and 'performance' are present,
    - performance.total_time matches the ProcessingResult.total_processing_time.
    
    Returns:
        bool: True when all assertions pass (test success).
    """
    print("\n=== Testing ProcessingResult Serialization ===")
    
    # Create mock extraction result
    extraction = ExtractionResult(
        full_text="Test document content",
        tables=[{"table": "data"}],
        equations=[{"eq": "E=mc^2"}],
        extraction_time=1.5
    )
    
    # Create mock processing result
    from core.embedders import ChunkWithEmbedding
    import numpy as np
    
    mock_chunks = [
        ChunkWithEmbedding(
            text="Test chunk 1",
            embedding=np.random.rand(2048),
            start_char=0,
            end_char=12,
            start_token=0,
            end_token=3,
            chunk_index=0,
            total_chunks=2,
            context_window_used=100
        ),
        ChunkWithEmbedding(
            text="Test chunk 2",
            embedding=np.random.rand(2048),
            start_char=13,
            end_char=25,
            start_token=3,
            end_token=6,
            chunk_index=1,
            total_chunks=2,
            context_window_used=100
        )
    ]
    
    result = ProcessingResult(
        extraction=extraction,
        chunks=mock_chunks,
        processing_metadata={'test': 'metadata'},
        total_processing_time=5.0,
        extraction_time=1.5,
        chunking_time=1.0,
        embedding_time=2.5,
        success=True
    )
    
    # Test serialization
    result_dict = result.to_dict()
    assert result_dict['success'] == True
    assert len(result_dict['chunks']) == 2
    assert result_dict['chunks'][0]['text'] == "Test chunk 1"
    assert len(result_dict['chunks'][0]['embedding']) == 2048
    print("✓ ProcessingResult serialization successful")
    
    # Verify all fields are present
    assert 'extraction' in result_dict
    assert 'processing_metadata' in result_dict
    assert 'performance' in result_dict
    assert result_dict['performance']['total_time'] == 5.0
    print("✓ All fields properly serialized")
    
    return True


def test_arxiv_manager():
    """
    Run integration checks for ArXivManager and ArXivValidator.
    
    Validates new and old arXiv ID formats, ensures invalid IDs are rejected, parses versioned IDs and year/month components, verifies generated PDF paths, and initializes an ArXivManager with a default ProcessingConfig. Returns False if the arXiv manager module cannot be imported; otherwise returns True. Test failures raise AssertionError.
    """
    print("\n=== Testing ArXiv Manager ===")
    
    try:
        from core.tools.arxiv.arxiv_manager import ArXivManager, ArXivValidator
        
        # Test ArXiv ID validation
        validator = ArXivValidator()
        
        # Test new format
        is_valid, error = validator.validate_arxiv_id("2301.00303")
        assert is_valid == True
        print("✓ New format ArXiv ID validation successful")
        
        # Test old format
        is_valid, error = validator.validate_arxiv_id("math-ph/0701001")
        assert is_valid == True, f"Old format validation failed: {error}"
        print("✓ Old format ArXiv ID validation successful")
        
        # Test invalid format
        is_valid, error = validator.validate_arxiv_id("invalid-id")
        assert is_valid == False
        assert error is not None
        print("✓ Invalid ID correctly rejected")
        
        # Test ID parsing
        components = validator.parse_arxiv_id("2301.00303v2")
        assert components['base_id'] == "2301.00303"
        assert components['version'] == "v2"
        assert components['year_month'] == "2301"
        print("✓ ArXiv ID parsing successful")
        
        # Test path generation
        pdf_path = validator.get_pdf_path("2301.00303")
        assert str(pdf_path).endswith("2301/2301.00303.pdf")
        print("✓ PDF path generation successful")
        
        # Initialize manager (without database)
        manager = ArXivManager(processing_config=ProcessingConfig())
        assert manager.processor is not None
        assert manager.validator is not None
        print("✓ ArXivManager initialization successful")
        
    except ImportError as e:
        print(f"⚠ Could not import ArXiv manager: {e}")
        return False
    
    return True


def run_performance_benchmark():
    """
    Run a simple performance benchmark of available chunking strategies.
    
    Creates a large sample text, instantiates a DocumentProcessor with a token-based
    configuration, and measures chunk creation throughput for the 'token',
    'semantic', and 'sliding' strategies. Prints the number of chunks produced,
    elapsed time, and chunks-per-second for each strategy.
    
    Returns:
        bool: True on completion.
    """
    print("\n=== Running Performance Benchmark ===")
    
    # Create large sample text
    large_text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
    tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
    quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo.
    """ * 100  # ~30K characters
    
    processor = DocumentProcessor(ProcessingConfig(
        chunk_size_tokens=256,
        chunk_overlap_tokens=50,
        chunking_strategy='token'
    ))
    
    # Benchmark chunking strategies
    strategies = ['token', 'semantic', 'sliding']
    
    for strategy in strategies:
        kwargs = {}
        if strategy == 'token':
            kwargs = {'chunk_size': 256, 'chunk_overlap': 50}
        elif strategy == 'semantic':
            kwargs = {'max_chunk_size': 256}
        elif strategy == 'sliding':
            kwargs = {'window_size': 256, 'step_size': 128}
        
        chunker = ChunkingStrategyFactory.create_strategy(strategy, **kwargs)
        
        start_time = time.time()
        chunks = chunker.create_chunks(large_text)
        elapsed = time.time() - start_time
        
        print(f"  {strategy:12s}: {len(chunks):3d} chunks in {elapsed:.3f}s "
              f"({len(chunks)/elapsed:.1f} chunks/sec)")
    
    return True


def main():
    """
    Run the module's test suite and report results.
    
    Executes a predefined list of test functions for the document processor (initialization,
    chunking strategies, result serialization, optional ArXiv integration, and a performance
    benchmark). Prints per-test outcomes and a final summary to standard output.
    
    Returns:
        bool: True if all tests passed (no failures), False otherwise.
    """
    print("=" * 60)
    print("DOCUMENT PROCESSOR TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Initialization", test_document_processor_initialization),
        ("Chunking Strategies", test_chunking_strategies),
        ("Result Serialization", test_processing_result_serialization),
        ("ArXiv Manager", test_arxiv_manager),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)