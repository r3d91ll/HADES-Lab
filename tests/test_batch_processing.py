#!/usr/bin/env python3
"""
Batch Processing Test
=====================

Comprehensive test of the document processor with 100 documents
to validate batching, error handling, and performance.
"""

import os
import time
import logging
import tempfile
import random
import string
from pathlib import Path
from typing import List, Tuple
import sys

# Set up Python path
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from core.processors.document_processor import (
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult
)
from core.processors.chunking_strategies import ChunkingStrategyFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_document(doc_id: int, size: str = "medium") -> str:
    """
    Generate a synthetic, academic-style document for testing.
    
    This returns a text string composed of a title, an abstract, and a number of sections with sample prose and a short mathematical formulation. Use size to control approximate length.
    
    Parameters:
        doc_id (int): Numeric identifier included in the generated title.
        size (str): One of 'small', 'medium', 'large', or 'empty'. Controls number of sections:
            - 'small' => ~3 sections
            - 'medium' => ~10 sections (default)
            - 'large' => ~30 sections
            - 'empty' => returns an empty string
    
    Returns:
        str: Generated document content as plain text. For 'empty' size, returns "".
    """
    if size == "empty":
        return ""
    
    # Generate realistic academic-style content
    title = f"Research Paper {doc_id}: Advances in {random.choice(['Machine Learning', 'Natural Language Processing', 'Computer Vision', 'Graph Theory', 'Quantum Computing'])}"
    
    abstract = f"""
    Abstract: This paper presents novel contributions to the field of computational science.
    We introduce a new algorithm that achieves state-of-the-art performance on benchmark datasets.
    Our approach combines theoretical insights with practical implementation considerations.
    The results demonstrate significant improvements over baseline methods.
    """
    
    # Generate paragraphs based on size
    sizes = {
        "small": 3,
        "medium": 10,
        "large": 30
    }
    num_paragraphs = sizes.get(size, 10)
    
    paragraphs = [title, abstract]
    
    for i in range(num_paragraphs):
        paragraph = f"""
        Section {i+1}: {''.join(random.choices(string.ascii_lowercase + ' ', k=random.randint(100, 300)))}
        
        We observe that the proposed method exhibits interesting properties. 
        Specifically, when applied to the test dataset, we find that performance metrics improve.
        The theoretical analysis suggests that this improvement is due to the novel architecture.
        Furthermore, empirical results validate our hypothesis about the underlying mechanisms.
        
        Mathematical formulation: Let X be the input space and Y be the output space.
        We define a mapping f: X → Y such that the loss function L is minimized.
        The optimization problem can be expressed as: min_θ E[L(f_θ(x), y)].
        """
        paragraphs.append(paragraph)
    
    return "\n\n".join(paragraphs)


def create_test_documents(num_docs: int, temp_dir: Path) -> List[Tuple[Path, int]]:
    """
    Create a set of synthetic test documents on disk and return their paths and IDs.
    
    Generates `num_docs` text files (named "doc_0000.txt", "doc_0001.txt", ...) in `temp_dir`. Each file contains synthetic content produced by `generate_synthetic_document`, with a randomized size selected from the distribution: small (20%), medium (50%), large (25%), and empty (5%) to exercise edge cases. Files are encoded as UTF-8.
    
    Parameters:
        num_docs (int): Number of documents to create.
        temp_dir (Path): Directory where files will be written; must exist and be writable.
    
    Returns:
        List[Tuple[Path, int]]: A list of (file_path, doc_id) tuples for each created document.
    
    Side effects:
        Writes files to disk using the ".txt" extension and logs progress every 10 documents.
    """
    documents = []
    
    # Create a mix of document sizes
    size_distribution = {
        "small": 0.2,
        "medium": 0.5,
        "large": 0.25,
        "empty": 0.05  # 5% empty documents to test edge cases
    }
    
    for i in range(num_docs):
        # Determine document size
        rand = random.random()
        cumsum = 0
        size = "medium"
        for s, prob in size_distribution.items():
            cumsum += prob
            if rand < cumsum:
                size = s
                break
        
        # Generate content
        content = generate_synthetic_document(i, size)
        
        # Save to file
        file_path = temp_dir / f"doc_{i:04d}.txt"
        file_path.write_text(content, encoding='utf-8')
        
        documents.append((file_path, i))
        
        if (i + 1) % 10 == 0:
            logger.info(f"Created {i + 1}/{num_docs} test documents")
    
    return documents


def test_batch_processing(num_docs: int = 100):
    """
    Run an end-to-end batch-processing test using synthetic documents.
    
    Creates a temporary directory with `num_docs` synthetic text documents, then processes them in batches across multiple chunking strategies ('traditional', 'late', 'semantic') using ProcessingConfig/DocumentProcessor. Aggregates per-document results (success/failure, chunk counts, processing times), logs throughput and summary statistics, and exercises error-handling scenarios for each strategy via test_error_cases.
    
    Parameters:
        num_docs (int): Number of synthetic documents to create and process (default: 100).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH PROCESSING TEST - {num_docs} Documents")
    logger.info(f"{'='*60}\n")
    
    # Create temporary directory for test documents
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Create test documents
        logger.info("Step 1: Creating test documents...")
        start_time = time.time()
        documents = create_test_documents(num_docs, temp_path)
        creation_time = time.time() - start_time
        logger.info(f"✓ Created {num_docs} documents in {creation_time:.2f}s")
        
        # Step 2: Initialize processor with different strategies
        strategies = ['traditional', 'late', 'semantic']
        
        for strategy in strategies:
            logger.info(f"\n{'='*40}")
            logger.info(f"Testing with {strategy.upper()} chunking strategy")
            logger.info(f"{'='*40}")
            
            # Configure processor
            config = ProcessingConfig(
                chunking_strategy=strategy,
                chunk_size_tokens=256,
                chunk_overlap_tokens=50,
                use_gpu=False,  # For testing
                use_fp16=False,  # For testing
                batch_size=10
            )
            
            processor = DocumentProcessor(config)
            
            # Step 3: Process documents in batches
            logger.info(f"Processing {num_docs} documents...")
            
            results = []
            failed_docs = []
            empty_docs = []
            processing_times = []
            chunk_counts = []
            
            batch_size = config.batch_size
            start_time = time.time()
            
            for batch_start in range(0, num_docs, batch_size):
                batch_end = min(batch_start + batch_size, num_docs)
                batch_docs = documents[batch_start:batch_end]
                
                # Process batch
                batch_paths = [(doc_path, None) for doc_path, _ in batch_docs]
                batch_ids = [f"doc_{doc_id}" for _, doc_id in batch_docs]
                
                batch_results = processor.process_batch(batch_paths, batch_ids)
                
                # Analyze results
                for result, (doc_path, doc_id) in zip(batch_results, batch_docs):
                    results.append(result)
                    
                    if not result.success:
                        failed_docs.append((doc_id, result.errors))
                    
                    if not result.chunks:
                        empty_docs.append(doc_id)
                    else:
                        chunk_counts.append(len(result.chunks))
                    
                    processing_times.append(result.total_processing_time)
                
                # Progress update
                if (batch_end % 20 == 0) or batch_end == num_docs:
                    elapsed = time.time() - start_time
                    rate = batch_end / elapsed if elapsed > 0 else 0
                    logger.info(f"  Processed {batch_end}/{num_docs} documents ({rate:.1f} docs/sec)")
            
            total_time = time.time() - start_time
            
            # Step 4: Analyze results
            logger.info("\n📊 Results Analysis:")
            logger.info(f"  Total processing time: {total_time:.2f}s")
            logger.info(f"  Average rate: {num_docs/total_time:.2f} docs/second")
            logger.info(f"  Successful: {len([r for r in results if r.success])}/{num_docs}")
            logger.info(f"  Failed: {len(failed_docs)}")
            logger.info(f"  Empty results: {len(empty_docs)}")
            
            if chunk_counts:
                avg_chunks = sum(chunk_counts) / len(chunk_counts)
                logger.info(f"  Average chunks per document: {avg_chunks:.1f}")
                logger.info(f"  Total chunks created: {sum(chunk_counts)}")
            
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                logger.info(f"  Average processing time per doc: {avg_time:.3f}s")
            
            # Report any failures
            if failed_docs:
                logger.warning("\n⚠️  Failed documents:")
                for doc_id, errors in failed_docs[:5]:  # Show first 5
                    logger.warning(f"    Doc {doc_id}: {errors}")
                if len(failed_docs) > 5:
                    logger.warning(f"    ... and {len(failed_docs) - 5} more")
            
            # Test error handling
            logger.info("\n🧪 Testing error handling...")
            test_error_cases(processor)
    
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING TEST COMPLETE")
    logger.info(f"{'='*60}\n")


def test_error_cases(processor: DocumentProcessor):
    """
    Run a set of small error-condition checks against the provided DocumentProcessor.
    
    Performs four focused tests to verify the processor's handling of common error cases:
    1. Non-existent file — verifies processing a missing path either fails gracefully or raises an exception.
    2. Empty document — verifies an empty file produces no chunks or is handled via an exception.
    3. Invalid chunking parameters — constructs a processor with chunk_overlap_tokens > chunk_size_tokens and ensures processing fails or raises a ValueError.
    4. Batch with mixed valid/invalid documents — submits a batch containing a valid file, an empty file, a non-existent path, and a duplicate valid file, and reports how many entries succeeded.
    
    Creates temporary files/directories as needed and attempts to clean them up. This function does not return a value; it logs outcomes for each check.
    """
    # Test 1: Non-existent file
    try:
        result = processor.process_document("/non/existent/file.pdf")
        if not result.success:
            logger.info("  ✓ Non-existent file handled correctly")
        else:
            logger.error("  ✗ Non-existent file should have failed")
    except Exception as e:
        logger.info(f"  ✓ Non-existent file raised exception: {type(e).__name__}")
    
    # Test 2: Empty document
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        empty_file = f.name
    
    try:
        result = processor.process_document(empty_file)
        if not result.chunks:
            logger.info("  ✓ Empty document handled correctly")
        else:
            logger.error(f"  ✗ Empty document produced {len(result.chunks)} chunks")
    except Exception as e:
        logger.info(f"  ✓ Empty document handled with exception: {type(e).__name__}")
    finally:
        os.unlink(empty_file)
    
    # Test 3: Invalid chunking parameters
    try:
        bad_config = ProcessingConfig(
            chunk_size_tokens=100,
            chunk_overlap_tokens=150  # Overlap > chunk size
        )
        bad_processor = DocumentProcessor(bad_config)
        
        # Try to process something
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for bad chunking parameters.")
            test_file = f.name
        
        result = bad_processor.process_document(test_file)
        if not result.success:
            logger.info("  ✓ Invalid chunking parameters caught")
        else:
            logger.error("  ✗ Invalid chunking parameters should have failed")
        
        os.unlink(test_file)
    except ValueError as e:
        logger.info(f"  ✓ Invalid chunking parameters raised ValueError: {e}")
    except Exception as e:
        logger.error(f"  ✗ Unexpected exception: {e}")
    
    # Test 4: Batch with mixed valid/invalid documents
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mix of valid and problematic documents
        valid_doc = temp_path / "valid.txt"
        valid_doc.write_text("This is valid content with multiple words.")
        
        empty_doc = temp_path / "empty.txt"
        empty_doc.write_text("")
        
        docs = [
            (valid_doc, None),
            (empty_doc, None),
            (Path("/non/existent.txt"), None),
            (valid_doc, None)  # Valid again
        ]
        
        results = processor.process_batch(docs, ["doc1", "doc2", "doc3", "doc4"])
        
        valid_count = sum(1 for r in results if r.success)
        logger.info(f"  ✓ Batch with mixed documents: {valid_count}/4 successful")


def performance_stress_test():
    """
    Run a suite of stress tests that process a very large synthetic document across multiple chunk-size/overlap configurations.
    
    For each predefined configuration this function:
    - Writes a single large test file to a temporary directory.
    - Constructs a ProcessingConfig and DocumentProcessor (using the 'traditional' chunking strategy).
    - Processes the file with processor.process_document(), measures elapsed time, and logs results:
      - success/failure
      - number of chunks produced
      - processing time and chunking rate (chunks/sec)
    
    Side effects:
    - Creates and deletes temporary files/directories.
    - Emits progress and result logs via the module logger.
    
    This function does not return a value.
    """
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE STRESS TEST")
    logger.info("="*60 + "\n")
    
    test_configs = [
        ("Small chunks, high overlap", 100, 50),
        ("Medium chunks, medium overlap", 256, 64),
        ("Large chunks, low overlap", 512, 50),
        ("Very large chunks, no overlap", 1000, 0)
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create one large document for stress testing
        large_content = generate_synthetic_document(999, "large") * 10  # Very large
        large_doc = temp_path / "large_doc.txt"
        large_doc.write_text(large_content)
        
        for desc, chunk_size, overlap in test_configs:
            logger.info(f"Testing: {desc}")
            logger.info(f"  Chunk size: {chunk_size}, Overlap: {overlap}")
            
            config = ProcessingConfig(
                chunk_size_tokens=chunk_size,
                chunk_overlap_tokens=overlap,
                chunking_strategy='traditional'
            )
            
            processor = DocumentProcessor(config)
            
            start = time.time()
            result = processor.process_document(large_doc)
            elapsed = time.time() - start
            
            if result.success:
                logger.info(f"  ✓ Processed in {elapsed:.2f}s")
                logger.info(f"    Chunks created: {len(result.chunks)}")
                logger.info(f"    Chunking rate: {len(result.chunks)/elapsed:.1f} chunks/sec")
            else:
                logger.error(f"  ✗ Failed: {result.errors}")
            
            logger.info("")


def main():
    """
    Run all batch processing tests.
    """
    # Run main batch test with 100 documents
    test_batch_processing(100)
    
    # Run performance stress test
    performance_stress_test()
    
    logger.info("\n✅ All batch processing tests completed!")


if __name__ == "__main__":
    main()