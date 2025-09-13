#!/usr/bin/env python3
"""
Overnight Test - 2000 Documents
================================

Comprehensive overnight test to validate the document processing architecture
with a large corpus of documents. Tests all chunking strategies and error
handling at scale.
"""

import os
import sys
import time
import json
import random
import string
import tempfile
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure detailed logging
log_dir = Path("tests/logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"overnight_test_{timestamp}.log"

# Set up both file and console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Use real embedder by default for accurate testing
MOCK_EMBEDDER = False  # Set to True only for quick debugging

if MOCK_EMBEDDER:
    logger.info("Using MOCK embedder for faster processing")
    
    class MockEmbedder:
        """Mock embedder for testing without loading heavy models."""
        
        def __init__(self, *args, **kwargs):
            """
            Initialize the instance and set the default embedding dimensionality.
            
            Sets the instance attribute `embedding_dim` to 2048. Accepts arbitrary positional and keyword arguments for compatibility with callers; they are not used by this initializer.
            """
            self.embedding_dim = 2048
        
        def embed_texts(self, texts: List[str], batch_size: int = 1):
            """
            Generate random embeddings for a batch of input texts (testing only).
            
            This returns a NumPy array of shape (len(texts), self.embedding_dim) with dtype float32 containing random normal values. The `batch_size` parameter is accepted for API compatibility but is ignored by this mock implementation.
            """
            import numpy as np
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
        
        def embed_with_late_chunking(self, text: str):
            """
            Create overlapping text chunks and attach mock embeddings for late-chunking tests.
            
            This mock implementation splits `text` into overlapping word-based chunks (chunk size 256 words with a 50-word overlap) and returns a list of `ChunkWithEmbedding` objects. Each chunk receives a randomly sampled float32 embedding of length `self.embedding_dim`. Character offsets (`start_char`, `end_char`) are approximate (computed as token indices * 5) and token offsets (`start_token`, `end_token`) correspond to word indices. The returned chunks have their `chunk_index` and `total_chunks` fields populated.
            
            Parameters:
                text (str): The source document text to chunk.
            
            Returns:
                List[ChunkWithEmbedding]: Mock chunks with random embeddings suitable for testing late-chunking behavior.
            """
            import numpy as np
            from core.embedders import ChunkWithEmbedding
            
            words = text.split()
            chunk_size = 256
            chunks = []
            
            for i in range(0, len(words), chunk_size - 50):  # 50 word overlap
                chunk_text = ' '.join(words[i:i+chunk_size])
                chunk = ChunkWithEmbedding(
                    text=chunk_text,
                    embedding=np.random.randn(self.embedding_dim).astype(np.float32),
                    start_char=i * 5,  # Approximate
                    end_char=(i + len(chunk_text)) * 5,
                    start_token=i,
                    end_token=min(i + chunk_size, len(words)),
                    chunk_index=len(chunks),
                    total_chunks=0,
                    context_window_used=chunk_size
                )
                chunks.append(chunk)
            
            for chunk in chunks:
                chunk.total_chunks = len(chunks)
            
            return chunks
    
    # Monkey patch the embedder
    import core.framework.embedders
    core.framework.embedders.JinaV4Embedder = MockEmbedder
else:
    logger.info("Using REAL Jina embedder - this will be slower")

from core.workflows.workflow_pdf import (
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult
)


def generate_document(doc_id: int, doc_type: str = "mixed") -> str:
    """
    Generate a synthetic document text for testing.
    
    doc_type selects the style of generated content and may be one of:
      - "research": academic-paper style with title, abstract, and multiple sections.
      - "technical": structured technical documentation (Overview, Installation, Configuration, API Reference, Examples).
      - "narrative": story/report style composed of multiple paragraphs.
      - "mixed": randomly selects one of the other types (weighted sampling among research, technical, narrative, huge, empty).
      - "empty": returns an empty string.
      - "huge": produces a very large, multi-section document.
    
    Parameters:
        doc_id (int): Numeric identifier used in titles and seeded content fragments.
        doc_type (str): Desired document style (see choices above). Defaults to "mixed".
    
    Returns:
        str: The generated document content (possibly empty). Content is randomized; repeated calls may produce different results.
    """
    
    if doc_type == "empty":
        return ""
    
    if doc_type == "mixed":
        # Randomly choose type with weights
        doc_type = random.choices(
            ['research', 'technical', 'narrative', 'huge', 'empty'],
            weights=[0.4, 0.3, 0.25, 0.04, 0.01],
            k=1
        )[0]
    
    if doc_type == "empty":
        return ""
    
    # Base content based on type
    if doc_type == "research":
        title = f"Paper {doc_id}: {random.choice(['Neural', 'Quantum', 'Statistical', 'Computational'])} {random.choice(['Methods', 'Approaches', 'Algorithms', 'Models'])} for {random.choice(['NLP', 'Vision', 'Learning', 'Optimization'])}"
        
        abstract = f"""
        Abstract: This paper presents novel contributions to computational science.
        We introduce algorithm #{doc_id} that achieves state-of-the-art performance.
        Our approach combines {random.choice(['deep learning', 'reinforcement learning', 'graph theory', 'optimization'])}
        with {random.choice(['attention mechanisms', 'transformer architectures', 'convolution networks', 'recursive models'])}.
        Results demonstrate {random.uniform(5, 25):.1f}% improvement over baselines.
        """
        
        sections = [
            "Introduction: " + " ".join([f"word{i}" for i in range(200)]),
            "Related Work: " + " ".join([f"reference{i}" for i in range(150)]),
            "Methodology: " + " ".join([f"equation{i}" for i in range(300)]),
            "Experiments: " + " ".join([f"result{i}" for i in range(250)]),
            "Conclusion: " + " ".join([f"summary{i}" for i in range(100)])
        ]
        
        content = f"{title}\n\n{abstract}\n\n" + "\n\n".join(sections)
        
    elif doc_type == "technical":
        title = f"Technical Document {doc_id}: {random.choice(['API', 'System', 'Module', 'Component'])} Documentation"
        
        content = f"{title}\n\n"
        content += "## Overview\n" + " ".join([f"overview{i}" for i in range(100)]) + "\n\n"
        content += "## Installation\n" + " ".join([f"step{i}" for i in range(50)]) + "\n\n"
        content += "## Configuration\n" + " ".join([f"config{i}" for i in range(150)]) + "\n\n"
        content += "## API Reference\n" + " ".join([f"api{i}" for i in range(300)]) + "\n\n"
        content += "## Examples\n" + " ".join([f"example{i}" for i in range(200)])
        
    elif doc_type == "narrative":
        title = f"Document {doc_id}: {random.choice(['Analysis', 'Report', 'Summary', 'Review'])}"
        
        paragraphs = []
        for p in range(random.randint(5, 15)):
            paragraph = " ".join([f"sentence{i}" for i in range(random.randint(50, 150))])
            paragraphs.append(paragraph)
        
        content = f"{title}\n\n" + "\n\n".join(paragraphs)
        
    elif doc_type == "huge":
        title = f"Large Document {doc_id}: Comprehensive Analysis"
        content = f"{title}\n\n"
        
        # Generate a very large document
        for section in range(20):
            content += f"\n## Section {section + 1}\n"
            content += " ".join([f"word{i}" for i in range(1000)])
            content += "\n\n"
    
    else:
        content = f"Document {doc_id}: Default content " + " ".join([f"word{i}" for i in range(100)])
    
    return content


def create_test_corpus(num_docs: int, output_dir: Path) -> Dict[str, Any]:
    """
    Generate a synthetic corpus of documents, write each document to output_dir, and return statistics about the created corpus.
    
    This will create num_docs text files named "doc_00000.txt", "doc_00001.txt", ... in output_dir by calling generate_document for each index. Progress is logged periodically. The output_dir must exist and be writable.
    
    Parameters:
        num_docs (int): Number of documents to generate.
        output_dir (Path): Directory where document files will be written.
    
    Returns:
        Dict[str, Any]: Statistics about the created corpus with keys:
            - total_docs: int, requested number of documents
            - empty_docs: int, count of zero-length documents
            - small_docs: int, documents with <500 chars
            - medium_docs: int, documents with 500–4999 chars
            - large_docs: int, documents with 5000–19999 chars
            - huge_docs: int, documents with >=20000 chars
            - total_chars: int, total characters across all documents
            - creation_time: float, total time in seconds spent creating the corpus
    """
    logger.info(f"Creating test corpus of {num_docs} documents in {output_dir}")
    
    stats = {
        'total_docs': num_docs,
        'empty_docs': 0,
        'small_docs': 0,
        'medium_docs': 0,
        'large_docs': 0,
        'huge_docs': 0,
        'total_chars': 0,
        'creation_time': 0
    }
    
    start_time = time.time()
    
    # Create documents with progress reporting
    for i in range(num_docs):
        # Generate document
        doc_type = random.choices(
            ['research', 'technical', 'narrative', 'mixed', 'empty', 'huge'],
            weights=[0.35, 0.25, 0.25, 0.10, 0.02, 0.03],
            k=1
        )[0]
        
        content = generate_document(i, doc_type)
        
        # Save document
        doc_path = output_dir / f"doc_{i:05d}.txt"
        doc_path.write_text(content, encoding='utf-8')
        
        # Update statistics
        doc_size = len(content)
        stats['total_chars'] += doc_size
        
        if doc_size == 0:
            stats['empty_docs'] += 1
        elif doc_size < 500:
            stats['small_docs'] += 1
        elif doc_size < 5000:
            stats['medium_docs'] += 1
        elif doc_size < 20000:
            stats['large_docs'] += 1
        else:
            stats['huge_docs'] += 1
        
        # Progress reporting
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (num_docs - i - 1) / rate
            logger.info(f"Created {i + 1}/{num_docs} documents ({rate:.1f} docs/sec, ETA: {eta:.1f}s)")
    
    stats['creation_time'] = time.time() - start_time
    
    logger.info(f"Corpus creation complete in {stats['creation_time']:.1f}s")
    logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
    
    return stats


def run_overnight_test(num_docs: int = 2000):
    """
    Run the full overnight end-to-end test workflow that builds a synthetic corpus, processes it under several chunking configurations, and collects performance and correctness metrics.
    
    This function:
    - Creates a temporary test corpus of synthetic text documents (num_docs) on disk.
    - Runs multiple document-processing configurations (different chunking strategies and sizes) using DocumentProcessor, processing documents in batches.
    - Aggregates per-run statistics (success/failure counts, chunk counts, processing rates, timing) and records sample failures.
    - Persists intermediate and final results to a timestamped JSON results file and writes detailed logs to the configured log file.
    
    Parameters:
        num_docs (int): Number of synthetic documents to generate and process (default 2000).
    
    Returns:
        dict: A dictionary containing:
          - 'corpus_stats': metadata about the generated corpus,
          - 'test_runs': list of per-configuration summaries (config, totals, rates, failed samples, etc.),
          - 'summary': final summary including best configuration, best processing rate, completion time, and file paths.
    
    Side effects:
    - Writes generated document files into a temporary directory.
    - Writes intermediate and final JSON results to the configured results file and emits logs to the configured log file.
    """
    logger.info("="*70)
    logger.info(f"OVERNIGHT TEST - {num_docs} Documents")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("="*70)
    
    results_file = log_dir / f"overnight_results_{timestamp}.json"
    
    # Create temporary directory for test corpus
    with tempfile.TemporaryDirectory(prefix="overnight_test_") as temp_dir:
        temp_path = Path(temp_dir)
        logger.info(f"Test corpus directory: {temp_path}")
        
        # Step 1: Create test corpus
        logger.info("\n" + "="*50)
        logger.info("STEP 1: Creating Test Corpus")
        logger.info("="*50)
        
        corpus_stats = create_test_corpus(num_docs, temp_path)
        
        # Get list of all documents
        doc_files = sorted(temp_path.glob("*.txt"))
        logger.info(f"Found {len(doc_files)} documents to process")
        
        # Step 2: Test different configurations
        test_configs = [
            {
                'name': 'Traditional Chunking - Small Chunks',
                'strategy': 'traditional',
                'chunk_size': 256,
                'overlap': 50
            },
            {
                'name': 'Traditional Chunking - Large Chunks',
                'strategy': 'traditional',
                'chunk_size': 512,
                'overlap': 100
            },
            {
                'name': 'Late Chunking - Optimal',
                'strategy': 'late',
                'chunk_size': 256,
                'overlap': 50
            }
        ]
        
        all_results = {
            'corpus_stats': corpus_stats,
            'test_runs': [],
            'summary': {}
        }
        
        for test_config in test_configs:
            logger.info("\n" + "="*50)
            logger.info(f"TESTING: {test_config['name']}")
            logger.info("="*50)
            
            # Configure processor
            config = ProcessingConfig(
                chunking_strategy=test_config['strategy'],
                chunk_size_tokens=test_config['chunk_size'],
                chunk_overlap_tokens=test_config['overlap'],
                batch_size=20  # Process 20 at a time
            )
            
            processor = DocumentProcessor(config)
            
            # Process documents with detailed tracking
            test_start = time.time()
            results = []
            failed_docs = []
            empty_results = []
            processing_times = []
            chunk_counts = []
            
            batch_size = config.batch_size
            
            for batch_start in range(0, len(doc_files), batch_size):
                batch_end = min(batch_start + batch_size, len(doc_files))
                batch_files = doc_files[batch_start:batch_end]
                
                # Process batch
                batch_paths = [(f, None) for f in batch_files]
                batch_ids = [f"doc_{i}" for i in range(batch_start, batch_end)]
                
                batch_start_time = time.time()
                batch_results = processor.process_batch(batch_paths, batch_ids)
                batch_time = time.time() - batch_start_time
                
                # Analyze batch results
                for result, doc_file in zip(batch_results, batch_files):
                    results.append(result)
                    
                    if not result.success:
                        failed_docs.append({
                            'file': doc_file.name,
                            'errors': result.errors
                        })
                    
                    if not result.chunks:
                        empty_results.append(doc_file.name)
                    else:
                        chunk_counts.append(len(result.chunks))
                    
                    processing_times.append(result.total_processing_time)
                
                # Progress update
                if batch_end % 100 == 0 or batch_end == len(doc_files):
                    elapsed = time.time() - test_start
                    rate = batch_end / elapsed if elapsed > 0 else 0
                    eta = (len(doc_files) - batch_end) / rate if rate > 0 else 0
                    
                    logger.info(
                        f"Progress: {batch_end}/{len(doc_files)} "
                        f"({rate:.1f} docs/sec, ETA: {timedelta(seconds=int(eta))})"
                    )
            
            test_time = time.time() - test_start
            
            # Calculate statistics
            successful = len([r for r in results if r.success])
            total_chunks = sum(chunk_counts)
            avg_chunks = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0
            avg_proc_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            test_summary = {
                'config': test_config,
                'total_docs': len(doc_files),
                'successful': successful,
                'failed': len(failed_docs),
                'empty_results': len(empty_results),
                'total_chunks': total_chunks,
                'avg_chunks_per_doc': avg_chunks,
                'total_time': test_time,
                'docs_per_second': len(doc_files) / test_time,
                'avg_processing_time': avg_proc_time,
                'failed_samples': failed_docs[:5]  # First 5 failures
            }
            
            all_results['test_runs'].append(test_summary)
            
            # Log summary
            logger.info("\n" + "-"*40)
            logger.info(f"Results for {test_config['name']}:")
            logger.info(f"  Success Rate: {successful}/{len(doc_files)} ({100*successful/len(doc_files):.1f}%)")
            logger.info(f"  Processing Rate: {test_summary['docs_per_second']:.1f} docs/sec")
            logger.info(f"  Total Chunks: {total_chunks:,}")
            logger.info(f"  Avg Chunks/Doc: {avg_chunks:.1f}")
            logger.info(f"  Total Time: {timedelta(seconds=int(test_time))}")
            
            if failed_docs:
                logger.warning(f"  Failed Documents: {len(failed_docs)}")
                for fail in failed_docs[:3]:
                    logger.warning(f"    - {fail['file']}: {fail['errors']}")
            
            # Save intermediate results
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"  Results saved to: {results_file}")
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("FINAL SUMMARY")
        logger.info("="*70)
        
        best_rate = max(run['docs_per_second'] for run in all_results['test_runs'])
        best_config = next(run['config']['name'] for run in all_results['test_runs'] 
                          if run['docs_per_second'] == best_rate)
        
        all_results['summary'] = {
            'total_documents': num_docs,
            'best_configuration': best_config,
            'best_rate': best_rate,
            'completed_at': datetime.now().isoformat(),
            'log_file': str(log_file),
            'results_file': str(results_file)
        }
        
        logger.info(f"Best Configuration: {best_config}")
        logger.info(f"Best Rate: {best_rate:.1f} docs/sec")
        logger.info(f"Log File: {log_file}")
        logger.info(f"Results File: {results_file}")
        
        # Save final results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info("\n" + "="*70)
        logger.info(f"TEST COMPLETE at {datetime.now().isoformat()}")
        logger.info("="*70)
        
        return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run overnight document processing test")
    parser.add_argument(
        "--docs", 
        type=int, 
        default=2000,
        help="Number of documents to test (default: 2000)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock embedder for quick testing (not for production validation)"
    )
    
    args = parser.parse_args()
    
    if args.mock:
        MOCK_EMBEDDER = True
        logger.info("WARNING: Using MOCK embedder - results won't reflect real performance!")
    
    try:
        results = run_overnight_test(args.docs)
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Results saved to: {results['summary']['results_file']}")
        print(f"📝 Logs saved to: {results['summary']['log_file']}")
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}", exc_info=True)
        sys.exit(1)