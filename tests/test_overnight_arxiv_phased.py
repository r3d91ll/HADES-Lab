#!/usr/bin/env python3
"""
Overnight Test - Phase-Separated ArXiv Processing
=================================================

Tests the document processing architecture with phase separation:
- Phase 1: Extract all PDFs with Docling (GPU) -> Stage to JSON
- Phase 2: Process all staged JSONs with Jina embeddings (GPU)

This mirrors the production ACID pipeline architecture for optimal performance.
"""

import os
import sys
import time
import json
import random
import torch
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
log_dir = Path("tests/logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"overnight_phased_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our processors
from core.extractors import DoclingExtractor
from core.embedders import JinaV4Embedder

# Staging directory for inter-phase communication
STAGING_DIR = Path("/dev/shm/overnight_staging")
STAGING_DIR.mkdir(exist_ok=True)

# ArXiv PDF repository
ARXIV_PDF_BASE = Path("/bulk-store/arxiv-data/pdf")


def collect_arxiv_papers(target_count: int = 2000, start_year: str = "1501") -> List[Tuple[Path, str]]:
    """
    Collect paths to ArXiv PDF files from the local ARXIV_PDF_BASE repository.
    
    Scans numeric year-month subdirectories under ARXIV_PDF_BASE (filtered to names >= start_year),
    randomly samples PDF files from each directory until up to target_count papers are collected,
    and returns a list of (pdf_path, arxiv_id) tuples where arxiv_id is the PDF filename stem.
    
    Parameters:
        target_count (int): Maximum number of papers to collect.
        start_year (str): Minimum directory name (inclusive) to consider (e.g., "202001" or "1501").
    
    Returns:
        List[Tuple[Path, str]]: Collected (Path to PDF, arXiv id) pairs, up to target_count items.
    
    Raises:
        ValueError: If no year-month directories are found at or after start_year.
    """
    logger.info(f"Collecting {target_count} ArXiv PDFs from {ARXIV_PDF_BASE}")
    
    papers = []
    yymm_dirs = sorted([d for d in ARXIV_PDF_BASE.iterdir() if d.is_dir() and d.name.isdigit()])
    yymm_dirs = [d for d in yymm_dirs if d.name >= start_year]
    
    if not yymm_dirs:
        raise ValueError(f"No ArXiv directories found from {start_year} onwards")
    
    for yymm_dir in yymm_dirs:
        if len(papers) >= target_count:
            break
            
        pdf_files = list(yymm_dir.glob("*.pdf"))
        if pdf_files:
            sample_size = min(len(pdf_files), target_count - len(papers))
            sampled_pdfs = random.sample(pdf_files, sample_size) if sample_size < len(pdf_files) else pdf_files
            
            for pdf_path in sampled_pdfs:
                arxiv_id = pdf_path.stem
                papers.append((pdf_path, arxiv_id))
                
                if len(papers) >= target_count:
                    break
    
    logger.info(f"Collected {len(papers)} ArXiv papers")
    return papers[:target_count]


def extract_single_document(args: Tuple[Path, str, DoclingExtractor]) -> Dict[str, Any]:
    """
    Extract a single PDF to a staged JSON file (worker entrypoint).
    
    This worker expects `args` to be a 3-tuple: (pdf_path: Path, arxiv_id: str, _), where the third element is a placeholder
    for an extractor instance (it is ignored). The function creates a new DoclingExtractor for the process, extracts the
    document, augments the extraction result with `arxiv_id`, `pdf_path`, and `success=True`, and writes the result to
    STAGING_DIR/{arxiv_id}.json.
    
    Returns a dict summarizing the outcome:
    - On success: {'arxiv_id': arxiv_id, 'success': True, 'staged_path': str(path_to_staged_json)}
    - On failure: {'arxiv_id': arxiv_id, 'success': False, 'error': str(exception)}
    """
    pdf_path, arxiv_id, _ = args
    
    try:
        # Create new extractor instance for this worker
        extractor = DoclingExtractor(use_ocr=False, extract_tables=True, use_fallback=True)
        
        # Extract
        result = extractor.extract(str(pdf_path))
        
        # Add metadata
        result['arxiv_id'] = arxiv_id
        result['pdf_path'] = str(pdf_path)
        result['success'] = True
        
        # Save to staging
        staging_file = STAGING_DIR / f"{arxiv_id}.json"
        with open(staging_file, 'w') as f:
            json.dump(result, f)
        
        return {'arxiv_id': arxiv_id, 'success': True, 'staged_path': str(staging_file)}
        
    except Exception as e:
        logger.error(f"Failed to extract {arxiv_id}: {e}")
        return {'arxiv_id': arxiv_id, 'success': False, 'error': str(e)}


def phase1_extraction(papers: List[Tuple[Path, str]], num_workers: int = 8) -> Dict[str, Any]:
    """
    Run Phase 1 extraction: parallel Docling extraction of provided PDFs and stage per-document JSON files.
    
    Processes the given list of papers in a ProcessPoolExecutor, calling the worker extract_single_document for each item. This function:
    - Clears the staging directory before starting.
    - Submits one extraction task per paper and collects completed results (workers are given 60s per future.result call).
    - Records successes and failures (failed tasks are captured and returned rather than raised).
    - Clears GPU memory cache after extraction if CUDA is available.
    
    Parameters:
        papers (List[Tuple[Path, str]]): Iterable of (pdf_path, arxiv_id) tuples to extract.
        num_workers (int): Number of parallel worker processes to use.
    
    Returns:
        Dict[str, Any]: Summary of the extraction phase containing:
            - 'phase' (str): "extraction"
            - 'total_papers' (int): number of papers processed
            - 'successful' (int): count of successful extractions
            - 'failed' (int): count of failed extractions
            - 'phase_time' (float): elapsed time in seconds for the phase
            - 'papers_per_minute' (float): throughput computed over the phase_time
            - 'staged_files' (int): number of JSON files present in the staging directory after the run
    
    Side effects:
        - Deletes existing JSON files in STAGING_DIR before running.
        - Writes per-document JSON files to STAGING_DIR via worker processes.
        - Logs progress and errors to the configured logger.
        - May clear CUDA cache via torch.cuda.empty_cache() if CUDA is available.
    """
    logger.info("="*70)
    logger.info(f"PHASE 1: EXTRACTION - {len(papers)} papers with {num_workers} workers")
    logger.info("="*70)
    
    start_time = time.time()
    successful = []
    failed = []
    
    # Clear staging directory
    for old_file in STAGING_DIR.glob("*.json"):
        old_file.unlink()
    
    # Create dummy extractor for args (actual extractors created in workers)
    dummy_extractor = None
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(extract_single_document, (pdf_path, arxiv_id, dummy_extractor)): arxiv_id
            for pdf_path, arxiv_id in papers
        }
        
        # Process as completed
        completed_count = 0
        for future in as_completed(futures):
            arxiv_id = futures[future]
            try:
                result = future.result(timeout=60)
                completed_count += 1
                
                if result['success']:
                    successful.append(result)
                else:
                    failed.append(result)
                
                # Progress update
                if completed_count % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    eta = (len(papers) - completed_count) / rate if rate > 0 else 0
                    logger.info(
                        f"Extraction progress: {completed_count}/{len(papers)} "
                        f"({rate*60:.1f} papers/min, ETA: {timedelta(seconds=int(eta))})"
                    )
                    
            except Exception as e:
                logger.error(f"Worker failed for {arxiv_id}: {e}")
                failed.append({'arxiv_id': arxiv_id, 'success': False, 'error': str(e)})
    
    phase_time = time.time() - start_time
    
    # Clear GPU memory after extraction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    results = {
        'phase': 'extraction',
        'total_papers': len(papers),
        'successful': len(successful),
        'failed': len(failed),
        'phase_time': phase_time,
        'papers_per_minute': (len(papers) / phase_time) * 60,
        'staged_files': len(list(STAGING_DIR.glob("*.json")))
    }
    
    logger.info(f"Phase 1 complete: {results['successful']}/{results['total_papers']} successful")
    logger.info(f"Extraction rate: {results['papers_per_minute']:.1f} papers/minute")
    
    return results


def process_single_staged_file(args: Tuple[Path, int]) -> Dict[str, Any]:
    """
    Process a single staged extraction JSON and produce embeddings for its text.
    
    This worker reads a staged JSON file produced by the extraction phase, selects the text content
    (preference order: `full_text`, `text`, `markdown`), assigns a GPU based on the provided
    worker_id (falls back to CPU if CUDA is unavailable), instantiates a JinaV4Embedder on that
    device, and generates embeddings using late chunking.
    
    Parameters:
        args (tuple): (staged_file: Path, worker_id: int). `worker_id` is used to map the process to a GPU.
    
    Returns:
        dict: Result payload with keys:
            - 'arxiv_id' (str): identifier for the document (from JSON or filename stem).
            - 'success' (bool): True on success, False on failure.
            - On success:
                - 'num_chunks' (int): number of text chunks embedded.
                - 'chunks' (list): chunk metadata/embeddings as returned by the embedder.
                - 'gpu_used' (int or str): GPU index used or 'cpu'.
            - On failure:
                - 'error' (str): error message explaining the failure (e.g., 'No text content' or exception text).
    """
    staged_file, worker_id = args
    
    try:
        # Distribute workers across GPUs
        # Workers 0-3 -> GPU 0, Workers 4-7 -> GPU 1, etc.
        gpu_id = worker_id % torch.cuda.device_count() if torch.cuda.is_available() else 0
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        # Set CUDA device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Load staged extraction
        with open(staged_file, 'r') as f:
            extraction = json.load(f)
        
        arxiv_id = extraction.get('arxiv_id', staged_file.stem)
        text = extraction.get('full_text', '') or extraction.get('text', '') or extraction.get('markdown', '')
        
        if not text:
            return {
                'arxiv_id': arxiv_id,
                'success': False,
                'error': 'No text content'
            }
        
        # Create embedder for this worker on specific GPU
        embedder = JinaV4Embedder(
            device=device,
            use_fp16=True,
            chunk_size_tokens=256,
            chunk_overlap_tokens=50
        )
        
        # Generate embeddings with late chunking
        chunks_with_embeddings = embedder.embed_with_late_chunking(text)
        
        return {
            'arxiv_id': arxiv_id,
            'success': True,
            'num_chunks': len(chunks_with_embeddings),
            'chunks': chunks_with_embeddings,  # Store if needed
            'gpu_used': gpu_id if torch.cuda.is_available() else 'cpu'
        }
        
    except Exception as e:
        logger.error(f"Failed to embed {staged_file.name}: {e}")
        return {
            'arxiv_id': staged_file.stem,
            'success': False,
            'error': str(e)
        }


def phase2_embedding(num_workers: int = 8) -> Dict[str, Any]:
    """
    Run Phase 2: generate embeddings for all staged JSON documents using multiple workers (GPU-aware).
    
    Scans the staging directory for JSON files produced by Phase 1 and processes them in parallel batches to limit GPU memory pressure. Each staged file is handed to a worker that selects a device (GPU by worker id or CPU), loads the staged JSON, extracts the document text, and produces embeddings (with late chunking). Results are aggregated into counts of successful and failed files and total embedding chunks.
    
    Parameters:
        num_workers (int): Number of parallel worker processes to use; also used to round-robin assign GPU device ids (worker_id % num_workers). Increase to parallelize across more GPUs or CPU workers.
    
    Returns:
        dict: Summary of the embedding phase containing:
            - phase (str): "embedding"
            - total_files (int): number of staged JSON files discovered
            - successful (int): count of successfully embedded files
            - failed (int): count of files that failed embedding
            - total_chunks (int): total number of embedding chunks produced across successful files
            - avg_chunks_per_doc (float): average chunks per successful document (0 if none)
            - phase_time (float): elapsed time in seconds for the embedding phase
            - papers_per_minute (float): processing rate computed as total_files / phase_time * 60
    """
    logger.info("="*70)
    logger.info(f"PHASE 2: EMBEDDING - Processing staged JSONs with {num_workers} workers")
    logger.info("="*70)
    
    # Get all staged files
    staged_files = list(STAGING_DIR.glob("*.json"))
    logger.info(f"Found {len(staged_files)} staged files to process")
    
    if not staged_files:
        logger.error("No staged files found!")
        return {'phase': 'embedding', 'error': 'No staged files'}
    
    start_time = time.time()
    successful = []
    failed = []
    total_chunks = 0
    
    # Process in parallel with limited workers for GPU memory
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks in batches to control GPU memory
        batch_size = num_workers * 2  # Process 2x workers at a time
        
        for i in range(0, len(staged_files), batch_size):
            batch = staged_files[i:i+batch_size]
            
            # Assign worker IDs for GPU distribution
            futures = {
                executor.submit(process_single_staged_file, (staged_file, idx % num_workers)): staged_file
                for idx, staged_file in enumerate(batch, start=i)
            }
            
            # Process batch
            for future in as_completed(futures):
                staged_file = futures[future]
                try:
                    result = future.result(timeout=120)
                    
                    if result['success']:
                        successful.append(result)
                        total_chunks += result.get('num_chunks', 0)
                    else:
                        failed.append(result)
                    
                except Exception as e:
                    logger.error(f"Worker failed for {staged_file.name}: {e}")
                    failed.append({
                        'arxiv_id': staged_file.stem,
                        'success': False,
                        'error': str(e)
                    })
            
            # Progress update
            processed = len(successful) + len(failed)
            if processed % 50 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(staged_files) - processed) / rate if rate > 0 else 0
                logger.info(
                    f"Embedding progress: {processed}/{len(staged_files)} "
                    f"({rate*60:.1f} papers/min, ETA: {timedelta(seconds=int(eta))})"
                )
                
                # Check GPU memory and usage
                if torch.cuda.is_available():
                    gpu_usage = {}
                    for result in successful[-50:]:  # Check last 50 results
                        gpu = result.get('gpu_used', 'cpu')
                        if gpu != 'cpu':
                            gpu_usage[gpu] = gpu_usage.get(gpu, 0) + 1
                    
                    for i in range(torch.cuda.device_count()):
                        mem_used = torch.cuda.memory_allocated(i) / 1024**3
                        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        usage_count = gpu_usage.get(i, 0)
                        logger.info(f"  GPU {i}: {mem_used:.1f}/{mem_total:.1f} GB (used by {usage_count} recent tasks)")
    
    phase_time = time.time() - start_time
    
    results = {
        'phase': 'embedding',
        'total_files': len(staged_files),
        'successful': len(successful),
        'failed': len(failed),
        'total_chunks': total_chunks,
        'avg_chunks_per_doc': total_chunks / len(successful) if successful else 0,
        'phase_time': phase_time,
        'papers_per_minute': (len(staged_files) / phase_time) * 60 if phase_time > 0 else 0
    }
    
    logger.info(f"Phase 2 complete: {results['successful']}/{results['total_files']} successful")
    logger.info(f"Embedding rate: {results['papers_per_minute']:.1f} papers/minute")
    logger.info(f"Total chunks generated: {results['total_chunks']}")
    
    return results


def run_phased_test(num_docs: int = 2000, extraction_workers: int = 8, embedding_workers: int = 8):
    """
    Run the two-phase (extraction -> embedding) overnight ArXiv processing test and return aggregated results.
    
    This orchestrator runs:
    - Phase 1: collects up to `num_docs` ArXiv PDFs and extracts each into a staged JSON using multiple workers.
    - Phase 2: reads staged JSONs and generates embeddings with GPU-aware worker distribution.
    
    Side effects:
    - Persists intermediate and final results to a timestamped JSON file in the configured log directory.
    - Emits progress and summary information to the configured logger.
    - Writes per-document staged JSONs to the shared staging directory.
    
    Parameters:
        num_docs (int): Target number of documents to process (default: 2000).
        extraction_workers (int): Number of worker processes for Phase 1 extraction (default: 8).
        embedding_workers (int): Number of worker processes for Phase 2 embedding (default: 8).
    
    Returns:
        dict: A nested results dictionary containing:
            - test_config: configuration used for the run
            - corpus_info: metadata about collected papers
            - phases: per-phase result objects ('extraction' and 'embedding')
            - summary: overall timing, rates, chunk counts, and completion timestamp
    """
    logger.info("="*70)
    logger.info(f"PHASE-SEPARATED ARXIV TEST - {num_docs} Papers")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("="*70)
    
    results_file = log_dir / f"overnight_phased_results_{timestamp}.json"
    
    # Collect papers
    logger.info("\nCollecting ArXiv papers...")
    papers = collect_arxiv_papers(num_docs)
    
    all_results = {
        'test_config': {
            'num_docs': num_docs,
            'extraction_workers': extraction_workers,
            'embedding_workers': embedding_workers,
            'staging_dir': str(STAGING_DIR)
        },
        'corpus_info': {
            'total_papers': len(papers),
            'sample_papers': [str(p) for p, _ in papers[:5]]
        },
        'phases': {}
    }
    
    # Phase 1: Extraction
    extraction_results = phase1_extraction(papers, extraction_workers)
    all_results['phases']['extraction'] = extraction_results
    
    # Save intermediate results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Phase 2: Embedding
    embedding_results = phase2_embedding(embedding_workers)
    all_results['phases']['embedding'] = embedding_results
    
    # Calculate overall metrics
    total_time = extraction_results['phase_time'] + embedding_results['phase_time']
    overall_rate = num_docs / total_time * 60
    
    all_results['summary'] = {
        'total_papers_processed': num_docs,
        'total_time_seconds': total_time,
        'extraction_time': extraction_results['phase_time'],
        'embedding_time': embedding_results['phase_time'],
        'overall_papers_per_minute': overall_rate,
        'extraction_rate': extraction_results['papers_per_minute'],
        'embedding_rate': embedding_results['papers_per_minute'],
        'total_chunks': embedding_results.get('total_chunks', 0),
        'avg_chunks_per_doc': embedding_results.get('avg_chunks_per_doc', 0),
        'completed_at': datetime.now().isoformat()
    }
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Log summary
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    logger.info(f"Overall Rate: {overall_rate:.1f} papers/minute")
    logger.info(f"Extraction: {extraction_results['papers_per_minute']:.1f} papers/minute")
    logger.info(f"Embedding: {embedding_results['papers_per_minute']:.1f} papers/minute")
    logger.info(f"Total Time: {timedelta(seconds=int(total_time))}")
    
    # Check against target
    target_rate = 11.3  # papers/minute from PRD
    if overall_rate >= target_rate:
        logger.info(f"✅ PERFORMANCE TARGET MET: {overall_rate:.1f} >= {target_rate} papers/minute")
    else:
        logger.warning(f"⚠️ Below target: {overall_rate:.1f} < {target_rate} papers/minute")
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Log file: {log_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run phase-separated overnight test")
    parser.add_argument(
        "--docs",
        type=int,
        default=2000,
        help="Number of documents to test (default: 2000)"
    )
    parser.add_argument(
        "--extraction-workers",
        type=int,
        default=8,
        help="Number of extraction workers (default: 8)"
    )
    parser.add_argument(
        "--embedding-workers", 
        type=int,
        default=8,
        help="Number of embedding workers (default: 8)"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_phased_test(
            args.docs,
            args.extraction_workers,
            args.embedding_workers
        )
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Results: {results['summary']['overall_papers_per_minute']:.1f} papers/minute")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        sys.exit(1)