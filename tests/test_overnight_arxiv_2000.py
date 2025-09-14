#!/usr/bin/env python3
"""
Overnight Test - 2000 Real ArXiv PDFs
======================================

Test the document processing architecture with actual ArXiv PDFs from local repository.
This tests the real pipeline: PDF extraction with Docling + Jina embeddings.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure detailed logging
log_dir = Path("tests/logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"overnight_arxiv_test_{timestamp}.log"

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

# Import our processors
from core.workflows.workflow_pdf import (
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult
)
from tools.arxiv.arxiv_manager import ArXivManager, ArXivValidator

# ArXiv PDF repository paths
ARXIV_PDF_BASE = Path("/bulk-store/arxiv-data/pdf")
ARXIV_LATEX_BASE = Path("/bulk-store/arxiv-data/src")
ARXIV_METADATA = Path("/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json")


def _normalize_yymm(yymm: str) -> tuple[int, int]:
    """Normalize YYMM to (year, month) tuple for correct comparison."""
    if len(yymm) != 4 or not yymm.isdigit():
        raise ValueError(f"Invalid YYMM: {yymm}")
    yy, mm = int(yymm[:2]), int(yymm[2:])
    if not 1 <= mm <= 12:
        raise ValueError(f"Invalid month in YYMM: {yymm}")
    # Handle century: 90-99 = 1990s, 00-89 = 2000s
    year = 1900 + yy if yy >= 90 else 2000 + yy
    return (year, mm)


def collect_arxiv_papers(target_count: int = 2000, start_year: str = "1501", seed: int | None = None) -> List[Tuple[Path, str]]:
    """
    Collect up to `target_count` ArXiv PDF file paths from the local ARXIV_PDF_BASE directory.
    
    Searches year-month subdirectories (numeric names) under ARXIV_PDF_BASE starting from `start_year` (YYMM). In each directory it samples PDFs (without replacement) until `target_count` entries are gathered. Each returned tuple contains the Path to the PDF and the arXiv identifier derived from the file stem (e.g., "1501.00001").
    
    Parameters:
        target_count (int): Maximum number of papers to collect.
        start_year (str): Inclusive YYMM directory name to start searching from (e.g., "1501" = January 2015).
        seed (int | None): Optional RNG seed for reproducible sampling.
    
    Returns:
        List[Tuple[Path, str]]: List of (pdf_path, arxiv_id) tuples, length <= target_count.
    
    Raises:
        ValueError: If no year-month directories are found at or after `start_year`.
    """
    logger.info(f"Collecting {target_count} ArXiv PDFs from {ARXIV_PDF_BASE}")
    
    papers = []
    rng = random.Random(seed) if seed is not None else random
    
    # Iterate through year-month directories with proper century handling
    yymm_dirs = sorted(
        [d for d in ARXIV_PDF_BASE.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: _normalize_yymm(d.name)
    )
    
    # Filter to start from specified year (normalize century)
    start_cutoff = _normalize_yymm(start_year)
    yymm_dirs = [d for d in yymm_dirs if _normalize_yymm(d.name) >= start_cutoff]
    
    if not yymm_dirs:
        raise ValueError(f"No ArXiv directories found from {start_year} onwards in {ARXIV_PDF_BASE}")
    
    logger.info(f"Found {len(yymm_dirs)} year-month directories from {start_year} onwards")
    
    for yymm_dir in yymm_dirs:
        if len(papers) >= target_count:
            break
            
        # Get PDFs from this directory
        pdf_files = list(yymm_dir.glob("*.pdf"))
        
        if pdf_files:
            # Sample PDFs from this directory
            sample_size = min(len(pdf_files), target_count - len(papers))
            sampled_pdfs = rng.sample(pdf_files, sample_size) if sample_size < len(pdf_files) else pdf_files
            
            for pdf_path in sampled_pdfs:
                # Extract ArXiv ID from filename
                arxiv_id = pdf_path.stem  # e.g., "1501.00001"
                papers.append((pdf_path, arxiv_id))
                
                if len(papers) >= target_count:
                    break
            
            logger.info(f"Collected {len(sampled_pdfs)} papers from {yymm_dir.name}, total: {len(papers)}")
    
    logger.info(f"Collected {len(papers)} ArXiv papers")
    return papers[:target_count]


def check_latex_availability(papers: List[Tuple[Path, str]]) -> Dict[str, Any]:
    """
    Return counts and percentage of papers that have LaTeX source directories with `.tex` files.
    
    Parameters:
        papers (List[Tuple[Path, str]]): List of tuples (pdf_path, arxiv_id) where `arxiv_id` begins with YYMM.
    
    Returns:
        dict: {
            'total': int total number of papers,
            'with_latex': int number of papers whose LaTeX directory exists and contains at least one `.tex` file,
            'latex_percentage': float percentage (0-100) of papers with LaTeX sources (0 if papers list is empty)
        }
    """
    latex_count = 0
    
    for pdf_path, arxiv_id in papers:
        yymm = arxiv_id[:4]  # First 4 chars are YYMM
        latex_dir = ARXIV_LATEX_BASE / yymm / arxiv_id
        
        if latex_dir.exists():
            # Check for .tex files
            tex_files = list(latex_dir.glob("*.tex"))
            if tex_files:
                latex_count += 1
    
    return {
        'total': len(papers),
        'with_latex': latex_count,
        'latex_percentage': 100 * latex_count / len(papers) if papers else 0
    }


def run_arxiv_overnight_test(num_docs: int = 2000, start_year: str = "1501"):
    """
    Run an end-to-end overnight performance and correctness test on real ArXiv PDFs.
    
    This function:
    - Collects up to `num_docs` ArXiv PDF paths from the local ArXiv store.
    - Optionally detects available LaTeX sources for each paper.
    - Executes the document processing pipeline under two configurations (traditional chunking and late chunking), processing papers in batches.
    - Aggregates per-document and per-run metrics (success/failure, chunk counts, processing/extraction/embedding times, throughput).
    - Persists intermediate and final results to a timestamped JSON results file and emits detailed progress logs.
    
    Parameters:
        num_docs (int): Maximum number of ArXiv papers to include in the test (default 2000).
    
    Returns:
        dict: Aggregated results containing:
          - 'corpus_info': metadata about the input corpus and LaTeX availability,
          - 'test_runs': list of per-configuration summaries (metrics, failures, timing),
          - 'summary': final summary with best configuration, best throughput, timestamps, and file paths.
    
    Side effects:
    - Writes intermediate and final JSON results files and extensive logs.
    - Reads PDF and optional LaTeX files from the configured ARXIV_PDF_BASE and ARXIV_LATEX_BASE paths.
    - May perform GPU memory queries if PyTorch is available.
    
    Note:
    - Exceptions raised during paper collection or other unexpected failures will propagate after logging; per-batch processing errors are captured and recorded without aborting the entire test run.
    """
    logger.info("="*70)
    logger.info(f"OVERNIGHT ARXIV TEST - {num_docs} Real PDFs")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("="*70)
    
    results_file = log_dir / f"overnight_arxiv_results_{timestamp}.json"
    
    # Step 1: Collect ArXiv papers
    logger.info("\n" + "="*50)
    logger.info("STEP 1: Collecting ArXiv Papers")
    logger.info("="*50)
    
    try:
        papers = collect_arxiv_papers(num_docs, start_year=start_year)
    except Exception as e:
        logger.error(f"Failed to collect papers: {e}")
        raise
    
    # Check LaTeX availability
    latex_stats = check_latex_availability(papers)
    logger.info(f"LaTeX sources: {latex_stats['with_latex']}/{latex_stats['total']} ({latex_stats['latex_percentage']:.1f}%)")
    
    # Step 2: Test different configurations
    test_configs = [
        {
            'name': 'Traditional Chunking - Optimized',
            'strategy': 'traditional',
            'chunk_size': 256,
            'overlap': 50,
            'batch_size': 10
        },
        {
            'name': 'Late Chunking - Production Config',
            'strategy': 'late',
            'chunk_size': 256,
            'overlap': 50,
            'batch_size': 8
        }
    ]
    
    all_results = {
        'corpus_info': {
            'total_papers': len(papers),
            'latex_stats': latex_stats,
            'pdf_base': str(ARXIV_PDF_BASE),
            'sample_papers': [str(p) for p, _ in papers[:5]]  # First 5 as sample
        },
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
            batch_size=test_config['batch_size'],
            use_gpu=True,
            use_fp16=True,  # Use FP16 for memory efficiency
            use_ocr=False,  # Disable OCR for speed
            extract_tables=True,  # Keep table extraction
            extract_equations=True  # Keep equation extraction
        )
        
        # Option 1: Use generic DocumentProcessor directly
        processor = DocumentProcessor(config)
        
        # Option 2: Use ArXivManager (uncomment to test ArXiv-specific processing)
        # arxiv_manager = ArXivManager(processing_config=config)
        
        # Process papers with detailed tracking
        test_start = time.time()
        results = []
        failed_docs = []
        processing_times = []
        chunk_counts = []
        extraction_times = []
        embedding_times = []
        
        batch_size = config.batch_size
        
        for batch_start in range(0, len(papers), batch_size):
            batch_end = min(batch_start + batch_size, len(papers))
            batch_papers = papers[batch_start:batch_end]
            
            # Prepare batch
            batch_paths = []
            batch_ids = []
            
            for pdf_path, arxiv_id in batch_papers:
                # Check for LaTeX
                yymm = arxiv_id[:4]
                latex_dir = ARXIV_LATEX_BASE / yymm / arxiv_id
                latex_path = None
                
                if latex_dir.exists():
                    tex_files = list(latex_dir.glob("*.tex"))
                    if tex_files:
                        # Prefer main.tex or paper.tex
                        for preferred in ['main.tex', 'paper.tex', f'{arxiv_id}.tex']:
                            preferred_path = latex_dir / preferred
                            if preferred_path.exists():
                                latex_path = preferred_path
                                break
                        if not latex_path and tex_files:
                            latex_path = tex_files[0]
                
                batch_paths.append((pdf_path, latex_path))
                batch_ids.append(arxiv_id)
            
            # Process batch
            try:
                batch_start_time = time.time()
                batch_results = processor.process_batch(batch_paths, batch_ids)
                batch_time = time.time() - batch_start_time
                
                # Analyze batch results
                for result, (pdf_path, arxiv_id) in zip(batch_results, batch_papers):
                    results.append(result)
                    
                    if not result.success:
                        failed_docs.append({
                            'arxiv_id': arxiv_id,
                            'pdf': str(pdf_path),
                            'errors': result.errors
                        })
                        logger.warning(f"Failed: {arxiv_id} - {result.errors}")
                    else:
                        if result.chunks:
                            chunk_counts.append(len(result.chunks))
                        processing_times.append(result.total_processing_time)
                        extraction_times.append(result.extraction_time)
                        embedding_times.append(result.embedding_time)
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                for _, arxiv_id in batch_papers:
                    failed_docs.append({
                        'arxiv_id': arxiv_id,
                        'errors': [str(e)]
                    })
            
            # Progress update
            if batch_end % 50 == 0 or batch_end == len(papers):
                elapsed = time.time() - test_start
                rate = batch_end / elapsed if elapsed > 0 else 0
                papers_per_minute = rate * 60
                eta = (len(papers) - batch_end) / rate if rate > 0 else 0
                
                logger.info(
                    f"Progress: {batch_end}/{len(papers)} "
                    f"({papers_per_minute:.1f} papers/min, "
                    f"ETA: {timedelta(seconds=int(eta))})"
                )
                
                # Log memory usage if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            mem_used = torch.cuda.memory_allocated(i) / 1024**3
                            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                            logger.info(f"  GPU {i}: {mem_used:.1f}/{mem_total:.1f} GB")
                except:
                    pass
        
        test_time = time.time() - test_start
        
        # Calculate statistics
        successful = len([r for r in results if r.success])
        total_chunks = sum(chunk_counts)
        avg_chunks = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0
        avg_proc_time = sum(processing_times) / len(processing_times) if processing_times else 0
        avg_extraction = sum(extraction_times) / len(extraction_times) if extraction_times else 0
        avg_embedding = sum(embedding_times) / len(embedding_times) if embedding_times else 0
        
        test_summary = {
            'config': test_config,
            'total_papers': len(papers),
            'successful': successful,
            'failed': len(failed_docs),
            'success_rate': 100 * successful / len(papers) if papers else 0,
            'total_chunks': total_chunks,
            'avg_chunks_per_doc': avg_chunks,
            'total_time_seconds': test_time,
            'papers_per_minute': (len(papers) / test_time) * 60,
            'avg_processing_time': avg_proc_time,
            'avg_extraction_time': avg_extraction,
            'avg_embedding_time': avg_embedding,
            'failed_samples': failed_docs[:10]  # First 10 failures
        }
        
        all_results['test_runs'].append(test_summary)
        
        # Log summary
        logger.info("\n" + "-"*40)
        logger.info(f"Results for {test_config['name']}:")
        logger.info(f"  Success Rate: {successful}/{len(papers)} ({test_summary['success_rate']:.1f}%)")
        logger.info(f"  Processing Rate: {test_summary['papers_per_minute']:.1f} papers/minute")
        logger.info(f"  Total Chunks: {total_chunks:,}")
        logger.info(f"  Avg Chunks/Paper: {avg_chunks:.1f}")
        logger.info(f"  Total Time: {timedelta(seconds=int(test_time))}")
        logger.info(f"  Avg Times: Extract={avg_extraction:.2f}s, Embed={avg_embedding:.2f}s")
        
        if failed_docs:
            logger.warning(f"  Failed Papers: {len(failed_docs)}")
            for fail in failed_docs[:3]:
                logger.warning(f"    - {fail['arxiv_id']}: {fail['errors'][:100]}")
        
        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"  Results saved to: {results_file}")
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    
    best_rate = max(run['papers_per_minute'] for run in all_results['test_runs'])
    best_config = next(run['config']['name'] for run in all_results['test_runs'] 
                      if run['papers_per_minute'] == best_rate)
    
    all_results['summary'] = {
        'total_papers_processed': num_docs,
        'best_configuration': best_config,
        'best_rate_papers_per_minute': best_rate,
        'completed_at': datetime.now().isoformat(),
        'log_file': str(log_file),
        'results_file': str(results_file)
    }
    
    logger.info(f"Best Configuration: {best_config}")
    logger.info(f"Best Rate: {best_rate:.1f} papers/minute")
    
    # Compare to target performance
    target_rate = 11.3  # papers/minute from PRD
    if best_rate >= target_rate:
        logger.info(f"✅ PERFORMANCE TARGET MET: {best_rate:.1f} >= {target_rate} papers/minute")
    else:
        logger.warning(f"⚠️ Below target: {best_rate:.1f} < {target_rate} papers/minute")
    
    logger.info(f"\nLog File: {log_file}")
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
    
    parser = argparse.ArgumentParser(description="Run overnight test with real ArXiv PDFs")
    parser.add_argument(
        "--docs", 
        type=int, 
        default=2000,
        help="Number of ArXiv papers to test (default: 2000)"
    )
    parser.add_argument(
        "--start-year",
        type=str,
        default="1501",
        help="Starting YYMM for paper selection (default: 1501 = Jan 2015)"
    )
    
    args = parser.parse_args()
    
    try:
        # Check that ArXiv data exists
        if not ARXIV_PDF_BASE.exists():
            logger.error(f"ArXiv PDF directory not found: {ARXIV_PDF_BASE}")
            logger.error("Please ensure ArXiv data is mounted at /bulk-store/arxiv-data/")
            sys.exit(1)
        
        results = run_arxiv_overnight_test(args.docs, start_year=args.start_year)
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Results: {results['summary']['results_file']}")
        print(f"📝 Logs: {results['summary']['log_file']}")
        print(f"🚀 Best rate: {results['summary']['best_rate_papers_per_minute']:.1f} papers/minute")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        sys.exit(1)