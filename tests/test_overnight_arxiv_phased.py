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
from core.framework.extractors.docling_extractor import DoclingExtractor
from core.framework.embedders import JinaV4Embedder

# Staging directory for inter-phase communication
STAGING_DIR = Path("/dev/shm/overnight_staging")
STAGING_DIR.mkdir(exist_ok=True)

# ArXiv PDF repository
ARXIV_PDF_BASE = Path("/bulk-store/arxiv-data/pdf")


def collect_arxiv_papers(target_count: int = 2000, start_year: str = "1501") -> List[Tuple[Path, str]]:
    """Collect paths to real ArXiv PDFs from the local repository."""
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
    """Extract a single document (worker function)."""
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
    Phase 1: Extract all documents with Docling and stage to JSON.
    Uses multiple workers for parallel processing.
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
    """Process a single staged JSON file (worker function)."""
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
    Phase 2: Process all staged JSONs with Jina embeddings.
    Uses multiple GPU workers for parallel processing.
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
    """Run the phase-separated overnight test."""
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