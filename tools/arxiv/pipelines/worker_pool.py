"""
Worker pool for parallel ACID-compliant processing.

Each worker gets its own ArangoDB connection and processes papers independently.
The ACID guarantees come from the lock-based coordination and transactions.
"""

import os
import time
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict

from .arango_acid_processor import ArangoACIDProcessor, ProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Results from processing a batch of papers"""
    total_papers: int
    successful: int
    failed: int
    total_time: float
    papers_per_minute: float
    results: List[ProcessingResult]
    lock_conflicts: int = 0
    timeouts: int = 0


class ArangoWorkerPool:
    """
    Manages pool of workers for parallel processing.
    
    Key features:
    - Each worker gets its own ArangoDB connection
    - Lock-based coordination prevents conflicts
    - Automatic retry on lock conflicts
    - Progress tracking and reporting
    """
    
    def __init__(self, num_workers: int = 4, config: Dict[str, Any] = None):
        """
        Initialize worker pool.
        
        Args:
            num_workers: Number of parallel workers (research scale: 4-8)
            config: Configuration for processors
        """
        self.num_workers = min(num_workers, mp.cpu_count())  # Don't exceed CPU cores
        self.config = config or {}
        logger.info(f"Initialized worker pool with {self.num_workers} workers")
    
    def process_batch(
        self,
        paper_paths: List[Tuple[str, str]],
        timeout_per_paper: int = 300,
        retry_on_lock: bool = True
    ) -> BatchResult:
        """
        Process batch of papers with worker pool.
        
        Args:
            paper_paths: List of (paper_id, pdf_path) tuples
            timeout_per_paper: Timeout in seconds for each paper
            retry_on_lock: Whether to retry papers that hit lock conflicts
            
        Returns:
            BatchResult with processing statistics
        """
        start_time = time.time()
        results = []
        lock_conflicts = 0
        timeouts = 0
        
        # Track papers that need retry due to lock conflicts
        retry_queue = []
        
        logger.info(f"Processing batch of {len(paper_paths)} papers with {self.num_workers} workers")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all papers for processing
            future_to_paper = {}
            for paper_id, pdf_path in paper_paths:
                future = executor.submit(
                    process_single_paper,
                    paper_id,
                    pdf_path,
                    self.config
                )
                future_to_paper[future] = (paper_id, pdf_path)
            
            # Collect results as they complete
            for future in as_completed(future_to_paper):
                paper_id, pdf_path = future_to_paper[future]
                
                try:
                    result = future.result(timeout=timeout_per_paper)
                    
                    # Check for lock conflicts more robustly - check error type or error code
                    error_msg = str(result.error).lower() if result.error else ""
                    is_lock_conflict = (
                        "already locked" in error_msg or 
                        "lock conflict" in error_msg or
                        "duplicate key" in error_msg or
                        (hasattr(result, 'error_code') and result.error_code == 'LOCK_CONFLICT')
                    )
                    
                    if not result.success and is_lock_conflict:
                        lock_conflicts += 1
                        if retry_on_lock:
                            retry_queue.append((paper_id, pdf_path))
                            logger.info(f"Paper {paper_id} hit lock conflict, queued for retry")
                        else:
                            results.append(result)
                    else:
                        results.append(result)
                        if result.success:
                            logger.info(
                                f"Processed {paper_id}: "
                                f"{result.num_chunks} chunks, "
                                f"{result.num_equations} equations, "
                                f"{result.num_tables} tables, "
                                f"{result.num_images} images "
                                f"in {result.processing_time:.2f}s"
                            )
                        else:
                            logger.error(f"Processing failed for paper {paper_id}: {result.error}")
                    
                except TimeoutError:
                    timeouts += 1
                    logger.error(f"Processing timeout for paper {paper_id}: exceeded {timeout_per_paper}s limit")
                    results.append(ProcessingResult(
                        paper_id=paper_id,
                        success=False,
                        processing_time=timeout_per_paper,
                        error=f"Timeout after {timeout_per_paper}s"
                    ))
                except Exception as e:
                    logger.error(f"Unexpected error while processing paper {paper_id}: {type(e).__name__}: {e}")
                    results.append(ProcessingResult(
                        paper_id=paper_id,
                        success=False,
                        processing_time=0,
                        error=str(e)
                    ))
        
        # Retry papers that hit lock conflicts (sequential to avoid conflicts)
        if retry_queue:
            logger.info(f"Retrying {len(retry_queue)} papers that hit lock conflicts")
            for paper_id, pdf_path in retry_queue:
                try:
                    result = process_single_paper(paper_id, pdf_path, self.config)
                    results.append(result)
                    if result.success:
                        logger.info(f"Retry successful for {paper_id}")
                    else:
                        logger.error(f"Retry attempt failed for paper {paper_id}: {result.error}")
                except Exception as e:
                    logger.error(f"Retry exception for paper {paper_id}: {type(e).__name__}: {e}")
                    results.append(ProcessingResult(
                        paper_id=paper_id,
                        success=False,
                        processing_time=0,
                        error=f"Retry failed: {str(e)}"
                    ))
        
        # Calculate statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        papers_per_minute = (successful / total_time) * 60 if total_time > 0 else 0
        
        batch_result = BatchResult(
            total_papers=len(paper_paths),
            successful=successful,
            failed=failed,
            total_time=total_time,
            papers_per_minute=papers_per_minute,
            results=results,
            lock_conflicts=lock_conflicts,
            timeouts=timeouts
        )
        
        # Log summary
        logger.info(
            f"Batch complete: {successful}/{len(paper_paths)} successful, "
            f"{failed} failed, {lock_conflicts} lock conflicts, "
            f"{timeouts} timeouts, "
            f"{papers_per_minute:.2f} papers/minute"
        )
        
        return batch_result
    
    def process_batch_with_progress(
        self,
        paper_paths: List[Tuple[str, str]],
        callback=None,
        **kwargs
    ) -> BatchResult:
        """
        Process batch with progress callback.
        
        Args:
            paper_paths: List of (paper_id, pdf_path) tuples
            callback: Function called with (completed, total) after each paper
            **kwargs: Additional arguments for process_batch
            
        Returns:
            BatchResult with processing statistics
        """
        if not callback:
            return self.process_batch(paper_paths, **kwargs)
        
        start_time = time.time()
        results = []
        completed = 0
        total = len(paper_paths)
        
        # Process with progress updates
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_paper = {}
            for paper_id, pdf_path in paper_paths:
                future = executor.submit(
                    process_single_paper,
                    paper_id,
                    pdf_path,
                    self.config
                )
                future_to_paper[future] = (paper_id, pdf_path)
            
            for future in as_completed(future_to_paper):
                paper_id, pdf_path = future_to_paper[future]
                
                try:
                    result = future.result(timeout=kwargs.get('timeout_per_paper', 300))
                    results.append(result)
                except Exception as e:
                    results.append(ProcessingResult(
                        paper_id=paper_id,
                        success=False,
                        processing_time=0,
                        error=str(e)
                    ))
                
                completed += 1
                callback(completed, total)
        
        # Calculate statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        papers_per_minute = (successful / total_time) * 60 if total_time > 0 else 0
        
        return BatchResult(
            total_papers=total,
            successful=successful,
            failed=failed,
            total_time=total_time,
            papers_per_minute=papers_per_minute,
            results=results
        )


def process_single_paper(paper_id: str, pdf_path: str, config: Dict[str, Any]) -> ProcessingResult:
    """
    Worker function for parallel processing.
    
    This function runs in a separate process with its own ArangoDB connection.
    
    Args:
        paper_id: ArXiv paper ID
        pdf_path: Path to PDF file
        config: Configuration dictionary
        
    Returns:
        ProcessingResult with success status and statistics
    """
    # Each worker creates its own processor (and thus ArangoDB connection)
    processor = ArangoACIDProcessor(config)
    return processor.process_paper(paper_id, pdf_path)


def estimate_processing_time(num_papers: int, num_workers: int = 4) -> Dict[str, float]:
    """
    Estimate processing time for a given number of papers.
    
    Args:
        num_papers: Number of papers to process
        num_workers: Number of parallel workers
        
    Returns:
        Dictionary with time estimates in different units
    """
    # Conservative estimates based on research-scale processing
    papers_per_minute_per_worker = 0.5  # Conservative: 2 minutes per paper
    
    effective_rate = papers_per_minute_per_worker * num_workers
    minutes = num_papers / effective_rate
    
    return {
        'minutes': minutes,
        'hours': minutes / 60,
        'days': minutes / (60 * 24),
        'papers_per_hour': effective_rate * 60
    }