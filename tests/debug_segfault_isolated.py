#!/usr/bin/env python3
"""
Debug Segmentation Fault - Isolated Testing
===========================================

Test each document individually with timeout and signal handling
to identify exactly which document causes the segfault.
"""

import sys
import logging
import signal
import traceback
from pathlib import Path
from multiprocessing import Process, Queue
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.extractors import DoclingExtractor

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_with_timeout(pdf_path: str, result_queue: Queue, timeout: int = 30):
    """Extract a document with timeout protection."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Extraction timed out after {timeout} seconds")
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        extractor = DoclingExtractor(
            use_ocr=False,
            extract_tables=False,  # Disable tables to reduce complexity
            use_fallback=True
        )
        
        result = extractor.extract(pdf_path)
        
        # Cancel timeout
        signal.alarm(0)
        
        if result:
            full_text = result.get('full_text', '') or result.get('text', '') or result.get('markdown', '')
            result_queue.put({
                'success': True,
                'text_length': len(full_text) if full_text else 0
            })
        else:
            result_queue.put({'success': False, 'error': 'No result returned'})
            
    except TimeoutError as e:
        signal.alarm(0)
        result_queue.put({'success': False, 'error': str(e)})
    except Exception as e:
        signal.alarm(0)
        result_queue.put({'success': False, 'error': str(e)})


def test_document_isolated(arxiv_id: str, pdf_path: str, timeout: int = 30):
    """Test a single document in a separate process."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {arxiv_id}")
    logger.info(f"PDF: {pdf_path}")
    
    # Check file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error(f"✗ PDF not found")
        return False
    
    size_mb = pdf_file.stat().st_size / (1024 * 1024)
    logger.info(f"PDF size: {size_mb:.2f} MB")
    
    # Run extraction in separate process
    result_queue = Queue()
    process = Process(
        target=extract_with_timeout,
        args=(pdf_path, result_queue, timeout)
    )
    
    process.start()
    process.join(timeout=timeout + 5)  # Give extra time for cleanup
    
    if process.is_alive():
        logger.error(f"✗ Process still alive after timeout, terminating...")
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            logger.error(f"✗ Process won't terminate, killing...")
            process.kill()
            process.join()
        return False
    
    # Check exit code
    if process.exitcode != 0:
        if process.exitcode == -11:  # SIGSEGV
            logger.error(f"✗ SEGMENTATION FAULT detected!")
            return False
        elif process.exitcode == -9:  # SIGKILL
            logger.error(f"✗ Process was killed")
            return False
        else:
            logger.error(f"✗ Process exited with code {process.exitcode}")
            return False
    
    # Get result
    if not result_queue.empty():
        result = result_queue.get()
        if result['success']:
            logger.info(f"✓ Extraction succeeded")
            logger.info(f"  Text length: {result['text_length']} characters")
            return True
        else:
            logger.error(f"✗ Extraction failed: {result['error']}")
            return False
    else:
        logger.error(f"✗ No result returned from process")
        return False


def main():
    # Test documents that previously failed
    test_documents = [
        ('1608.02280', '/bulk-store/arxiv-data/pdf/1608/1608.02280.pdf'),
        ('1608.02281', '/bulk-store/arxiv-data/pdf/1608/1608.02281.pdf'),
        ('1608.02282', '/bulk-store/arxiv-data/pdf/1608/1608.02282.pdf'),
        ('1608.02285', '/bulk-store/arxiv-data/pdf/1608/1608.02285.pdf'),  # Known segfault
        ('1608.02295', '/bulk-store/arxiv-data/pdf/1608/1608.02295.pdf'),
        ('1608.02296', '/bulk-store/arxiv-data/pdf/1608/1608.02296.pdf'),
        ('1608.02302', '/bulk-store/arxiv-data/pdf/1608/1608.02302.pdf'),
        ('1608.02303', '/bulk-store/arxiv-data/pdf/1608/1608.02303.pdf'),
        ('1608.02337', '/bulk-store/arxiv-data/pdf/1608/1608.02337.pdf'),
        ('1608.02383', '/bulk-store/arxiv-data/pdf/1608/1608.02383.pdf'),
    ]
    
    results = {
        'success': [],
        'failed': [],
        'segfault': []
    }
    
    logger.info(f"Testing {len(test_documents)} documents individually")
    logger.info("Each document runs in a separate process with timeout protection")
    
    for arxiv_id, pdf_path in test_documents:
        try:
            success = test_document_isolated(arxiv_id, pdf_path, timeout=30)
            if success:
                results['success'].append(arxiv_id)
            else:
                # Check if it was a segfault by trying to identify from logs
                results['failed'].append(arxiv_id)
        except Exception as e:
            logger.error(f"Unexpected error testing {arxiv_id}: {e}")
            results['failed'].append(arxiv_id)
        
        # Small delay between tests
        time.sleep(1)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"Success: {len(results['success'])} documents")
    if results['success']:
        logger.info(f"  {results['success']}")
    
    logger.info(f"Failed: {len(results['failed'])} documents")
    if results['failed']:
        logger.info(f"  {results['failed']}")
    
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()