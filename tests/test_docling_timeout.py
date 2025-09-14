#!/usr/bin/env python3
"""
Test Docling with Timeout
=========================

Simple test to identify which documents cause Docling to hang or crash.
"""

import sys
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_document(pdf_path: str) -> dict:
    """Extract a single document."""
    from core.extractors import DoclingExtractor
    
    try:
        extractor = DoclingExtractor(
            use_ocr=False,
            extract_tables=False,  # Disable to reduce complexity
            use_fallback=False  # Don't use fallback to test pure Docling
        )
        
        result = extractor.extract(pdf_path)
        
        if result:
            full_text = result.get('full_text', '') or result.get('text', '') or result.get('markdown', '')
            return {
                'success': True,
                'text_length': len(full_text) if full_text else 0
            }
        else:
            return {'success': False, 'error': 'No result'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}


def test_with_timeout(arxiv_id: str, pdf_path: str, timeout: int = 20):
    """Test extraction with timeout."""
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        logger.error(f"{arxiv_id}: PDF not found")
        return False
    
    size_mb = pdf_file.stat().st_size / (1024 * 1024)
    logger.info(f"{arxiv_id}: Testing ({size_mb:.1f} MB)")
    
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn')) as executor:
        future = executor.submit(extract_document, pdf_path)
        
        try:
            result = future.result(timeout=timeout)
            if result['success']:
                logger.info(f"{arxiv_id}: ✓ SUCCESS ({result['text_length']} chars)")
                return True
            else:
                logger.error(f"{arxiv_id}: ✗ FAILED - {result['error']}")
                return False
                
        except TimeoutError:
            logger.error(f"{arxiv_id}: ✗ TIMEOUT after {timeout}s")
            executor.shutdown(wait=False, cancel_futures=True)
            return False
        except Exception as e:
            logger.error(f"{arxiv_id}: ✗ CRASH - {e}")
            return False


def main():
    # Previously failed documents
    test_docs = [
        ('1608.02280', '/bulk-store/arxiv-data/pdf/1608/1608.02280.pdf'),
        ('1608.02281', '/bulk-store/arxiv-data/pdf/1608/1608.02281.pdf'),
        ('1608.02282', '/bulk-store/arxiv-data/pdf/1608/1608.02282.pdf'),
        ('1608.02285', '/bulk-store/arxiv-data/pdf/1608/1608.02285.pdf'),  # Known segfault
        ('1608.02295', '/bulk-store/arxiv-data/pdf/1608/1608.02295.pdf'),
    ]
    
    logger.info(f"Testing {len(test_docs)} documents with {20}s timeout each\n")
    
    success_count = 0
    for arxiv_id, pdf_path in test_docs:
        if test_with_timeout(arxiv_id, pdf_path, timeout=20):
            success_count += 1
        time.sleep(1)  # Small delay between tests
    
    logger.info(f"\nSummary: {success_count}/{len(test_docs)} succeeded")


if __name__ == "__main__":
    main()