#!/usr/bin/env python3
"""
Test Robust Extraction
======================

Test the new RobustExtractor with timeout protection and fallback.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.framework.extractors.robust_extractor import RobustExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_robust_extraction():
    """Test robust extraction on problematic documents."""
    
    # Initialize robust extractor with short timeout
    extractor = RobustExtractor(
        use_ocr=False,
        extract_tables=False,  # Disable for speed
        timeout=15,  # 15 second timeout
        use_fallback=True
    )
    
    # Test documents (including known problematic ones)
    test_docs = [
        ('1608.02280', '/bulk-store/arxiv-data/pdf/1608/1608.02280.pdf'),  # Works
        ('1608.02281', '/bulk-store/arxiv-data/pdf/1608/1608.02281.pdf'),  # Timeout
        ('1608.02285', '/bulk-store/arxiv-data/pdf/1608/1608.02285.pdf'),  # Segfault
        ('1608.02295', '/bulk-store/arxiv-data/pdf/1608/1608.02295.pdf'),  # Timeout
        ('1608.02383', '/bulk-store/arxiv-data/pdf/1608/1608.02383.pdf'),  # Large
    ]
    
    results = {
        'docling': [],
        'fallback': [],
        'failed': []
    }
    
    logger.info(f"Testing {len(test_docs)} documents with RobustExtractor")
    logger.info(f"Timeout: 15s, Fallback: Enabled\n")
    
    for arxiv_id, pdf_path in test_docs:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            logger.error(f"{arxiv_id}: PDF not found")
            results['failed'].append(arxiv_id)
            continue
            
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        logger.info(f"{arxiv_id}: Testing ({size_mb:.1f} MB)")
        
        result = extractor.extract(pdf_path)
        
        if result:
            extractor_used = result.get('extractor', 'docling')
            text = result.get('full_text', '') or result.get('text', '') or result.get('markdown', '')
            
            if extractor_used == 'pymupdf_fallback':
                logger.info(f"  ✓ SUCCESS via FALLBACK ({len(text)} chars)")
                results['fallback'].append(arxiv_id)
            else:
                logger.info(f"  ✓ SUCCESS via DOCLING ({len(text)} chars)")
                results['docling'].append(arxiv_id)
        else:
            logger.error(f"  ✗ FAILED completely")
            results['failed'].append(arxiv_id)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"Docling success: {len(results['docling'])} documents")
    if results['docling']:
        logger.info(f"  {results['docling']}")
    
    logger.info(f"Fallback success: {len(results['fallback'])} documents")
    if results['fallback']:
        logger.info(f"  {results['fallback']}")
    
    logger.info(f"Complete failures: {len(results['failed'])} documents")
    if results['failed']:
        logger.info(f"  {results['failed']}")
    
    total_success = len(results['docling']) + len(results['fallback'])
    logger.info(f"\nOverall success rate: {total_success}/{len(test_docs)} = {total_success/len(test_docs)*100:.0f}%")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    test_robust_extraction()