#!/usr/bin/env python3
"""
Debug Single Document
=====================

Test a single document to see exactly what fails.
"""

import sys
import logging
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.extractors import DoclingExtractor
from core.embedders import JinaV4Embedder

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_document(arxiv_id: str, pdf_path: str):
    """Test extraction and embedding of a single document."""
    
    logger.info(f"Testing {arxiv_id} from {pdf_path}")
    
    # Check file
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    size_mb = pdf_file.stat().st_size / (1024 * 1024)
    logger.info(f"PDF exists: {size_mb:.2f} MB")
    
    # Test extraction
    logger.info("="*40)
    logger.info("Testing EXTRACTION...")
    try:
        extractor = DoclingExtractor(
            use_ocr=False,
            extract_tables=True,
            use_fallback=True
        )
        
        extracted = extractor.extract(pdf_path)
        
        # Check what we got
        if extracted:
            logger.info(f"✓ Extraction succeeded")
            logger.info(f"  Keys in result: {list(extracted.keys())}")
            
            # Check for text content
            full_text = extracted.get('full_text', '') or extracted.get('text', '') or extracted.get('markdown', '')
            if full_text:
                logger.info(f"  Text length: {len(full_text)} characters")
                logger.info(f"  First 200 chars: {full_text[:200]}...")
            else:
                logger.warning("  No text content found!")
                logger.info(f"  Available keys: {extracted.keys()}")
        else:
            logger.error("✗ Extraction returned None")
            
    except Exception as e:
        logger.error(f"✗ Extraction failed with error: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Test embedding if extraction worked
    if extracted and full_text:
        logger.info("="*40)
        logger.info("Testing EMBEDDING...")
        try:
            embedder = JinaV4Embedder(
                device='cuda',
                use_fp16=True,
                chunk_size_tokens=1000,
                chunk_overlap_tokens=200
            )
            
            chunks = embedder.embed_with_late_chunking(full_text)
            
            logger.info(f"✓ Embedding succeeded")
            logger.info(f"  Generated {len(chunks)} chunks")
            if chunks:
                logger.info(f"  First chunk has {len(chunks[0].embedding)} dimensions")
                logger.info(f"  First chunk text preview: {chunks[0].text[:100]}...")
                
        except Exception as e:
            logger.error(f"✗ Embedding failed with error: {e}")
            logger.error(traceback.format_exc())

def main():
    # Test one of the failed extraction documents
    failed_extraction = {
        'arxiv_id': '1608.02285',
        'pdf_path': '/bulk-store/arxiv-data/pdf/1608/1608.02285.pdf'
    }
    
    # Test one of the successful extraction but failed embedding documents
    failed_embedding = {
        'arxiv_id': '1608.02383',
        'pdf_path': '/bulk-store/arxiv-data/pdf/1608/1608.02383.pdf'
    }
    
    logger.info("Testing document that failed extraction:")
    test_document(**failed_extraction)
    
    logger.info("\n" + "="*60 + "\n")
    
    logger.info("Testing document that extracted but failed embedding:")
    test_document(**failed_embedding)

if __name__ == "__main__":
    main()