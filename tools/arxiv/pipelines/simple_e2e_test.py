#!/usr/bin/env python3
"""
Simple E2E Test - Direct PDF Processing
========================================

Tests the restructured core modules by processing a single PDF directly,
without requiring PostgreSQL metadata.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from restructured core modules ONLY
from core.embedders import EmbedderFactory
from core.extractors import ExtractorFactory
from core.database.arango import ArangoDBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run simple E2E test."""

    # Check for a PDF file
    pdf_path = "/bulk-store/arxiv-data/pdf/2401/2401.00001v1.pdf"
    if not os.path.exists(pdf_path):
        # Try to find any PDF
        pdf_dir = Path("/bulk-store/arxiv-data/pdf")
        pdfs = list(pdf_dir.glob("*/*.pdf"))[:1]
        if not pdfs:
            logger.error("No PDFs found to test")
            return
        pdf_path = str(pdfs[0])

    logger.info(f"Testing with PDF: {pdf_path}")

    # Initialize extractor
    logger.info("Creating extractor...")
    extractor = ExtractorFactory.create('docling', ocr_enabled=False)

    # Extract content
    logger.info("Extracting content...")
    extraction = extractor.extract(pdf_path)

    if not extraction or not extraction.get('full_text'):
        logger.error("No text extracted")
        return

    logger.info(f"Extracted {len(extraction['full_text'])} characters")

    # Initialize embedder
    logger.info("Creating embedder...")
    embedder = EmbedderFactory.create(
        'jina_v4',
        device='cuda',
        batch_size=4,
        use_fp16=True
    )

    # Generate embeddings with late chunking
    logger.info("Generating embeddings with late chunking...")
    chunks_with_embeddings = embedder.embed_with_late_chunking(
        extraction['full_text']
    )

    logger.info(f"Created {len(chunks_with_embeddings)} chunks with embeddings")

    # Initialize ArangoDB
    logger.info("Connecting to ArangoDB...")
    arango_config = {
        'database': 'academy_store',
        'host': 'http://localhost:8529',
        'username': 'root',
        'password': os.environ.get('ARANGO_PASSWORD')
    }

    if not arango_config['password']:
        logger.error("ARANGO_PASSWORD not set")
        return

    arango_manager = ArangoDBManager(arango_config, pool_size=1)

    # Store a test document
    test_doc = {
        '_key': 'e2e_test_paper',
        'arxiv_id': 'test/e2e',
        'title': 'E2E Test Paper',
        'status': 'PROCESSED',
        'num_chunks': len(chunks_with_embeddings)
    }

    logger.info("Storing test document...")
    arango_manager.insert_document('arxiv_papers', test_doc, overwrite=True)

    # Store chunks and embeddings
    for idx, chunk_data in enumerate(chunks_with_embeddings[:2]):  # Just store first 2
        chunk_doc = {
            '_key': f"e2e_test_chunk_{idx}",
            'paper_id': 'test/e2e',
            'chunk_index': idx,
            'text': chunk_data.text[:500],  # Truncate for test
            'chunk_start': chunk_data.start_char,
            'chunk_end': chunk_data.end_char
        }
        chunk_result = arango_manager.insert_document('arxiv_chunks', chunk_doc, overwrite=True)
        logger.info(f"Stored chunk {idx}")

        embedding_doc = {
            '_key': f"e2e_test_emb_{idx}",
            'paper_id': 'test/e2e',
            'chunk_id': chunk_result['_id'],
            'vector': chunk_data.embedding.tolist(),
            'model': 'jina-v4',
            'dimension': len(chunk_data.embedding)
        }
        arango_manager.insert_document('arxiv_embeddings', embedding_doc, overwrite=True)
        logger.info(f"Stored embedding {idx}")

    logger.info("\n" + "="*60)
    logger.info("E2E TEST SUCCESSFUL!")
    logger.info("="*60)
    logger.info("✓ Extractor working (Docling)")
    logger.info("✓ Embedder working (Jina v4)")
    logger.info("✓ ArangoDB storage working")
    logger.info(f"✓ Processed {len(extraction['full_text'])} chars → {len(chunks_with_embeddings)} chunks")
    logger.info("="*60)


if __name__ == "__main__":
    main()