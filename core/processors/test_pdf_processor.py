#!/usr/bin/env python3
"""
Test script for the decoupled PDF processor.

Tests that PDFProcessor can process PDFs without any ArXiv-specific logic.
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.processors.pdf_processor import PDFProcessor, ProcessorConfig, PDFTask


def test_pdf_processor():
    """Test the PDF processor with a sample PDF."""
    
    # Configuration
    config = ProcessorConfig(
        extraction_workers=2,
        extraction_batch_size=1,
        embedding_workers=2,
        embedding_batch_size=1,
        use_fp16=True,
        gpu_devices=[0],  # Single GPU for testing
        staging_dir='/dev/shm/test_pdf_staging'
    )
    
    storage_config = {
        'host': os.environ.get('ARANGO_HOST', 'localhost'),
        'port': 8529,
        'database': 'academy_store',
        'username': 'root',
        'password': os.environ.get('ARANGO_PASSWORD', ''),
        'collection_prefix': 'test'  # Use test prefix to avoid collision
    }
    
    # Initialize processor
    print("Initializing PDF processor...")
    processor = PDFProcessor(config, storage_config)
    
    # Create a test task (using an ArXiv PDF but treating it generically)
    test_pdf = "/bulk-store/arxiv-data/pdf/1706/1706.03762v7.pdf"  # Attention is All You Need
    
    if not Path(test_pdf).exists():
        print(f"Test PDF not found: {test_pdf}")
        print("Please provide a valid PDF path for testing.")
        return False
    
    task = PDFTask(
        pdf_path=test_pdf,
        document_id="test_document_001",
        metadata={
            'title': 'Test Document',
            'source': 'test',
            'test_run': True
        }
    )
    
    print(f"Processing PDF: {test_pdf}")
    print(f"Document ID: {task.document_id}")
    
    # Process the PDF
    results = processor.process_batch([task])
    
    # Check results
    if results:
        result = results[0]
        print(f"\nProcessing Result:")
        print(f"  Success: {result.success}")
        print(f"  Document ID: {result.document_id}")
        print(f"  Chunks: {result.num_chunks}")
        print(f"  Equations: {result.num_equations}")
        print(f"  Tables: {result.num_tables}")
        print(f"  Images: {result.num_images}")
        print(f"  Processing Time: {result.processing_time:.2f} seconds")
        
        if result.error:
            print(f"  Error: {result.error}")
        
        return result.success
    else:
        print("No results returned")
        return False


def test_arxiv_compatibility():
    """Test that ArXiv papers can be processed with generic processor."""
    
    config = ProcessorConfig(
        extraction_workers=1,
        embedding_workers=1,
        collection_prefix='arxiv'  # Use ArXiv prefix for compatibility
    )
    
    storage_config = {
        'host': os.environ.get('ARANGO_HOST', 'localhost'),
        'port': 8529,
        'database': 'academy_store',
        'username': 'root',
        'password': os.environ.get('ARANGO_PASSWORD', ''),
        'collection_prefix': 'arxiv'
    }
    
    processor = PDFProcessor(config, storage_config)
    
    # Process an ArXiv paper with ArXiv metadata
    arxiv_id = "1706.03762"
    pdf_path = f"/bulk-store/arxiv-data/pdf/1706/{arxiv_id}v7.pdf"
    
    if Path(pdf_path).exists():
        result = processor.process_pdf(
            pdf_path=pdf_path,
            document_id=arxiv_id,
            metadata={
                'arxiv_id': arxiv_id,
                'title': 'Attention Is All You Need',
                'authors': ['Vaswani et al.'],
                'categories': ['cs.CL', 'cs.LG']
            }
        )
        
        print(f"\nArXiv Compatibility Test:")
        print(f"  ArXiv ID: {arxiv_id}")
        print(f"  Success: {result.success}")
        print(f"  Chunks: {result.num_chunks}")
        
        return result.success
    else:
        print(f"ArXiv PDF not found: {pdf_path}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Decoupled PDF Processor")
    print("=" * 60)
    
    # Check environment
    if not os.environ.get('ARANGO_PASSWORD'):
        print("Warning: ARANGO_PASSWORD not set in environment")
        print("Set it with: export ARANGO_PASSWORD='your-password'")
        sys.exit(1)
    
    # Run tests
    print("\n1. Testing generic PDF processing...")
    test1_passed = test_pdf_processor()
    
    print("\n2. Testing ArXiv compatibility...")
    test2_passed = test_arxiv_compatibility()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Generic PDF Processing: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"  ArXiv Compatibility: {'PASSED' if test2_passed else 'FAILED'}")
    print("=" * 60)
    
    sys.exit(0 if (test1_passed and test2_passed) else 1)