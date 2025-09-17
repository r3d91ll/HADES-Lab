#!/usr/bin/env python3
"""
Quick Pipeline Test
===================

Quick test to verify the ArXiv pipeline works with a small set of papers.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Fix Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir.parent))

from pipelines.arxiv_pipeline import ACIDPhasedPipeline
from core.database import ArangoDBManager

def run_quick_test():
    """Run a quick test with 5 papers."""
    
    # Test papers (well-known AI papers that should exist)
    test_arxiv_ids = [
        "1706.03762",  # Attention is All You Need
        "1810.04805",  # BERT
        "2005.14165",  # GPT-3
        "2302.13971",  # LLaMA
        "2305.10601",  # Tree of Thoughts
    ]
    
    print("="*60)
    print("Quick ArXiv Pipeline Test")
    print("="*60)
    print(f"Testing with {len(test_arxiv_ids)} papers")
    
    # Check environment
    arango_password = os.getenv('ARANGO_PASSWORD')
    if not arango_password:
        print("❌ ARANGO_PASSWORD not set")
        return False
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "acid_pipeline_phased.yaml"
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False
    
    print(f"✅ Using config: {config_path}")
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    
    # Set the password in environment (pipeline reads from there)
    os.environ['ARANGO_PASSWORD'] = arango_password
    
    try:
        pipeline = ACIDPhasedPipeline(str(config_path))
        print("✅ Pipeline initialized")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False
    
    # Process papers
    print(f"\nProcessing {len(test_arxiv_ids)} papers...")
    start_time = time.time()
    
    try:
        # The ACIDPhasedPipeline expects papers to exist in the filesystem
        # Let's check if they exist first
        pdf_base = Path("/bulk-store/arxiv-data/pdf")
        
        available_papers = []
        for arxiv_id in test_arxiv_ids:
            # ArXiv PDFs are stored as YYMM/arxiv_id.pdf
            # Extract year/month from ID if it has dots
            if '.' in arxiv_id:
                yymm = arxiv_id.split('.')[0]
                pdf_path = pdf_base / yymm / f"{arxiv_id}.pdf"
                if pdf_path.exists():
                    available_papers.append(arxiv_id)
                    print(f"  ✅ Found: {arxiv_id}")
                else:
                    print(f"  ⚠️  Not found: {arxiv_id} at {pdf_path}")
        
        if not available_papers:
            print("❌ No test papers found in filesystem")
            return False
        
        print(f"\nProcessing {len(available_papers)} available papers...")
        
        # The pipeline processes from filesystem, not from a list
        # We'll use the 'local' source and let it find papers
        pipeline.run(source='local', count=len(available_papers))
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Processing complete in {elapsed:.1f} seconds")
        
        # Verify in database
        print("\nVerifying database...")
        
        config = {
            'host': os.getenv('ARANGO_HOST', 'http://192.168.1.69:8529'),
            'database': 'academy_store',
            'username': 'root',
            'password': arango_password
        }
        
        db_manager = ArangoDBManager(config)
        db = db_manager.db
        
        for arxiv_id in available_papers[:3]:  # Check first 3
            clean_id = arxiv_id.replace('.', '_')
            if db.collection('arxiv_papers').has(clean_id):
                doc = db.collection('arxiv_papers').get(clean_id)
                print(f"  ✅ {arxiv_id}: {doc.get('status', 'unknown')}")
            else:
                print(f"  ⚠️  {arxiv_id}: not in database")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_quick_test()
    
    if success:
        print("\n✅ Quick test PASSED!")
        print("\nReady for large-scale testing:")
        print("1. cd ../scripts")
        print("2. python collect_ai_papers.py")
        print("3. cd ../tests")  
        print("4. ./run_large_scale_test.sh")
    else:
        print("\n❌ Quick test FAILED")
        print("Please fix issues before running large-scale tests")
    
    sys.exit(0 if success else 1)