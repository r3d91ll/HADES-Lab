#!/usr/bin/env python3
"""
Performance Test for ACID Pipeline
===================================

Tests that the ACID-compliant pipeline maintains the required
6.2 papers/minute throughput baseline.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from arango import ArangoClient
import yaml
import numpy as np

# Import the ACID processor
from core.tools.arxiv.arxiv_pipeline import ACIDPhasedPipeline
from core.database.arango.arango_client import ArangoDBManager

def load_config() -> Dict[str, Any]:
    """
    Load the ACID phased pipeline YAML configuration and inject the ArangoDB password from the environment.
    
    Reads "configs/acid_pipeline_phased.yaml" located three levels up from this file, parses it with yaml.safe_load, and returns the resulting dict with an added 'password' key set from the ARANGO_PASSWORD environment variable (empty string if unset).
    
    Returns:
        dict: Configuration dictionary for the pipeline.
    """
    config_path = Path(__file__).parent.parent.parent / "configs" / "acid_pipeline_phased.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Add password from environment
    config['password'] = os.environ.get('ARANGO_PASSWORD', '')
    return config


def create_test_pdfs(num_papers: int = 50) -> List[str]:
    """
    Create test PDF paths.
    In production, these would be real ArXiv PDFs.
    """
    # Use real PDF paths from the bulk store
    pdf_base = "/bulk-store/arxiv-data/pdf"
    
    # Some known PDFs that exist in the system
    test_pdfs = [
        f"{pdf_base}/1201/1201.0205.pdf",
        f"{pdf_base}/1201/1201.0206.pdf",
        f"{pdf_base}/1201/1201.0207.pdf",
        f"{pdf_base}/1201/1201.0208.pdf",
        f"{pdf_base}/1201/1201.0209.pdf",
        f"{pdf_base}/1201/1201.0210.pdf",
        f"{pdf_base}/1201/1201.0211.pdf",
        f"{pdf_base}/1201/1201.0212.pdf",
    ]
    
    # Check which ones actually exist
    existing_pdfs = []
    for pdf_path in test_pdfs:
        if Path(pdf_path).exists():
            existing_pdfs.append(pdf_path)
    
    if not existing_pdfs:
        print("WARNING: No test PDFs found, using dummy paths")
        # Create dummy paths for testing
        existing_pdfs = [f"/tmp/test_{i}.pdf" for i in range(num_papers)]
        # Create dummy files
        for pdf_path in existing_pdfs:
            Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
            Path(pdf_path).write_text("Dummy PDF content for testing")
    
    # Repeat the list to get enough papers
    result = []
    while len(result) < num_papers:
        result.extend(existing_pdfs)
    
    return result[:num_papers]


def test_throughput(processor: ArangoACIDProcessor, num_papers: int = 50) -> Dict[str, Any]:
    """
    Test processing throughput.
    
    Target: 6.2 papers/minute baseline
    """
    print(f"\nTesting throughput with {num_papers} papers...")
    
    # Get test PDFs
    pdf_paths = create_test_pdfs(num_papers)
    
    # Track results
    results = {
        'total_papers': num_papers,
        'successful': 0,
        'failed': 0,
        'start_time': time.time(),
        'end_time': None,
        'total_time': None,
        'papers_per_minute': None,
        'meets_baseline': False,
        'errors': []
    }
    
    # Process papers
    for i, pdf_path in enumerate(pdf_paths):
        paper_id = f"perf_test_{datetime.now().timestamp()}_{i}"
        
        try:
            result = processor.process_paper(paper_id, pdf_path)
            
            if result.success:
                results['successful'] += 1
            else:
                results['failed'] += 1
                results['errors'].append(result.error)
            
            # Progress update every 10 papers
            if (i + 1) % 10 == 0:
                elapsed = time.time() - results['start_time']
                rate = (i + 1) / (elapsed / 60)
                print(f"  Processed {i+1}/{num_papers} papers, rate: {rate:.2f} papers/min")
        
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(str(e))
            print(f"  Error processing paper {i+1}: {e}")
    
    # Calculate final metrics
    results['end_time'] = time.time()
    results['total_time'] = results['end_time'] - results['start_time']
    results['papers_per_minute'] = results['successful'] / (results['total_time'] / 60)
    results['meets_baseline'] = results['papers_per_minute'] >= 6.2
    
    return results


def test_concurrent_processing(processor: ArangoACIDProcessor, num_workers: int = 4) -> Dict[str, Any]:
    """
    Test concurrent processing with multiple workers.
    """
    print(f"\nTesting concurrent processing with {num_workers} workers...")
    
    import threading
    import queue
    
    # Create work queue
    work_queue = queue.Queue()
    results_queue = queue.Queue()
    
    # Add work items
    pdf_paths = create_test_pdfs(20)  # Fewer for concurrent test
    for i, pdf_path in enumerate(pdf_paths):
        paper_id = f"concurrent_test_{datetime.now().timestamp()}_{i}"
        work_queue.put((paper_id, pdf_path))
    
    def worker():
        """Worker thread function."""
        while True:
            try:
                paper_id, pdf_path = work_queue.get(timeout=1)
                start = time.time()
                result = processor.process_paper(paper_id, pdf_path)
                elapsed = time.time() - start
                results_queue.put({
                    'success': result.success,
                    'time': elapsed,
                    'error': result.error
                })
                work_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                results_queue.put({
                    'success': False,
                    'time': 0,
                    'error': str(e)
                })
    
    # Start workers
    threads = []
    start_time = time.time()
    
    for _ in range(num_workers):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    # Wait for completion
    work_queue.join()
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    
    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    successful = sum(1 for r in results if r['success'])
    papers_per_minute = successful / (total_time / 60)
    
    return {
        'num_workers': num_workers,
        'total_papers': len(pdf_paths),
        'successful': successful,
        'failed': len(results) - successful,
        'total_time': total_time,
        'papers_per_minute': papers_per_minute,
        'meets_baseline': papers_per_minute >= 6.2
    }


def test_transaction_overhead() -> Dict[str, Any]:
    """
    Test the overhead of ACID transactions vs non-transactional operations.
    """
    print("\nTesting transaction overhead...")
    
    config = load_config()
    client = ArangoClient(hosts=config.get('arango_host', 'http://192.168.1.69:8529'))
    db = client.db(
        config.get('database', 'academy_store'),
        username=config.get('username', 'root'),
        password=config['password']
    )
    
    # Ensure test collection exists
    if not db.has_collection('perf_test'):
        db.create_collection('perf_test')
    
    num_docs = 100
    
    # Test 1: Non-transactional inserts
    start = time.time()
    for i in range(num_docs):
        db.collection('perf_test').insert({
            '_key': f'non_txn_{i}',
            'data': f'test data {i}'
        }, overwrite=True)
    non_txn_time = time.time() - start
    
    # Test 2: Transactional inserts
    start = time.time()
    txn_db = db.begin_transaction(
        write=['perf_test'],
        read=[],
        exclusive=[],
        sync=True,
        allow_implicit=False,
        lock_timeout=5
    )
    
    try:
        for i in range(num_docs):
            txn_db.collection('perf_test').insert({
                '_key': f'txn_{i}',
                'data': f'test data {i}'
            }, overwrite=True)
        txn_db.commit_transaction()
    except Exception as e:
        txn_db.abort_transaction()
        raise
    
    txn_time = time.time() - start
    
    # Clean up
    db.collection('perf_test').truncate()
    
    overhead_percent = ((txn_time - non_txn_time) / non_txn_time) * 100
    
    return {
        'num_documents': num_docs,
        'non_transactional_time': non_txn_time,
        'transactional_time': txn_time,
        'overhead_seconds': txn_time - non_txn_time,
        'overhead_percent': overhead_percent,
        'acceptable': overhead_percent < 50  # Less than 50% overhead is acceptable
    }


def main():
    """Run performance tests."""
    print("="*80)
    print("ACID PIPELINE PERFORMANCE TESTS")
    print("="*80)
    print(f"Baseline target: 6.2 papers/minute")
    
    # Load config and create processor
    config = load_config()
    processor = ArangoACIDProcessor(config)
    
    # Run tests
    test_results = {}
    
    # Test 1: Sequential throughput
    print("\n" + "="*60)
    print("TEST 1: SEQUENTIAL THROUGHPUT")
    print("="*60)
    try:
        test_results['sequential'] = test_throughput(processor, num_papers=30)
        
        print(f"\nResults:")
        print(f"  Papers processed: {test_results['sequential']['successful']}/{test_results['sequential']['total_papers']}")
        print(f"  Total time: {test_results['sequential']['total_time']:.2f} seconds")
        print(f"  Rate: {test_results['sequential']['papers_per_minute']:.2f} papers/minute")
        print(f"  Meets baseline (6.2): {'✅ YES' if test_results['sequential']['meets_baseline'] else '❌ NO'}")
    except Exception as e:
        print(f"✗ Sequential test failed: {e}")
        traceback.print_exc()
        test_results['sequential'] = {'meets_baseline': False, 'error': str(e)}
    
    # Test 2: Concurrent processing
    print("\n" + "="*60)
    print("TEST 2: CONCURRENT PROCESSING")
    print("="*60)
    try:
        test_results['concurrent'] = test_concurrent_processing(processor, num_workers=4)
        
        print(f"\nResults:")
        print(f"  Workers: {test_results['concurrent']['num_workers']}")
        print(f"  Papers processed: {test_results['concurrent']['successful']}/{test_results['concurrent']['total_papers']}")
        print(f"  Total time: {test_results['concurrent']['total_time']:.2f} seconds")
        print(f"  Rate: {test_results['concurrent']['papers_per_minute']:.2f} papers/minute")
        print(f"  Meets baseline (6.2): {'✅ YES' if test_results['concurrent']['meets_baseline'] else '❌ NO'}")
    except Exception as e:
        print(f"✗ Concurrent test failed: {e}")
        traceback.print_exc()
        test_results['concurrent'] = {'meets_baseline': False, 'error': str(e)}
    
    # Test 3: Transaction overhead
    print("\n" + "="*60)
    print("TEST 3: TRANSACTION OVERHEAD")
    print("="*60)
    try:
        test_results['overhead'] = test_transaction_overhead()
        
        print(f"\nResults:")
        print(f"  Non-transactional: {test_results['overhead']['non_transactional_time']:.3f}s")
        print(f"  Transactional: {test_results['overhead']['transactional_time']:.3f}s")
        print(f"  Overhead: {test_results['overhead']['overhead_percent']:.1f}%")
        print(f"  Acceptable (<50%): {'✅ YES' if test_results['overhead']['acceptable'] else '❌ NO'}")
    except Exception as e:
        print(f"✗ Overhead test failed: {e}")
        traceback.print_exc()
        test_results['overhead'] = {'acceptable': False, 'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    all_passed = (
        test_results.get('sequential', {}).get('meets_baseline', False) and
        test_results.get('concurrent', {}).get('meets_baseline', False) and
        test_results.get('overhead', {}).get('acceptable', False)
    )
    
    print(f"Sequential throughput: {'✅ PASSED' if test_results.get('sequential', {}).get('meets_baseline', False) else '❌ FAILED'}")
    print(f"Concurrent processing: {'✅ PASSED' if test_results.get('concurrent', {}).get('meets_baseline', False) else '❌ FAILED'}")
    print(f"Transaction overhead: {'✅ PASSED' if test_results.get('overhead', {}).get('acceptable', False) else '❌ FAILED'}")
    
    if all_passed:
        print("\n🎉 ALL PERFORMANCE TESTS PASSED! 🎉")
        print("ACID compliance achieved WITHOUT sacrificing performance!")
    else:
        print("\n⚠️ Some performance tests failed")
        print("Performance optimization may be needed")
    
    # Clean up test data
    print("\nCleaning up test data...")
    try:
        client = ArangoClient(hosts=config.get('arango_host', 'http://192.168.1.69:8529'))
        db = client.db(
            config.get('database', 'academy_store'),
            username=config.get('username', 'root'),
            password=config['password']
        )
        
        # Clean test documents with safety checks
        for prefix in ['perf_test_', 'concurrent_test_']:
            # Check how many documents would be affected
            result = db.aql.execute(
                "FOR doc IN arxiv_papers FILTER STARTS_WITH(doc._key, @prefix) COLLECT WITH COUNT INTO cnt RETURN cnt",
                bind_vars={'prefix': prefix}
            )
            count = list(result)[0] if result else 0
            
            # Only clean if reasonable number of test documents
            if count > 0 and count < 1000:  # Safety limit
                print(f"  Cleaning {count} test papers with prefix '{prefix}'")
                db.aql.execute(
                    "FOR doc IN arxiv_papers FILTER STARTS_WITH(doc._key, @prefix) REMOVE doc IN arxiv_papers",
                    bind_vars={'prefix': prefix}
                )
                db.aql.execute(
                    "FOR doc IN arxiv_chunks FILTER STARTS_WITH(doc.paper_id, @prefix) REMOVE doc IN arxiv_chunks",
                    bind_vars={'prefix': prefix}
                )
                db.aql.execute(
                    "FOR doc IN arxiv_embeddings FILTER STARTS_WITH(doc.paper_id, @prefix) REMOVE doc IN arxiv_embeddings",
                    bind_vars={'prefix': prefix}
                )
            elif count >= 1000:
                print(f"  WARNING: Too many documents ({count}) match prefix '{prefix}' - skipping cleanup for safety")
        
        # Clean test collection
        if db.has_collection('perf_test'):
            db.delete_collection('perf_test')
        
        print("✓ Test data cleaned up")
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()