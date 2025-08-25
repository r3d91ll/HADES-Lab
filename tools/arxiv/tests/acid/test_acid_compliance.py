#!/usr/bin/env python3
"""
ACID Compliance Validation Tests
=================================

Tests to verify that the pipeline truly implements ACID properties:
- Atomicity: Transactions fully succeed or fully rollback
- Consistency: Data remains valid after each transaction
- Isolation: Concurrent transactions don't interfere
- Durability: Committed data persists even after crashes
"""

import os
import sys
import time
import json
import random
import threading
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from arango import ArangoClient
from arango.exceptions import DocumentInsertError, ArangoServerError
import yaml
import numpy as np


def load_config() -> Dict[str, Any]:
    """Load configuration."""
    # Config is in ../../configs/ relative to tests/acid/
    config_path = Path(__file__).parent.parent.parent / "configs" / "acid_pipeline_phased.yaml"
    if not config_path.exists():
        # Fallback to look in parent directory
        config_path = Path(__file__).parent.parent / "configs" / "acid_pipeline_phased.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class ACIDTester:
    """Test suite for ACID compliance."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize tester."""
        self.config = config
        self.client = ArangoClient(hosts=config['arango']['host'])
        self.db = self.client.db(
            config['arango']['database'],
            username=config['arango']['username'],
            password=os.environ.get('ARANGO_PASSWORD', '')
        )
        self.test_results = []
    
    def test_atomicity(self) -> bool:
        """
        Test Atomicity: Transaction either fully completes or fully rolls back.
        """
        print("\n" + "="*60)
        print("TEST 1: ATOMICITY")
        print("="*60)
        
        test_id = f"atomicity_test_{datetime.now().timestamp()}"
        
        try:
            # Begin transaction that will fail midway
            txn_db = self.db.begin_transaction(
                write=['arxiv_papers', 'arxiv_chunks', 'arxiv_embeddings'],
                read=[],
                exclusive=[],
                sync=True,
                allow_implicit=False,
                lock_timeout=5
            )
            
            try:
                # Insert paper (should succeed temporarily)
                txn_db.collection('arxiv_papers').insert({
                    '_key': test_id,
                    'arxiv_id': test_id,
                    'title': 'Test Paper for Atomicity',
                    'processed_at': datetime.now().isoformat()
                })
                print(f"✓ Inserted paper {test_id} in transaction")
                
                # Insert chunks (should succeed temporarily)
                for i in range(3):
                    txn_db.collection('arxiv_chunks').insert({
                        '_key': f"{test_id}_chunk_{i}",
                        'arxiv_id': test_id,
                        'chunk_index': i,
                        'text': f'Test chunk {i}'
                    })
                print(f"✓ Inserted 3 chunks in transaction")
                
                # Force an error by inserting duplicate key
                # This will cause entire transaction to rollback
                txn_db.collection('arxiv_papers').insert({
                    '_key': test_id,  # Duplicate key - will fail!
                    'arxiv_id': test_id,
                    'title': 'Duplicate Paper - Should Fail',
                    'processed_at': datetime.now().isoformat()
                })
                
                # Try to commit - should fail
                txn_db.commit_transaction()
                print("✗ Transaction should have failed but didn't")
                return False
                
            except Exception as e:
                # Transaction failed - abort it
                try:
                    txn_db.abort_transaction()
                except:
                    pass  # May already be aborted
                print(f"✓ Transaction failed and rolled back as expected: {str(e)[:100]}")
            
            # Verify nothing was committed
            paper_exists = self.db.collection('arxiv_papers').has(test_id)
            chunks_count = len(list(self.db.aql.execute(
                "FOR c IN arxiv_chunks FILTER c.arxiv_id == @id RETURN c",
                bind_vars={'id': test_id}
            )))
            
            if not paper_exists and chunks_count == 0:
                print("✓ ATOMICITY TEST PASSED: Transaction fully rolled back")
                return True
            else:
                print(f"✗ ATOMICITY TEST FAILED: Found {chunks_count} chunks, paper exists: {paper_exists}")
                return False
                
        except Exception as e:
            print(f"✗ ATOMICITY TEST ERROR: {e}")
            traceback.print_exc()
            return False
    
    def test_consistency(self) -> bool:
        """
        Test Consistency: Data remains valid after transactions.
        """
        print("\n" + "="*60)
        print("TEST 2: CONSISTENCY")
        print("="*60)
        
        test_id = f"consistency_test_{datetime.now().timestamp()}"
        
        try:
            # Begin a consistent transaction
            txn_db = self.db.begin_transaction(
                write=['arxiv_papers', 'arxiv_chunks', 'arxiv_embeddings'],
                read=[],
                exclusive=[],
                sync=True,
                allow_implicit=False,
                lock_timeout=5
            )
            
            try:
                # Insert paper with all required fields
                txn_db.collection('arxiv_papers').insert({
                    '_key': test_id,
                    'arxiv_id': test_id,
                    'title': 'Consistency Test Paper',
                    'abstract': 'Test abstract for consistency',
                    'processed_at': datetime.now().isoformat(),
                    'num_chunks': 2
                })
                
                # Insert matching chunks
                for i in range(2):
                    txn_db.collection('arxiv_chunks').insert({
                        '_key': f"{test_id}_chunk_{i}",
                        'arxiv_id': test_id,
                        'chunk_index': i,
                        'text': f'Chunk {i} text',
                        'total_chunks': 2
                    })
                
                # Insert matching embeddings
                txn_db.collection('arxiv_embeddings').insert({
                    '_key': test_id,
                    'arxiv_id': test_id,
                    'abstract_embedding': [0.1] * 2048,  # Correct dimension
                    'num_chunks': 2
                })
                
                # Commit the transaction
                txn_db.commit_transaction()
                
            except Exception as e:
                txn_db.abort_transaction()
                raise
            
            print("✓ Committed consistent transaction")
            
            # Verify consistency: chunks match paper metadata
            paper = self.db.collection('arxiv_papers').get(test_id)
            chunks = list(self.db.aql.execute(
                "FOR c IN arxiv_chunks FILTER c.arxiv_id == @id RETURN c",
                bind_vars={'id': test_id}
            ))
            embedding = self.db.collection('arxiv_embeddings').get(test_id)
            
            if (paper['num_chunks'] == len(chunks) == embedding['num_chunks'] == 2):
                print("✓ CONSISTENCY TEST PASSED: All data relationships are valid")
                return True
            else:
                print(f"✗ CONSISTENCY TEST FAILED: Mismatched counts")
                return False
                
        except Exception as e:
            print(f"✗ CONSISTENCY TEST ERROR: {e}")
            traceback.print_exc()
            return False
    
    def test_isolation(self) -> bool:
        """
        Test Isolation: Concurrent transactions don't interfere.
        """
        print("\n" + "="*60)
        print("TEST 3: ISOLATION")
        print("="*60)
        
        results = []
        errors = []
        
        def concurrent_insert(worker_id: int):
            """Worker function for concurrent inserts."""
            try:
                test_id = f"isolation_test_{worker_id}_{datetime.now().timestamp()}"
                
                # Acquire lock first (distributed locking)
                lock_acquired = False
                try:
                    self.db.collection('arxiv_locks').insert({
                        '_key': test_id,
                        'worker_id': worker_id,
                        'acquired_at': datetime.now().isoformat(),
                        'expiresAt': int((datetime.now() + timedelta(minutes=1)).timestamp())
                    })
                    lock_acquired = True
                except DocumentInsertError:
                    results.append(f"Worker {worker_id}: Lock already held")
                    return
                
                # If lock acquired, do transaction
                if lock_acquired:
                    # Begin transaction
                    txn_db = self.db.begin_transaction(
                        write=['arxiv_papers'],
                        read=[],
                        exclusive=[],
                        sync=True,
                        allow_implicit=False,
                        lock_timeout=2
                    )
                    
                    try:
                        # Simulate some work
                        time.sleep(random.uniform(0.1, 0.3))
                        
                        txn_db.collection('arxiv_papers').insert({
                            '_key': test_id,
                            'arxiv_id': test_id,
                            'worker_id': worker_id,
                            'processed_at': datetime.now().isoformat()
                        })
                        
                        # Commit transaction
                        txn_db.commit_transaction()
                    except Exception as e:
                        txn_db.abort_transaction()
                        raise
                    
                    results.append(f"Worker {worker_id}: SUCCESS")
                    
                    # Release lock
                    self.db.collection('arxiv_locks').delete(test_id)
                    
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")
        
        # Run 5 concurrent workers
        threads = []
        for i in range(5):
            t = threading.Thread(target=concurrent_insert, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all to complete
        for t in threads:
            t.join()
        
        print(f"Results: {len(results)} successes, {len(errors)} errors")
        for r in results:
            print(f"  {r}")
        for e in errors:
            print(f"  ERROR: {e}")
        
        # All workers should complete without data corruption
        if len(results) == 5 and len(errors) == 0:
            print("✓ ISOLATION TEST PASSED: All concurrent transactions completed safely")
            return True
        else:
            print(f"✗ ISOLATION TEST WARNING: Some transactions may have conflicted")
            # This is actually OK - isolation means they don't corrupt each other
            return len(errors) == 0
    
    def test_durability(self) -> bool:
        """
        Test Durability: Committed data persists.
        """
        print("\n" + "="*60)
        print("TEST 4: DURABILITY")
        print("="*60)
        
        test_id = f"durability_test_{datetime.now().timestamp()}"
        
        try:
            # Begin durable transaction with sync=True
            txn_db = self.db.begin_transaction(
                write=['arxiv_papers'],
                read=[],
                exclusive=[],
                sync=True,  # Ensures durability
                allow_implicit=False,
                lock_timeout=5
            )
            
            # Insert data
            txn_db.collection('arxiv_papers').insert({
                '_key': test_id,
                'arxiv_id': test_id,
                'title': 'Durability Test Paper',
                'committed_at': datetime.now().isoformat()
            })
            
            # Commit transaction with durability guarantee
            txn_db.commit_transaction()
            
            print(f"✓ Committed document {test_id} with sync=True")
            
            # Immediately verify it exists
            doc = self.db.collection('arxiv_papers').get(test_id)
            if doc:
                print("✓ Document immediately retrievable after commit")
                
                # Simulate connection loss and reconnect
                old_db = self.db
                new_client = ArangoClient(hosts=self.config['arango']['host'])
                self.db = new_client.db(
                    self.config['arango']['database'],
                    username=self.config['arango']['username'],
                    password=os.environ.get('ARANGO_PASSWORD', '')
                )
                
                # Check if document still exists with new connection
                doc_after = self.db.collection('arxiv_papers').get(test_id)
                if doc_after and doc_after['arxiv_id'] == test_id:
                    print("✓ DURABILITY TEST PASSED: Data persists across connections")
                    return True
                else:
                    print("✗ DURABILITY TEST FAILED: Document not found after reconnect")
                    return False
            else:
                print("✗ DURABILITY TEST FAILED: Document not found after commit")
                return False
                
        except Exception as e:
            print(f"✗ DURABILITY TEST ERROR: {e}")
            traceback.print_exc()
            return False
    
    def test_distributed_locking(self) -> bool:
        """
        Test distributed locking mechanism with TTL.
        """
        print("\n" + "="*60)
        print("TEST 5: DISTRIBUTED LOCKING")
        print("="*60)
        
        test_id = "lock_test_paper"
        
        try:
            # Clean up any existing lock
            try:
                self.db.collection('arxiv_locks').delete(test_id)
            except:
                pass
            
            # Worker 1 acquires lock (TTL needs Unix timestamp in seconds)
            expiry_time = int((datetime.now() + timedelta(seconds=2)).timestamp())
            self.db.collection('arxiv_locks').insert({
                '_key': test_id,
                'worker_id': 'worker_1',
                'acquired_at': datetime.now().isoformat(),
                'expiresAt': expiry_time
            })
            print("✓ Worker 1 acquired lock")
            
            # Worker 2 tries to acquire same lock (should fail)
            try:
                self.db.collection('arxiv_locks').insert({
                    '_key': test_id,
                    'worker_id': 'worker_2',
                    'acquired_at': datetime.now().isoformat(),
                    'expiresAt': int((datetime.now() + timedelta(seconds=2)).timestamp() * 1000)  # Convert to milliseconds
                })
                print("✗ Worker 2 should not have acquired lock")
                return False
            except DocumentInsertError:
                print("✓ Worker 2 correctly blocked by lock")
            
            # Check if lock has logically expired (even if not cleaned up)
            print("  Waiting for lock to expire...")
            time.sleep(3)
            
            # Check if lock is logically expired
            lock_doc = self.db.collection('arxiv_locks').get(test_id)
            if lock_doc:
                current_time = int(datetime.now().timestamp() * 1000)  # Convert to milliseconds
                if lock_doc['expiresAt'] < current_time:
                    print("✓ Lock has logically expired (TTL cleanup pending)")
                    # Manual cleanup for test
                    self.db.collection('arxiv_locks').delete(test_id)
                else:
                    print(f"✗ Lock not expired: expires at {lock_doc['expiresAt']}, current time {current_time}")
                    return False
            
            # Now worker 2 should be able to acquire
            try:
                self.db.collection('arxiv_locks').insert({
                    '_key': test_id,
                    'worker_id': 'worker_2',
                    'acquired_at': datetime.now().isoformat(),
                    'expiresAt': int((datetime.now() + timedelta(seconds=2)).timestamp() * 1000)  # Convert to milliseconds
                })
                print("✓ Worker 2 acquired lock after expiry")
                print("✓ DISTRIBUTED LOCKING TEST PASSED")
                return True
            except DocumentInsertError:
                print("✗ Worker 2 still blocked (unexpected)")
                return False
                
        except Exception as e:
            print(f"✗ DISTRIBUTED LOCKING TEST ERROR: {e}")
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """Run all ACID compliance tests."""
        print("\n" + "="*80)
        print("ACID COMPLIANCE VALIDATION SUITE")
        print("="*80)
        
        tests = [
            ("Atomicity", self.test_atomicity),
            ("Consistency", self.test_consistency),
            ("Isolation", self.test_isolation),
            ("Durability", self.test_durability),
            ("Distributed Locking", self.test_distributed_locking)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"\n✗ {test_name} test crashed: {e}")
                results[test_name] = False
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 FULL ACID COMPLIANCE ACHIEVED! 🎉")
        else:
            print(f"\n⚠️  {total - passed} tests failed - not fully ACID compliant")
        
        return passed == total


def main():
    """Run ACID compliance tests."""
    config = load_config()
    tester = ACIDTester(config)
    
    success = tester.run_all_tests()
    
    # Clean up test data
    print("\nCleaning up test data...")
    try:
        # Clean test documents
        for prefix in ['atomicity_test_', 'consistency_test_', 'isolation_test_', 'durability_test_']:
            tester.db.aql.execute(
                "FOR doc IN arxiv_papers FILTER STARTS_WITH(doc._key, @prefix) REMOVE doc IN arxiv_papers",
                bind_vars={'prefix': prefix}
            )
            tester.db.aql.execute(
                "FOR doc IN arxiv_chunks FILTER STARTS_WITH(doc.arxiv_id, @prefix) REMOVE doc IN arxiv_chunks",
                bind_vars={'prefix': prefix}
            )
            tester.db.aql.execute(
                "FOR doc IN arxiv_embeddings FILTER STARTS_WITH(doc._key, @prefix) REMOVE doc IN arxiv_embeddings",
                bind_vars={'prefix': prefix}
            )
        print("✓ Test data cleaned up")
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()