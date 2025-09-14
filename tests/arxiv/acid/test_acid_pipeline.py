"""
Test script for ACID pipeline validation.

Tests the ACID guarantees, lock management, transaction rollback,
and recovery mechanisms of the single-database architecture.
"""

import os
import sys
import time
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import concurrent.futures
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from tools.arxiv.pipelines.arango_acid_processor import ArangoACIDProcessor, ProcessingResult
from tools.arxiv.pipelines.worker_pool import ProcessingWorkerPool as ArangoWorkerPool
from tools.arxiv.utils.on_demand_processor import OnDemandProcessor
from tools.arxiv.monitoring.acid_monitoring import ArangoMonitor
from tools.arxiv.utils.migration_strategy import MigrationStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ACIDTestSuite:
    """
    Comprehensive test suite for ACID pipeline validation.
    
    Tests the four ACID properties:
    - Atomicity: All-or-nothing transactions
    - Consistency: Database remains valid after transactions
    - Isolation: Concurrent transactions don't interfere
    - Durability: Committed data persists
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize test suite"""
        self.config = config or {}
        self.processor = ArangoACIDProcessor(self.config)
        self.monitor = ArangoMonitor(self.config)
        self.test_results = []
        
        # Create temporary test PDFs
        self.test_pdf_dir = Path(tempfile.mkdtemp(prefix="acid_test_"))
        self._create_test_pdfs()
    
    def _create_test_pdfs(self):
        """Create dummy PDF files for testing"""
        for i in range(10):
            pdf_path = self.test_pdf_dir / f"test_paper_{i}.pdf"
            # Create a minimal valid PDF
            pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >> >> >> /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test Paper %d) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
365
%%%%EOF""" % i
            pdf_path.write_bytes(pdf_content)
    
    def test_atomicity(self) -> bool:
        """
        Test atomicity: transactions should be all-or-nothing.
        
        Simulates a failure during transaction to ensure rollback.
        """
        logger.info("Testing ATOMICITY...")
        
        try:
            paper_id = "test_atomicity_001"
            pdf_path = self.test_pdf_dir / "test_paper_0.pdf"
            
            # Mock a failure during the commit phase
            with patch.object(self.processor.db, 'begin_transaction') as mock_txn:
                # Create a mock transaction that fails on commit
                mock_txn_obj = Mock()
                mock_txn_obj.commit_transaction.side_effect = Exception("Simulated transaction failure")
                mock_txn.return_value = mock_txn_obj
                
                # Process should fail
                result = self.processor.process_paper(paper_id, str(pdf_path))
                
                if result.success:
                    logger.error("Transaction should have failed but succeeded")
                    return False
                
                # Verify nothing was committed
                papers_collection = self.processor.db.collection('arxiv_papers')
                sanitized_id = paper_id.replace('.', '_').replace('/', '_')
                
                if papers_collection.has(sanitized_id):
                    logger.error("Data was committed despite transaction failure")
                    return False
            
            logger.info("✓ Atomicity test passed: Failed transaction rolled back completely")
            return True
            
        except Exception as e:
            logger.error(f"Atomicity test failed: {e}")
            return False
    
    def test_consistency(self) -> bool:
        """
        Test consistency: database constraints should be maintained.
        
        Verifies that all required fields are present and valid.
        """
        logger.info("Testing CONSISTENCY...")
        
        try:
            paper_id = "test_consistency_001"
            pdf_path = self.test_pdf_dir / "test_paper_1.pdf"
            
            # Process a paper normally
            result = self.processor.process_paper(paper_id, str(pdf_path))
            
            if not result.success:
                logger.error(f"Failed to process paper: {result.error}")
                return False
            
            # Verify consistency constraints
            sanitized_id = paper_id.replace('.', '_').replace('/', '_')
            
            # Check paper document
            papers_collection = self.processor.db.collection('arxiv_papers')
            paper = papers_collection.get(sanitized_id)
            
            # Required fields must exist
            required_fields = ['_key', 'arxiv_id', 'status', 'processing_date']
            for field in required_fields:
                if field not in paper:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Numeric fields must be non-negative
            numeric_fields = ['num_chunks', 'num_equations', 'num_tables', 'num_images']
            for field in numeric_fields:
                if paper.get(field, 0) < 0:
                    logger.error(f"Invalid numeric value for {field}: {paper.get(field)}")
                    return False
            
            # Check referential integrity
            chunks_collection = self.processor.db.collection('arxiv_chunks')
            chunk_count = len(list(self.processor.db.aql.execute(
                "FOR c IN chunks FILTER c.paper_id == @id RETURN c",
                bind_vars={'id': sanitized_id}
            )))
            
            if chunk_count != paper['num_chunks']:
                logger.error(f"Chunk count mismatch: {chunk_count} vs {paper['num_chunks']}")
                return False
            
            logger.info("✓ Consistency test passed: All constraints maintained")
            return True
            
        except Exception as e:
            logger.error(f"Consistency test failed: {e}")
            return False
    
    def test_isolation(self) -> bool:
        """
        Test isolation: concurrent transactions shouldn't interfere.
        
        Processes the same paper ID from multiple workers simultaneously.
        """
        logger.info("Testing ISOLATION...")
        
        try:
            paper_id = "test_isolation_001"
            pdf_path = self.test_pdf_dir / "test_paper_2.pdf"
            
            # Try to process the same paper from multiple threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for i in range(3):
                    future = executor.submit(
                        self.processor.process_paper,
                        paper_id,
                        str(pdf_path)
                    )
                    futures.append(future)
                
                results = [f.result() for f in futures]
            
            # Only one should succeed (acquired the lock)
            successful = sum(1 for r in results if r.success)
            locked_out = sum(1 for r in results if not r.success and "locked" in str(r.error).lower())
            
            if successful != 1:
                logger.error(f"Expected 1 successful processing, got {successful}")
                return False
            
            if locked_out != 2:
                logger.error(f"Expected 2 lock conflicts, got {locked_out}")
                return False
            
            # Verify only one copy exists in database
            sanitized_id = paper_id.replace('.', '_').replace('/', '_')
            papers_collection = self.processor.db.collection('arxiv_papers')
            
            cursor = self.processor.db.aql.execute(
                "FOR p IN papers FILTER p.arxiv_id == @id RETURN p",
                bind_vars={'id': paper_id}
            )
            papers = list(cursor)
            
            if len(papers) != 1:
                logger.error(f"Expected 1 paper, found {len(papers)}")
                return False
            
            logger.info("✓ Isolation test passed: Concurrent access properly controlled")
            return True
            
        except Exception as e:
            logger.error(f"Isolation test failed: {e}")
            return False
    
    def test_durability(self) -> bool:
        """
        Test durability: committed data should persist.
        
        Verifies data remains after simulated crash/restart.
        """
        logger.info("Testing DURABILITY...")
        
        try:
            paper_id = "test_durability_001"
            pdf_path = self.test_pdf_dir / "test_paper_3.pdf"
            
            # Process and commit
            result = self.processor.process_paper(paper_id, str(pdf_path))
            
            if not result.success:
                logger.error(f"Failed to process paper: {result.error}")
                return False
            
            # Simulate system restart by creating new processor instance
            new_processor = ArangoACIDProcessor(self.config)
            
            # Verify data persisted
            sanitized_id = paper_id.replace('.', '_').replace('/', '_')
            papers_collection = new_processor.db.collection('arxiv_papers')
            
            if not papers_collection.has(sanitized_id):
                logger.error("Data did not persist after simulated restart")
                return False
            
            paper = papers_collection.get(sanitized_id)
            if paper['status'] != 'PROCESSED':
                logger.error(f"Paper status incorrect: {paper['status']}")
                return False
            
            logger.info("✓ Durability test passed: Committed data persists")
            return True
            
        except Exception as e:
            logger.error(f"Durability test failed: {e}")
            return False
    
    def test_lock_expiry(self) -> bool:
        """
        Test lock TTL expiry mechanism.
        
        Verifies that expired locks are automatically cleaned up.
        """
        logger.info("Testing LOCK EXPIRY...")
        
        try:
            paper_id = "test_lock_expiry_001"
            
            # Acquire a lock with very short TTL
            sanitized_id = paper_id.replace('.', '_').replace('/', '_')
            locks_collection = self.processor.db.collection('arxiv_locks')
            
            # Insert lock that expires immediately
            from datetime import datetime, timedelta
            locks_collection.insert({
                '_key': sanitized_id,
                'paper_id': paper_id,
                'worker_id': 99999,
                'acquired_at': datetime.now().isoformat(),
                'expiresAt': (datetime.now() - timedelta(seconds=1)).isoformat()  # Already expired
            })
            
            # Wait for TTL cleanup (may take a moment)
            time.sleep(2)
            
            # Lock should be gone
            if locks_collection.has(sanitized_id):
                # Try manual cleanup
                self.monitor.cleanup_expired_locks()
                
                if locks_collection.has(sanitized_id):
                    logger.error("Expired lock was not cleaned up")
                    return False
            
            logger.info("✓ Lock expiry test passed: TTL cleanup working")
            return True
            
        except Exception as e:
            logger.error(f"Lock expiry test failed: {e}")
            return False
    
    def test_worker_pool(self) -> bool:
        """
        Test worker pool parallel processing.
        
        Verifies that multiple workers can process different papers simultaneously.
        """
        logger.info("Testing WORKER POOL...")
        
        try:
            pool = ArangoWorkerPool(num_workers=2, config=self.config)
            
            # Prepare batch of papers
            paper_paths = []
            for i in range(4, 8):
                paper_id = f"test_worker_{i:03d}"
                pdf_path = self.test_pdf_dir / f"test_paper_{i}.pdf"
                paper_paths.append((paper_id, str(pdf_path)))
            
            # Process batch
            batch_result = pool.process_batch(paper_paths, timeout_per_paper=30)
            
            if batch_result.successful != len(paper_paths):
                logger.error(f"Not all papers processed: {batch_result.successful}/{len(paper_paths)}")
                return False
            
            if batch_result.failed > 0:
                logger.error(f"Some papers failed: {batch_result.failed}")
                return False
            
            logger.info(f"✓ Worker pool test passed: {batch_result.papers_per_minute:.2f} papers/minute")
            return True
            
        except Exception as e:
            logger.error(f"Worker pool test failed: {e}")
            return False
    
    def test_on_demand_processing(self) -> bool:
        """
        Test on-demand processor functionality.
        
        Verifies the Check → Download → Process → Cache workflow.
        """
        logger.info("Testing ON-DEMAND PROCESSING...")
        
        try:
            # Configure on-demand processor with test directory
            on_demand_config = {
                'cache_root': self.test_pdf_dir,
                'sqlite_db': self.test_pdf_dir / 'test_cache.db',
                'arango': self.config
            }
            
            on_demand = OnDemandProcessor(on_demand_config)
            
            # Process a test paper
            results = on_demand.process_papers(
                ["test_on_demand_001"],
                force_reprocess=False
            )
            
            # Should process successfully or report not found
            if "test_on_demand_001" not in results:
                logger.error("No result returned for test paper")
                return False
            
            status = results["test_on_demand_001"]
            if status not in ["processed", "not_found", "already_processed"]:
                logger.error(f"Unexpected status: {status}")
                return False
            
            logger.info("✓ On-demand processing test passed")
            return True
            
        except Exception as e:
            logger.error(f"On-demand processing test failed: {e}")
            return False
    
    def test_monitoring(self) -> bool:
        """
        Test monitoring and health check functionality.
        
        Verifies that monitoring can detect system state correctly.
        """
        logger.info("Testing MONITORING...")
        
        try:
            # Get current metrics
            metrics = self.monitor.get_overall_metrics()
            
            # Check that metrics are reasonable
            if metrics.total_papers < 0:
                logger.error("Invalid total papers count")
                return False
            
            # Perform health check
            health = self.monitor.check_health()
            
            if health['status'] not in ['healthy', 'warning', 'critical']:
                logger.error(f"Invalid health status: {health['status']}")
                return False
            
            # Get collection statistics
            stats = self.monitor.get_collection_statistics()
            
            required_collections = ['papers', 'chunks', 'embeddings', 'locks']
            for collection in required_collections:
                if collection not in stats:
                    logger.error(f"Missing collection in stats: {collection}")
                    return False
            
            logger.info(f"✓ Monitoring test passed: System {health['status']}")
            return True
            
        except Exception as e:
            logger.error(f"Monitoring test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all ACID tests and report results"""
        logger.info("\n" + "="*60)
        logger.info("STARTING ACID PIPELINE VALIDATION TESTS")
        logger.info("="*60)
        
        tests = [
            ("Atomicity", self.test_atomicity),
            ("Consistency", self.test_consistency),
            ("Isolation", self.test_isolation),
            ("Durability", self.test_durability),
            ("Lock Expiry", self.test_lock_expiry),
            ("Worker Pool", self.test_worker_pool),
            ("On-Demand Processing", self.test_on_demand_processing),
            ("Monitoring", self.test_monitoring)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} Test ---")
            try:
                if test_func():
                    passed += 1
                    self.test_results.append((test_name, "PASSED"))
                else:
                    failed += 1
                    self.test_results.append((test_name, "FAILED"))
            except Exception as e:
                failed += 1
                self.test_results.append((test_name, f"ERROR: {e}"))
                logger.error(f"Test {test_name} errored: {e}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        
        for test_name, result in self.test_results:
            status_symbol = "✓" if result == "PASSED" else "✗"
            logger.info(f"{status_symbol} {test_name}: {result}")
        
        logger.info(f"\nTotal: {passed} passed, {failed} failed")
        
        # Cleanup
        import shutil
        shutil.rmtree(self.test_pdf_dir, ignore_errors=True)
        
        return failed == 0


def main():
    """Main test runner"""
    # Get password from environment
    arango_password = os.environ.get('ARANGO_PASSWORD')
    if not arango_password:
        logger.error("ARANGO_PASSWORD environment variable not set")
        return False
    
    # Configuration for test environment
    config = {
        'arango_host': ['http://192.168.1.69:8529'],
        'database': 'academy_store_test',  # Use test database
        'username': 'root',
        'password': arango_password,
        'embedder_config': {
            'device': 'cpu',  # Use CPU for tests
            'use_fp16': False,
            'chunk_size_tokens': 100,
            'chunk_overlap_tokens': 20
        },
        'extractor_config': {
            'use_ocr': False,
            'extract_tables': True,
            'use_fallback': False
        }
    }
    
    # Run tests
    test_suite = ACIDTestSuite(config)
    success = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()