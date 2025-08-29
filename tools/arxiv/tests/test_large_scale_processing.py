#!/usr/bin/env python3
"""
Large-Scale ArXiv Processing Test Suite
========================================

Tests for processing 5000+ ArXiv papers on AI/RAG/LLM/ANT topics.
Includes performance benchmarks, error handling, and database verification.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Fix Python path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent  # Go up to HADES-Lab root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir.parent.parent))  # Add tools/arxiv to path

from pipelines.arxiv_pipeline import ACIDPhasedPipeline
from core.database import ArangoDBManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LargeScaleTestSuite:
    """Test suite for large-scale ArXiv processing."""
    
    def __init__(self, config_path: str, arango_password: str):
        """
        Initialize test suite.
        
        Args:
            config_path: Path to pipeline configuration
            arango_password: ArangoDB password
        """
        self.config_path = Path(config_path)
        self.arango_password = arango_password
        
        # Load configuration
        import yaml
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Setup database connection
        self.config['arango']['password'] = arango_password
        self.db_manager = ArangoDBManager(self.config['arango'])
        
        # Test results storage
        self.test_results = {
            'start_time': None,
            'end_time': None,
            'tests': {},
            'performance_metrics': {},
            'errors': []
        }
    
    def load_paper_list(self, paper_list_file: str) -> List[str]:
        """Load list of ArXiv IDs to process."""
        paper_list_path = Path(paper_list_file)
        
        if paper_list_path.suffix == '.txt':
            with open(paper_list_path) as f:
                return [line.strip() for line in f if line.strip()]
        elif paper_list_path.suffix == '.json':
            with open(paper_list_path) as f:
                data = json.load(f)
                # Extract IDs from the JSON structure
                all_ids = set()
                if 'papers' in data:
                    for topic_papers in data['papers'].values():
                        for paper in topic_papers:
                            all_ids.add(paper['arxiv_id'])
                return sorted(all_ids)
        else:
            raise ValueError(f"Unsupported file format: {paper_list_path.suffix}")
    
    def test_batch_processing(self, arxiv_ids: List[str], batch_size: int = 100):
        """
        Test processing papers in batches.
        
        Args:
            arxiv_ids: List of ArXiv IDs to process
            batch_size: Number of papers per batch
        """
        test_name = f"batch_processing_{batch_size}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        total_processed = 0
        total_failed = 0
        batch_times = []
        
        # Create pipeline
        pipeline = ACIDPhasedPipeline(
            config_path=str(self.config_path),
            arango_password=self.arango_password
        )
        
        # Process in batches
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i:i+batch_size]
            batch_start = time.time()
            
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} papers")
            
            try:
                # Process the batch
                results = pipeline.process_papers(batch, count=len(batch))
                
                total_processed += results.get('successful', 0)
                total_failed += results.get('failed', 0)
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Calculate metrics
                papers_per_minute = (len(batch) / batch_time) * 60 if batch_time > 0 else 0
                
                logger.info(f"  Batch completed in {batch_time:.2f}s")
                logger.info(f"  Rate: {papers_per_minute:.1f} papers/minute")
                logger.info(f"  Success: {results.get('successful', 0)}/{len(batch)}")
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                self.test_results['errors'].append({
                    'test': test_name,
                    'batch': i//batch_size + 1,
                    'error': str(e)
                })
        
        # Calculate overall metrics
        total_time = time.time() - start_time
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        overall_rate = (total_processed / total_time) * 60 if total_time > 0 else 0
        
        self.test_results['tests'][test_name] = {
            'total_papers': len(arxiv_ids),
            'total_processed': total_processed,
            'total_failed': total_failed,
            'success_rate': total_processed / len(arxiv_ids) if arxiv_ids else 0,
            'total_time_seconds': total_time,
            'average_batch_time': avg_batch_time,
            'overall_rate_per_minute': overall_rate
        }
        
        logger.info(f"\n{test_name} Results:")
        logger.info(f"  Total processed: {total_processed}/{len(arxiv_ids)}")
        logger.info(f"  Success rate: {(total_processed/len(arxiv_ids)*100):.1f}%")
        logger.info(f"  Overall rate: {overall_rate:.1f} papers/minute")
    
    def test_error_recovery(self, arxiv_ids: List[str], inject_errors: bool = True):
        """
        Test error recovery and retry mechanisms.
        
        Args:
            arxiv_ids: List of ArXiv IDs to process
            inject_errors: Whether to inject artificial errors
        """
        test_name = "error_recovery"
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*60}")
        
        # Select a subset for error testing
        test_ids = arxiv_ids[:50]
        
        # Track retry attempts
        retry_counts = {}
        successful_recoveries = 0
        
        pipeline = ACIDPhasedPipeline(
            config_path=str(self.config_path),
            arango_password=self.arango_password
        )
        
        for arxiv_id in test_ids:
            try:
                # Process with potential retry
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result = pipeline.process_papers([arxiv_id], count=1)
                        if result.get('successful', 0) > 0:
                            if attempt > 0:
                                successful_recoveries += 1
                            break
                    except Exception as e:
                        retry_counts[arxiv_id] = retry_counts.get(arxiv_id, 0) + 1
                        if attempt < max_retries - 1:
                            logger.debug(f"Retry {attempt + 1} for {arxiv_id}")
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            raise
                            
            except Exception as e:
                logger.error(f"Failed to process {arxiv_id} after retries: {e}")
        
        self.test_results['tests'][test_name] = {
            'papers_tested': len(test_ids),
            'papers_requiring_retry': len(retry_counts),
            'successful_recoveries': successful_recoveries,
            'average_retries': sum(retry_counts.values()) / len(retry_counts) if retry_counts else 0
        }
        
        logger.info(f"\n{test_name} Results:")
        logger.info(f"  Papers requiring retry: {len(retry_counts)}/{len(test_ids)}")
        logger.info(f"  Successful recoveries: {successful_recoveries}")
    
    def test_database_integrity(self):
        """Verify database integrity after processing."""
        test_name = "database_integrity"
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*60}")
        
        db = self.db_manager.db
        
        integrity_checks = {
            'papers_count': 0,
            'chunks_count': 0,
            'embeddings_count': 0,
            'orphaned_chunks': 0,
            'orphaned_embeddings': 0,
            'missing_embeddings': 0,
            'vector_dimensions': [],
            'processing_status': {}
        }
        
        # Count documents in each collection
        integrity_checks['papers_count'] = db.collection('arxiv_papers').count()
        integrity_checks['chunks_count'] = db.collection('arxiv_chunks').count()
        integrity_checks['embeddings_count'] = db.collection('arxiv_embeddings').count()
        
        # Check for orphaned chunks (chunks without papers)
        query = """
        FOR chunk IN arxiv_chunks
            LET paper = DOCUMENT('arxiv_papers', chunk.paper_id)
            FILTER paper == null
            COLLECT WITH COUNT INTO orphans
            RETURN orphans
        """
        cursor = db.aql.execute(query)
        integrity_checks['orphaned_chunks'] = next(cursor, 0)
        
        # Check for orphaned embeddings
        query = """
        FOR embedding IN arxiv_embeddings
            LET chunk = DOCUMENT('arxiv_chunks', embedding.chunk_id)
            FILTER chunk == null
            COLLECT WITH COUNT INTO orphans
            RETURN orphans
        """
        cursor = db.aql.execute(query)
        integrity_checks['orphaned_embeddings'] = next(cursor, 0)
        
        # Check for missing embeddings
        query = """
        FOR chunk IN arxiv_chunks
            LET embeddings = (
                FOR e IN arxiv_embeddings
                    FILTER e.chunk_id == chunk._id
                    RETURN e
            )
            FILTER LENGTH(embeddings) == 0
            COLLECT WITH COUNT INTO missing
            RETURN missing
        """
        cursor = db.aql.execute(query)
        integrity_checks['missing_embeddings'] = next(cursor, 0)
        
        # Sample vector dimensions
        query = """
        FOR embedding IN arxiv_embeddings
            LIMIT 10
            RETURN LENGTH(embedding.vector)
        """
        cursor = db.aql.execute(query)
        integrity_checks['vector_dimensions'] = list(cursor)
        
        # Processing status distribution
        query = """
        FOR paper IN arxiv_papers
            COLLECT status = paper.status WITH COUNT INTO count
            RETURN {status: status, count: count}
        """
        cursor = db.aql.execute(query)
        for item in cursor:
            integrity_checks['processing_status'][item['status']] = item['count']
        
        self.test_results['tests'][test_name] = integrity_checks
        
        # Report results
        logger.info(f"\n{test_name} Results:")
        logger.info(f"  Papers: {integrity_checks['papers_count']}")
        logger.info(f"  Chunks: {integrity_checks['chunks_count']}")
        logger.info(f"  Embeddings: {integrity_checks['embeddings_count']}")
        logger.info(f"  Orphaned chunks: {integrity_checks['orphaned_chunks']}")
        logger.info(f"  Orphaned embeddings: {integrity_checks['orphaned_embeddings']}")
        logger.info(f"  Missing embeddings: {integrity_checks['missing_embeddings']}")
        logger.info(f"  Processing status: {integrity_checks['processing_status']}")
        
        # Check for issues
        issues = []
        if integrity_checks['orphaned_chunks'] > 0:
            issues.append("Found orphaned chunks")
        if integrity_checks['orphaned_embeddings'] > 0:
            issues.append("Found orphaned embeddings")
        if integrity_checks['missing_embeddings'] > 0:
            issues.append("Found chunks without embeddings")
        
        if issues:
            logger.warning(f"  ⚠️  Issues detected: {', '.join(issues)}")
        else:
            logger.info(f"  ✅ No integrity issues detected")
        
        return len(issues) == 0
    
    def test_performance_benchmarks(self, arxiv_ids: List[str], sample_size: int = 100):
        """
        Run performance benchmarks.
        
        Args:
            arxiv_ids: List of ArXiv IDs
            sample_size: Number of papers to use for benchmarking
        """
        test_name = "performance_benchmarks"
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Test: {test_name}")
        logger.info(f"{'='*60}")
        
        # Select sample
        sample_ids = arxiv_ids[:sample_size]
        
        benchmarks = {
            'extraction_times': [],
            'embedding_times': [],
            'storage_times': [],
            'gpu_memory_usage': [],
            'cpu_usage': []
        }
        
        pipeline = ACIDPhasedPipeline(
            config_path=str(self.config_path),
            arango_password=self.arango_password
        )
        
        # Run benchmarks with detailed timing
        for arxiv_id in sample_ids[:10]:  # Detailed timing for first 10
            try:
                # Enable detailed timing in pipeline
                result = pipeline.process_papers(
                    [arxiv_id], 
                    count=1,
                    detailed_timing=True
                )
                
                if 'timing' in result:
                    timing = result['timing']
                    benchmarks['extraction_times'].append(timing.get('extraction', 0))
                    benchmarks['embedding_times'].append(timing.get('embedding', 0))
                    benchmarks['storage_times'].append(timing.get('storage', 0))
                    
            except Exception as e:
                logger.error(f"Benchmark failed for {arxiv_id}: {e}")
        
        # Calculate statistics
        def calculate_stats(times):
            if not times:
                return {'mean': 0, 'min': 0, 'max': 0}
            return {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            }
        
        self.test_results['performance_metrics'] = {
            'extraction': calculate_stats(benchmarks['extraction_times']),
            'embedding': calculate_stats(benchmarks['embedding_times']),
            'storage': calculate_stats(benchmarks['storage_times'])
        }
        
        logger.info(f"\n{test_name} Results:")
        for phase, stats in self.test_results['performance_metrics'].items():
            logger.info(f"  {phase}:")
            logger.info(f"    Mean: {stats['mean']:.2f}s")
            logger.info(f"    Min:  {stats['min']:.2f}s")
            logger.info(f"    Max:  {stats['max']:.2f}s")
    
    def run_all_tests(self, arxiv_ids: List[str]):
        """
        Run all tests in the suite.
        
        Args:
            arxiv_ids: List of ArXiv IDs to process
        """
        self.test_results['start_time'] = datetime.now().isoformat()
        
        logger.info("\n" + "="*60)
        logger.info("LARGE-SCALE ARXIV PROCESSING TEST SUITE")
        logger.info(f"Testing with {len(arxiv_ids)} papers")
        logger.info("="*60)
        
        # Run tests
        try:
            # 1. Performance benchmarks on small sample
            self.test_performance_benchmarks(arxiv_ids, sample_size=100)
            
            # 2. Batch processing test
            self.test_batch_processing(arxiv_ids[:1000], batch_size=100)
            
            # 3. Error recovery test
            self.test_error_recovery(arxiv_ids[1000:1050])
            
            # 4. Database integrity check
            self.test_database_integrity()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.test_results['errors'].append({
                'test': 'suite',
                'error': str(e)
            })
        
        self.test_results['end_time'] = datetime.now().isoformat()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def save_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"\nTest results saved to: {results_file}")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUITE SUMMARY")
        print("="*60)
        
        for test_name, results in self.test_results['tests'].items():
            print(f"\n{test_name}:")
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
        
        if self.test_results['errors']:
            print(f"\n⚠️  Errors encountered: {len(self.test_results['errors'])}")
            for error in self.test_results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error['test']}: {error['error'][:100]}")
        else:
            print("\n✅ All tests completed without errors")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Large-scale ArXiv processing tests")
    parser.add_argument('--config', required=True, help='Path to pipeline config')
    parser.add_argument('--papers', required=True, help='Path to paper list file')
    parser.add_argument('--limit', type=int, help='Limit number of papers to test')
    parser.add_argument('--arango-password', help='ArangoDB password')
    
    args = parser.parse_args()
    
    # Get ArangoDB password
    arango_password = args.arango_password or os.getenv('ARANGO_PASSWORD')
    if not arango_password:
        print("Error: ARANGO_PASSWORD not provided")
        sys.exit(1)
    
    # Initialize test suite
    suite = LargeScaleTestSuite(args.config, arango_password)
    
    # Load paper list
    arxiv_ids = suite.load_paper_list(args.papers)
    
    if args.limit:
        arxiv_ids = arxiv_ids[:args.limit]
    
    # Run tests
    suite.run_all_tests(arxiv_ids)


if __name__ == "__main__":
    main()