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
        Run a targeted error-recovery test over a small subset of ArXiv IDs and record retry metrics.
        
        This test processes up to the first 50 IDs from `arxiv_ids`, exercising the pipeline's retry behavior with up to three attempts per paper and exponential backoff between attempts. A per-paper run is considered a successful recovery if the pipeline result reports a success rate >= 80%. Metrics recorded in self.test_results['tests']['error_recovery'] include:
        - papers_tested
        - papers_requiring_retry
        - successful_recoveries (recoveries that succeeded after at least one retry)
        - average_retries
        
        Side effects:
        - Invokes ACIDPhasedPipeline.process_papers for each tested paper.
        - Logs progress and errors.
        - Updates self.test_results with the aggregated metrics.
        
        Parameters:
            arxiv_ids (List[str]): Sequence of ArXiv IDs; only the first 50 are used for this test.
            inject_errors (bool): Flag intended to enable artificial error injection; currently unused by this implementation.
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
                        # Consider it successful if all papers processed or above threshold
                        success_rate = result.get('successful', 0) / max(1, result.get('total', 1))
                        if success_rate >= 0.8:  # 80% success threshold
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
            self.test_performance_benchmarks(arxiv_ids, sample_size=100)
            
            # 2. Batch processing test
            batch_size_limit = min(1000, len(arxiv_ids))
            self.test_batch_processing(arxiv_ids[:batch_size_limit], batch_size=100)
            
            # 3. Error recovery test
            error_test_start = min(1000, len(arxiv_ids))
            error_test_end = min(1050, len(arxiv_ids))
            if error_test_start < error_test_end:
                self.test_error_recovery(arxiv_ids[error_test_start:error_test_end])
            
            self.test_error_recovery(arxiv_ids[1000:1050])
            
            # 4. Database integrity check
            self.test_database_integrity()
            
        except Exception as e:
            """
        Orchestrate and run the full suite of tests against the provided ArXiv paper IDs.
        
        This records the suite start time, logs a header, and runs the configured sequence of tests:
        performance benchmarks, batch processing (capped at 1,000 papers), two error-recovery ranges (1000–1050 and a conditional slice), and a database integrity check.
        Any exceptions raised during the sequence are caught and recorded in self.test_results['errors'].
        
        Parameters:
            arxiv_ids (List[str]): Ordered list of ArXiv IDs to use for the suite; length and ordering determine which subsets are used for batched and error-recovery tests.
        
        Returns:
            None
        """
        logger.error(f"Test suite failed: {e}")
            self.test_results['errors'].append({
                'test': 'suite',
                'error': str(e)
            })
        
        self.test_results['end_time'] = datetime.now().isoformat()

        # Compute HADES Conveyance metrics
        self.compute_hades_metrics()

        # Save results
        self.save_results()

        # Print summary
        self.print_summary()
    
    def compute_hades_metrics(self):
        """
        Compute the HADES Conveyance score from accumulated test results and store a detailed scorecard in self.test_results['hades'].
        
        This method derives the Conveyance metric C using the formula:
            C = (W · R · H / T) · Ctx^α
        
        Where:
        - W (What): content/signal quality derived from batch processing success rates (falls back to overall success rate or a default). Values are averaged and clamped to [0.01, 1.0].
        - R (Where): relational/topological integrity derived from database integrity issue counts (orphaned chunks, missing embeddings, inconsistent status); mapped and clamped to [0.01, 1.0].
        - H (Who): agent capability derived from error recovery results (successful_recoveries / papers_tested), clamped to [0.1, 1.0].
        - T (Time): latency/cost derived from performance benchmarks (mean_total_time or converted from overall_rate); constrained to be > 0.
        - Ctx: combined contextual quality computed from config context fields (local_coherence, instruction_fit, actionability, grounding) and context_weights; clamped to [0.01, 1.0].
        - α (alpha): exponent read from config with bounds [1.5, 2.0].
        
        Effects:
        - Computes W, R, H, T, Ctx, and alpha using values found in self.test_results and self.config (with sensible defaults).
        - Calculates Conveyance C and stores a structured scorecard under self.test_results['hades'] containing the score, formula, components, context breakdown, and a human-readable interpretation (including a performance class).
        - Emits an informational scorecard to the module logger.
        
        No value is returned; the result is persisted to self.test_results['hades'] and logged.
        """
        logger.info("\nComputing HADES Conveyance Metrics...")

        # W = What (signal/content quality) - derive from success rates
        W = 0.0
        success_count = 0

        # Check batch processing tests for success rates
        for size in [10, 50, 100]:
            test_key = f'batch_processing_{size}'
            if test_key in self.test_results['tests']:
                test_data = self.test_results['tests'][test_key]
                if 'success_rate' in test_data:
                    W += test_data['success_rate'] / 100.0  # Convert percentage to 0-1
                    success_count += 1

        # If no batch tests, check overall processing
        if success_count == 0 and 'overall_success_rate' in self.test_results.get('summary', {}):
            W = self.test_results['summary']['overall_success_rate'] / 100.0
            success_count = 1

        W = W / success_count if success_count > 0 else 0.5  # Average or default to 0.5
        W = max(0.01, min(1.0, W))  # Constrain to (0, 1]

        # R = Where (relational positioning) - derive from database integrity
        R = 1.0  # Default to perfect
        if 'database_integrity' in self.test_results['tests']:
            integrity = self.test_results['tests']['database_integrity']
            total_issues = 0

            if 'orphaned_chunks' in integrity:
                total_issues += integrity['orphaned_chunks']
            if 'missing_embeddings' in integrity:
                total_issues += integrity['missing_embeddings']
            if 'inconsistent_status' in integrity:
                total_issues += integrity['inconsistent_status']

            # Map issues to score: 0 issues = 1.0, more issues = lower score
            R = 1.0 / (1.0 + total_issues * 0.1) if total_issues > 0 else 1.0

        R = max(0.01, min(1.0, R))

        # H = Who (agent capability) - derive from error recovery
        H = 0.5  # Default capability
        if 'error_recovery' in self.test_results['tests']:
            recovery = self.test_results['tests']['error_recovery']
            if 'successful_recoveries' in recovery and 'papers_tested' in recovery:
                if recovery['papers_tested'] > 0:
                    H = recovery['successful_recoveries'] / recovery['papers_tested']

        H = max(0.1, min(1.0, H))

        # T = Time (latency/cost) - derive from performance benchmarks
        T = 1.0  # Default to 1 second
        if 'performance_benchmarks' in self.test_results['tests']:
            perf = self.test_results['tests']['performance_benchmarks']
            if 'mean_total_time' in perf:
                T = perf['mean_total_time']
            elif 'overall_rate' in perf and perf['overall_rate'] > 0:
                T = 60.0 / perf['overall_rate']  # Convert papers/min to seconds/paper

        T = max(0.01, T)  # Ensure T > 0

        # Context components (using config values or defaults)
        L = self.config.get('context', {}).get('local_coherence', 0.8)
        I = self.config.get('context', {}).get('instruction_fit', 0.7)
        A = self.config.get('context', {}).get('actionability', 0.6)
        G = self.config.get('context', {}).get('grounding', 0.9)

        # Context weights (default to equal weighting)
        wL = self.config.get('context_weights', {}).get('wL', 0.25)
        wI = self.config.get('context_weights', {}).get('wI', 0.25)
        wA = self.config.get('context_weights', {}).get('wA', 0.25)
        wG = self.config.get('context_weights', {}).get('wG', 0.25)

        # Compute combined context
        Ctx = wL * L + wI * I + wA * A + wG * G
        Ctx = max(0.01, min(1.0, Ctx))

        # Alpha exponent (from config or default)
        alpha = self.config.get('alpha', 1.75)
        alpha = max(1.5, min(2.0, alpha))

        # Compute Conveyance (efficiency view)
        # C = (W * R * H / T) * Ctx^α
        if W > 0 and R > 0 and H > 0 and T > 0:
            C = (W * R * H / T) * (Ctx ** alpha)
        else:
            C = 0.0  # Zero-propagation gate

        # Store detailed scorecard
        self.test_results['hades'] = {
            'conveyance_score': C,
            'formula': 'C = (W·R·H/T)·Ctx^α',
            'components': {
                'W_what_quality': W,
                'R_where_topology': R,
                'H_who_capability': H,
                'T_time_latency': T
            },
            'context': {
                'L_local_coherence': {'value': L, 'weight': wL},
                'I_instruction_fit': {'value': I, 'weight': wI},
                'A_actionability': {'value': A, 'weight': wA},
                'G_grounding': {'value': G, 'weight': wG},
                'combined_Ctx': Ctx,
                'alpha_exponent': alpha
            },
            'interpretation': {
                'efficiency_view': True,
                'zero_propagation': C == 0.0,
                'performance_class': 'high' if C > 1.0 else 'medium' if C > 0.5 else 'low' if C > 0 else 'failed'
            }
        }

        # Log the scorecard
        logger.info("\n" + "="*60)
        logger.info("HADES CONVEYANCE SCORECARD")
        logger.info("="*60)
        logger.info(f"Component Values:")
        logger.info(f"  W (What/Quality):     {W:.3f}")
        logger.info(f"  R (Where/Topology):   {R:.3f}")
        logger.info(f"  H (Who/Capability):   {H:.3f}")
        logger.info(f"  T (Time/Latency):     {T:.3f}s")
        logger.info(f"Context Factors:")
        logger.info(f"  L (Local Coherence):  {L:.3f} (weight: {wL:.2f})")
        logger.info(f"  I (Instruction Fit):  {I:.3f} (weight: {wI:.2f})")
        logger.info(f"  A (Actionability):    {A:.3f} (weight: {wA:.2f})")
        logger.info(f"  G (Grounding):        {G:.3f} (weight: {wG:.2f})")
        logger.info(f"  Ctx (Combined):       {Ctx:.3f}")
        logger.info(f"  α (Exponent):         {alpha:.2f}")
        logger.info("─" * 40)
        logger.info(f"Conveyance Score (C): {C:.4f}")
        logger.info(f"Performance Class:    {self.test_results['hades']['interpretation']['performance_class'].upper()}")
        logger.info("="*60)

    def save_results(self):
        """
        Persist the collected test results to a timestamped JSON file in the current working directory.
        
        The file is written as pretty-printed JSON with an indent of 2 and named using the pattern
        `test_results_YYYYmmdd_HHMMSS.json` based on the current local time. This function has the
        side effect of creating or overwriting the file and logging the saved file path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)

        logger.info(f"\nTest results saved to: {results_file}")
    
    def print_summary(self):
        """
        Print a concise, human-readable summary of the collected test results to stdout.
        
        Displays a header followed by each recorded test name and its result fields. Numeric (float) values are formatted to two decimal places. If any errors were recorded in self.test_results['errors'], prints a count and up to the first five error entries with each error's test name and a truncated error message (first 100 characters). Otherwise prints a success message.
        """
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