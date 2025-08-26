#!/usr/bin/env python3
"""
Conveyance Measurement System
============================

Implements the core Information Reconstructionism validation:
Measures conveyance improvement when LaTeX is included vs PDF-only.

This validates the theoretical framework: C = (W·R·H)/T · Ctx^α
where including LaTeX should increase Ctx (context) and thus conveyance.

Following Actor-Network Theory: This system measures the translation
quality between different actants (PDF vs LaTeX sources).
"""

import os
import sys
import logging
import psycopg2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from arango import ArangoClient

# Add HADES root to path
hades_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(hades_root))

from core.framework.embedders import JinaV4Embedder

logger = logging.getLogger(__name__)


@dataclass
class ConveyanceTestResult:
    """
    Result of a single conveyance measurement.
    
    In Actor-Network Theory terms, this represents the translation
    quality between different document representations.
    """
    arxiv_id: str
    pdf_conveyance: float
    latex_conveyance: float
    conveyance_delta: float
    test_queries: List[str]
    measurement_date: datetime
    pdf_chunks: int
    latex_chunks: int


class ConveyanceMeasurementSystem:
    """
    System for measuring conveyance improvement with dual embeddings.
    
    This implements the mathematical framework where conveyance is measured
    as the system's ability to retrieve relevant information given a query.
    
    The hypothesis: C_hybrid > C_pdf_only due to increased context (Ctx^α).
    """
    
    def __init__(self, pg_password: str, arango_password: str,
                 config_path: str = None):
        """
        Initialize the conveyance measurement system.
        
        Args:
            pg_password: PostgreSQL password
            arango_password: ArangoDB password
            config_path: Path to embedder config (optional)
        """
        self.pg_password = pg_password
        self.arango_password = arango_password
        
        # Initialize database connections
        self._init_postgresql()
        self._init_arangodb()
        
        # Initialize embedder for queries
        if config_path:
            self.embedder = JinaV4Embedder(config_path=config_path)
        else:
            # Use default config
            embedder_config = hades_root / 'configs' / 'embedder.yaml'
            self.embedder = JinaV4Embedder(config_path=str(embedder_config))
        
        logger.info("ConveyanceMeasurementSystem initialized")
    
    def _init_postgresql(self):
        """Initialize PostgreSQL connection."""
        try:
            self.pg_conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="Avernus",
                user="postgres",
                password=self.pg_password
            )
            self.pg_cur = self.pg_conn.cursor()
            logger.info("Connected to PostgreSQL for conveyance measurement")
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
    def _init_arangodb(self, host: str = None):
        """Initialize ArangoDB connection."""
        arango_host = host or os.getenv('ARANGO_HOST', 'http://192.168.1.69:8529')
        client = ArangoClient(hosts=arango_host)
    
    def _init_arangodb(self):
        """Initialize ArangoDB connection."""
        try:
            client = ArangoClient(hosts="http://192.168.1.69:8529")
            self.arango_db = client.db(
                "academy_store",
                username="root",
                password=self.arango_password
            )
            logger.info("Connected to ArangoDB for conveyance measurement")
        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            raise
    
    def get_papers_ready_for_measurement(self, limit: int = 100) -> List[Dict]:
        """
        Get papers that have both PDF and LaTeX embeddings but no conveyance measurement.
        
        Args:
            limit: Maximum number of papers to return
            
        Returns:
            List of paper metadata ready for measurement
        """
        query = """
            SELECT 
                p.id as arxiv_id,
                p.title,
                p.categories,
                p.pdf_chunk_count,
                p.latex_chunk_count,
                p.equation_count,
                p.table_count
            FROM arxiv_papers p
            WHERE p.pdf_embedded = true
              AND p.latex_embedded = true
              AND p.conveyance_delta IS NULL
              AND p.pdf_chunk_count > 0
              AND p.latex_chunk_count > 0
            ORDER BY p.equation_count DESC, p.table_count DESC
            LIMIT %s
        """
        
        self.pg_cur.execute(query, (limit,))
        
        papers = []
        for row in self.pg_cur.fetchall():
            papers.append({
                'arxiv_id': row[0],
                'title': row[1],
                'categories': row[2],
                'pdf_chunk_count': row[3],
                'latex_chunk_count': row[4],
                'equation_count': row[5] or 0,
                'table_count': row[6] or 0
            })
        
        return papers
    
    def generate_test_queries(self, paper: Dict) -> List[str]:
        """
        Generate relevant test queries for a paper.
        
        Args:
            paper: Paper metadata
            
        Returns:
            List of test queries designed to measure conveyance
        """
        title = paper['title']
        categories = paper.get('categories', '').split()[0] if paper.get('categories') else ''
        
        # Generate domain-specific queries based on category
        queries = []
        
        # General queries
        queries.extend([
            f"What is the main contribution of {title}?",
            f"How does {title} work?",
            "What are the key equations and formulas?",
            "What experimental results are reported?",
            "What are the limitations mentioned?"
        ])
        
        # Category-specific queries
        if 'cs.' in categories.lower():
            queries.extend([
                "What algorithm is proposed?",
                "What is the computational complexity?",
                "How is the system evaluated?"
            ])
        elif 'math.' in categories.lower():
            queries.extend([
                "What theorem is proven?",
                "What mathematical methods are used?",
                "What are the key definitions?"
            ])
        elif 'physics' in categories.lower():
            queries.extend([
                "What physical phenomenon is studied?",
                "What experimental setup is described?",
                "What theoretical model is proposed?"
            ])
        
        # Equation-specific queries if paper has equations
        if paper.get('equation_count', 0) > 0:
            queries.extend([
                "What are the governing equations?",
                "How are the equations derived?",
                "What do the mathematical symbols represent?"
            ])
        
        # Table-specific queries if paper has tables
        if paper.get('table_count', 0) > 0:
            queries.extend([
                "What results are shown in the tables?",
                "How do different methods compare?",
                "What are the numerical values reported?"
            ])
        
        return queries[:10]  # Limit to 10 queries to keep measurement tractable
    
    def measure_embedding_conveyance(self, arxiv_id: str, 
                                   test_queries: List[str],
                                   collection_name: str) -> float:
        """
        Measure conveyance for a specific embedding collection.
        
        Args:
            arxiv_id: Paper ID
            test_queries: List of queries to test
            collection_name: ArangoDB collection name
            
        Returns:
            Average conveyance score (0.0 to 1.0)
        """
        sanitized_id = arxiv_id.replace('/', '_').replace('.', '_')
        
        try:
            # Get the paper's embeddings
            paper_doc = self.arango_db.collection(collection_name).get(sanitized_id)
            if not paper_doc or not paper_doc.get('chunk_embeddings'):
                logger.warning(f"No embeddings found for {arxiv_id} in {collection_name}")
                return 0.0
            
            # Extract embeddings as numpy arrays
            paper_embeddings = []
            for chunk in paper_doc['chunk_embeddings']:
                embedding = np.array(chunk['embedding'])
                paper_embeddings.append(embedding)
            
            if not paper_embeddings:
                return 0.0
            
            paper_embeddings = np.array(paper_embeddings)
            
            # Generate query embeddings
            query_embeddings = []
            for query in test_queries:
                try:
                    # Embed the query
                    query_emb = self.embedder.embed_texts([query], task="retrieval")[0]
                    query_embeddings.append(query_emb)
                except Exception as e:
                    logger.warning(f"Failed to embed query '{query}': {e}")
                    continue
            
            if not query_embeddings:
                logger.warning(f"No valid query embeddings for {arxiv_id}")
                return 0.0
            
            query_embeddings = np.array(query_embeddings)
            
            # Calculate semantic similarity scores
            similarities = []
            for query_emb in query_embeddings:
                # Calculate cosine similarity with all paper chunks
                chunk_similarities = []
                for paper_emb in paper_embeddings:
                    query_norm = np.linalg.norm(query_emb)
                    paper_norm = np.linalg.norm(paper_emb)
                    if query_norm == 0 or paper_norm == 0:
                        similarity = 0.0
                    else:
                        similarity = np.dot(query_emb, paper_emb) / (query_norm * paper_norm)
                    chunk_similarities.append(similarity)
                
                # Use max similarity as the paper's response to this query
                max_similarity = max(chunk_similarities) if chunk_similarities else 0.0
                similarities.append(max_similarity)
            
            # Average similarity across all queries
            avg_conveyance = np.mean(similarities) if similarities else 0.0
            
            logger.debug(f"Conveyance for {arxiv_id} in {collection_name}: {avg_conveyance:.3f}")
            return float(avg_conveyance)
            
        except Exception as e:
            logger.error(f"Error measuring conveyance for {arxiv_id} in {collection_name}: {e}")
            return 0.0
    
    def measure_paper_conveyance(self, paper: Dict) -> Optional[ConveyanceTestResult]:
        """
        Measure conveyance for a single paper using both PDF and LaTeX embeddings.
        
        Args:
            paper: Paper metadata
            
        Returns:
            ConveyanceTestResult or None if measurement failed
        """
        arxiv_id = paper['arxiv_id']
        logger.info(f"Measuring conveyance for {arxiv_id}")
        
        try:
            # Generate test queries
            test_queries = self.generate_test_queries(paper)
            
            # Measure PDF-only conveyance
            pdf_conveyance = self.measure_embedding_conveyance(
                arxiv_id, test_queries, 'arxiv_pdf_embeddings'
            )
            
            # Measure LaTeX conveyance  
            latex_conveyance = self.measure_embedding_conveyance(
                arxiv_id, test_queries, 'arxiv_latex_embeddings'
            )
            
            # Calculate improvement delta
            if pdf_conveyance > 0:
                conveyance_delta = (latex_conveyance - pdf_conveyance) / pdf_conveyance
            else:
                conveyance_delta = 0.0
            
            result = ConveyanceTestResult(
                arxiv_id=arxiv_id,
                pdf_conveyance=pdf_conveyance,
                latex_conveyance=latex_conveyance,
                conveyance_delta=conveyance_delta,
                test_queries=test_queries,
                measurement_date=datetime.now(),
                pdf_chunks=paper.get('pdf_chunk_count', 0),
                latex_chunks=paper.get('latex_chunk_count', 0)
            )
            
            logger.info(f"✓ Measured {arxiv_id}: PDF={pdf_conveyance:.3f}, "
                       f"LaTeX={latex_conveyance:.3f}, Δ={conveyance_delta:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to measure conveyance for {arxiv_id}: {e}")
            return None
    
    def store_measurement_result(self, result: ConveyanceTestResult):
        """
        Store conveyance measurement result in PostgreSQL.
        
        Args:
            result: ConveyanceTestResult to store
        """
        try:
            update_query = """
                UPDATE arxiv_papers 
                SET 
                    conveyance_pdf_only = %s,
                    conveyance_with_latex = %s,
                    conveyance_delta = %s,
                    conveyance_measured_date = %s
                WHERE id = %s
            """
            
            self.pg_cur.execute(
                update_query,
                (
                    result.pdf_conveyance,
                    result.latex_conveyance,
                    result.conveyance_delta,
                    result.measurement_date,
                    result.arxiv_id
                )
            )
            self.pg_conn.commit()
            
            logger.debug(f"Stored conveyance measurement for {result.arxiv_id}")
            
        except Exception as e:
            logger.error(f"Failed to store measurement for {result.arxiv_id}: {e}")
            self.pg_conn.rollback()
    
    def run_measurement_batch(self, batch_size: int = 10) -> Dict:
        """
        Run conveyance measurements for a batch of papers.
        
        Args:
            batch_size: Number of papers to measure
            
        Returns:
            Dictionary with measurement statistics
        """
        logger.info("="*60)
        logger.info("CONVEYANCE MEASUREMENT SYSTEM")
        logger.info("="*60)
        
        # Get papers ready for measurement
        papers = self.get_papers_ready_for_measurement(batch_size)
        
        if not papers:
            logger.info("No papers ready for conveyance measurement")
            return {'measured': 0, 'failed': 0, 'improvement_samples': []}
        
        logger.info(f"Measuring conveyance for {len(papers)} papers")
        
        stats = {
            'measured': 0,
            'failed': 0,
            'improvement_samples': [],
            'start_time': datetime.now()
        }
        
        for paper in papers:
            result = self.measure_paper_conveyance(paper)
            
            if result:
                # Store result
                self.store_measurement_result(result)
                
                # Update stats
                stats['measured'] += 1
                stats['improvement_samples'].append(result.conveyance_delta)
                
            else:
                stats['failed'] += 1
        
        # Calculate final statistics
        stats['end_time'] = datetime.now()
        stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
        
        if stats['improvement_samples']:
            stats['avg_improvement'] = np.mean(stats['improvement_samples'])
            stats['std_improvement'] = np.std(stats['improvement_samples'])
            stats['positive_improvements'] = sum(1 for x in stats['improvement_samples'] if x > 0)
        
        return stats
    
    def get_measurement_summary(self) -> Dict:
        """
        Get summary statistics of all conveyance measurements.
        
        Returns:
            Dictionary with summary statistics
        """
        query = """
            SELECT 
                COUNT(*) as total_measured,
                AVG(conveyance_pdf_only) as avg_pdf_conveyance,
                AVG(conveyance_with_latex) as avg_latex_conveyance,
                AVG(conveyance_delta) as avg_improvement,
                STDDEV(conveyance_delta) as std_improvement,
                COUNT(CASE WHEN conveyance_delta > 0 THEN 1 END) as positive_improvements,
                COUNT(CASE WHEN conveyance_delta > 0.1 THEN 1 END) as significant_improvements,
                MAX(conveyance_delta) as max_improvement,
                MIN(conveyance_delta) as min_improvement
            FROM arxiv_papers
            WHERE conveyance_measured_date IS NOT NULL
        """
        
        self.pg_cur.execute(query)
        row = self.pg_cur.fetchone()
        
        if row and row[0] > 0:
            return {
                'total_measured': row[0],
                'avg_pdf_conveyance': float(row[1]) if row[1] else 0.0,
                'avg_latex_conveyance': float(row[2]) if row[2] else 0.0,
                'avg_improvement': float(row[3]) if row[3] else 0.0,
                'std_improvement': float(row[4]) if row[4] else 0.0,
                'positive_improvements': row[5],
                'significant_improvements': row[6],
                'max_improvement': float(row[7]) if row[7] else 0.0,
                'min_improvement': float(row[8]) if row[8] else 0.0
            }
        else:
            return {'total_measured': 0}


def main():
    """Main entry point for conveyance measurement."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Conveyance Measurement System')
    parser.add_argument('--pg-password', required=True, help='PostgreSQL password')
    parser.add_argument('--arango-password', required=True, help='ArangoDB password')
    parser.add_argument('--batch-size', type=int, default=10, help='Papers per batch')
    parser.add_argument('--embedder-config', help='Path to embedder config')
    parser.add_argument('--summary-only', action='store_true', 
                       help='Show summary without running new measurements')
    
    args = parser.parse_args()
    
    system = None
    try:
        # Initialize measurement system
        system = ConveyanceMeasurementSystem(
            pg_password=args.pg_password,
            arango_password=args.arango_password,
            config_path=args.embedder_config
        )
        
        if args.summary_only:
            # Show summary only
            summary = system.get_measurement_summary()
            print("\n" + "="*60)
            print("CONVEYANCE MEASUREMENT SUMMARY")
            print("="*60)
            
            if summary.get('total_measured', 0) > 0:
                print(f"Total papers measured: {summary['total_measured']:,}")
                print(f"Average PDF conveyance: {summary['avg_pdf_conveyance']:.3f}")
                print(f"Average LaTeX conveyance: {summary['avg_latex_conveyance']:.3f}")
                print(f"Average improvement: {summary['avg_improvement']:.3f} ± {summary['std_improvement']:.3f}")
                print(f"Papers with improvement: {summary['positive_improvements']}/{summary['total_measured']}")
                print(f"Significant improvements (>10%): {summary['significant_improvements']}")
                print(f"Best improvement: {summary['max_improvement']:.3f}")
                print(f"Worst improvement: {summary['min_improvement']:.3f}")
            else:
                print("No measurements found")
        else:
            # Run measurements
            stats = system.run_measurement_batch(args.batch_size)
            
            # Print results
            print("\n" + "="*60)
            print("MEASUREMENT RESULTS")
            print("="*60)
            print(f"Measured: {stats['measured']} papers")
            print(f"Failed: {stats['failed']} papers")
            print(f"Duration: {stats['duration']:.1f} seconds")
            
            if stats.get('improvement_samples'):
                print(f"Average improvement: {stats['avg_improvement']:.3f} ± {stats['std_improvement']:.3f}")
                print(f"Papers with positive improvement: {stats['positive_improvements']}/{stats['measured']}")
    
    finally:
        # Clean up database connections
        if system and hasattr(system, 'pg_conn'):
            try:
                system.pg_conn.close()
                logger.info("Closed PostgreSQL connection")
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connection: {e}")
        
        if system and hasattr(system, 'pg_cur'):
            try:
                system.pg_cur.close()
                logger.info("Closed PostgreSQL cursor")
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL cursor: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()