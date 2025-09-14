#!/usr/bin/env python3
"""
End-to-End Ingestion Pipeline for Core Restructure Validation
==============================================================

A clean-room implementation using only the restructured core modules.
This script validates that the Phase 1-4 restructure works correctly
by processing papers from PostgreSQL through to ArangoDB.

Theory Connection:
Implements C = (W·R·H/T)·Ctx^α where:
- W (What): Quality of extracted content and embeddings
- R (Where): Direct filesystem access and graph storage
- H (Who): Processing capability (GPU acceleration, worker pools)
- T (Time): End-to-end processing latency
- Ctx: Context preservation through late chunking
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from restructured core modules ONLY
from core.embedders import JinaV4Embedder, EmbedderFactory
from core.extractors import DoclingExtractor, ExtractorFactory
from core.database import DatabaseFactory
from core.config import ConfigManager, BaseConfig
from core.monitoring import PerformanceMonitor, ProgressTracker
# No need for ArangoStorage - we'll use ArangoDBManager directly

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PaperTask:
    """Task for processing a single paper."""
    arxiv_id: str
    pdf_path: Optional[str] = None
    latex_path: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    categories: Optional[str] = None
    authors: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of processing a single paper."""
    arxiv_id: str
    success: bool
    chunks_created: int = 0
    embeddings_created: int = 0
    equations_extracted: int = 0
    tables_extracted: int = 0
    error: Optional[str] = None
    processing_time: float = 0.0


class E2EIngestionPipeline:
    """
    End-to-end ingestion pipeline using restructured core modules.

    This implementation validates the core restructure by processing
    papers through the complete pipeline using only the new module structure.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config

        # Initialize monitoring
        self.monitor = PerformanceMonitor(
            component_name='E2E_Ingestion',
            log_dir=Path('logs')
        )

        # Initialize database connections
        self._init_databases()

        # Initialize extractors and embedders
        self._init_processors()

        # Statistics
        self.stats = {
            'total_papers': 0,
            'successful': 0,
            'failed': 0,
            'total_chunks': 0,
            'total_embeddings': 0,
            'start_time': datetime.now()
        }

    def _init_databases(self):
        """Initialize database connections using DatabaseFactory."""
        # PostgreSQL for metadata
        self.pg_conn = DatabaseFactory.get_postgres(
            database='arxiv',
            host=self.config.get('pg_host', 'localhost'),
            username=self.config.get('pg_user', 'todd'),
            password='olympus123'
        )

        # ArangoDB for storage - use existing ArangoDBManager
        from core.database.arango import ArangoDBManager

        arango_config = {
            'database': 'academy_store',
            'host': f"http://{self.config.get('arango_host', 'localhost')}:8529",
            'username': self.config.get('arango_user', 'root'),
            'password': os.environ.get('ARANGO_PASSWORD')
        }
        self.arango_manager = ArangoDBManager(arango_config, pool_size=4)
        self.arango_db = self.arango_manager.db

        logger.info("✓ Database connections established")

    def _init_processors(self):
        """Initialize extractors and embedders using factories."""
        # Create extractor
        self.extractor = ExtractorFactory.create(
            'docling',
            ocr_enabled=self.config.get('use_ocr', False),
            extract_tables=self.config.get('extract_tables', True)
        )

        # Create embedder
        self.embedder = EmbedderFactory.create(
            'jina_v4',
            device='cuda' if self.config.get('use_gpu', True) else 'cpu',
            batch_size=self.config.get('batch_size', 24),
            use_fp16=self.config.get('use_fp16', True)
        )

        logger.info("✓ Processors initialized")

    def get_papers_to_process(self, limit: int = 100) -> List[PaperTask]:
        """
        Get papers from PostgreSQL that need processing.

        Args:
            limit: Maximum number of papers to retrieve

        Returns:
            List of PaperTask objects
        """
        query = """
            SELECT
                p.arxiv_id,
                p.title,
                p.abstract,
                p.categories,
                p.authors,
                p.pdf_path,
                p.latex_path
            FROM papers p
            WHERE p.has_pdf = true
            AND NOT EXISTS (
                SELECT 1 FROM processed_papers pp
                WHERE pp.arxiv_id = p.arxiv_id
            )
            ORDER BY p.arxiv_id
            LIMIT %s
        """

        tasks = []
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (limit,))

            for row in cursor.fetchall():
                # Build full path for PDF
                if row['pdf_path']:
                    pdf_path = f"/bulk-store/arxiv-data/pdf/{row['pdf_path']}"
                else:
                    continue  # Skip if no PDF

                # Build full path for LaTeX if available
                latex_path = None
                if row['latex_path']:
                    latex_path = f"/bulk-store/arxiv-data/latex/{row['latex_path']}"

                task = PaperTask(
                    arxiv_id=row['arxiv_id'],
                    pdf_path=pdf_path,
                    latex_path=latex_path,
                    title=row['title'],
                    abstract=row['abstract'],
                    categories=row['categories'],
                    authors=row['authors']
                )
                tasks.append(task)

        logger.info(f"Retrieved {len(tasks)} papers to process")
        return tasks

    def process_paper(self, task: PaperTask) -> ProcessingResult:
        """
        Process a single paper through extraction and embedding.

        Args:
            task: Paper processing task

        Returns:
            ProcessingResult with status and metrics
        """
        start_time = time.time()
        result = ProcessingResult(arxiv_id=task.arxiv_id, success=False)

        try:
            # Step 1: Extract content from PDF
            extraction = self.extractor.extract(task.pdf_path)
            if not extraction or not extraction.get('full_text'):
                raise ValueError("No text extracted from PDF")

            # Step 2: Generate embeddings with late chunking
            chunks_with_embeddings = self.embedder.embed_with_late_chunking(
                extraction['full_text']
            )

            # Step 3: Store in ArangoDB
            paper_doc = {
                '_key': task.arxiv_id.replace('/', '_').replace('.', '_'),
                'arxiv_id': task.arxiv_id,
                'title': task.title,
                'abstract': task.abstract,
                'categories': task.categories,
                'authors': task.authors,
                'status': 'PROCESSED',
                'processing_date': datetime.now().isoformat(),
                'num_chunks': len(chunks_with_embeddings),
                'has_latex': task.latex_path is not None
            }

            # Store paper document
            self.arango_manager.insert_document('arxiv_papers', paper_doc)

            # Store chunks and embeddings
            for idx, chunk_data in enumerate(chunks_with_embeddings):
                chunk_doc = {
                    '_key': f"{paper_doc['_key']}_chunk_{idx}",
                    'paper_id': task.arxiv_id,
                    'chunk_index': idx,
                    'text': chunk_data.text,
                    'chunk_start': chunk_data.start_char,
                    'chunk_end': chunk_data.end_char
                }
                chunk_result = self.arango_manager.insert_document('arxiv_chunks', chunk_doc)

                # Store embedding
                embedding_doc = {
                    '_key': f"{paper_doc['_key']}_emb_{idx}",
                    'paper_id': task.arxiv_id,
                    'chunk_id': chunk_result['_id'],
                    'vector': chunk_data.embedding.tolist(),
                    'model': 'jina-v4',
                    'dimension': len(chunk_data.embedding)
                }
                self.arango_manager.insert_document('arxiv_embeddings', embedding_doc)

            # Extract and store structures (equations, tables)
            structures = extraction.get('structures', {})

            # Store equations if present
            equations = structures.get('equations', [])
            for i, eq in enumerate(equations):
                eq_doc = {
                    '_key': f"{paper_doc['_key']}_eq_{i}",
                    'paper_id': task.arxiv_id,
                    'latex': eq.get('latex', ''),
                    'label': eq.get('label'),
                    'type': eq.get('type', 'display')
                }
                self.arango_manager.insert_document('arxiv_equations', eq_doc)

            # Store tables if present
            tables = structures.get('tables', [])
            for i, table in enumerate(tables):
                table_doc = {
                    '_key': f"{paper_doc['_key']}_table_{i}",
                    'paper_id': task.arxiv_id,
                    'caption': table.get('caption', ''),
                    'content': table.get('content', ''),
                    'label': table.get('label')
                }
                self.arango_manager.insert_document('arxiv_tables', table_doc)

            # Update result
            result.success = True
            result.chunks_created = len(chunks_with_embeddings)
            result.embeddings_created = len(chunks_with_embeddings)
            result.equations_extracted = len(equations)
            result.tables_extracted = len(tables)

            # Mark as processed in PostgreSQL
            self._mark_processed(task.arxiv_id, success=True)

        except Exception as e:
            logger.error(f"Failed to process {task.arxiv_id}: {e}")
            result.error = str(e)
            self._mark_processed(task.arxiv_id, success=False, error=str(e))

        result.processing_time = time.time() - start_time
        return result

    def _mark_processed(self, arxiv_id: str, success: bool, error: Optional[str] = None):
        """Mark a paper as processed in PostgreSQL."""
        with self.pg_conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO processed_papers (arxiv_id, success, error, processed_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (arxiv_id)
                DO UPDATE SET
                    success = EXCLUDED.success,
                    error = EXCLUDED.error,
                    processed_at = EXCLUDED.processed_at
            """, (arxiv_id, success, error))
            self.pg_conn.commit()

    def run(self, limit: int = 100, num_workers: int = 4):
        """
        Run the E2E ingestion pipeline.

        Args:
            limit: Maximum number of papers to process
            num_workers: Number of parallel workers
        """
        logger.info(f"Starting E2E ingestion pipeline with {num_workers} workers")

        # Get papers to process
        tasks = self.get_papers_to_process(limit)
        if not tasks:
            logger.info("No papers to process")
            return

        # Create progress tracker
        tracker = ProgressTracker(
            name="E2E_Ingestion",
            description="Processing papers"
        )

        # Add a step for the processing
        tracker.add_step("papers", "Processing papers", len(tasks))
        tracker.start_step("papers")

        # Process papers in parallel
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process_paper, task): task
                      for task in tasks}

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Update statistics
                    self.stats['total_papers'] += 1
                    if result.success:
                        self.stats['successful'] += 1
                        self.stats['total_chunks'] += result.chunks_created
                        self.stats['total_embeddings'] += result.embeddings_created
                    else:
                        self.stats['failed'] += 1

                    # Update progress
                    tracker.update_step("papers", completed=1)

                except Exception as e:
                    logger.error(f"Worker failed for {task.arxiv_id}: {e}")
                    self.stats['failed'] += 1

        # Complete tracking
        tracker.complete_step("papers")

        # Print final statistics
        self._print_statistics()

    def _print_statistics(self):
        """Print processing statistics."""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()

        logger.info("\n" + "="*60)
        logger.info("E2E INGESTION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total papers: {self.stats['total_papers']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total chunks: {self.stats['total_chunks']}")
        logger.info(f"Total embeddings: {self.stats['total_embeddings']}")
        logger.info(f"Duration: {duration:.2f} seconds")

        if self.stats['successful'] > 0:
            rate = self.stats['successful'] / (duration / 60)
            logger.info(f"Processing rate: {rate:.2f} papers/minute")

        success_rate = (self.stats['successful'] / self.stats['total_papers'] * 100
                       if self.stats['total_papers'] > 0 else 0)
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info("="*60)


def main():
    """Main entry point for the E2E ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="E2E Ingestion Pipeline for Core Restructure Validation"
    )
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of papers to process'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--pg-host',
        default='localhost',
        help='PostgreSQL host'
    )
    parser.add_argument(
        '--arango-host',
        default='localhost',
        help='ArangoDB host'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        default=True,
        help='Use GPU for processing'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=24,
        help='Batch size for embeddings'
    )

    args = parser.parse_args()

    # Build configuration
    config = {
        'pg_host': args.pg_host,
        'pg_user': os.environ.get('PGUSER', 'todd'),
        'arango_host': args.arango_host,
        'arango_user': 'root',
        'use_gpu': args.use_gpu,
        'batch_size': args.batch_size,
        'use_fp16': True,
        'use_ocr': False,
        'extract_tables': True
    }

    # Ensure we have ArangoDB password
    if 'ARANGO_PASSWORD' not in os.environ:
        logger.error("ARANGO_PASSWORD environment variable not set")
        sys.exit(1)

    # Create and run pipeline
    pipeline = E2EIngestionPipeline(config)

    try:
        pipeline.run(limit=args.count, num_workers=args.workers)
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()