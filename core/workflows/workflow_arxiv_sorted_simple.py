#!/usr/bin/env python3
"""
ArXiv Size-Sorted Processing Workflow (Simplified)
==================================================

Minimal implementation that processes ArXiv abstracts sorted by size
for maximum GPU throughput.

Core flow:
1. Load metadata from JSON (if needed)
2. Query unprocessed records sorted by abstract length
3. Process with GPU workers
4. Store embeddings and chunks
"""

import os
import json
import logging
import multiprocessing as mp
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from queue import Empty
import threading

# Core imports
from core.database.database_factory import DatabaseFactory
from core.embedders.embedders_factory import EmbedderFactory

logger = logging.getLogger(__name__)


def worker_process_with_gpu(worker_id: int, gpu_id: int, model_name: str,
                           input_queue: mp.Queue, output_queue: mp.Queue, stop_event: Any,
                           embedding_batch_size: int = 48, chunk_size_tokens: int = 500,
                           chunk_overlap_tokens: int = 200) -> None:
    """GPU worker process - sets CUDA device before imports.

    Args:
        worker_id: Worker identifier
        gpu_id: GPU device ID
        model_name: Model name for embedder
        input_queue: Queue for input batches
        output_queue: Queue for results
        stop_event: Event to signal shutdown
        embedding_batch_size: Batch size for embedding
        chunk_size_tokens: Chunk size in tokens
        chunk_overlap_tokens: Overlap between chunks
    """
    # Set GPU BEFORE any CUDA imports
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    print(f"Worker {worker_id} started on GPU {gpu_id}, PID {os.getpid()}")

    # Import after GPU is set
    from core.embedders.embedders_factory import EmbedderFactory

    # Initialize batches_processed BEFORE try block to avoid UnboundLocalError
    batches_processed = 0

    try:
        # Initialize embedder with passed configuration
        embedder_config = {
            'device': 'cuda:0',  # Always 0 since we isolated GPU
            'use_fp16': True,
            'batch_size': embedding_batch_size,  # Use passed parameter
            'chunk_size_tokens': chunk_size_tokens,  # Use passed parameter
            'chunk_overlap_tokens': chunk_overlap_tokens  # Use passed parameter
        }

        embedder = EmbedderFactory.create(
            model_name=model_name,
            **embedder_config
        )

        print(f"Worker {worker_id} ready with batch_size={embedding_batch_size}, "
              f"chunk_size={chunk_size_tokens}, overlap={chunk_overlap_tokens}")

        # Process batches
        while not stop_event.is_set():
            try:
                batch = input_queue.get(timeout=1.0)

                if batch is None:  # Poison pill
                    break

                # Extract abstracts
                abstracts = []
                records = []
                for record in batch:
                    if record.get('abstract'):
                        abstracts.append(record['abstract'])
                        records.append(record)

                if abstracts:
                    # Generate embeddings with late chunking
                    all_chunks = embedder.embed_batch_with_late_chunking(
                        abstracts,
                        task="retrieval.passage"
                    )

                    # Package results
                    results = []
                    for record, chunks in zip(records, all_chunks):
                        for chunk in chunks:
                            results.append({
                                'record': record,
                                'chunk': chunk
                            })

                    output_queue.put(results)
                    batches_processed += 1

                    if batches_processed % 10 == 0:
                        print(f"Worker {worker_id}: {batches_processed} batches")

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    except Exception as e:
        logger.error(f"Worker {worker_id} init failed: {e}")
    finally:
        print(f"Worker {worker_id} finished - {batches_processed} batches")


class ArxivSortedWorkflow:
    """Simplified size-sorted ArXiv processing workflow.
    
    HADES Framework Implementation:
    - W (What): High-quality embeddings via jina-v4 model
    - R (Where): Size-sorted processing for optimal GPU cache locality
    - H (Who): Multi-worker GPU agents with configurable parallelism
    - T (Time): Minimized via batching and sorted processing
    - Ctx: Workflow context (GPU count, batch sizes, data characteristics)
    - α: Default 1.7 for context amplification
    
    Conveyance C = (W · R · H / T) · Ctx^α measured via throughput metrics.
    """

    def __init__(self,
                 database: str = 'arxiv_repository',
                 username: str = 'root',
                 metadata_file: str = '/bulk-store/arxiv-data/metadata/arxiv-kaggle-latest.json',
                 max_records: Optional[int] = None,
                 batch_size: int = 100,
                 embedding_batch_size: int = 48,
                 num_workers: int = 2,
                 drop_collections: bool = False,
                 chunk_size_tokens: int = 500,
                 chunk_overlap_tokens: int = 200) -> None:
        """Initialize workflow with minimal config."""

        self.database = database
        self.username = username
        self.metadata_file = Path(metadata_file)
        self.max_records = max_records
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.num_workers = num_workers
        self.drop_collections = drop_collections
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        # Collections
        self.metadata_collection = 'arxiv_metadata'
        self.chunks_collection = 'arxiv_abstract_chunks'
        self.embeddings_collection = 'arxiv_abstract_embeddings'

        # Counters
        self.processed_count = 0
        self.failed_count = 0

        # Setup database - improved password handling
        password_env = os.environ.get('ARANGO_PASSWORD_ENV', 'ARANGO_PASSWORD')
        password = os.environ.get(password_env, '')

        if not password:
            logger.warning(f"No password found in environment variable '{password_env}'. "
                         f"Either set {password_env} or use --password-env to specify a different variable.")

        os.environ['ARANGO_PASSWORD'] = password

        self.db = DatabaseFactory.get_arango(
            database=self.database,
            username=self.username,
            use_unix=True
        )

        # Setup multiprocessing
        ctx = mp.get_context('spawn')
        self.input_queue: mp.Queue = ctx.Queue(maxsize=50)
        self.output_queue: mp.Queue = ctx.Queue(maxsize=50)
        self.stop_event: Any = ctx.Event()

        # Workers and storage thread
        self.workers: List[Any] = []
        self.storage_thread: Optional[threading.Thread] = None

    def execute(self) -> bool:
        """Main execution flow."""
        start_time = datetime.now()

        try:
            # Initialize collections
            self._init_collections()

            # Start workers
            self._start_workers()

            # Start storage thread
            self.storage_thread = threading.Thread(target=self._storage_worker)
            self.storage_thread.start()

            # Process data
            if self.drop_collections:
                logger.info("Loading metadata from file")
                self._load_metadata()

            logger.info("Querying unprocessed records")
            self._process_unprocessed_records()

            # Shutdown
            self._shutdown_workers()

            # Results
            duration = (datetime.now() - start_time).total_seconds()
            throughput = self.processed_count / duration if duration > 0 else 0

            print("\n" + "="*60)
            print("Results")
            print("="*60)
            print(f"Success: True")
            print(f"Processed: {self.processed_count}")
            print(f"Failed: {self.failed_count}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Throughput: {throughput:.2f} records/second")

            return True

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return False

    def _init_collections(self) -> None:
        """Initialize database collections."""
        if self.drop_collections:
            logger.info("Dropping existing collections")
            for coll in [self.metadata_collection, self.chunks_collection, self.embeddings_collection]:
                if self.db.has_collection(coll):
                    self.db.delete_collection(coll)

        # Ensure collections exist
        for coll in [self.metadata_collection, self.chunks_collection, self.embeddings_collection]:
            if not self.db.has_collection(coll):
                self.db.create_collection(coll)

    def _start_workers(self) -> None:
        """Start GPU worker processes."""
        import torch

        # Detect available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Detected {num_gpus} GPU(s)")
        else:
            num_gpus = 0
            logger.warning("No GPUs detected")

        # Exit if workers requested but no GPUs available
        if self.num_workers > 0 and num_gpus == 0:
            raise RuntimeError(
                f"Requested {self.num_workers} GPU workers but no GPUs available. "
                "Either set --workers 0 for CPU mode or ensure CUDA is available."
            )

        # Get spawn context for proper CUDA initialization
        ctx = mp.get_context('spawn')

        for i in range(self.num_workers):
            # CRITICAL FIX: Use actual GPU count, not worker count
            gpu_id = i % num_gpus if num_gpus > 0 else 0

            # Use spawn context for process creation
            p = ctx.Process(
                target=worker_process_with_gpu,
                args=(
                    i,
                    gpu_id,
                    'jinaai/jina-embeddings-v4',
                    self.input_queue,
                    self.output_queue,
                    self.stop_event,
                    self.embedding_batch_size,  # Pass CLI arg
                    self.chunk_size_tokens,      # Pass chunk config
                    self.chunk_overlap_tokens    # Pass chunk config
                )
            )
            p.start()
            self.workers.append(p)
            print(f"Started worker {i} on GPU {gpu_id}")

    def _load_metadata(self) -> int:
        """Load metadata from JSON file to database."""
        # Check if file exists
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        batch = []
        total_loaded = 0
        total_errors = 0

        with open(self.metadata_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)

                    if not record.get('abstract'):
                        continue

                    doc = {
                        '_key': record.get('id', '').replace('.', '_').replace('/', '_'),
                        'arxiv_id': record.get('id'),
                        'title': record.get('title'),
                        'abstract': record.get('abstract'),
                        'authors': record.get('authors'),
                        'categories': record.get('categories'),
                        'update_date': record.get('update_date'),
                        'abstract_length': len(record.get('abstract', ''))
                    }
                    batch.append(doc)
                    total_loaded += 1

                    if len(batch) >= 10000:
                        result = self.db[self.metadata_collection].import_bulk(
                            batch,
                            on_duplicate='replace',
                            details=True
                        )
                        # Check for errors in import
                        if result.get('errors', 0) > 0:
                            error_count = result['errors']
                            total_errors += error_count
                            logger.warning(f"Import had {error_count} errors. Details: {result.get('details', [])[:5]}")

                        logger.info(f"Loaded {total_loaded} records, {total_errors} errors so far")
                        batch = []

                except json.JSONDecodeError:
                    continue

            # Final batch
            if batch:
                result = self.db[self.metadata_collection].import_bulk(
                    batch,
                    on_duplicate='replace',
                    details=True
                )
                # Check for errors in final batch
                if result.get('errors', 0) > 0:
                    error_count = result['errors']
                    total_errors += error_count
                    logger.warning(f"Final import had {error_count} errors")

        actual_count = self.db[self.metadata_collection].count()
        logger.info(f"Loaded {actual_count} total metadata records with {total_errors} errors")

        if total_errors > 0:
            logger.warning(f"Total import errors: {total_errors}")

        return actual_count

    def _process_unprocessed_records(self) -> None:
        """Query and process unprocessed records sorted by size."""
        # Check if embeddings exist
        embeddings_count = self.db[self.embeddings_collection].count()

        if embeddings_count == 0:
            # Fast path - no embeddings yet
            logger.info("Fast path: no existing embeddings")
            query = f"""
            FOR doc IN {self.metadata_collection}
                FILTER doc.abstract != null
                SORT doc.abstract_length ASC
                {'LIMIT ' + str(self.max_records) if self.max_records else ''}
                RETURN doc
            """
        else:
            # Check which records have embeddings
            query = f"""
            FOR doc IN {self.metadata_collection}
                FILTER doc.abstract != null
                LET has_embedding = FIRST(
                    FOR e IN {self.embeddings_collection}
                        FILTER e.arxiv_id == doc.arxiv_id
                        LIMIT 1
                        RETURN 1
                )
                FILTER has_embedding == null
                SORT doc.abstract_length ASC
                {'LIMIT ' + str(self.max_records) if self.max_records else ''}
                RETURN doc
            """

        # Stream results with longer TTL to handle large datasets
        cursor = self.db.aql.execute(query, batch_size=10000, ttl=3600)  # 1 hour TTL

        batch = []
        batch_count = 0
        for record in cursor:
            batch.append(record)

            if len(batch) >= self.batch_size:
                self.input_queue.put(batch)
                batch_count += 1

                if batch_count % 100 == 0:
                    logger.info(f"Queued {batch_count} batches")

                batch = []

        # Final batch
        if batch:
            self.input_queue.put(batch)

        # Send poison pills
        for _ in range(self.num_workers):
            self.input_queue.put(None)

        logger.info(f"Queued all records for processing")

    def _storage_worker(self) -> None:
        """Storage thread that saves embeddings to database."""
        # CRITICAL FIX: Continue draining queue even after stop_event is set
        while not self.stop_event.is_set() or not self.output_queue.empty():
            try:
                results = self.output_queue.get(timeout=1.0)
                if results:
                    self._store_batch(results)
            except Empty:
                # Only continue if we're still running; exit if stopped and queue is empty
                if self.stop_event.is_set() and self.output_queue.empty():
                    break
                continue
            except Exception as e:
                logger.error(f"Storage error: {e}")
                self.failed_count += len(results) if 'results' in locals() else 0

    def _store_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Store chunks and embeddings (NOT metadata - already loaded)."""
        if not batch:
            return

        # Begin transaction
        txn = self.db.begin_transaction(
            write=[self.chunks_collection, self.embeddings_collection]
        )

        try:
            papers_seen = set()

            for item in batch:
                record = item['record']
                chunk = item['chunk']
                arxiv_id = record.get('arxiv_id') or record.get('id', '')

                if not arxiv_id:
                    continue

                sanitized_id = arxiv_id.replace('.', '_').replace('/', '_')

                # Track unique papers
                if arxiv_id not in papers_seen:
                    papers_seen.add(arxiv_id)

                # Store chunk
                chunk_id = f"{sanitized_id}_chunk_{chunk.chunk_index}"
                chunk_doc = {
                    '_key': chunk_id,
                    'arxiv_id': arxiv_id,
                    'paper_key': sanitized_id,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'text': chunk.text,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'created_at': datetime.now().isoformat()
                }
                txn.collection(self.chunks_collection).insert(chunk_doc, overwrite=True)

                # Store embedding
                embedding_doc = {
                    '_key': f"{chunk_id}_emb",
                    'chunk_id': chunk_id,
                    'arxiv_id': arxiv_id,
                    'paper_key': sanitized_id,
                    'embedding': chunk.embedding.tolist(),
                    'embedding_dim': len(chunk.embedding),
                    'model': 'jinaai/jina-embeddings-v4',
                    'created_at': datetime.now().isoformat()
                }
                txn.collection(self.embeddings_collection).insert(embedding_doc, overwrite=True)

            # Commit
            txn.commit_transaction()
            self.processed_count += len(papers_seen)

            if self.processed_count % 1000 == 0:
                logger.info(f"Stored {self.processed_count} papers")

        except Exception as e:
            txn.abort_transaction()
            logger.error(f"Failed to store batch: {e}")
            self.failed_count += len(batch)

    def _shutdown_workers(self) -> None:
        """Clean shutdown of workers and storage."""
        # Wait for workers
        logger.info("Waiting for workers to finish")
        for p in self.workers:
            p.join(timeout=60)

        # Stop storage
        self.stop_event.set()
        if self.storage_thread:
            self.storage_thread.join(timeout=30)


if __name__ == "__main__":
    import argparse

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="ArXiv size-sorted processing (simplified)")
    parser.add_argument('--database', default='arxiv_repository')
    parser.add_argument('--username', default='root')
    parser.add_argument('--password-env', default='ARANGO_PASSWORD')
    parser.add_argument('--count', type=int, default=None, help='Max records to process')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--embedding-batch-size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--chunk-size-tokens', type=int, default=500, help='Chunk size in tokens')
    parser.add_argument('--chunk-overlap-tokens', type=int, default=200, help='Overlap between chunks')
    parser.add_argument('--drop-collections', action='store_true')

    args = parser.parse_args()

    # Set password env var
    os.environ['ARANGO_PASSWORD_ENV'] = args.password_env

    # Print config
    print("="*60)
    print("ArXiv Size-Sorted Processing (Simplified)")
    print("="*60)
    print(f"Database: {args.database}")
    print(f"Username: {args.username}")
    print(f"Records: {args.count if args.count else 'all'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding batch: {args.embedding_batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Drop collections: {args.drop_collections}")
    print("="*60)

    # Run workflow
    workflow = ArxivSortedWorkflow(
        database=args.database,
        username=args.username,
        max_records=args.count,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        num_workers=args.workers,
        drop_collections=args.drop_collections,
        chunk_size_tokens=args.chunk_size_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens
    )

    success = workflow.execute()
    exit(0 if success else 1)
