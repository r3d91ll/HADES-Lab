#!/usr/bin/env python3
"""
ArXiv Initial Ingest Workflow
=============================

Processes ArXiv abstracts in their original order from the JSON file
using the new HTTP/2 optimized ArangoDB client.

Core flow:
1. Load metadata from JSON (if needed)
2. Query unprocessed records in original order
3. Process with GPU workers
4. Store embeddings and chunks via optimized client
"""

import json
import logging
import multiprocessing as mp
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging early to ensure error messages are visible
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from core.database.arango import CollectionDefinition

# Core imports
from core.database.database_factory import DatabaseFactory
from core.embedders.embedders_factory import EmbedderFactory

logger = logging.getLogger(__name__)


def worker_process_with_gpu(worker_id: int, gpu_id: int, model_name: str,
                           input_queue: mp.Queue, output_queue: mp.Queue, stop_event: Any,
                           embedding_batch_size: int = 48, chunk_size_tokens: int = 500,
                           chunk_overlap_tokens: int = 200,
                           embedder_type: str = 'jina') -> None:
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
        embedder_type: Identifier for embedder selection ('jina' or 'sentence-transformers')
    """
    # Set GPU BEFORE any CUDA imports
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    print(f"Worker {worker_id} started on GPU {gpu_id}, PID {os.getpid()}")

    # Import already done at module level

    # Initialize batches_processed BEFORE try block to avoid UnboundLocalError
    batches_processed = 0

    try:
        # Initialize embedder with passed configuration
        resolved_model = model_name
        embedder_kind = embedder_type.lower()
        if embedder_kind in {'sentence', 'sentence-transformers'} and 'sentence-transformers' not in resolved_model.lower():
            logger.warning(
                "Worker %s: SentenceTransformers requested without explicit model; using sentence-transformers/all-mpnet-base-v2.",
                worker_id,
            )
            resolved_model = 'sentence-transformers/all-mpnet-base-v2'
        elif embedder_kind not in {'jina', 'transformer', 'sentence', 'sentence-transformers'}:
            logger.warning("Worker %s: Unknown embedder_type '%s'; defaulting to JinaV4Embedder", worker_id, embedder_type)
            embedder_kind = 'jina'

        if embedder_kind in {'sentence', 'sentence-transformers'}:
            logger.warning(
                "Worker %s: SentenceTransformersEmbedder is deprecated and will be removed after migration; routing to Jina fallback.",
                worker_id,
            )

        embedder = EmbedderFactory.create(
            model_name=resolved_model,
            device='cuda:0',  # Always 0 since we isolated GPU
            use_fp16=True,
            batch_size=embedding_batch_size,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )

        print(f"Worker {worker_id} ready with batch_size={embedding_batch_size}, "
              f"chunk_size={chunk_size_tokens}, overlap={chunk_overlap_tokens}, embedder={embedder_kind}")

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
                    # Generate contextualised chunks using late chunking
                    all_chunks = embedder.embed_batch_with_late_chunking(
                        abstracts,
                        task="retrieval.passage"
                    )

                    # Package results
                    results = []
                    for record, chunks in zip(records, all_chunks, strict=False):
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
            except Exception:
                logger.exception("Worker %d error", worker_id)

    except Exception:
        logger.exception("Worker %d init failed", worker_id)
    finally:
        print(f"Worker {worker_id} finished - {batches_processed} batches")


class ArxivInitialIngestWorkflow:
    """ArXiv initial ingest workflow using HTTP/2 optimized client.

    HADES Framework Implementation:
    - W (What): High-quality embeddings via jina-v4 model
    - R (Where): Original document order from JSON file
    - H (Who): Multi-worker GPU agents with configurable parallelism
    - T (Time): Minimized via HTTP/2 optimized client (10x faster)
    - Ctx: Workflow context (GPU count, batch sizes, data characteristics)
    - α: Default 1.7 for context amplification

    Conveyance C = (W · R · H / T) · Ctx^α measured via throughput metrics.
    """

    def __init__(self,
                 database: str = 'arxiv_repository',
                 username: str = 'root',
                 metadata_file: str = 'data/arxiv-kaggle-latest.json',  # Using NVME copy for speed
                 max_records: int | None = None,
                 batch_size: int = 100,
                 embedding_batch_size: int = 48,
                 num_workers: int = 2,
                 drop_collections: bool = False,
                 chunk_size_tokens: int = 500,
                 chunk_overlap_tokens: int = 200,
                 embedding_model: str | None = None,
                 embedder_type: str | None = None) -> None:
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
        self.embedding_model = embedding_model or os.environ.get('HADES_EMBEDDER_MODEL', 'jinaai/jina-embeddings-v4')
        self.embedder_type = (embedder_type or os.environ.get('HADES_EMBEDDER', 'jina')).lower()

        # Collections
        self.metadata_collection = 'arxiv_metadata'
        self.chunks_collection = 'arxiv_abstract_chunks'
        self.embeddings_collection = 'arxiv_abstract_embeddings'
        self.structures_collection = 'arxiv_structures'

        # Counters
        self.processed_count = 0
        self.failed_count = 0

        # Setup optimized HTTP/2 memory client
        # Password should come from .env file or environment
        password = os.environ.get('ARANGO_PASSWORD')
        if not password:
            raise ValueError(
                "ARANGO_PASSWORD not set. Please create a .env file from .env.example "
                "or set the ARANGO_PASSWORD environment variable."
            )

        # Use the new optimized memory client for ALL operations
        self.memory_client = DatabaseFactory.get_arango_memory_service(
            database=self.database,
            username=self.username,
            password=password
        )

        # Setup multiprocessing
        ctx = mp.get_context('spawn')
        self.input_queue: mp.Queue = ctx.Queue(maxsize=50)
        self.output_queue: mp.Queue = ctx.Queue(maxsize=50)
        self.stop_event: Any = ctx.Event()

        # Workers and storage thread
        self.workers: list[Any] = []
        self.storage_thread: threading.Thread | None = None

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
            print("Success: True")
            print(f"Processed: {self.processed_count}")
            print(f"Failed: {self.failed_count}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Throughput: {throughput:.2f} records/second")

            return True

        except Exception as e:
            logger.exception("Workflow failed")
            return False
        finally:
            # Clean up memory client
            if hasattr(self, 'memory_client'):
                self.memory_client.close()

    def _init_collections(self) -> None:
        """Initialize database collections using optimized HTTP/2 client."""

        if self.drop_collections:
            logger.info("Dropping existing collections via HTTP/2 client")
            self.memory_client.drop_collections(
                [
                    self.metadata_collection,
                    self.chunks_collection,
                    self.embeddings_collection,
                    self.structures_collection,
                ],
                ignore_missing=True,
            )

        # Define collections with proper types and indexes
        collections = [
            CollectionDefinition(
                name=self.metadata_collection,
                type="document",
                indexes=[
                    {"type": "hash", "fields": ["arxiv_id"], "unique": False},
                    {"type": "skiplist", "fields": ["abstract_length"]}
                ]
            ),
            CollectionDefinition(
                name=self.chunks_collection,
                type="document",
                indexes=[
                    {"type": "hash", "fields": ["arxiv_id"], "unique": False},
                    {"type": "hash", "fields": ["paper_key"], "unique": False},
                    {"type": "hash", "fields": ["document_id"], "unique": False},
                    {"type": "hash", "fields": ["chunk_id"], "unique": False},
                ]
            ),
            CollectionDefinition(
                name=self.embeddings_collection,
                type="document",
                indexes=[
                    {"type": "hash", "fields": ["arxiv_id"], "unique": False},
                    {"type": "hash", "fields": ["chunk_id"], "unique": False},
                    {"type": "hash", "fields": ["document_id"], "unique": False},
                ]
            ),
            CollectionDefinition(
                name=self.structures_collection,
                type="document",
                indexes=[
                    {"type": "hash", "fields": ["arxiv_id"], "unique": False},
                    {"type": "hash", "fields": ["paper_key"], "unique": False},
                    {"type": "hash", "fields": ["document_id"], "unique": False}
                ]
            )
        ]

        logger.info("Creating collections via HTTP/2 client")
        self.memory_client.create_collections(collections)
        logger.info("Collections initialized successfully")

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
                    self.embedding_model,
                    self.input_queue,
                    self.output_queue,
                    self.stop_event,
                    self.embedding_batch_size,  # Pass CLI arg
                    self.chunk_size_tokens,      # Pass chunk config
                    self.chunk_overlap_tokens,   # Pass chunk config
                    self.embedder_type,
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

        with open(self.metadata_file) as f:
            for _, line in enumerate(f, 1):
                try:
                    record = json.loads(line)

                    if not record.get('abstract'):
                        continue

                    raw_authors = record.get('authors', [])
                    if isinstance(raw_authors, list):
                        authors = raw_authors
                    elif isinstance(raw_authors, str):
                        authors = [a.strip() for a in re.split(r'[;,]\s*', raw_authors) if a.strip()]
                    else:
                        authors = []

                    raw_categories = record.get('categories', [])
                    if isinstance(raw_categories, list):
                        categories = raw_categories
                    elif isinstance(raw_categories, str):
                        categories = raw_categories.split()
                    else:
                        categories = []

                    doc = {
                        '_key': record.get('id', '').replace('.', '_').replace('/', '_'),
                        'arxiv_id': record.get('id'),
                        'title': record.get('title'),
                        'abstract': record.get('abstract'),
                        'authors': authors,
                        'categories': categories,
                        'update_date': record.get('update_date'),
                        'abstract_length': len(record.get('abstract', ''))
                    }
                    batch.append(doc)
                    total_loaded += 1

                    if len(batch) >= 10000:
                        try:
                            # Use optimized bulk_insert
                            inserted = self.memory_client.bulk_insert(self.metadata_collection, batch)
                            logger.info(f"Loaded {total_loaded} records ({inserted} inserted)")
                        except Exception as e:
                            total_errors += len(batch)
                            logger.warning(f"Import error: {e}")

                        batch = []

                except json.JSONDecodeError:
                    continue

            # Final batch
            if batch:
                try:
                    inserted = self.memory_client.bulk_insert(self.metadata_collection, batch)
                    logger.info(f"Final batch: {inserted} inserted")
                except Exception as e:
                    total_errors += len(batch)
                    logger.warning(f"Final import error: {e}")

        # Get count using query
        count_result = self.memory_client.execute_query(f"RETURN LENGTH({self.metadata_collection})")
        actual_count = count_result[0] if count_result else 0
        logger.info(f"Loaded {actual_count} total metadata records with {total_errors} errors")

        if total_errors > 0:
            logger.warning(f"Total import errors: {total_errors}")

        return actual_count

    def _process_unprocessed_records(self) -> None:
        """Query and process unprocessed records in original order."""
        # Check if embeddings exist using query
        count_result = self.memory_client.execute_query(f"RETURN LENGTH({self.embeddings_collection})")
        embeddings_count = count_result[0] if count_result else 0

        if embeddings_count == 0:
            # Fast path - no embeddings yet, process in original order
            logger.info("Fast path: no existing embeddings, processing in original order")
            query = f"""
            FOR doc IN {self.metadata_collection}
                FILTER doc.abstract != null
                {'LIMIT ' + str(self.max_records) if self.max_records else ''}
                RETURN doc
            """
        else:
            # Check which records have embeddings, maintain original order
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
                {'LIMIT ' + str(self.max_records) if self.max_records else ''}
                RETURN doc
            """

        # Execute query and process results
        results = self.memory_client.execute_query(query, batch_size=10000)

        batch = []
        batch_count = 0
        for record in results:
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

        logger.info("Queued all records for processing")

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
            except Exception:
                logger.exception("Storage error")
                self.failed_count += len(results) if 'results' in locals() else 0

    def _store_batch(self, batch: list[dict[str, Any]]) -> None:
        """Store chunks and embeddings using optimized HTTP/2 client."""
        if not batch:
            return

        try:
            papers_seen = set()
            chunk_docs = []
            embedding_docs = []

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

                # Prepare chunk document
                chunk_id = f"{sanitized_id}_chunk_{chunk.chunk_index}"
                chunk_doc = {
                    '_key': chunk_id,
                    'arxiv_id': arxiv_id,
                    'document_id': arxiv_id,
                    'paper_key': sanitized_id,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'text': chunk.text,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'created_at': datetime.now().isoformat()
                }
                chunk_docs.append(chunk_doc)

                # Prepare embedding document
                embedding_doc = {
                    '_key': f"{chunk_id}_emb",
                    'chunk_id': chunk_id,
                    'arxiv_id': arxiv_id,
                    'document_id': arxiv_id,
                    'paper_key': sanitized_id,
                    'embedding': chunk.embedding.tolist(),
                    'embedding_dim': int(chunk.embedding.shape[0]) if hasattr(chunk.embedding, 'shape') else len(chunk.embedding),
                    'model': 'jinaai/jina-embeddings-v4',
                    'created_at': datetime.now().isoformat()
                }
                embedding_docs.append(embedding_doc)

            # Bulk insert chunks and embeddings using optimized client
            if chunk_docs:
                chunks_inserted = self.memory_client.bulk_insert(self.chunks_collection, chunk_docs)
                logger.debug(f"Inserted {chunks_inserted} chunks")

            if embedding_docs:
                embeddings_inserted = self.memory_client.bulk_insert(self.embeddings_collection, embedding_docs)
                logger.debug(f"Inserted {embeddings_inserted} embeddings")

            self.processed_count += len(papers_seen)

            if self.processed_count % 1000 == 0:
                logger.info(f"Stored {self.processed_count} papers")

        except Exception:
            logger.exception("Failed to store batch")
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

    parser = argparse.ArgumentParser(description="ArXiv initial ingest with HTTP/2 optimized client")
    parser.add_argument('--database', default='arxiv_repository')
    parser.add_argument('--username', default='root')
    parser.add_argument('--password', default=None, help='ArangoDB password (or use ARANGO_PASSWORD env var)')
    parser.add_argument('--password-env', default='ARANGO_PASSWORD')
    parser.add_argument('--metadata-file', default='data/arxiv-kaggle-latest.json', help='Path to ArXiv metadata JSON file')
    parser.add_argument('--count', type=int, default=None, help='Max records to process')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--embedding-batch-size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--chunk-size-tokens', type=int, default=500, help='Chunk size in tokens')
    parser.add_argument('--chunk-overlap-tokens', type=int, default=200, help='Overlap between chunks')
    parser.add_argument('--drop-collections', action='store_true')

    args = parser.parse_args()

    # Set password - from argument or environment
    if args.password:
        os.environ['ARANGO_PASSWORD'] = args.password
    elif not os.environ.get('ARANGO_PASSWORD'):
        env_password = os.environ.get(args.password_env)
        if env_password:
            os.environ['ARANGO_PASSWORD'] = env_password

    if args.password_env:
        os.environ['ARANGO_PASSWORD_ENV'] = args.password_env

    # Print config
    print("="*60)
    print("ArXiv Initial Ingest (HTTP/2 Optimized)")
    print("="*60)
    print(f"Database: {args.database}")
    print(f"Username: {args.username}")
    print(f"Records: {args.count if args.count else 'all'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding batch: {args.embedding_batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Drop collections: {args.drop_collections}")
    print(f"Chunk size: {args.chunk_size_tokens} tokens")
    print(f"Chunk overlap: {args.chunk_overlap_tokens} tokens")
    print("="*60)

    # Run workflow
    workflow = ArxivInitialIngestWorkflow(
        database=args.database,
        username=args.username,
        metadata_file=args.metadata_file,
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
