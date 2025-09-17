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
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
from queue import Empty
import threading

# Core imports
from core.database.database_factory import DatabaseFactory
from core.embedders.embedders_factory import EmbedderFactory

logger = logging.getLogger(__name__)


def worker_process_with_gpu(worker_id, gpu_id, model_name, input_queue, output_queue, stop_event):
    """
    GPU-backed worker process that initializes a GPU-local embedder and converts incoming records' abstracts into chunked embeddings.
    
    The process sets CUDA_VISIBLE_DEVICES to isolate a single GPU, disables tokenizer parallelism, imports and constructs an embedder on that device, then continuously reads batches from input_queue. Each batch should be an iterable of record dicts; a None value is treated as a poison pill and causes shutdown. Records with a non-empty 'abstract' field are embedded using late chunking; for each produced chunk the worker emits a dict {'record': <original record>, 'chunk': <chunk dict>} into output_queue. The loop stops when a poison pill is received or stop_event is set. Initialization and runtime errors are logged; the function returns None.
    
    Parameters:
        worker_id (int): Identifier used for logging and progress messages.
        gpu_id (int): GPU index to assign to this process via CUDA_VISIBLE_DEVICES.
        model_name (str): Model identifier passed to the EmbedderFactory.
        input_queue (multiprocessing.Queue): Queue providing batches (iterables of record dicts). Use None as a poison pill to signal termination.
        output_queue (multiprocessing.Queue): Queue where the worker will put lists of {'record', 'chunk'} dicts produced for storage.
        stop_event (multiprocessing.Event): Event that, when set, causes the worker to exit the processing loop.
    
    Returns:
        None
    """
    # Set GPU BEFORE any CUDA imports
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    print(f"Worker {worker_id} started on GPU {gpu_id}, PID {os.getpid()}")

    # Import after GPU is set
    from core.embedders.embedders_factory import EmbedderFactory

    try:
        # Initialize embedder
        embedder_config = {
            'device': 'cuda:0',  # Always 0 since we isolated GPU
            'use_fp16': True,
            'batch_size': 48,  # Embedding batch size
            'chunk_size_tokens': 500,
            'chunk_overlap_tokens': 200
        }

        embedder = EmbedderFactory.create(
            model_name=model_name,
            **embedder_config
        )

        print(f"Worker {worker_id} ready")

        # Process batches
        batches_processed = 0
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
                 database='arxiv_repository',
                 username='root',
                 metadata_file='/bulk-store/arxiv-data/metadata/arxiv-kaggle-latest.json',
                 max_records=None,
                 batch_size=100,
                 embedding_batch_size=48,
                 num_workers=2,
                 drop_collections=False):
        """
                 Initialize the ArxivSortedWorkflow instance and allocate resources required for processing.
                 
                 Parameters:
                     database (str): Name of the ArangoDB database to use.
                     username (str): Database username.
                     metadata_file (str | Path): Path to a line-delimited JSON metadata file (used when loading metadata).
                     max_records (int | None): Optional limit on number of records to process; None means no limit.
                     batch_size (int): Number of metadata records grouped and sent to worker processes per work batch.
                     embedding_batch_size (int): Maximum number of texts sent to the embedder in a single inference call.
                     num_workers (int): Number of GPU worker processes to spawn (each worker is assigned a GPU id via modulo mapping).
                     drop_collections (bool): If True, drop and recreate target collections before running (triggers metadata load).
                 
                 Notes:
                     - The constructor reads the database password from the environment variable named by ARANGO_PASSWORD_ENV
                       (falls back to ARANGO_PASSWORD), and exposes it as ARANGO_PASSWORD for the DatabaseFactory.
                     - This method initializes multiprocessing queues and a stop event, and prepares collection names and counters.
                 """

        self.database = database
        self.username = username
        self.metadata_file = Path(metadata_file)
        self.max_records = max_records
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.num_workers = num_workers
        self.drop_collections = drop_collections

        # Collections
        self.metadata_collection = 'arxiv_metadata'
        self.chunks_collection = 'arxiv_abstract_chunks'
        self.embeddings_collection = 'arxiv_abstract_embeddings'

        # Counters
        self.processed_count = 0
        self.failed_count = 0

        # Setup database
        password = os.environ.get(os.environ.get('ARANGO_PASSWORD_ENV', 'ARANGO_PASSWORD'), '')
        os.environ['ARANGO_PASSWORD'] = password

        self.db = DatabaseFactory.get_arango(
            database=self.database,
            username=self.username,
            use_unix=True
        )

        # Setup multiprocessing
        ctx = mp.get_context('spawn')
        self.input_queue = ctx.Queue(maxsize=50)
        self.output_queue = ctx.Queue(maxsize=50)
        self.stop_event = ctx.Event()

        # Workers and storage thread
        self.workers = []
        self.storage_thread = None

    def execute(self):
        """
        Run the full workflow: prepare collections, start GPU worker processes and the storage thread, optionally load metadata, stream and process unprocessed records, then shut down workers and report results.
        
        This method orchestrates the end-to-end pipeline and has the following observable effects:
        - Initializes or (optionally) recreates database collections.
        - Starts multiprocessing GPU workers and a background storage thread.
        - If configured to drop collections, loads metadata from the configured JSONL file.
        - Streams unprocessed records (sorted by abstract length) to workers for embedding/chunk generation.
        - Persists generated chunks and embeddings via the storage thread.
        - Gracefully shuts down workers and the storage thread, then prints a summary of processed/failed counts, duration, and throughput.
        
        Returns:
            bool: True on successful completion of the workflow; False if an exception occurs during execution.
        """
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

    def _init_collections(self):
        """
        Ensure required ArangoDB collections exist; optionally drop and recreate them.
        
        If self.drop_collections is true, deletes the metadata, chunks, and embeddings
        collections (if present). Then verifies that the three collections named by
        self.metadata_collection, self.chunks_collection, and self.embeddings_collection
        exist, creating any that are missing.
        """
        if self.drop_collections:
            logger.info("Dropping existing collections")
            for coll in [self.metadata_collection, self.chunks_collection, self.embeddings_collection]:
                if self.db.has_collection(coll):
                    self.db.delete_collection(coll)

        # Ensure collections exist
        for coll in [self.metadata_collection, self.chunks_collection, self.embeddings_collection]:
            if not self.db.has_collection(coll):
                self.db.create_collection(coll)

    def _start_workers(self):
        """Start GPU worker processes."""
        for i in range(self.num_workers):
            gpu_id = i % self.num_workers  # Simple GPU assignment

            p = mp.Process(
                target=worker_process_with_gpu,
                args=(
                    i,
                    gpu_id,
                    'jinaai/jina-embeddings-v4',
                    self.input_queue,
                    self.output_queue,
                    self.stop_event
                )
            )
            p.start()
            self.workers.append(p)
            print(f"Started worker {i} on GPU {gpu_id}")

    def _load_metadata(self):
        """
        Load metadata from the workflow's JSONL metadata file into the metadata collection.
        
        Reads the file at self.metadata_file line-by-line (JSON Lines). For each record with a non-empty `abstract` it builds a document containing a sanitized `_key`, `arxiv_id`, `title`, `abstract`, `authors`, `categories`, `update_date`, and `abstract_length`. Documents are imported in batches (10,000) with on-duplicate replacement. Malformed JSON lines are skipped.
        
        Returns:
            int: The number of documents present in the metadata collection after import.
        """
        batch = []
        total_loaded = 0

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
                        logger.info(f"Loaded {total_loaded} records")
                        batch = []

                except json.JSONDecodeError:
                    continue

            # Final batch
            if batch:
                self.db[self.metadata_collection].import_bulk(
                    batch,
                    on_duplicate='replace',
                    details=True
                )

        actual_count = self.db[self.metadata_collection].count()
        logger.info(f"Loaded {actual_count} total metadata records")
        return actual_count

    def _process_unprocessed_records(self):
        """
        Populate the input queue with metadata records that have not yet been embedded, ordered by abstract length.
        
        If the embeddings collection is empty a fast-path query returns all documents with an abstract; otherwise documents that already have an embedding are filtered out. Records are streamed from the database in large AQL batches, accumulated into batches of size self.batch_size, and each batch is placed onto self.input_queue for worker consumption. After all records are queued, a `None` poison pill is enqueued once per worker to signal completion.
        
        Side effects:
        - Issues AQL queries against self.db.
        - Puts lists of records into self.input_queue.
        - Enqueues `None` entries (poison pills) equal to self.num_workers.
        - Logs progress during queuing.
        """
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

    def _storage_worker(self):
        """
        Worker thread that consumes completed embedding batches from the output queue and persists them.
        
        Continuously polls self.output_queue (1.0s timeout) until self.stop_event is set. For each non-empty result retrieved, calls self._store_batch(result) to write chunks and embeddings to the database. Swallows queue Empty exceptions and continues polling. On any other exception the error is logged and self.failed_count is incremented by the number of items in the failed batch when available.
        """
        while not self.stop_event.is_set():
            try:
                results = self.output_queue.get(timeout=1.0)
                if results:
                    self._store_batch(results)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Storage error: {e}")
                self.failed_count += len(results) if 'results' in locals() else 0

    def _store_batch(self, batch: List[Dict]):
        """
        Persist a batch of generated text chunks and their embeddings to the database in a single transaction.
        
        This writes documents to the chunks and embeddings collections and updates internal counters on success. Each batch item is expected to be a dict with keys 'record' (the original metadata record) and 'chunk' (an object with chunk_index, total_chunks, text, start_char, end_char, and embedding). Items missing an arXiv identifier are skipped. On success the transaction is committed and processed_count is incremented by the number of unique papers in the batch; on failure the transaction is aborted and failed_count is increased by the batch size.
        
        Parameters:
            batch (List[Dict]): List of items to store. Each item must contain:
                - 'record': original record dict containing at least 'arxiv_id' or 'id'.
                - 'chunk': chunk object with attributes `chunk_index`, `total_chunks`, `text`,
                  `start_char`, `end_char`, and `embedding` (array-like).
        
        Side effects:
            - Inserts/overwrites documents in the chunks and embeddings collections.
            - Commits or aborts a database transaction.
            - Updates self.processed_count and self.failed_count.
        """
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

    def _shutdown_workers(self):
        """
        Cleanly shut down worker processes and the storage thread.
        
        Waits up to 60 seconds for each worker process in self.workers to finish. Signals the storage thread to stop by setting self.stop_event and waits up to 30 seconds for self.storage_thread to join if it exists.
        """
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
        drop_collections=args.drop_collections
    )

    success = workflow.execute()
    exit(0 if success else 1)