#!/usr/bin/env python3
"""
ArXiv Parallel Processing Workflow
===================================

Production-ready workflow for processing ArXiv metadata with parallel
embedding generation using multiple GPU workers.

This workflow implements the complete pipeline:
1. Load metadata from JSON file
2. Process abstracts with late chunking
3. Generate embeddings in parallel using multiple GPU workers
4. Store everything atomically in ArangoDB

Theory Connection (Conveyance Framework):
Optimizes C = (W·R·H/T)·Ctx^α by:
- Minimizing T through parallel GPU processing
- Maximizing H through multi-worker architecture
- Preserving Ctx through late chunking
"""

import os
import json
import logging
import time
import multiprocessing as mp
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
import threading

import numpy as np
from dataclasses import dataclass

# Core imports
from core.workflows.workflow_base import WorkflowBase
from core.workflows.state.state_manager import StateManager, CheckpointManager
from core.monitoring.progress_tracker import ProgressTracker
from core.monitoring.performance_monitor import PerformanceMonitor
from core.database.database_factory import DatabaseFactory
from core.embedders.embedders_factory import EmbedderFactory
from core.embedders.embedders_base import EmbeddingConfig
from core.logging.logging import LogManager

# Tools imports
from core.tools.arxiv.arxiv_metadata_config import ArxivMetadataConfig

logger = logging.getLogger(__name__)


def worker_process_with_gpu(config, input_queue, output_queue, stop_event, gpu_id):
    """
    Run a worker process bound to a single GPU that consumes record batches, generates late-chunked embeddings, and emits results.
    
    This function must be invoked in a separate process. It sets CUDA_VISIBLE_DEVICES (to isolate the assigned GPU) before importing any CUDA-dependent modules, initializes an embedder on that GPU, and then enters a loop that:
    - reads batches from input_queue (uses a 1s timeout),
    - treats None as a poison pill to stop,
    - extracts records with non-empty abstracts,
    - generates embeddings using late chunking,
    - sends a dictionary to output_queue containing either {'worker_id': ..., 'results': [...]} on success or {'worker_id': ..., 'error': ...} on failure.
    
    Operation continues until a poison pill is received or stop_event is set. Logging is emitted for lifecycle events (initialization, batch receipt/processing, errors, shutdown).
    
    Parameters documented only when necessary:
    - config: Worker configuration (WorkerConfig) containing worker_id, model_name, embedding_batch_size, use_fp16, etc.
    - gpu_id: Identifier for the GPU to bind (string as used for CUDA_VISIBLE_DEVICES).
    
    Side effects:
    - Mutates environment by setting CUDA_VISIBLE_DEVICES and TOKENIZERS_PARALLELISM.
    - Performs CUDA initialization and GPU memory allocation via the embedder.
    - Reads from input_queue and writes structured messages to output_queue.
    """
    # CRITICAL: Set GPU BEFORE any CUDA imports
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # NOW import CUDA-dependent modules
    from core.embedders.embedders_factory import EmbedderFactory
    from core.logging.logging import LogManager
    from queue import Empty
    import time

    # Setup structured logging
    worker_logger = LogManager.get_logger(
        f"worker_{config.worker_id}",
        f"gpu_{gpu_id}_{datetime.now().isoformat()}"
    )

    batches_processed = 0

    try:
        # Initialize embedder
        worker_logger.info("worker_initializing",
            worker_id=config.worker_id,
            gpu_id=gpu_id,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            model=config.model_name
        )

        embedder_config = {
            'device': 'cuda:0',  # Always cuda:0 since we isolated the GPU
            'use_fp16': config.use_fp16,
            'batch_size': config.embedding_batch_size,  # Use embedding batch size (128), not record batch size (1000)!
            'chunk_size_tokens': 500,  # Match the JinaV4Embedder settings
            'chunk_overlap_tokens': 200
        }

        embedder = EmbedderFactory.create(
            model_name=config.model_name,
            **embedder_config
        )

        worker_logger.info("worker_ready",
            worker_id=config.worker_id,
            gpu_id=gpu_id
        )

        # Process batches
        while not stop_event.is_set():
            try:
                batch = input_queue.get(timeout=1.0)

                if batch is None:  # Poison pill
                    worker_logger.info("worker_stopping",
                        worker_id=config.worker_id,
                        batches_processed=batches_processed
                    )
                    break

                worker_logger.info("batch_received",
                    worker_id=config.worker_id,
                    batch_size=len(batch),
                    batch_number=batches_processed + 1
                )

                # Process batch
                results = []
                abstracts_to_embed = []
                records_with_abstracts = []

                for record in batch:
                    if record.get('abstract'):
                        abstracts_to_embed.append(record['abstract'])
                        records_with_abstracts.append(record)

                if abstracts_to_embed:
                    # Use the working method from single-GPU workflow
                    worker_logger.info(f"Worker {config.worker_id}: Processing {len(abstracts_to_embed)} abstracts")
                    start_embed = time.time()
                    all_chunks = embedder.embed_batch_with_late_chunking(
                        abstracts_to_embed,
                        task="retrieval.passage"
                    )
                    embed_time = time.time() - start_embed
                    worker_logger.info(f"Worker {config.worker_id}: Embedded in {embed_time:.2f}s = {len(abstracts_to_embed)/embed_time:.1f} docs/sec")

                    # Package results
                    for record, record_chunks in zip(records_with_abstracts, all_chunks):
                        for chunk in record_chunks:
                            results.append({
                                'record': record,
                                'chunk': chunk
                            })

                batches_processed += 1

                worker_logger.info("batch_processed",
                    worker_id=config.worker_id,
                    batch_number=batches_processed,
                    results_count=len(results)
                )

                # Send results back
                output_queue.put({
                    'worker_id': config.worker_id,
                    'results': results
                })

            except Empty:
                continue
            except Exception as e:
                worker_logger.error("worker_error",
                    worker_id=config.worker_id,
                    error=str(e),
                    exc_info=True
                )
                output_queue.put({
                    'worker_id': config.worker_id,
                    'error': str(e)
                })

    except Exception as e:
        worker_logger.error("worker_init_failed",
            worker_id=config.worker_id,
            error=str(e),
            exc_info=True
        )
    finally:
        worker_logger.info("worker_shutdown",
            worker_id=config.worker_id,
            batches_processed=batches_processed
        )


@dataclass
class WorkerConfig:
    """Configuration for a single embedding worker."""
    worker_id: int
    gpu_device: str
    batch_size: int
    embedding_batch_size: int
    model_name: str
    use_fp16: bool


class EmbeddingWorker(mp.Process):
    """
    Worker process for generating embeddings on a specific GPU.

    Each worker:
    1. Runs on a dedicated GPU
    2. Processes batches from a shared queue
    3. Outputs embeddings to a result queue
    """

    def __init__(self,
                 config: WorkerConfig,
                 input_queue: mp.Queue,
                 output_queue: mp.Queue,
                 stop_event: mp.Event):
        """
                 Initialize the embedding worker process and store runtime objects.
                 
                 This stores the worker configuration, IPC queues, and shutdown event on the
                 process instance. The actual embedder is not initialized here (left as None
                 until the process is started).
                 
                 Args:
                     config: WorkerConfig with GPU and batching settings.
                     input_queue: mp.Queue that receives batches of metadata records to embed.
                     output_queue: mp.Queue where embedding result payloads (or error payloads)
                         will be posted.
                     stop_event: mp.Event used to request early shutdown.
                 """
        super().__init__()
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.embedder = None

    def run(self):
        """
        Run the worker process: initialize GPU-specific embedder, consume batches from the input queue, generate embeddings with late-chunking, and push results (or error payloads) to the output queue.
        
        The worker:
        - Sets CUDA_VISIBLE_DEVICES based on the worker's configured GPU and initializes an embedder bound to that GPU.
        - Repeatedly reads batches from self.input_queue. A None batch is treated as a poison pill and causes clean shutdown.
        - For each non-empty batch, calls self._process_batch(batch) to produce a list of chunked embedding results, then puts a dict {'worker_id': ..., 'results': ...} onto self.output_queue.
        - On processing errors, logs the exception and places an error payload {'worker_id': ..., 'error': str(e)} on self.output_queue.
        - Honors self.stop_event: the loop periodically times out on queue.get so the stop_event can be observed.
        - Logs lifecycle events (initialization, readiness, batch receipt/processing, errors, and shutdown).
        
        This method does not return a value; side effects include setting the CUDA_VISIBLE_DEVICES environment variable, initializing self.embedder, sending messages to self.output_queue, and writing structured logs.
        """
        # Set GPU for this worker - extract device ID from cuda:X format
        gpu_id = self.config.gpu_device.split(':')[-1] if ':' in self.config.gpu_device else '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

        # Setup structured logging for this worker
        worker_logger = LogManager.get_logger(
            f"worker_{self.config.worker_id}",
            f"gpu_{gpu_id}_{datetime.now().isoformat()}"
        )

        try:
            # Initialize embedder for this worker
            worker_logger.info("worker_initializing",
                worker_id=self.config.worker_id,
                gpu_id=gpu_id,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                model=self.config.model_name
            )

            embedder_config = {
                'device': 'cuda:0',  # Always use cuda:0 since we set CUDA_VISIBLE_DEVICES
                'use_fp16': self.config.use_fp16,
                'batch_size': self.config.embedding_batch_size,  # Use embedding batch size, not record batch size!
                'chunk_size_tokens': 500,  # Match JinaV4Embedder default
                'chunk_overlap_tokens': 200  # Match JinaV4Embedder default
                # Late chunking is implemented in the embedder itself
            }

            self.embedder = EmbedderFactory.create(
                model_name=self.config.model_name,
                **embedder_config
            )

            worker_logger.info("worker_ready",
                worker_id=self.config.worker_id,
                gpu_id=gpu_id
            )

            # Process batches
            batches_processed = 0
            while not self.stop_event.is_set():
                try:
                    # Get batch from queue (timeout to check stop event)
                    batch = self.input_queue.get(timeout=1.0)

                    if batch is None:  # Poison pill
                        worker_logger.info("worker_stopping",
                            worker_id=self.config.worker_id,
                            batches_processed=batches_processed,
                            reason="poison_pill"
                        )
                        break

                    worker_logger.info("batch_received",
                        worker_id=self.config.worker_id,
                        batch_size=len(batch),
                        batch_number=batches_processed + 1
                    )

                    # Process batch
                    results = self._process_batch(batch)
                    batches_processed += 1

                    worker_logger.info("batch_processed",
                        worker_id=self.config.worker_id,
                        batch_number=batches_processed,
                        results_count=len(results)
                    )

                    # Send results back
                    self.output_queue.put({
                        'worker_id': self.config.worker_id,
                        'results': results
                    })

                except Empty:
                    continue
                except Exception as e:
                    worker_logger.error("worker_error",
                        worker_id=self.config.worker_id,
                        error=str(e),
                        exc_info=True
                    )
                    self.output_queue.put({
                        'worker_id': self.config.worker_id,
                        'error': str(e)
                    })

        except Exception as e:
            worker_logger.error("worker_init_failed",
                worker_id=self.config.worker_id,
                error=str(e),
                exc_info=True
            )
        finally:
            worker_logger.info("worker_shutdown",
                worker_id=self.config.worker_id,
                batches_processed=batches_processed
            )

    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Process a batch of metadata records into late-chunked embeddings.
        
        Filters out records that lack an 'abstract', embeds the remaining abstracts using
        self.embedder.embed_batch_with_late_chunking(task="retrieval.passage"), and
        returns a flat list of result dictionaries mapping each original record to each
        generated chunk.
        
        Parameters:
            batch (List[Dict]): Input records; each record is expected to be a dict that
                may contain an 'abstract' key.
        
        Returns:
            List[Dict]: A list where each item is {'record': <original_record>, 'chunk': <chunk_dict>}.
        
        Raises:
            Exception: Re-raises any exception raised by the embedder while generating embeddings.
        """
        results = []

        # Extract abstracts for embedding (same as single-GPU workflow)
        abstracts_to_embed = []
        records_with_abstracts = []

        for record in batch:
            if record.get('abstract'):
                abstracts_to_embed.append(record['abstract'])
                records_with_abstracts.append(record)

        if abstracts_to_embed:
            try:
                # Use the WORKING method from single-GPU workflow
                all_chunks = self.embedder.embed_batch_with_late_chunking(
                    abstracts_to_embed,
                    task="retrieval.passage"
                )

                # Package results (same as single-GPU workflow)
                for record, record_chunks in zip(records_with_abstracts, all_chunks):
                    for chunk in record_chunks:
                        results.append({
                            'record': record,
                            'chunk': chunk
                        })

            except Exception as e:
                logger.error(f"Worker {self.config.worker_id} failed to embed batch: {e}")
                raise

        return results


class ArxivParallelWorkflow(WorkflowBase):
    """
    Parallel processing workflow for ArXiv metadata with multi-GPU embedding.

    Implements a producer-consumer pattern:
    - Main thread: Reads JSON and produces batches
    - Worker processes: Consume batches and generate embeddings
    - Storage thread: Consumes results and stores in database
    """

    def __init__(self, config: ArxivMetadataConfig):
        """
        Initialize the ArxivParallelWorkflow.
        
        Sets up logging, state and checkpoint managers, progress and performance monitors,
        processing queues and worker placeholders, counters, and a database handle
        placeholder using values from the provided ArxivMetadataConfig.
        
        Parameters:
            config (ArxivMetadataConfig): Workflow configuration (paths, batching and
                embedding settings, DB options, resume/drop flags, and other runtime
                options) used to initialize managers and runtime defaults.
        """
        super().__init__(config)
        self.metadata_config = config

        # Setup structured logging
        self.logger = LogManager.get_logger(
            "arxiv_parallel_workflow",
            f"run_{datetime.now().isoformat()}"
        )

        # State management
        self.state_manager = StateManager(
            state_file=config.state_file,
            process_name="arxiv_parallel_workflow"
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_file=config.checkpoint_file
        )

        # Monitoring
        self.progress_tracker = ProgressTracker(
            name="arxiv_parallel_workflow",
            description="Parallel ArXiv processing with multi-GPU embedding"
        )
        self.performance_monitor = PerformanceMonitor(
            component_name="arxiv_parallel_workflow"
        )

        # Processing state
        self.workers = []
        self.input_queue = None
        self.output_queue = None
        self.stop_event = None
        self.storage_thread = None

        # Counters
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0  # Track skipped records when resuming
        self.start_time = None

        # Database connection
        self.db = None

    def _initialize_components(self):
        """
        Initialize all runtime components required by the workflow.
        
        This sets up the ArangoDB connection and (optionally) drops and recreates the metadata, chunks, and embeddings
        collections; configures progress-tracker steps; creates inter-process communication primitives (spawn-context
        queues and a stop event); detects available GPUs and builds per-worker WorkerConfig objects; starts worker
        processes (using a spawn context to ensure GPU isolation) that run `worker_process_with_gpu`; and starts the
        storage thread that consumes worker outputs and writes to the database.
        
        Side effects:
        - Sets or mutates these instance attributes: db, input_queue, output_queue, stop_event, workers (appends Process
          objects), storage_thread.
        - May start multiple OS processes and a background thread.
        - May clear checkpoints and state when `metadata_config.drop_collections` is true.
        
        Behavior notes:
        - If `metadata_config.drop_collections` is true, existing collections are removed and checkpoint/state managers
          are cleared for a fresh run.
        - The progress tracker is initialized with totals derived from `metadata_config.max_records` or a large default.
        - GPU detection is used to adjust the effective number of workers if fewer GPUs are available than requested;
          workers are assigned to GPUs round-robin. If no GPUs are detected, workers use CPU.
        """
        self.logger.info("initializing_components")

        # Initialize database connection
        self.db = DatabaseFactory.get_arango(
            database=self.metadata_config.arango_database,
            username=self.metadata_config.arango_username,
            use_unix=True  # Use Unix socket for lowest latency
        )

        # Drop collections if requested
        if self.metadata_config.drop_collections:
            self.logger.warning("dropping_collections")
            for collection in [
                self.metadata_config.metadata_collection,
                self.metadata_config.chunks_collection,
                self.metadata_config.embeddings_collection
            ]:
                if self.db.has_collection(collection):
                    self.db.delete_collection(collection)
                    self.logger.info("collection_dropped", name=collection)

            # Also clear checkpoint when dropping collections for fresh start
            self.checkpoint_manager.clear()
            self.state_manager.clear()
            self.logger.info("checkpoint_cleared")

        # Ensure collections exist
        for collection in [
            self.metadata_config.metadata_collection,
            self.metadata_config.chunks_collection,
            self.metadata_config.embeddings_collection
        ]:
            if not self.db.has_collection(collection):
                self.db.create_collection(collection)

        # Initialize progress tracker steps
        total_records = self.metadata_config.max_records or 2828998
        self.progress_tracker.add_step(
            step_id="metadata_loading",
            name="Loading Metadata",
            total_items=total_records
        )
        self.progress_tracker.add_step(
            step_id="embedding_generation",
            name="Generating Embeddings",
            total_items=total_records
        )
        self.progress_tracker.add_step(
            step_id="database_storage",
            name="Storing in Database",
            total_items=total_records
        )
        self.logger.info("progress_tracker_initialized",
            total_records=total_records,
            steps=["metadata_loading", "embedding_generation", "database_storage"]
        )

        # Initialize queues using spawn context for inter-process communication
        ctx = mp.get_context('spawn')
        self.input_queue = ctx.Queue(maxsize=500)  # Limit queue size to control memory
        self.output_queue = ctx.Queue(maxsize=500)
        self.stop_event = ctx.Event()

        # Determine number of workers and GPU assignment
        num_workers = getattr(self.metadata_config, 'num_workers', 2)
        available_gpus = self._get_available_gpus()

        self.logger.info("gpu_detection",
            requested_workers=num_workers,
            available_gpus=available_gpus,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')
        )

        if len(available_gpus) < num_workers:
            self.logger.warning("insufficient_gpus",
                requested=num_workers,
                available=len(available_gpus)
            )
            num_workers = max(1, len(available_gpus))

        # Create worker configurations
        worker_configs = []
        for i in range(num_workers):
            gpu_device = f"cuda:{available_gpus[i % len(available_gpus)]}" if available_gpus else "cpu"
            worker_configs.append(WorkerConfig(
                worker_id=i,
                gpu_device=gpu_device,
                batch_size=self.metadata_config.batch_size,
                embedding_batch_size=self.metadata_config.embedding_batch_size,
                model_name=self.metadata_config.embedder_model,
                use_fp16=self.metadata_config.use_fp16
            ))
            self.logger.info("worker_config_created",
                worker_id=i,
                gpu_device=gpu_device
            )

        # Start worker processes using spawn context for proper GPU isolation
        self.logger.info("starting_workers", count=num_workers)

        for i, config in enumerate(worker_configs):
            # Extract just the GPU ID number
            gpu_id = config.gpu_device.split(':')[-1] if ':' in config.gpu_device else '0'

            # Create worker with spawn context
            worker = ctx.Process(
                target=worker_process_with_gpu,
                args=(config, self.input_queue, self.output_queue, self.stop_event, gpu_id)
            )
            worker.start()
            self.workers.append(worker)
            self.logger.info("worker_started",
                worker_id=config.worker_id,
                gpu_device=config.gpu_device,
                gpu_id=gpu_id
            )

        # Start storage thread
        self.storage_thread = Thread(target=self._storage_worker)
        self.storage_thread.start()

        self.logger.info("initialization_complete",
            num_workers=num_workers,
            gpu_assignments=[c.gpu_device for c in worker_configs]
        )

    def _get_available_gpus(self) -> List[int]:
        """
        Return a list of available GPU indices.
        
        This first attempts to detect GPUs via PyTorch (torch.cuda). If PyTorch is not installed or no GPUs are reported, it falls back to parsing the CUDA_VISIBLE_DEVICES environment variable (comma-separated indices). Returns an empty list if no GPUs are found by either method.
        
        Returns:
            List[int]: GPU device indices available to the process (e.g., [0, 1]). 
        """
        try:
            import torch
            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
        except ImportError:
            pass

        # Check CUDA_VISIBLE_DEVICES
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            return [int(d) for d in devices if d.strip().isdigit()]

        return []

    def _storage_worker(self):
        """
        Storage thread that consumes embedding results from the worker output queue and persists them to ArangoDB in batched, atomic transactions.
        
        This method runs in the main process. It continuously reads result payloads from self.output_queue until self.stop_event is set and the queue is drained. Expected payloads are dicts with either:
        - 'results': a list of items (each item is a dict containing a 'record' and its 'chunk'), or
        - an error payload containing an 'error' key (these are logged and skipped).
        
        Results are buffered and written in batches (internal batch_size = 500) via self._store_batch(...) to minimize transaction overhead. While buffering, the method periodically updates self.progress_tracker for the "embedding_generation" step in 500-item increments. On shutdown it flushes any remaining buffered items before exiting.
        
        Side effects:
        - Persists metadata, chunk, and embedding documents to the configured ArangoDB collections through self._store_batch.
        - Updates progress via self.progress_tracker.
        - Logs worker error payloads instead of raising them.
        
        Behavior notes:
        - Uses a short queue timeout to allow responsive shutdown.
        - Respects self.stop_event to determine when to flush remaining buffered results and terminate.
        """
        logger.info("Storage worker started")

        batch_buffer = []
        batch_size = 500  # Store in batches for efficiency

        while not self.stop_event.is_set() or not self.output_queue.empty():
            try:
                # Get result from queue
                result = self.output_queue.get(timeout=1.0)

                if 'error' in result:
                    logger.error(f"Worker {result['worker_id']} error: {result['error']}")
                    continue

                # Add to batch buffer
                batch_buffer.extend(result['results'])

                # Update embedding generation progress less frequently
                # Accumulate counts and update in batches
                if len(batch_buffer) % 500 == 0:  # Every 500 items
                    self.progress_tracker.update_step(
                        "embedding_generation",
                        completed=500  # Update in chunks
                    )

                # Store when batch is full
                if len(batch_buffer) >= batch_size:
                    self._store_batch(batch_buffer[:batch_size])
                    batch_buffer = batch_buffer[batch_size:]

            except Empty:
                # Timeout - check if we should flush partial batch
                if batch_buffer and self.stop_event.is_set():
                    self._store_batch(batch_buffer)
                    batch_buffer = []

        # Store any remaining records
        if batch_buffer:
            self._store_batch(batch_buffer)

        logger.info("Storage worker finished")

    def _load_existing_ids(self) -> set:
        """
        Return a set of arXiv IDs that already have embeddings, used to skip previously-processed records when resuming.
        
        If resume_from_checkpoint on the workflow config is False, returns an empty set immediately. Otherwise queries the embeddings collection (arxiv_abstract_embeddings) for stored arXiv IDs and returns them as a Python set for O(1) membership checks. On error (e.g., database query failure) the function logs the failure and returns an empty set so the workflow will conservatively reprocess records.
        """
        if not self.metadata_config.resume_from_checkpoint:
            return set()

        self.logger.info("loading_existing_ids_for_resume")
        start_time = time.time()

        try:
            # Query all existing IDs from the embeddings collection
            cursor = self.db.aql.execute('''
                FOR doc IN arxiv_abstract_embeddings
                    RETURN doc.arxiv_id
            ''', ttl=300, batch_size=10000)  # 5 minute timeout, large batch size

            # Convert to set for O(1) lookups
            existing_ids = set(cursor)

            elapsed = time.time() - start_time
            self.logger.info("existing_ids_loaded",
                count=len(existing_ids),
                elapsed_seconds=elapsed,
                memory_mb=len(existing_ids) * 50 / 1024 / 1024  # Rough estimate
            )

            return existing_ids

        except Exception as e:
            self.logger.error("failed_to_load_existing_ids", error=str(e))
            # If we can't load existing IDs, safer to process everything
            # rather than risk skipping records incorrectly
            return set()

    def _store_batch(self, batch: List[Dict]):
        """
        Store a batch of chunked embeddings and their metadata atomically in ArangoDB.
        
        Processes a list of items (each item is a dict with keys "record" and "chunk"), grouping by paper arXiv ID to insert one metadata document per paper and one chunk+embedding document per chunk. All writes for the batch are performed inside a single database transaction; on error the transaction is aborted and the batch is counted as failed.
        
        Detailed behavior:
        - Skips items that lack an arXiv ID.
        - Sanitizes arXiv IDs into document keys (replaces "." and "/" with "_").
        - Inserts (or overwrites) one metadata document per paper into the configured metadata collection.
        - Inserts (or overwrites) chunk documents into the configured chunks collection; chunk keys use the sanitized paper key and the chunk index.
        - Inserts (or overwrites) embedding documents into the configured embeddings collection; embeddings are stored as plain lists and include model and dimension metadata.
        - Commits the transaction on success; on failure attempts to abort the transaction and increments the workflow's failed_count.
        - On success increments processed_count by the number of items in the batch.
        - Updates the progress tracker for the "database_storage" step when batch size >= 100.
        - Logs throughput periodically (every 1000 processed records) using the workflow's start_time and metadata_config.max_records.
        
        Parameters:
            batch: iterable of dict items where each item contains:
                - 'record': the original metadata dict (must include 'id' for arXiv ID)
                - 'chunk': an object or mapping with attributes/keys used below:
                    chunk.chunk_index, chunk.total_chunks, chunk.text,
                    chunk.start_char, chunk.end_char, chunk.start_token, chunk.end_token,
                    chunk.context_window_used, chunk.embedding (array-like)
        
        Side effects:
        - Writes to three ArangoDB collections (metadata, chunks, embeddings) via a transaction.
        - Mutates self.processed_count and self.failed_count.
        - Updates the workflow's progress tracker and emits periodic log messages.
        """
        if not batch:
            return

        # Begin transaction
        txn = self.db.begin_transaction(
            write=[
                self.metadata_config.metadata_collection,
                self.metadata_config.chunks_collection,
                self.metadata_config.embeddings_collection
            ]
        )

        try:
            # Group by paper ID to avoid duplicates
            papers_seen = set()

            for item in batch:
                record = item['record']
                chunk = item['chunk']
                arxiv_id = record.get('id', '')

                if not arxiv_id:
                    continue

                sanitized_id = arxiv_id.replace('.', '_').replace('/', '_')

                # Store metadata (once per paper)
                if arxiv_id not in papers_seen:
                    papers_seen.add(arxiv_id)

                    metadata_doc = {
                        '_key': sanitized_id,
                        'arxiv_id': arxiv_id,
                        'submitter': record.get('submitter'),
                        'authors': record.get('authors'),
                        'title': record.get('title'),
                        'comments': record.get('comments'),
                        'journal_ref': record.get('journal-ref'),
                        'doi': record.get('doi'),
                        'report_no': record.get('report-no'),
                        'categories': record.get('categories'),
                        'license': record.get('license'),
                        'abstract': record.get('abstract'),
                        'versions': record.get('versions', []),
                        'update_date': record.get('update_date'),
                        'authors_parsed': record.get('authors_parsed', []),
                        'processed_at': datetime.now().isoformat(),
                        'has_abstract': bool(record.get('abstract'))
                    }

                    txn.collection(self.metadata_config.metadata_collection).insert(
                        metadata_doc, overwrite=True
                    )

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
                    'start_token': chunk.start_token,
                    'end_token': chunk.end_token,
                    'context_window_used': chunk.context_window_used,
                    'created_at': datetime.now().isoformat()
                }

                txn.collection(self.metadata_config.chunks_collection).insert(
                    chunk_doc, overwrite=True
                )

                # Store embedding
                embedding_doc = {
                    '_key': f"{chunk_id}_emb",
                    'chunk_id': chunk_id,
                    'arxiv_id': arxiv_id,
                    'paper_key': sanitized_id,
                    'embedding': chunk.embedding.tolist(),
                    'embedding_dim': len(chunk.embedding),
                    'model': self.metadata_config.embedder_model,
                    'created_at': datetime.now().isoformat()
                }

                txn.collection(self.metadata_config.embeddings_collection).insert(
                    embedding_doc, overwrite=True
                )

            # Commit transaction
            txn.commit_transaction()

            # Update counters
            self.processed_count += len(batch)

            # Update progress tracker less frequently for better performance
            # Only update for larger batches to reduce overhead
            if len(batch) >= 100:
                self.progress_tracker.update_step(
                    "database_storage",
                    completed=len(batch)
                )

            # Log progress less frequently to reduce I/O overhead
            # Changed from 100 to 1000 for better performance
            if self.processed_count % 1000 == 0:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                throughput = self.processed_count / elapsed
                if self.metadata_config.max_records:
                    logger.info(
                        f"Progress: {self.processed_count:,}/{self.metadata_config.max_records:,} "
                        f"({throughput:.2f} rec/s)"
                    )
                else:
                    logger.info(
                        f"Progress: {self.processed_count:,} processed "
                        f"({throughput:.2f} rec/s)"
                    )

        except Exception as e:
            logger.error(f"Failed to store batch: {e}")
            try:
                txn.abort_transaction()
            except:
                pass
            self.failed_count += len(batch)

    def _process_metadata_file(self):
        """
        Stream and enqueue ArXiv metadata records from the configured JSONL file for downstream embedding.
        
        Reads the metadata file path from self.metadata_config.metadata_file and streams it line-by-line (JSONL). Records without an abstract are skipped. If resume is enabled, existing arXiv IDs returned by self._load_existing_ids() are skipped. Valid records are accumulated into batches of size self.metadata_config.batch_size and each full batch is placed on self.input_queue for worker processes. After the file is exhausted any final partial batch is enqueued, then a poison pill (None) is sent once per worker to signal shutdown. Progress is reported via self.progress_tracker and basic counters (self.skipped_count) are maintained.
        
        Exceptions:
            FileNotFoundError: if the configured metadata file does not exist.
        
        Notes:
            - Malformed JSON lines are logged and skipped (no exception is propagated).
            - The method does not return a value; its effects are enqueuing batches and updating workflow state.
        """
        metadata_file = Path(self.metadata_config.metadata_file)

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load existing IDs if resuming
        existing_ids = self._load_existing_ids()

        self.logger.info("processing_metadata_file",
            file=str(metadata_file),
            max_records=self.metadata_config.max_records,
            batch_size=self.metadata_config.batch_size,
            num_workers=len(self.workers),
            resume_mode=bool(existing_ids),
            existing_count=len(existing_ids)
        )

        # Track progress
        self.progress_tracker.start_step("metadata_loading")

        batch = []
        total_processed = 0
        total_skipped = 0
        batches_queued = 0
        records_scanned = 0

        with open(metadata_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Check if we've reached max records (based on NEW records processed)
                if self.metadata_config.max_records and total_processed >= self.metadata_config.max_records:
                    break

                try:
                    record = json.loads(line)
                    records_scanned += 1

                    # Skip if no abstract
                    if not record.get('abstract'):
                        continue

                    # Skip if already processed (when resuming)
                    arxiv_id = record.get('id')
                    if arxiv_id and arxiv_id in existing_ids:
                        total_skipped += 1
                        # Log progress every 10,000 skipped records
                        if total_skipped % 10000 == 0:
                            self.logger.info("skipping_processed_records",
                                skipped_count=total_skipped,
                                line_number=line_num
                            )
                        continue

                    batch.append(record)
                    total_processed += 1

                    # Send batch to workers when full
                    if len(batch) >= self.metadata_config.batch_size:
                        # Don't split batches - queue the whole batch
                        # Workers will take batches from queue as they're ready
                        self.input_queue.put(batch)
                        batches_queued += 1

                        # Update metadata loading progress less frequently
                        if batches_queued % 10 == 0:
                            self.progress_tracker.update_step(
                                "metadata_loading",
                                completed=len(batch) * 10  # Account for batches since last update
                            )

                        # Only log every 10th batch to reduce I/O
                        if batches_queued % 10 == 0:
                            self.logger.info("batch_queued",
                                batch_number=batches_queued,
                                batch_size=len(batch),
                                queue_size=self.input_queue.qsize() if hasattr(self.input_queue, 'qsize') else 'unknown',
                                total_processed=total_processed,
                                total_skipped=total_skipped
                            )

                        batch = []

                except json.JSONDecodeError as e:
                    self.logger.warning("invalid_json",
                        line_num=line_num,
                        error=str(e)
                    )
                    continue

        # Process remaining batch
        if batch:
            self.input_queue.put(batch)
            batches_queued += 1

            # Update metadata loading progress for final batch
            self.progress_tracker.update_step(
                "metadata_loading",
                completed=len(batch)
            )

            self.logger.info("final_batch_queued",
                batch_size=len(batch),
                total_batches=batches_queued
            )

        # Send stop signal to workers
        for worker_id in range(len(self.workers)):
            self.input_queue.put(None)
            self.logger.info("poison_pill_sent", worker_id=worker_id)

        # Save skipped count for final reporting
        self.skipped_count = total_skipped

        self.logger.info("metadata_loading_complete",
            total_records=total_processed,
            total_skipped=total_skipped,
            records_scanned=records_scanned,
            total_batches=batches_queued,
            workers=len(self.workers)
        )

    def execute(self) -> Any:
        """
        Run the full multi-GPU ArXiv embedding workflow and return a summary result.
        
        This method orchestrates the end-to-end pipeline:
        - initializes database, queues, worker processes, and storage thread,
        - streams metadata and dispatches batches to GPU workers,
        - waits for workers to finish and then stops the storage thread,
        - computes timing and throughput metrics and returns a WorkflowResult summarizing success, counts, timing, and relevant metadata.
        
        On unhandled exceptions the method attempts a safe shutdown (sets the stop event and terminates any alive workers) and returns a failed WorkflowResult containing the error message.
        
        Returns:
            WorkflowResult: Structured summary of the run including success flag, item counts, start/end times, and metadata (throughput, duration, workers, config options, or error details on failure).
        """
        try:
            self.start_time = datetime.now()

            # Initialize components
            self._initialize_components()

            # Start monitoring
            #self.performance_monitor.start_monitoring()

            # Process metadata file
            self._process_metadata_file()

            # Wait for workers to finish
            logger.info("Waiting for workers to complete...")
            for worker in self.workers:
                worker.join(timeout=300)  # 5 minute timeout per worker

            # Signal storage thread to stop and wait
            self.stop_event.set()
            if self.storage_thread:
                self.storage_thread.join()

            # Stop monitoring
            #self.performance_monitor.stop_monitoring()

            # Calculate final metrics
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()

            logger.info(f"Workflow completed in {duration:.2f} seconds")
            logger.info(f"Processed: {self.processed_count:,} new records")
            if self.skipped_count > 0:
                logger.info(f"Skipped: {self.skipped_count:,} already-processed records")
            logger.info(f"Failed: {self.failed_count:,} records")
            logger.info(f"Throughput: {self.processed_count / duration:.2f} records/second")

            # Return result
            from core.workflows.workflow_base import WorkflowResult
            return WorkflowResult(
                workflow_name="arxiv_parallel_workflow",
                success=True,
                items_processed=self.processed_count,
                items_failed=self.failed_count,
                start_time=self.start_time,
                end_time=end_time,
                metadata={
                    'throughput': self.processed_count / duration if duration > 0 else 0,
                    'duration_seconds': duration,
                    'items_skipped': self.skipped_count,
                    'num_workers': len(self.workers),
                    'metadata_file': str(self.metadata_config.metadata_file),
                    'embedder_model': self.metadata_config.embedder_model,
                    'batch_size': self.metadata_config.batch_size,
                    'resume_mode': self.metadata_config.resume_from_checkpoint
                }
            )

        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)

            # Clean up
            if self.stop_event:
                self.stop_event.set()

            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()

            # Return failure result
            from core.workflows.workflow_base import WorkflowResult
            return WorkflowResult(
                workflow_name="arxiv_parallel_workflow",
                success=False,
                items_processed=self.processed_count,
                items_failed=self.failed_count,
                start_time=self.start_time or datetime.utcnow(),
                end_time=datetime.utcnow(),
                metadata={'error': str(e)}
            )

    def validate_inputs(self, **kwargs) -> bool:
        """Validate workflow inputs."""
        return True

    def cleanup(self):
        """
        Shut down the workflow and release multiprocessing resources.
        
        Stops the shared stop event (if present), forcefully terminates any worker
        processes that are still alive, and drains the input and output queues to
        remove pending items. Intended to be safe to call multiple times during
        abnormal shutdown or cleanup phases; side effects include terminating child
        processes and removing queued work.
        """
        # Stop event
        if self.stop_event:
            self.stop_event.set()

        # Terminate workers if still running
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)

        # Clear queues
        if self.input_queue:
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except:
                    break

        if self.output_queue:
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except:
                    break


if __name__ == "__main__":
    """
    CLI entry point for multi-GPU ArXiv processing.

    Usage:
        python -m core.workflows.workflow_arxiv_parallel --count 1000 --workers 2
    """
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Process ArXiv metadata with multi-GPU parallel processing"
    )

    parser.add_argument(
        '--count',
        type=int,
        default=None,
        help='Number of records to process (default: all documents)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Records per batch (default: 100)'
    )

    parser.add_argument(
        '--embedding-batch-size',
        type=int,
        default=32,
        help='Embeddings per GPU batch (default: 32)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help='Number of GPU workers (default: 2)'
    )

    parser.add_argument(
        '--drop-collections',
        action='store_true',
        help='Drop existing collections before processing'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )

    args = parser.parse_args()

    # Check environment
    if not os.environ.get('ARANGO_PASSWORD'):
        print("ERROR: ARANGO_PASSWORD environment variable not set")
        print("Please set: export ARANGO_PASSWORD='your-password'")
        sys.exit(1)

    # Determine record count
    # If --count not specified: process all documents
    if args.count is None:
        max_records = None  # Process all documents
        display_count = "all"
    else:
        max_records = args.count
        display_count = str(args.count)

    # Create configuration
    from core.tools.arxiv.arxiv_metadata_config import ArxivMetadataConfig

    # Calculate checkpoint interval safely
    if max_records:
        checkpoint_interval = min(1000, max_records // 10)
    else:
        checkpoint_interval = 10000  # Default for unlimited processing

    config = ArxivMetadataConfig(
        max_records=max_records,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        num_workers=args.workers,
        drop_collections=args.drop_collections,
        resume_from_checkpoint=args.resume,
        checkpoint_interval=checkpoint_interval,
        monitor_interval=100
    )

    print("=" * 60)
    print("ArXiv Parallel Processing")
    print("=" * 60)
    print(f"Records: {display_count}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding batch: {args.embedding_batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Drop collections: {args.drop_collections}")
    print(f"Resume: {args.resume}")
    print("=" * 60)

    # Run workflow
    workflow = ArxivParallelWorkflow(config)
    result = workflow.execute()

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Processed: {result.items_processed}")
    print(f"Failed: {result.items_failed}")
    print(f"Duration: {(result.end_time - result.start_time).total_seconds():.2f} seconds")

    if result.metadata.get('throughput'):
        print(f"Throughput: {result.metadata['throughput']:.2f} records/second")

    sys.exit(0 if result.success else 1)