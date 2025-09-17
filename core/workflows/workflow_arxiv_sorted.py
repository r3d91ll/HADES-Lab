#!/usr/bin/env python3
"""
ArXiv Size-Sorted Processing Workflow
=====================================

Optimized workflow for processing ArXiv metadata with size-based sorting
for improved throughput and GPU utilization.

This workflow implements the complete pipeline:
1. Load and sort metadata by abstract size (smallest first)
2. Process abstracts with late chunking in size order
3. Generate embeddings in parallel using multiple GPU workers
4. Store everything atomically in ArangoDB with position tracking

Theory Connection (Conveyance Framework):
Optimizes C = (W·R·H/T)·Ctx^α by:
- Minimizing T through size-ordered processing (small docs process faster)
- Maximizing H through better GPU batch utilization
- Preserving Ctx through late chunking
- Amplifying efficiency by processing similar-sized documents together
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
    Run a single worker process pinned to a specific GPU that embeds batches of ArXiv abstracts and streams results back to the main process.
    
    This function is intended to be executed in a multiprocessing child process. It enforces GPU isolation by setting CUDA_VISIBLE_DEVICES to gpu_id before importing any CUDA-dependent libraries, initializes a per-worker embedder, and then repeatedly consumes record batches from input_queue. Each batch is filtered for records that contain an 'abstract'; those abstracts are embedded using late-chunking and each produced chunk is paired with its originating record. Results are emitted to output_queue as dictionaries of the form {'worker_id': <int>, 'results': [ {'record': ..., 'chunk': ...}, ... ]}. If a batch of None is read from input_queue it is treated as a poison pill and causes graceful shutdown.
    
    Notable side effects:
    - Sets environment variables 'CUDA_VISIBLE_DEVICES' and 'TOKENIZERS_PARALLELISM'.
    - Imports CUDA-dependent modules after setting CUDA_VISIBLE_DEVICES.
    - Initializes an embedder configured to use 'cuda:0' (because the process's visible device list is isolated to the target GPU).
    - Emits structured log events via the per-worker logger and places error payloads ( {'worker_id': ..., 'error': <str>} ) on output_queue when exceptions occur.
    
    Parameters:
    - config: WorkerConfig dataclass containing worker_id, model_name, use_fp16, embedding_batch_size, and other per-worker settings.
    - gpu_id: string or int used to set CUDA_VISIBLE_DEVICES for this process.
    
    The function does not return a value.
    """
    # CRITICAL: Set GPU BEFORE any CUDA imports
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    print(f"Worker process started - GPU {gpu_id}, PID {os.getpid()}")

    # NOW import CUDA-dependent modules
    try:
        from core.embedders.embedders_factory import EmbedderFactory
        from core.logging.logging import LogManager
        from queue import Empty
        import time
        print(f"Worker {config.worker_id} - imports successful")
    except Exception as e:
        print(f"Worker {config.worker_id} - IMPORT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

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
                 Initialize an embedding worker process.
                 
                 Parameters:
                     config: WorkerConfig with per-worker settings (worker_id, gpu_device, batch sizes, model_name, use_fp16).
                     input_queue: multiprocessing.Queue providing batches (lists) of records to embed; None is used as a poison pill to stop the worker.
                     output_queue: multiprocessing.Queue where the worker places embedding results or error records.
                     stop_event: multiprocessing.Event that can be set by the parent to request early shutdown.
                 """
        super().__init__()
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.embedder = None

    def run(self):
        """
        Run the worker process: initialize the per-GPU embedder, consume record batches from the input queue, produce per-record embedding chunks, and send results or errors to the output queue.
        
        This method:
        - Sets CUDA_VISIBLE_DEVICES so the process is bound to the configured GPU and creates a per-worker logger.
        - Lazily initializes an embedder configured to run on the worker's GPU (the embedder uses late-chunking internally).
        - Loops until the shared stop event is set or a None "poison pill" is received:
          - Reads a batch from self.input_queue with a short timeout to allow for responsive shutdown.
          - Treats None as a signal to stop and exits the loop.
          - Processes each batch via self._process_batch, increments a local processed counter, and puts a result dict onto self.output_queue of the form {'worker_id': ..., 'results': ...}.
        - On batch- or initialization-level exceptions, logs the error and forwards an error entry to self.output_queue of the form {'worker_id': ..., 'error': <str>} so the parent process can handle failures.
        - Ensures final shutdown logging regardless of success or failure.
        
        Side effects:
        - Mutates the process environment variable CUDA_VISIBLE_DEVICES.
        - Initializes and retains self.embedder.
        - Emits structured log events and writes messages into input/output multiprocessing queues.
        
        Returns:
            None
        """
        # Set GPU for this worker - extract device ID from cuda:X format
        gpu_id = self.config.gpu_device.split(':')[-1] if ':' in self.config.gpu_device else '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

        # Setup structured logging for this worker
        worker_logger = LogManager.get_logger(
            f"worker_{self.config.worker_id}",
            f"gpu_{gpu_id}_{datetime.now().isoformat()}"
        )

        # Log that worker is starting
        print(f"WORKER {self.config.worker_id} STARTING on GPU {gpu_id}")

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
                        batch_number=batches_processed + 1,
                        first_record_id=batch[0].get('arxiv_id', 'unknown') if batch else 'empty'
                    )

                    # Process batch
                    results = self._process_batch(batch)
                    batches_processed += 1

                    worker_logger.info("batch_processed",
                        worker_id=self.config.worker_id,
                        batch_number=batches_processed,
                        results_count=len(results),
                        chunks_created=len(results) if results else 0
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
        Embed abstracts from a batch and return per-record embedding chunks.
        
        Only records that include a non-empty 'abstract' field are embedded; records without abstracts are ignored. For each input record with an abstract, this calls the instance embedder's embed_batch_with_late_chunking and produces one result entry per returned chunk: a dict with keys 'record' (the original record) and 'chunk' (a single embedding chunk produced for that record).
        
        Parameters:
            batch (List[Dict]): Iterable of record dictionaries; each record is expected to contain an 'abstract' string to be embedded.
        
        Returns:
            List[Dict]: A flat list of {'record': record, 'chunk': chunk} mappings for every chunk produced.
        
        Raises:
            Exception: Propagates any exception raised by the embedder (embedding failures are logged before re-raising).
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


class ArxivSortedWorkflow(WorkflowBase):
    """
    Size-sorted processing workflow for ArXiv metadata with multi-GPU embedding.

    Implements a size-ordered producer-consumer pattern:
    - Main thread: Reads JSON and produces batches
    - Worker processes: Consume batches and generate embeddings
    - Storage thread: Consumes results and stores in database
    """

    def __init__(self, config: ArxivMetadataConfig):
        """
        Initialize the ArxivSortedWorkflow with configuration and prepare runtime components.
        
        Creates and stores the provided ArxivMetadataConfig, initializes structured logging, state and checkpoint managers, progress and performance trackers, and default runtime fields used during execution (worker list, inter-process queues and events, storage thread handle, counters, start time placeholder, and database handle).
        Parameters:
            config (ArxivMetadataConfig): Workflow configuration containing runtime options, file paths for state/checkpointing, database and embedding settings.
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
        Initialize and wire up all runtime components required by the workflow.
        
        Sets up the ArangoDB connection and required collections (optionally dropping and clearing state when
        drop_collections is enabled), registers progress-tracker steps, creates inter-process IPC primitives
        (using the "spawn" multiprocessing context), detects available GPUs and builds per-worker configurations,
        launches per-GPU embedding worker processes, and starts the storage thread that persists embedding chunks
        to the database.
        
        Side effects:
        - Assigns self.db, self.input_queue, self.output_queue, self.stop_event, self.workers, and self.storage_thread.
        - May drop and recreate the metadata, chunks, and embeddings collections and clear checkpoint/state managers
          when metadata_config.drop_collections is true.
        - May reduce the configured number of workers to match available GPUs.
        - Starts multiprocessing worker processes (via spawn) and a storage Thread.
        
        No return value.
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
            print(f"Starting worker {config.worker_id} on GPU {gpu_id}")
            worker = ctx.Process(
                target=worker_process_with_gpu,
                args=(config, self.input_queue, self.output_queue, self.stop_event, gpu_id)
            )
            worker.start()
            self.workers.append(worker)
            print(f"Worker {config.worker_id} started, PID: {worker.pid}")
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
        Return a list of available GPU device indices.
        
        Tries to detect GPUs via PyTorch (torch.cuda). If PyTorch is available and reports CUDA devices, returns
        a list of indices [0..device_count-1]. If PyTorch is not available or reports no devices, falls back to
        parsing the CUDA_VISIBLE_DEVICES environment variable (comma-separated device IDs); non-integer entries are ignored.
        Returns an empty list when no GPUs are detected.
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
        Storage thread that consumes embedding results from the output queue and writes them to the database in transactional batches.
        
        This method runs in the main process and continuously drains self.output_queue until stop_event is set and the queue is empty. It:
        - Aggregates incoming result items into an internal buffer and calls self._store_batch(...) in batches (default batch size 500) for efficient transactional writes.
        - Skips and logs any result entries containing an 'error' key.
        - Periodically updates the "embedding_generation" progress step in 500-item increments.
        - Flushes any remaining buffer when stopping.
        
        Side effects:
        - Performs database writes via self._store_batch.
        - Updates self.progress_tracker.
        - Logs errors and lifecycle events.
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
        Return the set of arXiv IDs that already have embeddings in the database.
        
        If resume_from_checkpoint is False this returns an empty set. On any error while querying the database the method logs the failure and returns an empty set (safer to reprocess all records).
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
        Store a batch of record/chunk items into the database atomically.
        
        Processes a list of items (each expected to be a dict with keys 'record' and 'chunk'), inserts chunk documents into the configured chunks collection and corresponding embedding documents into the embeddings collection inside a single transaction, and updates in-memory counters and progress tracking.
        
        Parameters:
            batch (List[Dict]): List of items where each item contains:
                - 'record': a mapping that must include either 'arxiv_id' or 'id' identifying the paper.
                - 'chunk': an object with attributes used to build documents (chunk_index, total_chunks, text, start_char, end_char, start_token, end_token, context_window_used, embedding).
        
        Side effects:
            - Performs transactional writes to the metadata-configured chunks and embeddings collections.
            - Increments self.processed_count by the number of unique papers stored.
            - On error, aborts the transaction (if possible) and increments self.failed_count by len(batch).
            - Emits progress updates to self.progress_tracker and logs periodic throughput.
        
        Notes:
            - Paper IDs are sanitized (dots and slashes replaced with underscores) to form document keys.
            - Empty or items without an identifiable arXiv ID are skipped.
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
                # Handle both cases: 'id' from file loading, 'arxiv_id' from database query
                arxiv_id = record.get('arxiv_id') or record.get('id', '')

                if not arxiv_id:
                    continue

                sanitized_id = arxiv_id.replace('.', '_').replace('/', '_')

                # Track unique papers for counting
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

            # Update counters - count unique papers, not chunks
            unique_papers = len(papers_seen)
            self.processed_count += unique_papers

            self.logger.info("batch_stored",
                papers=unique_papers,
                chunks=len(batch),
                total_processed=self.processed_count)

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

    def _load_metadata_to_database(self):
        """
        Load metadata JSON lines into the configured metadata collection and return the number of documents present after the load.
        
        This routine:
        - Reads the metadata file (JSONL) configured at self.metadata_config.metadata_file.
        - Skips records that do not contain an 'abstract'.
        - Batches documents (default batch_size=10000) and imports them with on_duplicate='replace' so repeated runs can overwrite existing entries.
        - Continues on single-record or batch errors (logs warnings/errors but does not raise for malformed lines or partial batch failures).
        - Emits progress and diagnostic logs at regular intervals and on import results.
        - After processing the file, imports any remaining documents and verifies the final collection count.
        
        Returns:
            int: The actual number of documents in the metadata collection after the load (collection.count()).
        
        Raises:
            FileNotFoundError: If the configured metadata file does not exist.
        """
        metadata_file = Path(self.metadata_config.metadata_file)

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        self.logger.info("loading_metadata_to_database",
            file=str(metadata_file))

        batch = []
        batch_size = 10000
        total_loaded = 0
        total_lines = 0

        with open(metadata_file, 'r') as f:
            self.logger.info("starting_file_read",
                file=metadata_file,
                batch_size=batch_size)

            # Add detailed tracking variables
            lines_read = 0
            records_with_abstract = 0
            records_without_abstract = 0
            batches_imported = 0

            for line_num, line in enumerate(f, 1):
                total_lines = line_num
                lines_read += 1

                # More frequent progress logging for debugging
                if line_num % 1000 == 0:
                    self.logger.info("detailed_reading_progress",
                        line_num=line_num,
                        lines_read=lines_read,
                        loaded_so_far=total_loaded,
                        with_abstract=records_with_abstract,
                        without_abstract=records_without_abstract,
                        in_current_batch=len(batch),
                        batches_imported=batches_imported)

                # Extra logging at critical points
                if line_num == 9000:
                    self.logger.warning("AT_LINE_9000",
                        loaded=total_loaded,
                        batch_size=len(batch))
                elif line_num == 10000:
                    self.logger.warning("AT_LINE_10000",
                        loaded=total_loaded,
                        batch_size=len(batch))
                elif line_num == 11000:
                    self.logger.warning("PASSED_LINE_11000",
                        loaded=total_loaded,
                        batch_size=len(batch))

                try:
                    record = json.loads(line)

                    # Skip if no abstract
                    if not record.get('abstract'):
                        records_without_abstract += 1
                        continue

                    records_with_abstract += 1

                    # Store in database
                    doc = {
                        '_key': record.get('id', '').replace('.', '_').replace('/', '_'),
                        'arxiv_id': record.get('id'),
                        'title': record.get('title'),
                        'abstract': record.get('abstract'),
                        'authors': record.get('authors'),
                        'categories': record.get('categories'),
                        'update_date': record.get('update_date'),
                        'journal_ref': record.get('journal_ref'),
                        'doi': record.get('doi'),
                        'abstract_length': len(record.get('abstract', ''))
                    }
                    batch.append(doc)
                    total_loaded += 1

                    if len(batch) >= batch_size:
                        self.logger.info("batch_full_importing",
                            batch_size=len(batch),
                            line_num=line_num,
                            total_loaded_before=total_loaded)

                        # Import with on_duplicate='replace' to handle any duplicates
                        try:
                            result = self.db[self.metadata_config.metadata_collection].import_bulk(
                                batch,
                                on_duplicate='replace',
                                details=True
                            )

                            batches_imported += 1

                            # Calculate actual successful imports
                            successful = result.get('created', 0) + result.get('updated', 0)

                            self.logger.info("batch_imported_detailed",
                                batch_number=batches_imported,
                                batch_size=len(batch),
                                total_loaded=total_loaded,
                                created=result.get('created', 0),
                                updated=result.get('updated', 0),
                                errors=result.get('errors', 0),
                                successful=successful,
                                line_num=line_num)

                            # If there were errors, log them but continue
                            if result.get('errors', 0) > 0:
                                self.logger.warning("import_errors_detailed",
                                    batch_number=batches_imported,
                                    errors=result.get('errors'),
                                    details=result.get('details', [])[:10])  # Log first 10 error details

                                # Log high error count but CONTINUE loading
                                if result.get('errors', 0) > 100:
                                    self.logger.warning("high_error_count_but_continuing",
                                        batch_number=batches_imported,
                                        errors=result.get('errors'),
                                        successful=successful,
                                        batch_size=len(batch),
                                        continuing_at_line=line_num)
                                    # DON'T raise exception - continue loading rest of file!
                        except Exception as e:
                            self.logger.error("batch_import_failed",
                                error=str(e),
                                batch_size=len(batch),
                                batch_number=batches_imported,
                                line_num=line_num)
                            import traceback
                            self.logger.error("traceback", tb=traceback.format_exc())
                            # Continue with next batch instead of failing
                        finally:
                            batch = []
                            self.logger.info("batch_reset_continuing",
                                line_num=line_num,
                                total_loaded=total_loaded)

                    # Log progress every 100k records
                    if total_loaded % 100000 == 0:
                        self.logger.info("loading_milestone",
                            loaded=total_loaded,
                            line_number=line_num)

                except json.JSONDecodeError as e:
                    self.logger.warning("invalid_json",
                        line=line_num,
                        error=str(e))
                    continue
                except Exception as e:
                    self.logger.error("unexpected_error_in_line_processing",
                        line=line_num,
                        error=str(e),
                        type=type(e).__name__)
                    import traceback
                    self.logger.error("traceback", tb=traceback.format_exc())
                    # Don't let one bad record stop everything
                    continue

        # Log that we exited the file reading loop
        self.logger.warning("FILE_READING_LOOP_EXITED",
            total_lines_read=total_lines,
            total_loaded=total_loaded,
            records_with_abstract=records_with_abstract,
            records_without_abstract=records_without_abstract,
            batches_imported=batches_imported,
            final_batch_size=len(batch))

        # Import remaining batch
        if batch:
            result = self.db[self.metadata_config.metadata_collection].import_bulk(
                batch,
                on_duplicate='replace',
                details=True
            )
            self.logger.info("final_batch_imported",
                batch_size=len(batch),
                created=result.get('created', 0),
                updated=result.get('updated', 0),
                errors=result.get('errors', 0))

        self.logger.info("metadata_load_complete",
            total_loaded=total_loaded,
            total_lines_read=total_lines,
            final_batch_size=len(batch) if batch else 0)

        # Verify what actually made it to database
        actual_count = self.db[self.metadata_config.metadata_collection].count()
        self.logger.info("metadata_verification",
            loaded_count=total_loaded,
            database_count=actual_count,
            difference=total_loaded - actual_count)

        return actual_count  # Return what's actually in database, not what we tried to load

    def _get_unprocessed_records_sorted(self):
        """
        Return unprocessed metadata records sorted by abstract length (ascending).
        
        This queries the metadata collection and returns records that do not yet have
        corresponding embeddings, ordered from shortest to longest abstract. If the
        embeddings collection is empty the function uses a fast path that selects all
        documents with a non-null abstract; otherwise it filters out documents that
        already have at least one embedding. The query respects `self.metadata_config.max_records`
        when set.
        
        Returns:
            list[dict]: A list of result objects, each with keys:
                - 'id' (str): arXiv identifier (arxiv_id).
                - 'record' (dict): the full metadata document from the metadata collection.
                - 'abstract_length' (int): length of the abstract used for sorting.
        
        Notes:
            - All results are loaded into memory before being returned.
            - Returns an empty list if no matching records are found.
        """
        self.logger.info("querying_unprocessed_records")

        # Check if embeddings collection is empty (common after drop-collections)
        embeddings_count = self.db[self.metadata_config.embeddings_collection].count()

        if embeddings_count == 0:
            # Fast path: No embeddings exist, so ALL records are unprocessed
            # Just get all records with abstracts sorted by length
            self.logger.info("fast_path_no_embeddings_exist")

            if self.metadata_config.max_records:
                query = f"""
                FOR doc IN {self.metadata_config.metadata_collection}
                    FILTER doc.abstract != null
                    SORT doc.abstract_length ASC
                    LIMIT {self.metadata_config.max_records}
                    RETURN {{
                        'id': doc.arxiv_id,
                        'record': doc,
                        'abstract_length': doc.abstract_length
                    }}
                """
            else:
                query = f"""
                FOR doc IN {self.metadata_config.metadata_collection}
                    FILTER doc.abstract != null
                    SORT doc.abstract_length ASC
                    RETURN {{
                        'id': doc.arxiv_id,
                        'record': doc,
                        'abstract_length': doc.abstract_length
                    }}
                """
        else:
            # Slower path: Need to check which records have embeddings
            self.logger.info("checking_for_existing_embeddings",
                embeddings_count=embeddings_count)

            # Original query with embedding check
            query = f"""
            FOR doc IN {self.metadata_config.metadata_collection}
                FILTER doc.abstract != null
                LET has_embedding = FIRST(
                    FOR e IN {self.metadata_config.embeddings_collection}
                        FILTER e.arxiv_id == doc.arxiv_id
                        LIMIT 1
                        RETURN 1
                )
                FILTER has_embedding == null
                SORT doc.abstract_length ASC
                {'LIMIT ' + str(self.metadata_config.max_records) if self.metadata_config.max_records else ''}
                RETURN {{
                    'id': doc.arxiv_id,
                    'record': doc,
                    'abstract_length': doc.abstract_length
                }}
            """

        # Execute query and load all results into memory
        cursor = self.db.aql.execute(
            query,
            batch_size=10000,
            count=True
        )

        # Load all results into memory (we have 256GB RAM)
        records = list(cursor)

        self.logger.info("unprocessed_records_loaded",
            total_records=len(records))

        # Log size distribution
        if records:
            min_chars = records[0]['abstract_length']
            max_chars = records[-1]['abstract_length']
            median_chars = records[len(records)//2]['abstract_length'] if len(records) > 0 else 0

            self.logger.info("size_distribution",
                min_chars=min_chars,
                max_chars=max_chars,
                median_chars=median_chars,
                total_records=len(records))

        return records

    # Removed _store_processing_order and _get_resume_position methods - no longer needed
    # The query-based approach handles resume automatically by finding unprocessed records

    def _process_metadata_file(self):
        """Process the ArXiv metadata in size-sorted order."""
        self.logger.info("starting_sorted_processing")

        # If drop_collections, load metadata from file first
        if self.metadata_config.drop_collections:
            self.logger.info("loading_metadata_from_file")
            loaded_count = self._load_metadata_to_database()
            self.logger.info("finished_loading_metadata", count=loaded_count)

            # Verify the load worked
            actual_count = self.db[self.metadata_config.metadata_collection].count()
            self.logger.info("verified_collection_count", count=actual_count)

        # Query for unprocessed records sorted by size
        sorted_records = self._get_unprocessed_records_sorted()

        # Process in sorted order
        if sorted_records:
            self._process_sorted_records(sorted_records)
        else:
            self.logger.info("no_unprocessed_records")

    def _process_sorted_records(self, sorted_records):
        """
        Process size-sorted records, annotate them with ordering metadata, and enqueue them for embedding.
        
        Takes a list of pre-filtered, size-sorted records and streams them to worker processes in batches sized by self.metadata_config.batch_size. For each record the method adds two fields in-place on the record dict:
        - `size_order_position` (int): zero-based index in the sorted sequence
        - `abstract_length` (int): length used for sorting (copied from the input item)
        
        Batches are put onto self.input_queue; progress is reported to self.progress_tracker ("metadata_loading") periodically (every 10 batches) and after the final partial batch. After all records are enqueued, a `None` poison pill is sent once per worker to signal shutdown.
        
        Parameters:
            sorted_records (Sequence[Mapping]): Iterable of items previously returned by _get_unprocessed_records_sorted; each item is expected to contain at least `record` (a dict) and `abstract_length` (int).
        
        Returns:
            None
        """
        self.logger.info("processing_sorted_records",
            total_records=len(sorted_records),
            batch_size=self.metadata_config.batch_size,
            num_workers=len(self.workers)
        )

        # Track progress
        self.progress_tracker.start_step("metadata_loading")

        batch = []
        total_processed = 0
        batches_queued = 0

        # Process all records (they're already filtered for unprocessed)
        for position, item in enumerate(sorted_records):
            record = item['record']

            # Add position and abstract length for tracking
            record['size_order_position'] = position
            record['abstract_length'] = item['abstract_length']

            batch.append(record)
            total_processed += 1

            # Send batch to workers when full
            if len(batch) >= self.metadata_config.batch_size:
                self.input_queue.put(batch)
                batches_queued += 1

                # Update metadata loading progress less frequently
                if batches_queued % 10 == 0:
                    self.progress_tracker.update_step(
                        "metadata_loading",
                        completed=total_processed
                    )

                    # Log progress with size information
                    current_char_avg = sum(r['abstract_length'] for r in batch) / len(batch)
                    self.logger.info("batch_queued",
                        batch_number=batches_queued,
                        batch_size=len(batch),
                        position=position,
                        total=len(sorted_records),
                        avg_chars=current_char_avg,
                        queue_size=self.input_queue.qsize() if hasattr(self.input_queue, 'qsize') else 'unknown',
                        total_processed=total_processed,
                        progress_pct=(position/len(sorted_records)*100)
                    )

                batch = []

        # Process remaining batch
        if batch:
            self.input_queue.put(batch)
            batches_queued += 1

            self.progress_tracker.update_step(
                "metadata_loading",
                completed=total_processed
            )

            self.logger.info("final_batch_queued",
                batch_size=len(batch),
                total_batches=batches_queued
            )

        # Send stop signal to workers
        for worker_id in range(len(self.workers)):
            self.input_queue.put(None)
            self.logger.info("poison_pill_sent", worker_id=worker_id)

        self.logger.info("sorted_loading_complete",
            total_records=total_processed,
            skipped_from_resume=self.skipped_count,
            total_batches=batches_queued,
            workers=len(self.workers)
        )

    def execute(self) -> Any:
        """
        Run the full size-sorted ArXiv embedding workflow end-to-end.
        
        Initializes components (DB, worker processes, queues), loads and processes metadata in ascending abstract length order, waits for GPU embedding workers to finish, stops the storage thread, computes final metrics, and returns a WorkflowResult summarizing the run.
        
        Returns:
            WorkflowResult: summary of the run including:
                - workflow_name (str)
                - success (bool)
                - items_processed (int)
                - items_failed (int)
                - start_time (datetime)
                - end_time (datetime)
                - metadata (dict) with keys such as:
                    - throughput (float): processed records / duration (records/sec)
                    - duration_seconds (float)
                    - items_skipped (int)
                    - num_workers (int)
                    - metadata_file (str)
                    - embedder_model (str)
                    - batch_size (int)
                    - resume_mode (bool)
                    - error (str) — present only when success is False
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
        """
        Validate workflow inputs.
        
        This placeholder implementation currently accepts any keyword arguments and always returns True.
        Override in subclasses to perform concrete validation of configuration or runtime parameters.
        """
        return True

    def cleanup(self):
        """
        Shut down processing and release IPC resources.
        
        Sets the workflow stop signal, force-terminates any still-running worker processes (joining briefly), and drains the input and output multiprocessing queues to leave the process in a clean state.
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
        description="Process ArXiv metadata with size-sorted multi-GPU parallel processing"
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

    parser.add_argument(
        '--database',
        type=str,
        default='arxiv_repository',
        help='ArangoDB database name (default: arxiv_repository)'
    )

    parser.add_argument(
        '--username',
        type=str,
        default='arxiv_writer',
        help='Database username (default: arxiv_writer)'
    )

    parser.add_argument(
        '--password-env',
        type=str,
        default='ARXIV_WRITER_PASSWORD',
        help='Environment variable containing password (default: ARXIV_WRITER_PASSWORD)'
    )

    args = parser.parse_args()

    # Check environment for password
    password = os.environ.get(args.password_env)
    if not password:
        # Fallback to ARANGO_PASSWORD for backward compatibility
        password = os.environ.get('ARANGO_PASSWORD')
        if not password:
            print(f"ERROR: {args.password_env} environment variable not set")
            print(f"Please set: export {args.password_env}='your-password'")
            print("Or use: export ARANGO_PASSWORD='your-password' (fallback)")
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
        # Ensure checkpoint interval is at least 100 (validation requirement)
        checkpoint_interval = max(100, min(1000, max_records // 10))
    else:
        checkpoint_interval = 10000  # Default for unlimited processing

    # Set password in environment for DatabaseFactory to use
    os.environ['ARANGO_PASSWORD'] = password

    config = ArxivMetadataConfig(
        max_records=max_records,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        num_workers=args.workers,
        drop_collections=args.drop_collections,
        resume_from_checkpoint=args.resume,
        checkpoint_interval=checkpoint_interval,
        monitor_interval=100,
        arango_database=args.database,  # Use specified database
        arango_username=args.username   # Use specified username
        # Password comes from environment variable
    )

    print("=" * 60)
    print("ArXiv Size-Sorted Processing")
    print("=" * 60)
    print(f"Database: {args.database}")
    print(f"Username: {args.username}")
    print(f"Records: {display_count}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding batch: {args.embedding_batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Drop collections: {args.drop_collections}")
    print(f"Resume: {args.resume}")
    print("=" * 60)

    # Run workflow
    workflow = ArxivSortedWorkflow(config)
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