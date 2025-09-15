#!/usr/bin/env python3
"""
ArXiv Memory-Optimized Parallel Processing Workflow
====================================================

Loads entire dataset into RAM for maximum performance.
Processes 2.8M ArXiv papers with dual-GPU embedding generation.

Key improvements:
- Loads entire 4.6GB dataset into RAM once
- Stores metadata first, then processes embeddings
- Eliminates disk I/O during processing
- Creates single chunk per abstract (no chunking needed)

Theory Connection (Conveyance Framework):
Optimizes TIME (T) dimension by eliminating I/O bottlenecks,
directly improving Conveyance C = (W·R·H/T)·Ctx^α
"""

import os
import sys
import json
import time
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from threading import Thread
from queue import Empty
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.workflows.workflow_base import WorkflowBase, WorkflowConfig, WorkflowResult
from core.workflows.state import StateManager, CheckpointManager
from core.monitoring.progress_tracker import ProgressTracker
from core.monitoring.performance_monitor import PerformanceMonitor
from core.database.database_factory import DatabaseFactory
from core.embedders.embedders_factory import EmbedderFactory
from core.tools.arxiv.arxiv_metadata_config import ArxivMetadataConfig
from core.logging.logging import LogManager
from arango.exceptions import DocumentInsertError

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for embedding worker."""
    worker_id: int
    gpu_device: str
    batch_size: int
    model_name: str
    use_fp16: bool = True


def worker_process_with_gpu(config, input_queue, output_queue, stop_event, gpu_id):
    """
    GPU-isolated worker process with proper CUDA device setting.
    """
    # CRITICAL: Set GPU BEFORE any CUDA imports
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # NOW import CUDA-dependent modules
    from core.embedders.embedders_factory import EmbedderFactory
    from core.logging.logging import LogManager

    # Setup structured logging for this worker
    worker_logger = LogManager.get_logger(
        f"worker_{config.worker_id}",
        f"gpu_{gpu_id}_{datetime.now().isoformat()}"
    )

    try:
        # Initialize embedder for this worker
        worker_logger.info("worker_initializing",
            worker_id=config.worker_id,
            gpu_id=gpu_id,
            model=config.model_name
        )

        embedder_config = {
            'device': 'cuda:0',  # Always cuda:0 since we set CUDA_VISIBLE_DEVICES
            'use_fp16': config.use_fp16,
            'batch_size': config.batch_size,
            'chunk_size_tokens': 512,
            'chunk_overlap_tokens': 128
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
        batches_processed = 0
        while not stop_event.is_set():
            try:
                # Get batch from queue
                batch = input_queue.get(timeout=1.0)

                if batch is None:  # Poison pill
                    worker_logger.info("worker_stopping",
                        worker_id=config.worker_id,
                        batches_processed=batches_processed
                    )
                    break

                # Process batch - just embed abstracts directly
                abstracts = [record['abstract'] for record in batch if record.get('abstract')]

                if abstracts:
                    # Direct embedding - no chunking for abstracts
                    embeddings = embedder.embed_batch(
                        abstracts,
                        batch_size=config.batch_size,
                        task="retrieval.passage"
                    )

                    # Package results
                    results = []
                    emb_idx = 0
                    for record in batch:
                        if record.get('abstract'):
                            results.append({
                                'arxiv_id': record['id'],
                                'embedding': embeddings[emb_idx]
                            })
                            emb_idx += 1

                    # Send results back
                    output_queue.put({
                        'worker_id': config.worker_id,
                        'results': results
                    })

                    batches_processed += 1
                    worker_logger.info("batch_processed",
                        worker_id=config.worker_id,
                        batch_number=batches_processed,
                        results_count=len(results)
                    )

            except Empty:
                continue
            except Exception as e:
                worker_logger.error("batch_processing_failed",
                    worker_id=config.worker_id,
                    error=str(e)
                )
                output_queue.put({
                    'worker_id': config.worker_id,
                    'error': str(e)
                })

    except Exception as e:
        worker_logger.error("worker_init_failed",
            worker_id=config.worker_id,
            error=str(e)
        )


class ArxivMemoryWorkflow(WorkflowBase):
    """
    Memory-optimized workflow for ArXiv processing.

    Loads entire dataset into RAM, stores metadata first,
    then processes embeddings from memory.
    """

    def __init__(self, config: Optional[ArxivMetadataConfig] = None):
        """Initialize memory-optimized workflow."""
        if config is None:
            config = ArxivMetadataConfig()

        # Validate configuration
        config.validate_full()

        # Initialize base workflow
        workflow_config = WorkflowConfig(
            name="arxiv_memory_workflow",
            batch_size=config.batch_size,
            num_workers=getattr(config, 'num_workers', 2),
            use_gpu=config.use_gpu,
            checkpoint_enabled=config.resume_from_checkpoint,
            checkpoint_interval=config.checkpoint_interval
        )
        super().__init__(workflow_config)
        self.metadata_config = config

        # Setup logging
        self.logger = LogManager.get_logger(
            "arxiv_memory_workflow",
            f"run_{datetime.now().isoformat()}"
        )

        # State management
        self.state_manager = StateManager(
            str(config.state_file),
            "arxiv_memory_workflow"
        )
        self.checkpoint_manager = CheckpointManager(
            str(config.checkpoint_file)
        )

        # Progress tracking
        self.progress_tracker = ProgressTracker(
            name="arxiv_memory_workflow",
            description="Memory-optimized ArXiv processing"
        )

        # Processing state
        self.all_records = []  # All records in memory
        self.workers = []
        self.input_queue = None
        self.output_queue = None
        self.stop_event = None
        self.storage_thread = None

        # Counters
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None

        # Database connection
        self.db = None

    def _load_entire_dataset(self) -> List[Dict]:
        """
        Load entire ArXiv dataset into memory.

        Returns:
            List of all records with abstracts
        """
        metadata_file = Path(self.metadata_config.metadata_file)

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        self.logger.info("loading_entire_dataset",
            file=str(metadata_file),
            file_size_gb=metadata_file.stat().st_size / (1024**3)
        )

        print(f"Loading {metadata_file.name} into RAM...")
        print(f"File size: {metadata_file.stat().st_size / (1024**3):.2f} GB")

        all_records = []
        start_time = time.time()

        with open(metadata_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    elapsed = time.time() - start_time
                    rate = line_num / elapsed
                    print(f"  Loaded {line_num:,} records ({rate:.0f} records/sec)...")

                try:
                    record = json.loads(line)

                    # Only keep records with abstracts
                    if record.get('abstract'):
                        all_records.append(record)

                        # Apply max_records limit if set
                        if self.metadata_config.max_records and len(all_records) >= self.metadata_config.max_records:
                            break

                except json.JSONDecodeError:
                    continue

        load_time = time.time() - start_time

        # Estimate memory usage without duplicating data (avoids OOM)
        try:
            import psutil  # optional dependency
            memory_usage_bytes = psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            # Cheap fallback: approximate size without creating copies
            import sys
            # Rough estimate: number of records * average size per record
            # Assumes ~2KB per record (typical for ArXiv metadata)
            memory_usage_bytes = len(all_records) * 2048
        memory_usage_gb = memory_usage_bytes / (1024**3)

        self.logger.info("dataset_loaded",
            total_records=len(all_records),
            load_time_seconds=load_time,
            memory_usage_gb=memory_usage_gb
        )

        print(f"✅ Loaded {len(all_records):,} records in {load_time:.1f} seconds")
        print(f"   Memory usage: ~{memory_usage_gb:.1f} GB")

        return all_records

    def _store_metadata_batch(self, batch: List[Dict]) -> int:
        """
        Store metadata and create chunk records in database.

        Args:
            batch: List of records to store

        Returns:
            Number of successfully stored records
        """
        if not batch:
            return 0

        # Begin transaction
        txn = self.db.begin_transaction(
            write=[
                self.metadata_config.metadata_collection,
                self.metadata_config.chunks_collection
            ]
        )

        try:
            successful = 0

            for record in batch:
                arxiv_id = record.get('id', '')
                if not arxiv_id:
                    continue

                sanitized_id = arxiv_id.replace('.', '_').replace('/', '_')

                # Store metadata
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

                # Create single chunk for abstract
                abstract = record.get('abstract', '')
                if abstract:
                    chunk_doc = {
                        '_key': f"{sanitized_id}_chunk_0",
                        'arxiv_id': arxiv_id,
                        'paper_key': sanitized_id,
                        'chunk_index': 0,
                        'total_chunks': 1,
                        'text': abstract,
                        'start_char': 0,
                        'end_char': len(abstract),
                        'start_token': 0,
                        'end_token': len(abstract) // 4,  # Rough estimate
                        'context_window_used': len(abstract) // 4,
                        'created_at': datetime.now().isoformat()
                    }

                    txn.collection(self.metadata_config.chunks_collection).insert(
                        chunk_doc, overwrite=True
                    )

                successful += 1

            # Commit transaction
            txn.commit_transaction()

            return successful

        except Exception as e:
            self.logger.error(f"Failed to store metadata batch: {e}")
            try:
                txn.abort_transaction()
            except:
                pass
            return 0

    def _store_embeddings_batch(self, batch: List[Dict]) -> int:
        """
        Store embeddings in database.

        Args:
            batch: List of embedding results

        Returns:
            Number of successfully stored embeddings
        """
        if not batch:
            return 0

        # Begin transaction
        txn = self.db.begin_transaction(
            write=[self.metadata_config.embeddings_collection]
        )

        try:
            successful = 0

            for item in batch:
                arxiv_id = item['arxiv_id']
                embedding = item['embedding']

                sanitized_id = arxiv_id.replace('.', '_').replace('/', '_')

                # Store embedding (single chunk per abstract)
                embedding_doc = {
                    '_key': f"{sanitized_id}_chunk_0_emb",
                    'chunk_id': f"{sanitized_id}_chunk_0",
                    'arxiv_id': arxiv_id,
                    'paper_key': sanitized_id,
                    'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    'embedding_dim': len(embedding),
                    'model': self.metadata_config.embedder_model,
                    'created_at': datetime.now().isoformat()
                }

                txn.collection(self.metadata_config.embeddings_collection).insert(
                    embedding_doc, overwrite=True
                )

                successful += 1

            # Commit transaction
            txn.commit_transaction()

            return successful

        except Exception as e:
            self.logger.error(f"Failed to store embeddings batch: {e}")
            try:
                txn.abort_transaction()
            except:
                pass
            return 0

    def _storage_worker(self):
        """
        Worker thread for storing embedding results in database.
        """
        self.logger.info("storage_worker_started")

        batch_buffer = []
        batch_size = 100  # Store in batches

        while not self.stop_event.is_set() or not self.output_queue.empty():
            try:
                # Get result from queue
                result = self.output_queue.get(timeout=1.0)

                if 'error' in result:
                    self.logger.error("worker_error",
                        worker_id=result['worker_id'],
                        error=result['error']
                    )
                    continue

                # Add to batch buffer
                batch_buffer.extend(result['results'])

                # Update embedding generation progress
                self.progress_tracker.update_step(
                    "embedding_generation",
                    completed=len(result['results'])
                )

                # Store when batch is full
                if len(batch_buffer) >= batch_size:
                    stored = self._store_embeddings_batch(batch_buffer[:batch_size])

                    # Update storage progress
                    self.progress_tracker.update_step(
                        "embedding_storage",
                        completed=stored
                    )

                    batch_buffer = batch_buffer[batch_size:]
                    self.processed_count += stored

            except Empty:
                # Timeout - check if we should flush partial batch
                if batch_buffer and self.stop_event.is_set():
                    stored = self._store_embeddings_batch(batch_buffer)
                    self.progress_tracker.update_step(
                        "embedding_storage",
                        completed=stored
                    )
                    self.processed_count += stored
                    batch_buffer = []

        # Store any remaining records
        if batch_buffer:
            stored = self._store_embeddings_batch(batch_buffer)
            self.progress_tracker.update_step(
                "embedding_storage",
                completed=stored
            )
            self.processed_count += stored

        self.logger.info("storage_worker_finished")

    def _initialize_components(self):
        """Initialize workflow components."""
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

        # Ensure collections exist
        for collection in [
            self.metadata_config.metadata_collection,
            self.metadata_config.chunks_collection,
            self.metadata_config.embeddings_collection
        ]:
            if not self.db.has_collection(collection):
                self.db.create_collection(collection)

        # Initialize progress tracker steps
        total_records = len(self.all_records)
        self.progress_tracker.add_step(
            step_id="metadata_storage",
            name="Storing Metadata",
            total_items=total_records
        )
        self.progress_tracker.add_step(
            step_id="embedding_generation",
            name="Generating Embeddings",
            total_items=total_records
        )
        self.progress_tracker.add_step(
            step_id="embedding_storage",
            name="Storing Embeddings",
            total_items=total_records
        )

        # Initialize queues
        ctx = mp.get_context('spawn')
        self.input_queue = ctx.Queue(maxsize=100)
        self.output_queue = ctx.Queue(maxsize=100)
        self.stop_event = ctx.Event()

        # Start GPU workers
        num_workers = getattr(self.metadata_config, 'num_workers', 2)

        for i in range(num_workers):
            gpu_id = str(i)  # GPU 0, GPU 1, etc.

            config = WorkerConfig(
                worker_id=i,
                gpu_device=f"cuda:{i}",
                batch_size=self.metadata_config.embedding_batch_size,
                model_name=self.metadata_config.embedder_model,
                use_fp16=self.metadata_config.use_fp16
            )

            worker = ctx.Process(
                target=worker_process_with_gpu,
                args=(config, self.input_queue, self.output_queue, self.stop_event, gpu_id)
            )
            worker.start()
            self.workers.append(worker)

            self.logger.info("worker_started",
                worker_id=i,
                gpu_id=gpu_id
            )

        # Start storage thread
        self.storage_thread = Thread(target=self._storage_worker)
        self.storage_thread.start()

    def _execute_workflow(self) -> WorkflowResult:
        """Execute the memory-optimized workflow."""
        self.start_time = time.time()

        # Step 1: Load entire dataset into RAM
        print("\n" + "="*60)
        print("STEP 1: Loading Dataset into RAM")
        print("="*60)
        self.all_records = self._load_entire_dataset()

        # Step 2: Initialize components
        print("\n" + "="*60)
        print("STEP 2: Initializing Components")
        print("="*60)
        self._initialize_components()

        # Step 3: Store all metadata first
        print("\n" + "="*60)
        print("STEP 3: Storing Metadata and Chunks")
        print("="*60)
        self.progress_tracker.start_step("metadata_storage")

        metadata_stored = 0
        batch_size = 1000

        for i in range(0, len(self.all_records), batch_size):
            batch = self.all_records[i:i+batch_size]
            stored = self._store_metadata_batch(batch)
            metadata_stored += stored

            self.progress_tracker.update_step(
                "metadata_storage",
                completed=stored
            )

            if (i // batch_size) % 10 == 0:
                print(f"  Stored {metadata_stored:,} / {len(self.all_records):,} metadata records")

        self.progress_tracker.complete_step("metadata_storage")
        print(f"✅ Metadata storage complete: {metadata_stored:,} records")

        # Step 4: Process embeddings from memory
        print("\n" + "="*60)
        print("STEP 4: Generating and Storing Embeddings")
        print("="*60)
        self.progress_tracker.start_step("embedding_generation")
        self.progress_tracker.start_step("embedding_storage")

        # Queue all records for embedding
        batch = []
        for record in self.all_records:
            batch.append(record)

            if len(batch) >= self.metadata_config.batch_size:
                self.input_queue.put(batch)
                batch = []

        # Queue remaining batch
        if batch:
            self.input_queue.put(batch)

        # Send stop signals
        for _ in self.workers:
            self.input_queue.put(None)

        # Wait for workers to finish
        print("Processing embeddings...")
        for worker in self.workers:
            worker.join()

        # Signal storage thread to stop and wait
        self.stop_event.set()
        self.storage_thread.join()

        # Complete progress tracking
        self.progress_tracker.complete_step("embedding_generation")
        self.progress_tracker.complete_step("embedding_storage")

        # Calculate final statistics
        elapsed = time.time() - self.start_time
        throughput = self.processed_count / elapsed if elapsed > 0 else 0

        print("\n" + "="*60)
        print("WORKFLOW COMPLETE")
        print("="*60)
        print(f"  Total processed: {self.processed_count:,}")
        print(f"  Total time: {elapsed:.1f} seconds")
        print(f"  Throughput: {throughput:.1f} papers/second")

        return WorkflowResult(
            success=True,
            items_processed=self.processed_count,
            items_failed=self.failed_count,
            duration_seconds=elapsed,
            metadata={'throughput': throughput}
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Memory-optimized ArXiv processing"
    )

    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of records to process'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Records per batch'
    )

    parser.add_argument(
        '--embedding-batch-size',
        type=int,
        default=48,
        help='Embeddings per GPU batch'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help='Number of GPU workers'
    )

    parser.add_argument(
        '--drop-collections',
        action='store_true',
        help='Drop existing collections'
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
        sys.exit(1)

    # Create configuration
    from core.tools.arxiv.arxiv_metadata_config import ArxivMetadataConfig

    config = ArxivMetadataConfig(
        max_records=args.count,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        num_workers=args.workers,
        drop_collections=args.drop_collections,
        resume_from_checkpoint=args.resume
    )

    print("="*60)
    print("ArXiv Memory-Optimized Processing")
    print("="*60)
    print(f"Records: {args.count}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding batch: {args.embedding_batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Drop collections: {args.drop_collections}")
    print("="*60)

    # Run workflow
    workflow = ArxivMemoryWorkflow(config)
    result = workflow.execute()

    # Exit with appropriate code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()