#!/usr/bin/env python3
"""
ArXiv Metadata Processing Workflow
===================================

Processes the complete ArXiv metadata dataset (2.8M records) from Kaggle JSON file.
Creates metadata, abstract chunks, and embeddings in ArangoDB using Jina v4.

This workflow implements:
- Streaming JSON processing for memory efficiency
- MANDATORY late chunking for abstract embeddings (per CLAUDE.md)
- High-throughput batch processing (48+ papers/second target)
- Atomic transactions for data integrity
- Full integration with core infrastructure (state, monitoring, config)

Theory Connection (Conveyance Framework):
C = (W·R·H/T)·Ctx^α

- W (WHAT): Abstract semantic content via Jina v4 embeddings
- R (WHERE): ArangoDB graph relationships between metadata/chunks/embeddings
- H (WHO): SentenceTransformersEmbedder for high-throughput processing
- T (TIME): Optimized for 48+ papers/second throughput
- Ctx: Late chunking preserves context (Ctx remains high, avoiding zero-propagation)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Generator

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Core infrastructure imports
from core.workflows.workflow_base import WorkflowBase, WorkflowConfig, WorkflowResult
from core.workflows.state import StateManager, CheckpointManager
from core.workflows.storage.storage_local import StorageManager
from core.monitoring.progress_tracker import ProgressTracker, ProgressState
from core.monitoring.performance_monitor import PerformanceMonitor
from core.database.database_factory import DatabaseFactory
from core.embedders.embedders_factory import EmbedderFactory
from core.tools.arxiv.arxiv_metadata_config import ArxivMetadataConfig
from arango.exceptions import DocumentInsertError, CollectionDeleteError

logger = logging.getLogger(__name__)




class ArxivMetadataWorkflow(WorkflowBase):
    """
    Workflow for processing ArXiv metadata with abstract embeddings.

    Implements streaming processing for 2.8M records with late chunking
    and high-throughput embedding generation using core infrastructure.
    """

    def __init__(self, config: Optional[ArxivMetadataConfig] = None):
        """
        Create an ArxivMetadataWorkflow instance.
        
        If no config is provided, a default ArxivMetadataConfig is created and fully validated. The constructor initializes the workflow base (with a derived WorkflowConfig), sets up state and checkpoint managers, progress and performance trackers, and initializes placeholders for the embedder and database as well as counters used during execution.
        
        Parameters:
            config (Optional[ArxivMetadataConfig]): Workflow configuration; when omitted a default configuration is created and validated.
        """
        if config is None:
            config = ArxivMetadataConfig()

        # Validate configuration
        config.validate_full()

        # Initialize base workflow with WorkflowConfig
        workflow_config = WorkflowConfig(
            name="arxiv_metadata_workflow",
            batch_size=config.batch_size,
            num_workers=1,  # Single-threaded for streaming
            use_gpu=config.use_gpu,
            checkpoint_enabled=config.resume_from_checkpoint,
            checkpoint_interval=config.checkpoint_interval
        )
        super().__init__(workflow_config)
        self.metadata_config: ArxivMetadataConfig = config

        # Initialize core infrastructure components
        self.embedder = None
        self.db = None

        # State management
        self.state_manager = StateManager(
            str(config.state_file),
            "arxiv_metadata_workflow"
        )
        self.checkpoint_manager = CheckpointManager(
            str(config.checkpoint_file)
        )

        # Progress tracking
        self.progress_tracker = ProgressTracker(
            name="arxiv_metadata_workflow",
            description="Processing ArXiv metadata with embeddings"
        )
        self.performance_monitor = PerformanceMonitor(
            component_name="arxiv_metadata_workflow"
        )

        # Counters
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None

    def _initialize_components(self) -> None:
        """
        Initialize and configure the workflow's ArangoDB connection and embedding engine.
        
        This sets up:
        - self.db: an ArangoDB connection obtained via DatabaseFactory (prefers a Unix socket).
        - self.embedder: an embedder instance created via EmbedderFactory, configured for
          device, fp16 usage, embedding batch size, and late-chunking parameters (chunk size
          and overlap) based on the workflow's ArxivMetadataConfig.
        
        Side effects:
        - Mutates self.db and self.embedder.
        - Logs initialization progress and the configured target throughput.
        
        No return value.
        """
        logger.info("Initializing workflow components...")

        # Use DatabaseFactory to get ArangoDB connection (will try Unix socket first)
        self.db = DatabaseFactory.get_arango(
            database=self.metadata_config.arango_database,
            username=self.metadata_config.arango_username,
            use_unix=True  # Try Unix socket first
        )

        # Initialize embedder with high-throughput settings
        logger.info(f"Initializing {self.metadata_config.embedder_model} embedder...")
        embedder_config = {
            'device': self.metadata_config.gpu_device if self.metadata_config.use_gpu else 'cpu',
            'use_fp16': self.metadata_config.use_fp16,
            'batch_size': self.metadata_config.embedding_batch_size,
            'chunk_size_tokens': self.metadata_config.chunk_size_tokens,
            'chunk_overlap_tokens': self.metadata_config.chunk_overlap_tokens
        }

        # Use factory to create embedder (will use SentenceTransformersEmbedder)
        self.embedder = EmbedderFactory.create(
            model_name=self.metadata_config.embedder_model,
            **embedder_config
        )

        logger.info(f"Embedder initialized: {type(self.embedder).__name__}")
        logger.info(f"Target throughput: {self.metadata_config.target_throughput} papers/second")

    def _setup_collections(self) -> None:
        """
        Ensure required ArangoDB collections exist and optionally drop and recreate them.
        
        If `metadata_config.drop_collections` is true, existing metadata, chunks, and embeddings
        collections are deleted (if present). When collections are dropped, the checkpoint and
        state managers are cleared to allow a fresh run. After that (or if not dropping), each
        required collection is created/validated via StorageManager.ensure_collection.
        
        Raises:
            Exception: Propagates any error encountered while deleting or ensuring collections.
        """
        try:
            # Drop collections if requested
            if self.metadata_config.drop_collections:
                logger.warning("Dropping existing collections as requested...")

                collections_to_drop = [
                    self.metadata_config.metadata_collection,
                    self.metadata_config.chunks_collection,
                    self.metadata_config.embeddings_collection
                ]

                for coll_name in collections_to_drop:
                    if self.db.has_collection(coll_name):
                        try:
                            self.db.delete_collection(coll_name)
                            logger.info(f"Dropped collection: {coll_name}")
                        except CollectionDeleteError as e:
                            logger.error(f"Failed to drop collection {coll_name}: {e}")
                            raise

                # Also clear checkpoint when dropping collections for fresh start
                self.checkpoint_manager.clear()
                self.state_manager.clear()
                logger.info("Cleared checkpoint for fresh start")

            # Use StorageManager to ensure collections exist
            collections = [
                (self.metadata_config.metadata_collection, False),
                (self.metadata_config.chunks_collection, False),
                (self.metadata_config.embeddings_collection, False)
            ]

            for coll_name, is_edge in collections:
                StorageManager.ensure_collection(self.db, coll_name, edge=is_edge)
                logger.info(f"Ensured collection: {coll_name}")

        except Exception as e:
            logger.error(f"Failed to setup collections: {e}")
            raise

    def _stream_metadata_records(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yield metadata records parsed from the configured JSONL metadata file.
        
        Reads the file line-by-line (memory efficient), parsing each line as a JSON object and yielding the resulting dict for downstream processing. Skips records beyond `metadata_config.max_records` when set and skips records already marked processed by the checkpoint manager. On malformed JSON lines the function logs a warning, increments the `parse_errors` stat via the state manager, and continues. Raises FileNotFoundError if the configured metadata file does not exist.
        
        Yields:
            dict: A parsed metadata record (one JSON object per line).
        """
        metadata_path = self.metadata_config.metadata_file

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        logger.info(f"Streaming metadata from: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if self.metadata_config.max_records and line_num > self.metadata_config.max_records:
                    break

                try:
                    record = json.loads(line.strip())

                    # Skip already processed records if resuming
                    arxiv_id = record.get('id', f'line_{line_num}')
                    if self.checkpoint_manager.is_processed(arxiv_id):
                        continue

                    yield record
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    self.state_manager.increment_stat('parse_errors', 1)
                    continue

    def _process_batch(self, batch: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Process a batch of ArXiv metadata records: generate late-chunked embeddings and persist metadata, chunk, and embedding documents in a single atomic transaction.
        
        This method:
        - Extracts records that contain an abstract and generates embeddings using the embedder's batch late-chunking API.
        - Associates each embedding chunk with its parent record and writes three kinds of documents to the database within one transaction:
          - metadata documents (one per record),
          - chunk documents (one per chunk),
          - embedding documents (one per chunk embedding).
        - Uses sanitized ArXiv IDs as ArangoDB document keys.
        - Marks all successfully persisted record IDs as processed in the checkpoint manager after a committed transaction.
        
        Parameters:
            batch (List[Dict[str, Any]]): A list of metadata records (each expected to include at least an 'id' and optionally an 'abstract'). Records missing an 'id' are counted as failed and skipped.
        
        Returns:
            Tuple[int, int]: (successful_count, failed_count). On embedding generation failure or a transactional error, the batch is treated as failed; on success, successful_count is set to the number of records in the batch.
        """
        successful = 0
        failed = 0

        # Extract abstracts for embedding
        abstracts_to_embed = []
        records_with_abstracts = []

        for record in batch:
            if record.get('abstract'):
                abstracts_to_embed.append(record['abstract'])
                records_with_abstracts.append(record)

        # Generate embeddings with MANDATORY late chunking
        chunks_with_embeddings = []

        if abstracts_to_embed:
            try:
                # Use batch late chunking for maximum throughput
                all_chunks = self.embedder.embed_batch_with_late_chunking(
                    abstracts_to_embed,
                    task="retrieval.passage"
                )

                # Flatten and associate with records
                for record, record_chunks in zip(records_with_abstracts, all_chunks):
                    for chunk in record_chunks:
                        chunks_with_embeddings.append({
                            'record': record,
                            'chunk': chunk
                        })

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                return 0, len(batch)

        # Store in database with atomic transaction
        db = self.db  # Use the connection from DatabaseFactory

        # Begin transaction
        txn = db.begin_transaction(
            write=[
                self.metadata_config.metadata_collection,
                self.metadata_config.chunks_collection,
                self.metadata_config.embeddings_collection
            ]
        )

        transaction_committed = False

        try:
            # txn is already the transaction database
            txn_db = txn

            # Store metadata records
            for record in batch:
                arxiv_id = record.get('id', '')
                if not arxiv_id:
                    failed += 1
                    continue

                # Sanitize ID for ArangoDB key
                sanitized_id = arxiv_id.replace('.', '_').replace('/', '_')

                # Prepare metadata document
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

                txn_db.collection(self.metadata_config.metadata_collection).insert(
                    metadata_doc, overwrite=True
                )

                # Store chunks and embeddings
                for item in chunks_with_embeddings:
                    record = item['record']
                    chunk = item['chunk']
                    arxiv_id = record.get('id', '')
                    sanitized_id = arxiv_id.replace('.', '_').replace('/', '_')

                    # Create unique chunk ID
                    chunk_id = f"{sanitized_id}_chunk_{chunk.chunk_index}"

                    # Store chunk document
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

                    txn_db.collection(self.metadata_config.chunks_collection).insert(
                        chunk_doc, overwrite=True
                    )

                    # Store embedding document
                    embedding_doc = {
                        '_key': f"{chunk_id}_emb",
                        'chunk_id': chunk_id,
                        'arxiv_id': arxiv_id,
                        'paper_key': sanitized_id,
                        'embedding': chunk.embedding.tolist(),  # Convert numpy to list
                        'embedding_dim': len(chunk.embedding),
                        'model': self.metadata_config.embedder_model,
                        'created_at': datetime.now().isoformat()
                    }

                    txn_db.collection(self.metadata_config.embeddings_collection).insert(
                        embedding_doc, overwrite=True
                    )

                # Commit transaction
                txn.commit_transaction()
                transaction_committed = True
                successful = len(batch)

        except Exception as e:
            if not transaction_committed:
                logger.error(f"Transaction failed: {e}")
                try:
                    txn.abort_transaction()
                except Exception as abort_error:
                    logger.warning(f"Failed to abort transaction: {abort_error}")
                failed = len(batch)
            else:
                # This shouldn't happen - transaction was already committed
                logger.warning(f"Post-transaction error (data was saved): {e}")

        # Mark records as processed in checkpoint manager (outside transaction)
        if successful > 0:
            try:
                processed_ids = [r.get('id', '') for r in batch if r.get('id')]
                for arxiv_id in processed_ids:
                    self.checkpoint_manager.mark_processed(arxiv_id)
            except Exception as e:
                logger.debug(f"Checkpoint marking failed (non-critical): {e}")

        return successful, failed

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate required inputs for the workflow.
        
        Performs two checks:
        1. The configured metadata file exists on disk.
        2. The ARANGO_PASSWORD environment variable is set.
        
        Returns:
            bool: True if both checks pass; False otherwise. Errors are logged when a check fails.
        """
        # Check metadata file exists
        if not self.metadata_config.metadata_file.exists():
            logger.error(f"Metadata file not found: {self.metadata_config.metadata_file}")
            return False

        # Check database password
        if not os.environ.get('ARANGO_PASSWORD'):
            logger.error("ARANGO_PASSWORD environment variable not set")
            return False

        return True

    def execute(self, **kwargs) -> WorkflowResult:
        """
        Run the ArXiv metadata processing workflow: stream records from the configured metadata file, generate late-chunked embeddings, and persist metadata, chunk documents, and embeddings into ArangoDB in atomic batches.
        
        This method orchestrates initialization, optional resume from checkpoint, batched processing (using the configured batch size), progress tracking across phases (metadata loading, embedding generation, database storage), periodic checkpoint saves, and final cleanup. On successful completion it optionally clears saved checkpoints/state and returns a WorkflowResult containing throughput, duration, input file, embedder model, and batch size. On failure it catches exceptions, logs the error, and returns a failed WorkflowResult with error details.
        
        Parameters:
            **kwargs: Passed through to input validation (validate_inputs). No other kwargs are required by the workflow.
        
        Returns:
            WorkflowResult describing success/failure, items processed/failed, start/end times, and a metadata map with runtime metrics (e.g., throughput, duration_seconds, metadata_file, embedder_model, batch_size).
        """
        self.start_time = datetime.now()

        # Validate inputs
        if not self.validate_inputs(**kwargs):
            return WorkflowResult(
                workflow_name=self.config.name,
                success=False,
                items_processed=0,
                items_failed=0,
                start_time=self.start_time,
                end_time=datetime.now(),
                errors=["Input validation failed"]
            )

        try:
            # Initialize components
            self._initialize_components()

            # Setup collections
            self._setup_collections()

            # Load state if resuming
            if self.metadata_config.resume_from_checkpoint:
                if self.state_manager.load():
                    self.processed_count = self.state_manager.get_checkpoint('processed_count', 0)
                    self.failed_count = self.state_manager.get_checkpoint('failed_count', 0)
                    logger.info(f"Resuming from checkpoint: {self.processed_count} records processed")

            # Setup progress tracking with phases
            total_records = self.metadata_config.max_records or 2800000  # Estimate if processing all

            # Add progress steps
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

            # Start metadata loading phase
            self.progress_tracker.start_step("metadata_loading")

            # Process metadata in batches
            batch = []
            total_processed = 0

            for record in self._stream_metadata_records():
                # Records are already filtered in _stream_metadata_records using checkpoint_manager

                batch.append(record)

                # Process batch when full
                if len(batch) >= self.metadata_config.batch_size:
                    successful, failed = self._process_batch(batch)
                    self.processed_count += successful
                    self.failed_count += failed

                    # Update progress tracker
                    self.progress_tracker.update_step(
                        "metadata_loading",
                        completed=successful,
                        failed=failed
                    )

                    # Calculate and report throughput
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    if elapsed > 0:
                        throughput = self.processed_count / elapsed
                        self.performance_monitor.gauge(
                            "throughput",
                            throughput,
                            metadata={"unit": "records/sec"}
                        )

                        # Log progress at intervals
                        if total_processed % self.metadata_config.monitor_interval == 0:
                            logger.info(
                                f"Progress: {self.processed_count} processed, "
                                f"{self.failed_count} failed, "
                                f"throughput: {throughput:.1f} rec/s"
                            )

                    # Save checkpoint
                    if self.processed_count % self.metadata_config.checkpoint_interval == 0:
                        self.state_manager.set_checkpoint('processed_count', self.processed_count)
                        self.state_manager.set_checkpoint('failed_count', self.failed_count)
                        self.state_manager.save()
                        self.checkpoint_manager.save()

                    batch = []
                    total_processed += successful + failed

            # Process remaining batch
            if batch:
                successful, failed = self._process_batch(batch)
                self.processed_count += successful
                self.failed_count += failed

                self.progress_tracker.update_step(
                    "metadata_loading",
                    completed=successful,
                    failed=failed
                )

            # Complete progress tracking
            self.progress_tracker.complete_step("metadata_loading")
            self.progress_tracker.complete_step("embedding_generation")
            self.progress_tracker.complete_step("database_storage")

            # Clear checkpoint on successful completion
            if self.metadata_config.resume_from_checkpoint:
                self.checkpoint_manager.clear()
                self.state_manager.clear()

            # Calculate final metrics
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()

            logger.info(f"Workflow completed: {self.processed_count} records processed")
            logger.info(f"Throughput: {self.processed_count / duration:.2f} records/second")

            return WorkflowResult(
                workflow_name=self.config.name,
                success=True,
                items_processed=self.processed_count,
                items_failed=self.failed_count,
                start_time=self.start_time,
                end_time=end_time,
                metadata={
                    'throughput': self.processed_count / duration,
                    'duration_seconds': duration,
                    'metadata_file': self.metadata_config.metadata_file,
                    'embedder_model': self.metadata_config.embedder_model,
                    'batch_size': self.metadata_config.batch_size
                }
            )

        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)

            return WorkflowResult(
                workflow_name=self.config.name,
                success=False,
                items_processed=self.processed_count,
                items_failed=self.failed_count,
                start_time=self.start_time,
                end_time=datetime.now(),
                errors=[str(e)]
            )

    @property
    def supports_streaming(self) -> bool:
        """This workflow supports streaming processing."""
        return True