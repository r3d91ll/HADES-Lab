#!/usr/bin/env python3
"""
Generic Document Processor
==========================

Source-agnostic document processing pipeline that handles:
- PDF extraction (with optional LaTeX)
- Embedding generation
- Storage to ArangoDB

This processor doesn't know or care about document sources (ArXiv, Semantic Scholar, etc).
It simply processes what it's given.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DocumentTask:
    """Generic document processing task."""
    document_id: str  # Unique identifier (could be arxiv_id, DOI, filename, etc)
    pdf_path: str
    latex_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # Optional metadata from source
    
    def __post_init__(self):
        """
        Ensure the `metadata` field is initialized.
        
        If `metadata` was not provided (is None), set it to an empty dictionary on the instance.
        """
        if self.metadata is None:
            self.metadata = {}


class GenericDocumentProcessor:
    """
    Processes documents without knowing their source.
    
    Handles the core pipeline:
    1. Extraction (Docling)
    2. Embedding (Jina v4)
    3. Storage (ArangoDB)
    """
    
    def __init__(self, config: Dict[str, Any], collection_prefix: str = "documents"):
        """
        Create a GenericDocumentProcessor configured to run extraction, embedding, and storage for a collection of documents.
        
        Initializes processor state from `config`, ensures a staging directory exists, and derives ArangoDB collection names using `collection_prefix`.
        
        Parameters:
            config (dict): Configuration dictionary. Expected keys (optional):
                - "staging.directory": path for temporary per-document staging files (defaults to "/dev/shm/doc_staging").
                - extraction / phases.extraction, embedding / phases.embedding, and arango-related settings used elsewhere by the processor.
            collection_prefix (str): Prefix used to construct ArangoDB collection names; resulting collections are
                "{prefix}_papers", "{prefix}_chunks", "{prefix}_embeddings", and "{prefix}_structures".
        
        Notes:
            - The staging directory is created if it does not exist.
            - This initializer sets `self.staging_dir`, `self.config`, `self.collection_prefix`, and `self.collections`.
        """
        self.config = config
        self.collection_prefix = collection_prefix
        
        # Setup staging directory
        self.staging_dir = Path(config.get('staging', {}).get('directory', '/dev/shm/doc_staging'))
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        
        # Collections for this source
        self.collections = {
            'papers': f'{collection_prefix}_papers',
            'chunks': f'{collection_prefix}_chunks',
            'embeddings': f'{collection_prefix}_embeddings',
            'structures': f'{collection_prefix}_structures'
        }
        
        logger.info(f"Initialized GenericDocumentProcessor with prefix: {collection_prefix}")
        logger.info(f"Collections: {self.collections}")
    
    def process_documents(self, tasks: List[DocumentTask]) -> Dict[str, Any]:
        """
        Run the full processing pipeline for a batch of DocumentTask items: extraction, then embedding/storage.
        
        If extraction produces no staged files the method returns early with extraction details and an empty embedding result.
        
        Parameters:
            tasks (List[DocumentTask]): Documents to process.
        
        Returns:
            Dict[str, Any]: Summary of the run with keys:
                - success (bool): overall success (False if extraction produced no staged files).
                - extraction (dict): results from the extraction phase (includes 'success', 'failed', 'staged_files').
                - embedding (dict): results from the embedding/storage phase (includes 'success', 'failed', 'chunks_created').
                - total_processed (int, optional): number of successfully embedded documents (present when success is True).
        """
        logger.info(f"Processing {len(tasks)} documents")
        
        # Phase 1: Extraction
        extraction_results = self._extraction_phase(tasks)
        
        if not extraction_results['staged_files']:
            logger.error("No documents were successfully extracted")
            return {
                'success': False,
                'extraction': extraction_results,
                'embedding': {'success': [], 'failed': []}
            }
        
        # Phase 2: Embedding and Storage
        embedding_results = self._embedding_phase(extraction_results['staged_files'])
        
        return {
            'success': True,
            'extraction': extraction_results,
            'embedding': embedding_results,
            'total_processed': len(embedding_results['success'])
        }
    
    def _extraction_phase(self, tasks: List[DocumentTask]) -> Dict[str, Any]:
        """
        Run extraction for a batch of documents and write per-document staging JSONs.
        
        This method dispatches DocumentTask items to a pool of worker processes (with GPU device
        assignment taken from the configured extraction settings) to extract each document into
        the processor's staging directory. Extraction is performed per-task and collected into an
        aggregation of results; failures for individual documents are handled per-task so the
        overall phase continues.
        
        Returns:
            Dict[str, Any]: A summary dictionary with keys:
                - "success": List[str] — document_id values that were successfully extracted.
                - "failed": List[str] — document_id values that failed extraction.
                - "staged_files": List[str] — filesystem paths to the created staging JSON files
                  for successful extractions.
        
        Side effects:
            - Writes one staging JSON file per successfully extracted document under the
              instance's staging_dir.
            - Uses worker processes and may set CUDA_VISIBLE_DEVICES for GPU assignment.
        """
        logger.info(f"EXTRACTION PHASE: {len(tasks)} documents")
        
        # Check both 'extraction' and 'phases.extraction' for config compatibility
        extraction_config = self.config.get('phases', {}).get('extraction', {}) or self.config.get('extraction', {})
        num_workers = extraction_config.get('workers', 8)
        gpu_devices = extraction_config.get('gpu_devices', [0, 1])
        
        logger.info(f"Starting {num_workers} extraction workers on GPU devices {gpu_devices}")
        
        results: Dict[str, List[str]] = {
            'success': [],
            'failed': [],
            'staged_files': []
        }
        
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_extraction_worker,
            initargs=(gpu_devices, extraction_config)
        ) as executor:
            
            futures = {
                executor.submit(_extract_document, task, str(self.staging_dir)): task
                for task in tasks
            }
            
            with tqdm(total=len(tasks), desc="Extracting documents") as pbar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result(timeout=120)
                        if result['success']:
                            results['success'].append(result['document_id'])
                            results['staged_files'].append(result['staged_path'])
                        else:
                            results['failed'].append(result['document_id'])
                            logger.warning(f"Extraction failed for {result['document_id']}: {result.get('error')}")
                    except Exception as e:
                        results['failed'].append(task.document_id)
                        logger.error(f"Extraction error for {task.document_id}: {e}")
                    finally:
                        pbar.update(1)
        
        logger.info(f"Extraction complete: {len(results['success'])} success, {len(results['failed'])} failed")
        return results
    
    def _embedding_phase(self, staged_files: List[str]) -> Dict[str, Any]:
        """
        Run embedding and storage for a set of staged document JSON files.
        
        This method reads embedding configuration from self.config (preferring `phases.embedding` and falling back to `embedding`),
        spawns a pool of embedding worker processes (initialized via _init_embedding_worker), and dispatches _embed_and_store
        for each path in staged_files. It aggregates per-document outcomes into a summary dictionary.
        
        Parameters:
            staged_files (List[str]): Paths to per-document staging JSON files produced during extraction.
        
        Returns:
            Dict[str, Any]: A summary with keys:
                - 'success' (List[str]): document_ids that were embedded and stored successfully.
                - 'failed' (List[str]): staging file paths for documents that failed to embed/store.
                - 'chunks_created' (int): total number of chunks created across all successfully processed documents.
        """
        logger.info(f"EMBEDDING PHASE: {len(staged_files)} documents")
        
        # Check both 'embedding' and 'phases.embedding' for config compatibility
        embedding_config = self.config.get('phases', {}).get('embedding', {}) or self.config.get('embedding', {})
        num_workers = embedding_config.get('workers', 4)
        gpu_devices = embedding_config.get('gpu_devices', [0, 1])
        
        logger.info(f"Starting {num_workers} embedding workers on GPU devices {gpu_devices}")
        
        success: List[str] = []
        failed: List[str] = []
        results: Dict[str, Any] = {
            'success': success,
            'failed': failed,
            'chunks_created': 0
        }
        
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_embedding_worker,
            initargs=(gpu_devices, self.config.get('arango', {}), self.collections)
        ) as executor:
            
            futures = {
                executor.submit(_embed_and_store, staged_path): staged_path
                for staged_path in staged_files
            }
            
            with tqdm(total=len(staged_files), desc="Embedding documents") as pbar:
                for future in as_completed(futures):
                    staged_path = futures[future]
                    try:
                        result = future.result(timeout=300)
                        if result['success']:
                            success.append(result['document_id'])
                            results['chunks_created'] += result['num_chunks']
                        else:
                            failed.append(staged_path)
                            logger.warning(f"Embedding failed: {result.get('error')}")
                    except Exception as e:
                        failed.append(staged_path)
                        logger.error(f"Embedding error: {e}")
                    finally:
                        pbar.update(1)
        
        logger.info(f"Embedding complete: {len(results['success'])} success, {len(results['failed'])} failed")
        return results


# Worker functions (run in separate processes)

WORKER_DOCLING = None
WORKER_EMBEDDER = None
WORKER_DB_MANAGER = None


def _init_extraction_worker(gpu_devices, extraction_config):
    """
    Initialize the extraction worker process.
    
    Sets up per-worker GPU visibility and creates the module-level WORKER_DOCLING extractor instance used by extraction tasks.
    
    Parameters:
        gpu_devices (Sequence[int]): List of GPU device indices available to workers. If provided, this worker will set CUDA_VISIBLE_DEVICES to one device chosen by worker index modulo the list length.
        extraction_config (Mapping): Configuration for extraction. Recognized keys:
            - 'type': if equal to 'code', a CodeExtractor is created; otherwise a RobustExtractor is used.
            - 'timeout_seconds': timeout passed to RobustExtractor (default 30).
            - 'docling': dict with optional keys 'use_ocr', 'extract_tables', and 'use_fallback' used when creating RobustExtractor.
    
    Side effects:
        - Sets the environment variable CUDA_VISIBLE_DEVICES for the current process when gpu_devices is non-empty.
        - Assigns a process-global extractor to WORKER_DOCLING (either CodeExtractor or RobustExtractor), which the worker relies on for document extraction.
    """
    global WORKER_DOCLING
    import os
    
    # Assign GPU
    worker_info = mp.current_process()
    worker_id = worker_info._identity[0] if hasattr(worker_info, '_identity') else 0
    if gpu_devices:
        gpu_id = gpu_devices[worker_id % len(gpu_devices)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Check extraction type from config
    extraction_type = extraction_config.get('type', 'pdf')
    
    if extraction_type == 'code':
        # Use CodeExtractor for code files
        from core.extractors import CodeExtractor
        WORKER_DOCLING = CodeExtractor()
    else:
        # Use RobustExtractor for PDFs
        from core.extractors import RobustExtractor
        WORKER_DOCLING = RobustExtractor(
            use_ocr=extraction_config.get('docling', {}).get('use_ocr', False),
            extract_tables=extraction_config.get('docling', {}).get('extract_tables', True),
            timeout=extraction_config.get('timeout_seconds', 30),
            use_fallback=extraction_config.get('docling', {}).get('use_fallback', True)
        )


def _extract_document(task: DocumentTask, staging_dir: str) -> Dict[str, Any]:
    """
    Extract a single document into a staging JSON file by running the configured extractor and (optionally) attaching LaTeX source.
    
    This function uses the module-level extractor instance initialized in worker processes to extract text and structure from task.pdf_path, attempts to read task.latex_path if provided, and writes a staged JSON document into staging_dir. On success it returns the staged file path; on failure it returns an error message.
    
    Parameters:
        task (DocumentTask): The document task containing document_id, pdf_path, optional latex_path, and metadata.
        staging_dir (str): Directory where the per-document staging JSON will be written.
    
    Returns:
        Dict[str, Any]: A result dictionary with:
            - 'success' (bool): True on success, False on failure.
            - 'document_id' (str): The task's document_id.
            - 'staged_path' (str): Path to the written staging JSON (present when success is True).
            - 'error' (str): Error message (present when success is False).
    
    Raises:
        RuntimeError: If the module-level extractor worker is not initialized.
    """
    global WORKER_DOCLING
    
    if WORKER_DOCLING is None:
        raise RuntimeError("Worker not initialized")
    
    try:
        # Extract PDF
        extracted = WORKER_DOCLING.extract(task.pdf_path)
        
        # Add LaTeX if available
        if task.latex_path and Path(task.latex_path).exists():
            try:
                with open(task.latex_path, 'r', encoding='utf-8') as f:
                    extracted['latex_source'] = f.read()
                    extracted['has_latex'] = True
            except Exception as e:
                logger.warning(f"Could not read LaTeX: {e}")
                extracted['has_latex'] = False
        else:
            extracted['has_latex'] = False
        
        # Build complete document
        document = {
            'document_id': task.document_id,
            'pdf_path': task.pdf_path,
            'latex_path': task.latex_path,
            'metadata': task.metadata,
            'extracted': extracted,
            'processing_date': datetime.now().isoformat()
        }
        
        # Save to staging
        staged_path = Path(staging_dir) / f"{task.document_id.replace('/', '_')}.json"
        with open(staged_path, 'w') as f:
            json.dump(document, f)
        
        return {
            'success': True,
            'document_id': task.document_id,
            'staged_path': str(staged_path)
        }
        
    except Exception as e:
        return {
            'success': False,
            'document_id': task.document_id,
            'error': str(e)
        }


def _init_embedding_worker(gpu_devices, arango_config, collections):
    """
    Initialize per-process embedding resources.
    
    Sets up global WORKER_EMBEDDER and WORKER_DB_MANAGER for the current worker process. This includes:
    - Selecting and exporting a CUDA_VISIBLE_DEVICES value derived from gpu_devices and the worker's multiprocessing index.
    - Inserting the repository root into sys.path so local modules can be imported.
    - Creating a JinaV4Embedder configured for CUDA with FP16 and predetermined chunking parameters.
    - Creating an ArangoDBManager from arango_config and overriding its collections mapping with the provided collections.
    
    Parameters:
        gpu_devices (Sequence[int] | None): Sequence of GPU device ids available to workers; if provided, one is chosen based on the worker index.
        arango_config (Mapping): Configuration used to construct the ArangoDBManager.
        collections (Mapping[str, str]): Mapping of logical collection names to concrete ArangoDB collection names; used to override the manager's collections.
    
    Side effects:
        - Mutates module-level globals WORKER_EMBEDDER and WORKER_DB_MANAGER.
        - Mutates the process environment variable CUDA_VISIBLE_DEVICES.
        - Modifies sys.path for imports.
    """
    global WORKER_EMBEDDER, WORKER_DB_MANAGER
    import os
    from core.embedders import JinaV4Embedder
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.database.arango_db_manager import ArangoDBManager
    
    # Assign GPU
    worker_info = mp.current_process()
    worker_id = worker_info._identity[0] if hasattr(worker_info, '_identity') else 0
    if gpu_devices:
        gpu_id = gpu_devices[worker_id % len(gpu_devices)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Initialize embedder
    WORKER_EMBEDDER = JinaV4Embedder(
        device='cuda',
        use_fp16=True,
        chunk_size_tokens=1000,
        chunk_overlap_tokens=200
    )
    
    # Initialize DB manager with collections
    WORKER_DB_MANAGER = ArangoDBManager(arango_config)
    WORKER_DB_MANAGER.collections = collections  # Override with source-specific collections


def _embed_and_store(staged_path: str) -> Dict[str, Any]:
    """
    Embed the extracted text from a staged document and store paper, chunk, and embedding records in ArangoDB.
    
    Loads the staged JSON produced by the extraction phase, generates chunked embeddings via the worker embedder, and writes a paper record plus per-chunk records (text and embedding) inside a single ArangoDB transaction. The function acquires a per-document lock during the write, removes the staging file on success, and clears GPU memory before and after processing.
    
    Parameters:
        staged_path (str): Path to the staging JSON file created by the extraction step (contains keys like `document_id`, `metadata`, and `extracted`).
    
    Returns:
        Dict[str, Any]: Result object containing at minimum:
          - `success` (bool): True on success, False on error.
          - On success: `document_id` (str) and `num_chunks` (int).
          - On failure: `error` (str) with a short description.
    
    Raises:
        RuntimeError: If the module-level embedder or DB manager workers are not initialized.
    """
    global WORKER_EMBEDDER, WORKER_DB_MANAGER
    import torch
    
    if WORKER_EMBEDDER is None or WORKER_DB_MANAGER is None:
        raise RuntimeError("Worker not initialized")
    
    try:
        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Load staged document
        with open(staged_path, 'r') as f:
            doc = json.load(f)
        
        document_id = doc['document_id']
        extracted = doc.get('extracted', {})
        
        # Get text content
        full_text = (
            extracted.get('full_text', '') or 
            extracted.get('text', '') or 
            extracted.get('markdown', '')
        )
        
        if not full_text:
            return {
                'success': False,
                'error': 'No text content found'
            }
        
        # Generate embeddings
        chunks = WORKER_EMBEDDER.embed_with_late_chunking(full_text)
        
        # Store to ArangoDB
        sanitized_id = document_id.replace('.', '_').replace('/', '_')
        
        # Acquire lock with timeout (1 minute should be sufficient for lock acquisition)
        if not WORKER_DB_MANAGER.acquire_lock(document_id, timeout_minutes=1):
            logger.warning(f"Could not acquire lock for {document_id} within 60 seconds")
            return {
                'success': False,
                'error': 'Lock acquisition timeout - document may be processing elsewhere'
            }
        
        try:
            # Begin transaction
            collections = WORKER_DB_MANAGER.collections
            write_collections = [
                collections['papers'],
                collections['chunks'],
                collections['embeddings']
            ]
            
            txn_db = WORKER_DB_MANAGER.begin_transaction(
                write_collections=write_collections,
                lock_timeout=5
            )
            
            try:
                # Store paper metadata with Tree-sitter symbols if available
                paper_doc = {
                    '_key': sanitized_id,
                    'document_id': document_id,
                    'metadata': doc.get('metadata', {}),
                    'status': 'PROCESSED',
                    'processing_date': datetime.now().isoformat(),
                    'num_chunks': len(chunks),
                    'has_latex': extracted.get('has_latex', False)
                }
                
                # Add repository field if this is GitHub data
                if doc.get('metadata', {}).get('repo'):
                    paper_doc['repository'] = doc['metadata']['repo']
                
                # Add Tree-sitter symbol data if available
                if extracted.get('symbols'):
                    paper_doc['symbols'] = extracted['symbols']
                    paper_doc['symbol_hash'] = extracted.get('symbol_hash', '')
                    paper_doc['code_metrics'] = extracted.get('code_metrics', {})
                    paper_doc['code_structure'] = extracted.get('code_structure', {})
                    paper_doc['language'] = extracted.get('language')
                    paper_doc['has_tree_sitter'] = True
                else:
                    paper_doc['has_tree_sitter'] = False
                
                txn_db.collection(collections['papers']).insert(paper_doc, overwrite=True)
                
                # Store chunks and embeddings
                for i, chunk in enumerate(chunks):
                    chunk_key = f"{sanitized_id}_chunk_{i}"
                    
                    # Store chunk
                    txn_db.collection(collections['chunks']).insert({
                        '_key': chunk_key,
                        'document_id': sanitized_id,
                        'text': chunk.text,
                        'chunk_index': i,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char
                    }, overwrite=True)
                    
                    # Store embedding with symbol metadata for code files
                    embedding_doc = {
                        '_key': f"{chunk_key}_emb",
                        'document_id': sanitized_id,
                        'chunk_id': chunk_key,
                        'vector': chunk.embedding.tolist(),
                        'model': 'jina-v4'
                    }
                    
                    # Add symbol metadata if this is code with Tree-sitter data
                    if extracted.get('symbols'):
                        embedding_doc['has_symbols'] = True
                        embedding_doc['language'] = extracted.get('language')
                        # Store a summary of symbols for this chunk's context
                        # This helps the Jina v4 coding LoRA understand the code context
                        embedding_doc['symbol_context'] = {
                            'total_functions': len(extracted['symbols'].get('functions', [])),
                            'total_classes': len(extracted['symbols'].get('classes', [])),
                            'total_imports': len(extracted['symbols'].get('imports', [])),
                            'complexity': extracted.get('code_metrics', {}).get('complexity', 0)
                        }
                    
                    txn_db.collection(collections['embeddings']).insert(embedding_doc, overwrite=True)
                
                # Commit
                txn_db.commit_transaction()
                
            except Exception as e:
                txn_db.abort_transaction()
                raise
                
        finally:
            WORKER_DB_MANAGER.release_lock(document_id)
        
        # Clean up staging file
        Path(staged_path).unlink()
        
        # Clear GPU memory after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'success': True,
            'document_id': document_id,
            'num_chunks': len(chunks)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
