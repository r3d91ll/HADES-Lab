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
from dataclasses import dataclass
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
    metadata: Dict[str, Any] = None  # Optional metadata from source
    
    def __post_init__(self):
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
        Initialize processor.
        
        Args:
            config: Configuration dict with extraction, embedding, storage settings
            collection_prefix: Prefix for ArangoDB collections (e.g., "arxiv", "semantic_scholar")
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
        Process a batch of documents through the full pipeline.
        
        Args:
            tasks: List of DocumentTask objects to process
            
        Returns:
            Processing results with success/failure counts
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
        """Extract documents to staging area."""
        logger.info(f"EXTRACTION PHASE: {len(tasks)} documents")
        
        # Check both 'extraction' and 'phases.extraction' for config compatibility
        extraction_config = self.config.get('phases', {}).get('extraction', {}) or self.config.get('extraction', {})
        num_workers = extraction_config.get('workers', 8)
        gpu_devices = extraction_config.get('gpu_devices', [0, 1])
        
        logger.info(f"Starting {num_workers} extraction workers on GPU devices {gpu_devices}")
        
        results = {
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
        """Embed and store documents."""
        logger.info(f"EMBEDDING PHASE: {len(staged_files)} documents")
        
        # Check both 'embedding' and 'phases.embedding' for config compatibility
        embedding_config = self.config.get('phases', {}).get('embedding', {}) or self.config.get('embedding', {})
        num_workers = embedding_config.get('workers', 4)
        gpu_devices = embedding_config.get('gpu_devices', [0, 1])
        
        logger.info(f"Starting {num_workers} embedding workers on GPU devices {gpu_devices}")
        
        results = {
            'success': [],
            'failed': [],
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
                            results['success'].append(result['document_id'])
                            results['chunks_created'] += result['num_chunks']
                        else:
                            results['failed'].append(staged_path)
                            logger.warning(f"Embedding failed: {result.get('error')}")
                    except Exception as e:
                        results['failed'].append(staged_path)
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
    """Initialize extraction worker."""
    global WORKER_DOCLING
    import os
    from core.framework.extractors.docling_extractor import DoclingExtractor
    
    # Assign GPU
    worker_info = mp.current_process()
    worker_id = worker_info._identity[0] if hasattr(worker_info, '_identity') else 0
    if gpu_devices:
        gpu_id = gpu_devices[worker_id % len(gpu_devices)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Initialize Docling
    WORKER_DOCLING = DoclingExtractor(
        use_ocr=extraction_config.get('use_ocr', False),
        extract_tables=extraction_config.get('extract_tables', True),
        use_fallback=extraction_config.get('use_fallback', True)
    )


def _extract_document(task: DocumentTask, staging_dir: str) -> Dict[str, Any]:
    """Extract a single document."""
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
    """Initialize embedding worker."""
    global WORKER_EMBEDDER, WORKER_DB_MANAGER
    import os
    from core.framework.embedders import JinaV4Embedder
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from tools.arxiv.pipelines.arango_db_manager import ArangoDBManager
    
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
    """Embed and store a document."""
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
                # Store paper metadata
                txn_db.collection(collections['papers']).insert({
                    '_key': sanitized_id,
                    'document_id': document_id,
                    'metadata': doc.get('metadata', {}),
                    'status': 'PROCESSED',
                    'processing_date': datetime.now().isoformat(),
                    'num_chunks': len(chunks),
                    'has_latex': extracted.get('has_latex', False)
                }, overwrite=True)
                
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
                    
                    # Store embedding
                    txn_db.collection(collections['embeddings']).insert({
                        '_key': f"{chunk_key}_emb",
                        'document_id': sanitized_id,
                        'chunk_id': chunk_key,
                        'vector': chunk.embedding.tolist(),
                        'model': 'jina-v4'
                    }, overwrite=True)
                
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