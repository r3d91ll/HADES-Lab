#!/usr/bin/env python3
"""
ACID Pipeline with Phase-Separated Architecture
===============================================

This implementation separates Docling extraction (CPU) and Jina embedding (GPU)
into distinct phases to eliminate resource competition. Documents are processed
in two phases:

Phase 1 (EXTRACTION): All PDFs extracted to JSON in RamFS staging
Phase 2 (EMBEDDING): All staged JSONs embedded and stored to ArangoDB

This architecture ensures:
- No GPU memory competition between Docling and Jina
- Maximum worker utilization in each phase
- Observable Actor-Network Theory boundaries
- Full ACID compliance with atomic transactions
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import traceback
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue, Empty
import threading
import signal
import psutil
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from arango import ArangoClient
from arango.exceptions import DocumentInsertError, ArangoServerError
import yaml
from functools import wraps
import random

# Import shared ArangoDBManager and retry decorator
try:
    # Try relative import first (when used as module)
    from .arango_db_manager import ArangoDBManager, retry_with_backoff
except ImportError:
    # Fall back to absolute import (when run directly)
    from arango_db_manager import ArangoDBManager, retry_with_backoff

# Global instances for worker processes
WORKER_EMBEDDER = None
WORKER_DB_MANAGER = None
WORKER_DOCLING = None


# retry_with_backoff is imported from arango_db_manager


def _extract_single_document(task):
    """Extract a single document with GPU-accelerated Docling."""
    global WORKER_DOCLING
    import json
    from pathlib import Path
    from datetime import datetime
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Use the pre-initialized Docling instance
    if WORKER_DOCLING is None:
        raise RuntimeError("WORKER_DOCLING not initialized! Worker initialization failed.")
    
    extractor = WORKER_DOCLING
    
    try:
        # Extract PDF with GPU acceleration
        extracted = extractor.extract(task.pdf_path)
    
        # Merge with LaTeX if available
        if task.latex_path and Path(task.latex_path).exists():
            try:
                with open(task.latex_path, 'r', encoding='utf-8') as f:
                    latex_content = f.read()
                extracted['latex_source'] = latex_content
                extracted['has_latex'] = True
            except Exception as e:
                logger.warning(f"LaTeX source not available for paper {task.arxiv_id}: {type(e).__name__}: {e}")
                extracted['has_latex'] = False
        else:
            extracted['has_latex'] = False
        
        # Add metadata
        extracted['arxiv_id'] = task.arxiv_id
        extracted['pdf_path'] = task.pdf_path
        extracted['processing_date'] = datetime.now().isoformat()
        
        # Save to staging
        staged_path = Path(f"/dev/shm/acid_staging/{task.arxiv_id}.json")
        with open(staged_path, 'w') as f:
            json.dump(extracted, f)
        
        return {
            'success': True,
            'arxiv_id': task.arxiv_id,
            'staged_path': str(staged_path),
            'size_mb': staged_path.stat().st_size / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Extraction failed for paper {task.arxiv_id}: {type(e).__name__}: {e}")
        return {
            'success': False,
            'arxiv_id': task.arxiv_id,
            'error': str(e),
            'size_mb': 0
        }


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]):
    """Setup logging based on configuration."""
    log_config = config if isinstance(config, dict) else {}
    
    level = log_config.get('level', 'INFO')
    log_file = log_config.get('file', None)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console handler
    if log_config.get('console', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(file_handler)


# ArangoDBManager has been moved to arango_db_manager.py for shared use


@dataclass
class ProcessingTask:
    """Task for processing."""
    arxiv_id: str
    pdf_path: str
    latex_path: Optional[str] = None
    priority: int = 0
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PhaseManager:
    """
    Manages the two-phase processing architecture.
    
    Phase 1 (EXTRACTION): CPU-only Docling extraction to RamFS
    Phase 2 (EMBEDDING): GPU-only Jina embedding from RamFS
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize phase manager."""
        self.config = config
        self.current_phase = "IDLE"
        
        # Setup staging directory in RamFS
        self.staging_dir = Path(config.get('staging', {}).get('directory', '/dev/shm/acid_staging'))
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Staging directory: {self.staging_dir}")
        
        # Phase statistics
        self.phase_stats = {
            'EXTRACTION': {
                'start_time': None,
                'end_time': None,
                'papers_processed': 0,
                'papers_failed': 0,
                'total_size_mb': 0
            },
            'EMBEDDING': {
                'start_time': None,
                'end_time': None,
                'papers_processed': 0,
                'papers_failed': 0,
                'chunks_created': 0
            }
        }
        
        # Database manager (for EMBEDDING phase)
        self.db_manager = None
        
        # Worker pools (created per phase)
        self.worker_pool = None
        
    def start_extraction_phase(self, tasks: List[ProcessingTask]) -> Dict[str, Any]:
        """
        Phase 1: Extract all PDFs to JSON in staging.
        Uses GPU-accelerated Docling workers with parallel processing.
        """
        self.current_phase = "EXTRACTION"
        self.phase_stats['EXTRACTION']['start_time'] = datetime.now()
        logger.info(f"="*80)
        logger.info(f"PHASE 1: EXTRACTION - Processing {len(tasks)} papers")
        logger.info(f"="*80)
        
        # Ensure staging is clean
        self._clean_staging()
        
        # Get configuration
        num_workers = self.config.get('phases', {}).get('extraction', {}).get('workers', 8)
        gpu_devices = self.config.get('phases', {}).get('extraction', {}).get('gpu_devices', [0, 1])
        
        logger.info(f"Starting {num_workers} GPU extraction workers on devices {gpu_devices}")
        logger.info(f"Submitting {len(tasks)} individual extraction tasks to {num_workers} workers")
        
        results = {
            'success': [],
            'failed': [],
            'staged_files': []
        }
        
        # Process with GPU workers - each paper is a separate task
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context('spawn'),
            initializer=self._init_extraction_worker,
            initargs=(gpu_devices, self.config.get('phases', {}).get('extraction', {}).get('docling', {}))
        ) as executor:
            # Submit individual extraction tasks - all workers get work immediately
            futures = {}
            for task in tasks:
                future = executor.submit(_extract_single_document, task)
                futures[future] = task
            
            logger.info(f"Submitted {len(futures)} tasks to {num_workers} workers")
            
            # Process results as they complete
            with tqdm(total=len(tasks), desc="Extracting PDFs") as pbar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result(timeout=120)  # Timeout per document
                        if result['success']:
                            results['success'].append(result['arxiv_id'])
                            results['staged_files'].append(result['staged_path'])
                            self.phase_stats['EXTRACTION']['papers_processed'] += 1
                            self.phase_stats['EXTRACTION']['total_size_mb'] += result['size_mb']
                        else:
                            results['failed'].append(result['arxiv_id'])
                            self.phase_stats['EXTRACTION']['papers_failed'] += 1
                            logger.warning(f"Extraction incomplete for {result['arxiv_id']}: {result.get('error', 'Unknown error')}")
                        pbar.update(1)
                    except Exception as e:
                        results['failed'].append(task.arxiv_id)
                        self.phase_stats['EXTRACTION']['papers_failed'] += 1
                        logger.error(f"Unexpected extraction error for paper {task.arxiv_id}: {type(e).__name__}: {e}")
                        pbar.update(1)
        
        # Phase complete
        self.phase_stats['EXTRACTION']['end_time'] = datetime.now()
        elapsed = (self.phase_stats['EXTRACTION']['end_time'] - 
                  self.phase_stats['EXTRACTION']['start_time']).total_seconds()
        
        logger.info(f"="*80)
        logger.info(f"EXTRACTION PHASE COMPLETE")
        logger.info(f"  Processed: {self.phase_stats['EXTRACTION']['papers_processed']} papers")
        logger.info(f"  Failed: {self.phase_stats['EXTRACTION']['papers_failed']} papers")
        logger.info(f"  Total size: {self.phase_stats['EXTRACTION']['total_size_mb']:.2f} MB")
        logger.info(f"  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info(f"  Rate: {len(tasks)/elapsed*60:.1f} papers/minute")
        logger.info(f"="*80)
        
        # CRITICAL: Wait for extraction workers to fully terminate before cleaning GPU
        import time
        logger.info("Waiting for extraction workers to terminate...")
        time.sleep(5)  # Give workers time to fully shut down
        
        # Now clean up GPU memory after extraction workers are gone
        self._cleanup_gpu_memory()
        
        # Additional delay to ensure GPU is ready for embedding workers
        logger.info("Preparing for embedding phase...")
        time.sleep(3)
        
        return results
    
    def start_embedding_phase(self, staged_files: List[str]) -> Dict[str, Any]:
        """
        Phase 2: Embed all staged JSONs and store to ArangoDB.
        Uses GPU-only Jina workers.
        """
        self.current_phase = "EMBEDDING"
        self.phase_stats['EMBEDDING']['start_time'] = datetime.now()
        logger.info(f"="*80)
        logger.info(f"PHASE 2: EMBEDDING - Processing {len(staged_files)} staged documents")
        logger.info(f"="*80)
        
        # Initialize database connection
        self.db_manager = ArangoDBManager(self.config['arango'])
        
        # Create GPU embedding workers
        num_workers = self.config.get('embedding', {}).get('workers', 4)
        gpu_devices = self.config.get('embedding', {}).get('gpu_devices', [0, 1])
        logger.info(f"Starting {num_workers} GPU embedding workers on devices {gpu_devices}")
        
        results = {
            'success': [],
            'failed': [],
            'chunks_created': 0
        }
        
        # Process with GPU workers
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context('spawn'),
            initializer=self._init_embedding_worker,
            initargs=(gpu_devices, self.config['arango'])
        ) as executor:
            # Submit all embedding tasks
            futures = {}
            for i, staged_path in enumerate(staged_files):
                # Assign GPU in round-robin fashion
                gpu_id = gpu_devices[i % len(gpu_devices)]
                future = executor.submit(self._embed_and_store, staged_path, gpu_id)
                futures[future] = staged_path
            
            # Process results as they complete
            with tqdm(total=len(staged_files), desc="Embedding documents") as pbar:
                for future in as_completed(futures):
                    staged_path = futures[future]
                    try:
                        result = future.result(timeout=300)
                        if result['success']:
                            results['success'].append(result['arxiv_id'])
                            results['chunks_created'] += result['num_chunks']
                            self.phase_stats['EMBEDDING']['papers_processed'] += 1
                            self.phase_stats['EMBEDDING']['chunks_created'] += result['num_chunks']
                        else:
                            results['failed'].append(staged_path)
                            self.phase_stats['EMBEDDING']['papers_failed'] += 1
                            logger.warning(f"Embedding generation failed for {staged_path}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        results['failed'].append(staged_path)
                        self.phase_stats['EMBEDDING']['papers_failed'] += 1
                        logger.error(f"Unexpected embedding error for {staged_path}: {type(e).__name__}: {e}")
                    finally:
                        pbar.update(1)
        
        # Phase complete
        self.phase_stats['EMBEDDING']['end_time'] = datetime.now()
        elapsed = (self.phase_stats['EMBEDDING']['end_time'] - 
                  self.phase_stats['EMBEDDING']['start_time']).total_seconds()
        
        logger.info(f"="*80)
        logger.info(f"EMBEDDING PHASE COMPLETE")
        logger.info(f"  Processed: {self.phase_stats['EMBEDDING']['papers_processed']} papers")
        logger.info(f"  Failed: {self.phase_stats['EMBEDDING']['papers_failed']} papers")
        logger.info(f"  Chunks created: {self.phase_stats['EMBEDDING']['chunks_created']}")
        logger.info(f"  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info(f"  Rate: {len(staged_files)/elapsed*60:.1f} papers/minute")
        logger.info(f"="*80)
        
        return results
    
    def _clean_staging(self):
        """Clean staging directory."""
        if self.staging_dir.exists():
            for file in self.staging_dir.glob("*.json"):
                file.unlink()
            logger.info(f"Cleaned staging directory: {self.staging_dir}")
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory between phases."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cleared between phases")
        except Exception as e:
            logger.warning(f"GPU memory cleanup failed (non-critical): {type(e).__name__}: {e}")
    
    @staticmethod
    def _init_extraction_worker(gpu_devices=None, docling_config=None):
        """Initialize extraction worker with GPU access and Docling instance."""
        import torch
        import os
        import logging
        from core.framework.extractors.docling_extractor import DoclingExtractor
        
        # Set up logging
        logger = logging.getLogger(__name__)
        
        # Set up worker-specific GPU assignment
        worker_info = mp.current_process()
        worker_id = worker_info._identity[0] if hasattr(worker_info, '_identity') else 0
        
        if gpu_devices:
            gpu_id = gpu_devices[worker_id % len(gpu_devices)]
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.info(f"Extraction worker {worker_id} assigned to GPU {gpu_id}")
        
        # Set reasonable number of CPU threads per worker
        torch.set_num_threads(8)
        
        # Initialize Docling ONCE per worker (reused for all batches)
        global WORKER_DOCLING
        docling_config = docling_config or {}
        WORKER_DOCLING = DoclingExtractor(
            use_ocr=docling_config.get('use_ocr', False),
            extract_tables=docling_config.get('extract_tables', True),
            use_fallback=docling_config.get('use_fallback', True)
        )
        # Note: extract_equations and extract_images are handled internally by Docling
        
        # Log GPU availability
        if torch.cuda.is_available():
            logger.info(f"✓ Extraction worker {worker_id} initialized with GPU access and Docling instance")
        else:
            logger.info(f"✓ Extraction worker {worker_id} initialized (CPU-only mode) with Docling instance")
    
    @staticmethod
    def _init_embedding_worker(gpu_devices: List[int], arango_config: Dict[str, Any]):
        """Initialize embedding worker with pre-loaded Jina model and DB connection."""
        import torch
        import os
        import logging
        from core.framework.embedders import JinaV4Embedder
        # ArangoDBManager is already imported at module level
        
        # Set up logging for worker
        logger = logging.getLogger(__name__)
        
        # Set up worker-specific GPU assignment
        worker_info = mp.current_process()
        worker_id = worker_info._identity[0] if hasattr(worker_info, '_identity') else 0
        gpu_id = gpu_devices[worker_id % len(gpu_devices)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Initialize embedder ONCE per worker
        global WORKER_EMBEDDER, WORKER_DB_MANAGER
        
        logger.info(f"Embedding worker {worker_id} starting initialization on GPU {gpu_id}...")
        
        WORKER_EMBEDDER = JinaV4Embedder(
            device='cuda',
            use_fp16=True,
            chunk_size_tokens=1000,
            chunk_overlap_tokens=200
        )
        
        # Initialize database connection ONCE per worker
        WORKER_DB_MANAGER = ArangoDBManager(arango_config)
        
        logger.info(f"✓ Embedding worker {worker_id} initialized on GPU {gpu_id} with pre-loaded Jina v4 model")
    
    @staticmethod
    def _extract_document_batch(task_batch: List[ProcessingTask]) -> List[Dict[str, Any]]:
        """Extract a batch of documents with GPU-accelerated Docling."""
        global WORKER_DOCLING
        import json
        from pathlib import Path
        from datetime import datetime
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Use the pre-initialized Docling instance
        if WORKER_DOCLING is None:
            raise RuntimeError("WORKER_DOCLING not initialized! Worker initialization failed.")
        
        extractor = WORKER_DOCLING
        batch_results = []
        
        # Process each document in the batch
        for task in task_batch:
            try:
                # Extract PDF with GPU acceleration
                extracted = extractor.extract(task.pdf_path)
            
                # Merge with LaTeX if available
                if task.latex_path and Path(task.latex_path).exists():
                    try:
                        with open(task.latex_path, 'r', encoding='utf-8') as f:
                            latex_content = f.read()
                        extracted['latex_source'] = latex_content
                        extracted['has_latex'] = True
                    except Exception as e:
                        logger.warning(f"Failed to read LaTeX for {task.arxiv_id}: {e}")
                
                # Create unified document
                unified_doc = {
                    'arxiv_id': task.arxiv_id,
                    'pdf_path': task.pdf_path,
                    'latex_path': task.latex_path,
                    'extracted': extracted,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'extraction_method': 'docling_gpu_batch'
                }
                
                # Save to staging
                staging_dir = Path('/dev/shm/acid_staging')  # Use config path
                staging_dir.mkdir(parents=True, exist_ok=True)
                staged_path = staging_dir / f"{task.arxiv_id}.json"
                with open(staged_path, 'w', encoding='utf-8') as f:
                    json.dump(unified_doc, f, ensure_ascii=False, indent=2)
                
                # Get file size
                size_mb = staged_path.stat().st_size / (1024 * 1024)
                
                batch_results.append({
                    'success': True,
                    'arxiv_id': task.arxiv_id,
                    'staged_path': str(staged_path),
                    'size_mb': size_mb
                })
                
            except Exception as e:
                batch_results.append({
                    'success': False,
                    'arxiv_id': task.arxiv_id,
                    'error': str(e)
                })
                logger.error(f"Failed to extract {task.arxiv_id}: {e}")
        
        return batch_results
    
    @staticmethod
    def _embed_and_store(staged_path: str, gpu_id: int) -> Dict[str, Any]:
        """Embed document and store to ArangoDB using pre-loaded embedder and DB connection."""
        try:
            # Use the pre-loaded embedder and DB manager from worker initialization
            global WORKER_EMBEDDER, WORKER_DB_MANAGER
            
            if WORKER_EMBEDDER is None:
                # This should never happen if worker was properly initialized
                raise RuntimeError("WORKER_EMBEDDER not initialized! Worker initialization failed.")
            
            if WORKER_DB_MANAGER is None:
                raise RuntimeError("WORKER_DB_MANAGER not initialized! Worker initialization failed.")
            
            embedder = WORKER_EMBEDDER
            db_manager = WORKER_DB_MANAGER
            
            # Load staged document
            with open(staged_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            
            arxiv_id = doc['arxiv_id']
            
            # Extract text - check both 'extracted' key (Docling v2) and top-level (backward compat)
            # Docling v2 stores content under 'extracted' key
            extracted = doc.get('extracted', {})
            full_text = extracted.get('full_text', '') or extracted.get('markdown', '') or doc.get('full_text', '') or doc.get('markdown', '')
            
            # Generate embeddings with late chunking
            chunks = embedder.embed_with_late_chunking(full_text)
            
            # Acquire lock first (ACID pattern: Reserve → Compute → Commit → Release)
            if not db_manager.acquire_lock(arxiv_id):
                logger.warning(f"Paper {arxiv_id} is already in processing queue, skipping duplicate")
                return {
                    'success': False,
                    'error': 'Already being processed by another worker'
                }
            
            try:
                # Store to ArangoDB with real stream transaction
                sanitized_id = arxiv_id.replace('.', '_').replace('/', '_')
                
                # Begin stream transaction with all collections we'll write to
                write_collections = ['arxiv_papers', 'arxiv_chunks', 'arxiv_embeddings']
                structures = doc.get('structures', {})
                if structures.get('equations'):
                    write_collections.append('arxiv_equations')
                if structures.get('tables'):
                    write_collections.append('arxiv_tables')
                if structures.get('images'):
                    write_collections.append('arxiv_images')
                
                txn_db = db_manager.begin_transaction(
                    write_collections=write_collections,
                    lock_timeout=5
                )
                
                try:
                    # Store paper metadata
                    txn_db.collection('arxiv_papers').insert({
                        '_key': sanitized_id,
                        'arxiv_id': arxiv_id,
                        'title': doc.get('title', ''),
                        'abstract': doc.get('abstract', ''),
                        'status': 'PROCESSED',
                        'processing_date': datetime.now().isoformat(),
                        'num_chunks': len(chunks),
                        'num_equations': len(structures.get('equations', [])),
                        'num_tables': len(structures.get('tables', [])),
                        'num_images': len(structures.get('images', []))
                    }, overwrite=True)
                    
                    # Store chunks and embeddings separately (proper ACID schema)
                    for i, chunk in enumerate(chunks):
                        chunk_key = f"{sanitized_id}_chunk_{i}"
                        
                        # Store chunk text
                        txn_db.collection('arxiv_chunks').insert({
                            '_key': chunk_key,
                            'paper_id': sanitized_id,
                            'text': chunk.text,
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'start_char': chunk.start_char,
                            'end_char': chunk.end_char,
                            'start_token': chunk.start_token,
                            'end_token': chunk.end_token,
                            'context_window_used': chunk.context_window_used
                        }, overwrite=True)
                        
                        # Store embedding
                        txn_db.collection('arxiv_embeddings').insert({
                            '_key': f"{chunk_key}_emb",
                            'paper_id': sanitized_id,
                            'chunk_id': chunk_key,
                            'vector': chunk.embedding.tolist(),
                            'model': 'jina-v4',
                            'embedding_method': 'late_chunking',
                            'embedding_date': datetime.now().isoformat()
                        }, overwrite=True)
                    
                    # Store equations if present
                    for i, eq in enumerate(structures.get('equations', [])):
                        txn_db.collection('arxiv_equations').insert({
                            '_key': f"{sanitized_id}_eq_{i}",
                            'paper_id': sanitized_id,
                            'equation_index': i,
                            'latex': eq.get('latex', ''),
                            'label': eq.get('label', ''),
                            'type': eq.get('type', 'display')
                        }, overwrite=True)
                    
                    # Store tables if present
                    for i, tbl in enumerate(structures.get('tables', [])):
                        txn_db.collection('arxiv_tables').insert({
                            '_key': f"{sanitized_id}_table_{i}",
                            'paper_id': sanitized_id,
                            'table_index': i,
                            'caption': tbl.get('caption', ''),
                            'headers': tbl.get('headers', []),
                            'data_rows': tbl.get('rows', [])
                        }, overwrite=True)
                    
                    # Store images if present
                    for i, img in enumerate(structures.get('images', [])):
                        txn_db.collection('arxiv_images').insert({
                            '_key': f"{sanitized_id}_img_{i}",
                            'paper_id': sanitized_id,
                            'image_index': i,
                            'caption': img.get('caption', ''),
                            'type': img.get('type', 'figure')
                        }, overwrite=True)
                    
                    # Commit transaction
                    txn_db.commit_transaction()
                    
                except Exception as e:
                    txn_db.abort_transaction()
                    raise
                
            finally:
                # Always release lock
                db_manager.release_lock(arxiv_id)
            
            # Clean up staged file
            Path(staged_path).unlink()
            
            return {
                'success': True,
                'arxiv_id': arxiv_id,
                'num_chunks': len(chunks)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def _create_full_text(extracted: Dict[str, Any]) -> str:
        """Create unified text from extracted content."""
        parts = []
        
        # Add title with safe access
        title = extracted.get('title')
        if title:
            parts.append(f"# {title}\n")
        
        # Add abstract with safe access
        abstract = extracted.get('abstract')
        if abstract:
            parts.append(f"## Abstract\n{abstract}\n")
        
        # Add main text - check multiple possible fields
        # Docling may use 'full_text', 'text', or 'markdown'
        full_text = extracted.get('full_text') or extracted.get('text') or extracted.get('markdown')
        if full_text:
            parts.append(full_text)
        
        # Add LaTeX if present
        if extracted.get('latex_source'):
            parts.append("\n[BEGIN LATEX SOURCE]\n")
            parts.append(extracted['latex_source'])
            parts.append("\n[END LATEX SOURCE]\n")
        
        return "\n".join(parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get phase statistics."""
        return self.phase_stats


class ACIDPhasedPipeline:
    """
    Main pipeline orchestrator for phase-separated ACID processing.
    """
    
    def __init__(self, config_path: str):
        """Initialize pipeline."""
        self.config = load_config(config_path)
        self.phase_manager = PhaseManager(self.config)
        
        # Setup logging
        setup_logging(self.config.get('logging', {}))
        
        # Checkpoint management
        self.checkpoint_file = Path(self.config.get('checkpoint', {}).get('file', 'acid_phased_checkpoint.json'))
        self.checkpoint = self._load_checkpoint()
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown_requested = False
    
    def run(self, source: str = 'local', count: int = 100):
        """
        Run the phased pipeline.
        
        Args:
            source: Data source ('local', 'rag', or 'specific_list')
            count: Number of papers to process
        """
        logger.info(f"="*80)
        logger.info(f"ACID PHASED PIPELINE - Starting")
        logger.info(f"  Source: {source}")
        logger.info(f"  Count: {count}")
        logger.info(f"="*80)
        
        # Get tasks to process
        tasks = self._get_tasks(source, count)
        if not tasks:
            logger.warning("No extraction tasks found - all papers may have already been processed")
            return
        
        logger.info(f"Loaded {len(tasks)} tasks for processing")
        
        # Phase 1: Extraction
        extraction_results = self.phase_manager.start_extraction_phase(tasks)
        
        if not extraction_results['staged_files']:
            logger.error("No files were successfully extracted - check PDF paths and Docling configuration")
            return
        
        # Phase 2: Embedding
        embedding_results = self.phase_manager.start_embedding_phase(
            extraction_results['staged_files']
        )
        
        # Final report
        self._generate_report(extraction_results, embedding_results)
        
        # Save checkpoint
        self._save_checkpoint({
            'last_run': datetime.now().isoformat(),
            'papers_processed': len(embedding_results['success']),
            'extraction_stats': self.phase_manager.phase_stats['EXTRACTION'],
            'embedding_stats': self.phase_manager.phase_stats['EMBEDDING']
        })
    
    def _get_tasks(self, source: str, count: int) -> List[ProcessingTask]:
        """Get processing tasks based on source."""
        tasks = []
        
        if source == 'local':
            # Get local PDFs
            pdf_dir = Path(self.config['processing']['local']['pdf_dir'])
            pattern = self.config['processing']['local']['pattern']
            
            pdf_files = sorted(pdf_dir.glob(pattern))[:count]
            
            for pdf_path in pdf_files:
                # Extract arxiv_id from path
                arxiv_id = pdf_path.stem
                
                # Check for LaTeX
                latex_path = pdf_path.parent / f"{arxiv_id}.tex"
                if not latex_path.exists():
                    latex_path = None
                
                tasks.append(ProcessingTask(
                    arxiv_id=arxiv_id,
                    pdf_path=str(pdf_path),
                    latex_path=str(latex_path) if latex_path else None
                ))
        
        elif source == 'rag':
            # Get RAG paper IDs
            rag_ids = self.config['processing']['rag']['paper_ids'][:count]
            
            for arxiv_id in rag_ids:
                # Construct paths
                pdf_path = Path(self.config['processing']['local']['pdf_dir']) / f"{arxiv_id}.pdf"
                latex_path = Path(self.config['processing']['local']['pdf_dir']) / f"{arxiv_id}.tex"
                
                if pdf_path.exists():
                    tasks.append(ProcessingTask(
                        arxiv_id=arxiv_id,
                        pdf_path=str(pdf_path),
                        latex_path=str(latex_path) if latex_path.exists() else None
                    ))
        
        elif source == 'specific_list':
            # Get specific list paper IDs
            specific_ids = self.config['processing']['specific_list']['arxiv_ids'][:count]
            
            for arxiv_id in specific_ids:
                # Construct paths based on directory structure YYMM/arxiv_id.pdf
                pdf_dir = Path(self.config['processing']['local']['pdf_dir'])
                year_month = arxiv_id.split('.')[0]  # Extract YYMM from arxiv_id
                pdf_path = pdf_dir / year_month / f"{arxiv_id}.pdf"
                latex_path = pdf_dir / year_month / f"{arxiv_id}.tex"
                
                if pdf_path.exists():
                    tasks.append(ProcessingTask(
                        arxiv_id=arxiv_id,
                        pdf_path=str(pdf_path),
                        latex_path=str(latex_path) if latex_path.exists() else None
                    ))
                else:
                    logger.warning(f"PDF not found for {arxiv_id}: {pdf_path}")
        
        return tasks
    
    def _generate_report(self, extraction_results: Dict, embedding_results: Dict):
        """Generate final report."""
        logger.info(f"="*80)
        logger.info(f"FINAL REPORT")
        logger.info(f"="*80)
        
        # Overall stats
        total_attempted = len(extraction_results['success']) + len(extraction_results['failed'])
        total_success = len(embedding_results['success'])
        
        logger.info(f"Total papers attempted: {total_attempted}")
        logger.info(f"Successfully processed: {total_success}")
        logger.info(f"Overall success rate: {total_success/total_attempted*100:.1f}%")
        
        # Phase stats
        ext_stats = self.phase_manager.phase_stats['EXTRACTION']
        emb_stats = self.phase_manager.phase_stats['EMBEDDING']
        
        if ext_stats['end_time'] and ext_stats['start_time']:
            ext_time = (ext_stats['end_time'] - ext_stats['start_time']).total_seconds()
            logger.info(f"\nExtraction Phase:")
            logger.info(f"  Time: {ext_time:.1f}s")
            if ext_stats['papers_processed'] > 0:
                logger.info(f"  Rate: {ext_stats['papers_processed']/ext_time*60:.1f} papers/min")
                logger.info(f"  Avg size: {ext_stats['total_size_mb']/ext_stats['papers_processed']:.2f} MB/paper")
            else:
                logger.info(f"  Rate: 0.0 papers/min")
        
        if emb_stats['end_time'] and emb_stats['start_time']:
            emb_time = (emb_stats['end_time'] - emb_stats['start_time']).total_seconds()
            logger.info(f"\nEmbedding Phase:")
            logger.info(f"  Time: {emb_time:.1f}s")
            if emb_stats['papers_processed'] > 0:
                logger.info(f"  Rate: {emb_stats['papers_processed']/emb_time*60:.1f} papers/min")
                logger.info(f"  Avg chunks: {emb_stats['chunks_created']/emb_stats['papers_processed']:.1f} chunks/paper")
            else:
                logger.info(f"  Rate: 0.0 papers/min")
                logger.info(f"  Avg chunks: 0.0 chunks/paper")
        
        if ext_stats['end_time'] and ext_stats['start_time'] and emb_stats['end_time']:
            total_time = (emb_stats['end_time'] - ext_stats['start_time']).total_seconds()
            logger.info(f"\nTotal Pipeline:")
            logger.info(f"  Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            logger.info(f"  End-to-end rate: {total_success/total_time*60:.1f} papers/min")
        
        logger.info(f"="*80)
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Unable to load checkpoint file (will start fresh): {type(e).__name__}: {e}")
        return {}
    
    def _save_checkpoint(self, data: Dict[str, Any]):
        """Save checkpoint."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Critical: Failed to save checkpoint - progress may be lost on restart: {type(e).__name__}: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, requesting shutdown...")
        self.shutdown_requested = True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='ACID Pipeline with Phase-Separated Architecture')
    parser.add_argument('--config', type=str, default='configs/acid_pipeline.yaml',
                       help='Configuration file path')
    parser.add_argument('--source', type=str, default='local',
                       choices=['local', 'rag', 'specific_list'],
                       help='Data source')
    parser.add_argument('--count', type=int, default=100,
                       help='Number of papers to process')
    parser.add_argument('--arango-password', type=str,
                       help='ArangoDB password (overrides env)')
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.arango_password:
        os.environ['ARANGO_PASSWORD'] = args.arango_password
    
    # Create and run pipeline
    pipeline = ACIDPhasedPipeline(args.config)
    pipeline.run(source=args.source, count=args.count)


if __name__ == '__main__':
    # Import tqdm at module level for progress bars
    from tqdm import tqdm
    
    main()