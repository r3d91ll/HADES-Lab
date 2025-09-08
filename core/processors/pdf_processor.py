"""
Generic PDF processing pipeline for document extraction and embedding.

Implements the WHAT dimension of the Conveyance Framework:
- Extracts content from PDFs (semantic content)
- Generates embeddings (semantic representation)
- Stores to backend (persistent knowledge)

This processor is source-agnostic and can be used by any tool
that needs to process PDF documents.
"""

import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.framework.embedders import JinaV4Embedder
from core.framework.extractors.docling_extractor import DoclingExtractor
from core.framework.storage import ArangoStorage

logger = logging.getLogger(__name__)

# Global variables for worker processes
WORKER_DOCLING: Optional[DoclingExtractor] = None
WORKER_EMBEDDER: Optional[JinaV4Embedder] = None
WORKER_STORAGE: Optional[ArangoStorage] = None


@dataclass
class PDFTask:
    """Task for processing a PDF document."""
    pdf_path: str
    document_id: str
    metadata: Dict[str, Any] = None
    additional_files: Dict[str, str] = None  # e.g., {'latex': '/path/to/latex.tex'}

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.additional_files is None:
            self.additional_files = {}


@dataclass
class ProcessingResult:
    """Result of processing a PDF document."""
    success: bool
    document_id: str
    num_chunks: int = 0
    num_equations: int = 0
    num_tables: int = 0
    num_images: int = 0
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class ProcessorConfig:
    """Configuration for PDF processor."""
    # Extraction settings
    extraction_workers: int = 32
    extraction_batch_size: int = 24
    use_ocr: bool = False
    extract_tables: bool = True
    extract_equations: bool = True
    extract_images: bool = True
    
    # Embedding settings
    embedding_workers: int = 8
    embedding_batch_size: int = 24
    use_fp16: bool = True
    chunk_size_tokens: int = 1000
    chunk_overlap_tokens: int = 200
    
    # Storage settings
    storage_backend: str = 'arango'
    collection_prefix: str = 'documents'
    
    # GPU settings
    gpu_devices: List[int] = None
    
    # Staging settings
    staging_dir: str = '/dev/shm/pdf_staging'
    
    def __post_init__(self):
        if self.gpu_devices is None:
            self.gpu_devices = [0, 1] if torch.cuda.device_count() > 1 else [0]


class PDFProcessor:
    """
    Generic PDF processing pipeline.
    
    Processes PDFs through extraction, embedding, and storage phases.
    Maintains phase separation for optimal GPU memory management.
    
    From Conveyance Framework:
    - W (WHAT): Extracted content and embeddings
    - H (WHO): Processing capability with GPU acceleration
    - T (Time): Batch processing for efficiency
    - Ctx: Preserved through late chunking
    """
    
    def __init__(self, config: ProcessorConfig, storage_config: Dict[str, Any]):
        """
        Initialize PDF processor.
        
        Args:
            config: Processor configuration
            storage_config: Storage backend configuration (e.g., ArangoDB credentials)
        """
        self.config = config
        self.storage_config = storage_config
        
        # Create staging directory
        self.staging_dir = Path(config.staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDFProcessor initialized with {config.extraction_workers} extraction workers, "
                   f"{config.embedding_workers} embedding workers")
    
    def process_batch(self, tasks: List[PDFTask]) -> List[ProcessingResult]:
        """
        Process a batch of PDFs through the complete pipeline.
        
        Implements phase separation:
        1. Extract all documents (GPU-accelerated Docling)
        2. Clear GPU memory
        3. Embed all documents (Jina v4 with late chunking)
        
        Args:
            tasks: List of PDF processing tasks
            
        Returns:
            List of processing results
        """
        logger.info(f"Starting batch processing of {len(tasks)} PDFs")
        
        # Phase 1: Extraction
        extraction_results = self._extraction_phase(tasks)
        
        # Clear GPU memory between phases
        self._clear_gpu_memory()
        
        # Phase 2: Embedding and Storage
        final_results = self._embedding_phase(extraction_results)
        
        # Cleanup staging
        self._cleanup_staging(tasks)
        
        return final_results
    
    def process_pdf(self, pdf_path: str, document_id: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process a single PDF document.
        
        Args:
            pdf_path: Path to PDF file
            document_id: Unique identifier for document
            metadata: Optional metadata to store with document
            
        Returns:
            Processing result
        """
        task = PDFTask(pdf_path=pdf_path, document_id=document_id, metadata=metadata)
        results = self.process_batch([task])
        return results[0] if results else ProcessingResult(success=False, document_id=document_id,
                                                          error="Processing failed")
    
    def _extraction_phase(self, tasks: List[PDFTask]) -> List[Dict[str, Any]]:
        """Run extraction phase with parallel workers."""
        logger.info(f"Phase 1: Extracting {len(tasks)} documents with {self.config.extraction_workers} workers")
        
        # Initialize worker pool for extraction
        with mp.Pool(
            processes=self.config.extraction_workers,
            initializer=self._init_extraction_worker,
            initargs=(self.config.gpu_devices, self._get_docling_config())
        ) as pool:
            # Process in batches
            batch_size = self.config.extraction_batch_size
            all_results = []
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                batch_results = pool.apply_async(self._extract_document_batch, (batch,))
                all_results.extend(batch_results.get())
            
        logger.info(f"Extraction phase complete: {len(all_results)} documents processed")
        return all_results
    
    def _embedding_phase(self, extraction_results: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Run embedding phase with parallel workers."""
        # Filter successful extractions
        staged_paths = [r['staged_path'] for r in extraction_results if r.get('success')]
        
        if not staged_paths:
            logger.warning("No successful extractions to embed")
            return [ProcessingResult(success=False, document_id=r.get('document_id', 'unknown'),
                                   error=r.get('error', 'Extraction failed'))
                   for r in extraction_results]
        
        logger.info(f"Phase 2: Embedding {len(staged_paths)} documents with {self.config.embedding_workers} workers")
        
        # Initialize worker pool for embedding
        with mp.Pool(
            processes=self.config.embedding_workers,
            initializer=self._init_embedding_worker,
            initargs=(self.config.gpu_devices, self.storage_config, self._get_embedder_config())
        ) as pool:
            # Process embeddings
            embed_results = pool.map(self._embed_and_store, staged_paths)
        
        # Convert to ProcessingResult objects
        final_results = []
        for embed_result in embed_results:
            if embed_result['success']:
                final_results.append(ProcessingResult(
                    success=True,
                    document_id=embed_result['document_id'],
                    num_chunks=embed_result.get('num_chunks', 0),
                    num_equations=embed_result.get('num_equations', 0),
                    num_tables=embed_result.get('num_tables', 0),
                    num_images=embed_result.get('num_images', 0),
                    processing_time=embed_result.get('processing_time', 0)
                ))
            else:
                final_results.append(ProcessingResult(
                    success=False,
                    document_id=embed_result['document_id'],
                    error=embed_result.get('error', 'Unknown error')
                ))
        
        logger.info(f"Embedding phase complete: {len(final_results)} documents processed")
        return final_results
    
    @staticmethod
    def _init_extraction_worker(gpu_devices: List[int], docling_config: Dict[str, Any]):
        """Initialize extraction worker with Docling instance."""
        global WORKER_DOCLING
        
        # Set up GPU for worker
        worker_info = mp.current_process()
        worker_id = worker_info._identity[0] if hasattr(worker_info, '_identity') else 0
        
        if gpu_devices:
            gpu_id = gpu_devices[worker_id % len(gpu_devices)]
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.info(f"Extraction worker {worker_id} assigned to GPU {gpu_id}")
        
        # Initialize Docling
        WORKER_DOCLING = DoclingExtractor(
            use_ocr=docling_config.get('use_ocr', False),
            extract_tables=docling_config.get('extract_tables', True),
            use_fallback=docling_config.get('use_fallback', True)
        )
        
        logger.info(f"Extraction worker {worker_id} initialized")
    
    @staticmethod
    def _init_embedding_worker(gpu_devices: List[int], storage_config: Dict[str, Any], 
                              embedder_config: Dict[str, Any]):
        """Initialize embedding worker with Jina embedder and storage."""
        global WORKER_EMBEDDER, WORKER_STORAGE
        
        # Set up GPU for worker
        worker_info = mp.current_process()
        worker_id = worker_info._identity[0] if hasattr(worker_info, '_identity') else 0
        
        if gpu_devices:
            gpu_id = gpu_devices[worker_id % len(gpu_devices)]
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.info(f"Embedding worker {worker_id} assigned to GPU {gpu_id}")
        
        # Initialize embedder
        WORKER_EMBEDDER = JinaV4Embedder(
            device='cuda' if gpu_devices else 'cpu',
            use_fp16=embedder_config.get('use_fp16', True),
            chunk_size_tokens=embedder_config.get('chunk_size_tokens', 1000),
            chunk_overlap_tokens=embedder_config.get('chunk_overlap_tokens', 200)
        )
        
        # Initialize storage
        WORKER_STORAGE = ArangoStorage(storage_config)
        
        logger.info(f"Embedding worker {worker_id} initialized")
    
    @staticmethod
    def _extract_document_batch(task_batch: List[PDFTask]) -> List[Dict[str, Any]]:
        """Extract a batch of documents."""
        global WORKER_DOCLING
        
        if WORKER_DOCLING is None:
            raise RuntimeError("WORKER_DOCLING not initialized!")
        
        extractor = WORKER_DOCLING
        batch_results = []
        
        for task in task_batch:
            start_time = datetime.now()
            try:
                # Extract PDF
                extracted = extractor.extract(task.pdf_path)
                
                # Merge additional files if provided
                for file_type, file_path in task.additional_files.items():
                    if Path(file_path).exists():
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            extracted[f'{file_type}_source'] = content
                            extracted[f'has_{file_type}'] = True
                        except Exception as e:
                            logger.warning(f"Failed to read {file_type} for {task.document_id}: {e}")
                
                # Create unified document
                unified_doc = {
                    'document_id': task.document_id,
                    'pdf_path': task.pdf_path,
                    'metadata': task.metadata,
                    'extracted': extracted,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'extraction_method': 'docling_gpu_batch'
                }
                
                # Save to staging
                staging_dir = Path('/dev/shm/pdf_staging')
                staging_dir.mkdir(parents=True, exist_ok=True)
                staged_path = staging_dir / f"{task.document_id.replace('/', '_')}.json"
                
                with open(staged_path, 'w', encoding='utf-8') as f:
                    json.dump(unified_doc, f, ensure_ascii=False, indent=2)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                batch_results.append({
                    'success': True,
                    'document_id': task.document_id,
                    'staged_path': str(staged_path),
                    'processing_time': processing_time
                })
                
            except Exception as e:
                batch_results.append({
                    'success': False,
                    'document_id': task.document_id,
                    'error': str(e)
                })
                logger.error(f"Failed to extract {task.document_id}: {e}")
        
        return batch_results
    
    @staticmethod
    def _embed_and_store(staged_path: str) -> Dict[str, Any]:
        """Embed document and store to backend."""
        global WORKER_EMBEDDER, WORKER_STORAGE
        
        if WORKER_EMBEDDER is None or WORKER_STORAGE is None:
            raise RuntimeError("Worker not properly initialized!")
        
        try:
            start_time = datetime.now()
            
            # Load staged document
            with open(staged_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            
            document_id = doc['document_id']
            metadata = doc.get('metadata', {})
            
            # Extract text
            extracted = doc.get('extracted', {})
            full_text = (extracted.get('full_text', '') or 
                        extracted.get('markdown', '') or 
                        doc.get('full_text', '') or 
                        doc.get('markdown', ''))
            
            # Generate embeddings with late chunking
            chunks = WORKER_EMBEDDER.embed_with_late_chunking(full_text)
            
            # Store to backend
            result = WORKER_STORAGE.store_document(
                document_id=document_id,
                chunks=chunks,
                metadata=metadata,
                extracted_content=extracted
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'document_id': document_id,
                'num_chunks': result.get('num_chunks', len(chunks)),
                'num_equations': result.get('num_equations', 0),
                'num_tables': result.get('num_tables', 0),
                'num_images': result.get('num_images', 0),
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to embed/store document: {e}")
            return {
                'success': False,
                'document_id': staged_path.split('/')[-1].replace('.json', ''),
                'error': str(e)
            }
    
    def _clear_gpu_memory(self):
        """Clear GPU memory between phases."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cleared between phases")
        except Exception as e:
            logger.warning(f"GPU memory cleanup failed (non-critical): {e}")
    
    def _cleanup_staging(self, tasks: List[PDFTask]):
        """Clean up staging files."""
        for task in tasks:
            staged_path = self.staging_dir / f"{task.document_id.replace('/', '_')}.json"
            if staged_path.exists():
                try:
                    staged_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup staging file {staged_path}: {e}")
    
    def _get_docling_config(self) -> Dict[str, Any]:
        """Get Docling configuration."""
        return {
            'use_ocr': self.config.use_ocr,
            'extract_tables': self.config.extract_tables,
            'extract_equations': self.config.extract_equations,
            'extract_images': self.config.extract_images,
            'use_fallback': True
        }
    
    def _get_embedder_config(self) -> Dict[str, Any]:
        """Get embedder configuration."""
        return {
            'use_fp16': self.config.use_fp16,
            'chunk_size_tokens': self.config.chunk_size_tokens,
            'chunk_overlap_tokens': self.config.chunk_overlap_tokens
        }