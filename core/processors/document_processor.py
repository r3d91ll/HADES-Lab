#!/usr/bin/env python3
"""
Generic Document Processor
===========================

Source-agnostic document processing implementing Information Reconstructionism.
Handles expensive operations (PDF extraction, embedding, chunking) while remaining
independent of source-specific metadata and storage schemas.

This processor serves as an obligatory passage point (Actor-Network Theory) 
between raw documents and structured knowledge, transforming information
across the boundary from unstructured to structured formats.
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import numpy as np

from core.framework.extractors.docling_extractor import DoclingExtractor
from core.framework.extractors.latex_extractor import LaTeXExtractor
from core.framework.embedders import JinaV4Embedder, ChunkWithEmbedding

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """
    Configuration for generic document processing.
    
    Separates processing configuration from source-specific concerns,
    allowing each source to optimize its own metadata while sharing
    expensive processing operations.
    """
    # Extraction settings
    use_gpu: bool = True
    extract_tables: bool = True
    extract_equations: bool = True
    extract_images: bool = True
    use_ocr: bool = False
    
    # Embedding settings
    embedding_model: str = 'jina-v4'
    embedding_dim: int = 2048
    use_fp16: bool = True
    device: Optional[str] = None  # Auto-detect if None
    
    # Chunking settings
    chunk_size_tokens: int = 1000
    chunk_overlap_tokens: int = 200
    chunking_strategy: str = 'late'  # 'late', 'semantic', 'fixed'
    max_chunk_size: int = 8192
    
    # Processing settings
    batch_size: int = 1
    num_workers: int = 1
    timeout_seconds: int = 300
    
    # Performance settings
    cache_embeddings: bool = True
    use_ramfs_staging: bool = True
    staging_dir: str = '/dev/shm/document_staging'


@dataclass
class ExtractionResult:
    """
    Result from document extraction phase.
    
    Contains the raw extracted content before chunking and embedding,
    preserving all structural information from the original document.
    """
    full_text: str
    tables: List[Dict[str, Any]] = field(default_factory=list)
    equations: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    latex_source: Optional[str] = None
    has_latex: bool = False
    extraction_time: float = 0.0
    extractor_version: str = ""


@dataclass
class ProcessingResult:
    """
    Complete result from document processing.
    
    This is a source-agnostic result that contains all expensive computations.
    Source-specific managers can adapt this to their storage schemas while
    avoiding duplication of processing logic.
    """
    # Extraction results
    extraction: ExtractionResult
    
    # Chunked text with embeddings
    chunks: List[ChunkWithEmbedding]
    
    # Processing metadata
    processing_metadata: Dict[str, Any]
    
    # Performance metrics
    total_processing_time: float
    extraction_time: float
    chunking_time: float
    embedding_time: float
    
    # Status information
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'extraction': {
                'full_text': self.extraction.full_text,
                'tables': self.extraction.tables,
                'equations': self.extraction.equations,
                'images': self.extraction.images,
                'figures': self.extraction.figures,
                'metadata': self.extraction.metadata,
                'has_latex': self.extraction.has_latex,
                'extraction_time': self.extraction.extraction_time
            },
            'chunks': [
                {
                    'text': chunk.text,
                    'embedding': np.asarray(chunk.embedding).tolist(),
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'context_window_used': chunk.context_window_used
                }
                for chunk in self.chunks
            ] if self.chunks else [],
            'processing_metadata': self.processing_metadata,
            'performance': {
                'total_time': self.total_processing_time,
                'extraction_time': self.extraction_time,
                'chunking_time': self.chunking_time,
                'embedding_time': self.embedding_time
            },
            'success': self.success,
            'errors': self.errors,
            'warnings': self.warnings
        }


class DocumentProcessor:
    """
    Generic document processor for any PDF/LaTeX source.
    
    This processor implements the CONVEYANCE dimension of Information
    Reconstructionism by transforming documents from their source format
    into structured, embedded representations. It serves as a boundary
    object (ANT) that maintains coherence across different document sources
    while adapting to local needs.
    
    The processor is intentionally source-agnostic, handling only the
    expensive computational operations while leaving source-specific
    metadata and storage to specialized managers.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize document processor with configuration.
        
        Args:
            config: Processing configuration (uses defaults if None)
        """
        self.config = config or ProcessingConfig()
        
        # Initialize extractors
        self.docling_extractor = DoclingExtractor(
            use_ocr=self.config.use_ocr,
            extract_tables=self.config.extract_tables
        )
        self.latex_extractor = LaTeXExtractor() if self.config.extract_equations else None
        
        # Initialize embedder with late chunking support
        self.embedder = JinaV4Embedder(
            device=self.config.device,
            use_fp16=self.config.use_fp16,
            chunk_size_tokens=self.config.chunk_size_tokens,
            chunk_overlap_tokens=self.config.chunk_overlap_tokens
        )
        
        # Setup staging directory if using RamFS
        if self.config.use_ramfs_staging:
            self.staging_dir = Path(self.config.staging_dir)
            self.staging_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DocumentProcessor with config: {self.config}")
    
    def process_document(
        self,
        pdf_path: Union[str, Path],
        latex_path: Optional[Union[str, Path]] = None,
        document_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a document through extraction, chunking, and embedding.
        
        This method implements the complete processing pipeline while
        remaining agnostic to the document's source. It returns a
        structured result that source-specific managers can adapt
        to their storage schemas.
        
        Args:
            pdf_path: Path to PDF file
            latex_path: Optional path to LaTeX source
            document_id: Optional identifier for logging
            
        Returns:
            ProcessingResult containing all extracted and embedded content
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        # Convert paths to Path objects
        pdf_path = Path(pdf_path)
        if latex_path:
            latex_path = Path(latex_path)
        
        doc_id = document_id or pdf_path.stem
        logger.info(f"Processing document: {doc_id}")
        
        try:
            # Phase 1: Extract content from PDF
            extraction_start = time.time()
            extraction_result = self._extract_content(pdf_path, latex_path)
            extraction_time = time.time() - extraction_start
            
            # Phase 2: Create chunks (respecting context windows)
            chunking_start = time.time()
            if self.config.chunking_strategy == 'late':
                # Late chunking: embed full document then chunk
                chunks = self._create_late_chunks(extraction_result.full_text)
            elif self.config.chunking_strategy in ['traditional', 'token', 'semantic', 'fixed']:
                # Traditional chunking: chunk then embed
                chunks = self._create_traditional_chunks(extraction_result.full_text)
            else:
                raise ValueError(
                    f"Unknown chunking strategy: '{self.config.chunking_strategy}'. "
                    f"Allowed values are: 'late', 'traditional', 'token', 'semantic', 'fixed'"
                )
            chunking_time = time.time() - chunking_start
            
            # Check if chunking produced no results
            if not chunks:
                logger.warning(f"No chunks produced for document {doc_id}. Document may be empty.")
                # Return early with empty result
                return ProcessingResult(
                    extraction=extraction_result,
                    chunks=[],
                    processing_metadata={'error': 'No chunks produced'},
                    total_processing_time=time.time() - start_time,
                    extraction_time=extraction_time,
                    chunking_time=chunking_time,
                    embedding_time=0,
                    success=False,
                    errors=['Document produced no chunks'],
                    warnings=['Empty or unprocessable document']
                )
            
            # Phase 3: Generate embeddings (if not already done by late chunking)
            embedding_start = time.time()
            if chunks and not isinstance(chunks[0], ChunkWithEmbedding):
                chunks = self._embed_chunks(chunks)
            embedding_time = time.time() - embedding_start
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Build processing metadata
            processing_metadata = {
                'processor_version': '2.0.0',
                'extraction_method': 'docling',
                'embedding_model': self.config.embedding_model,
                'chunking_strategy': self.config.chunking_strategy,
                'chunk_count': len(chunks),
                'has_latex': extraction_result.has_latex,
                'has_tables': len(extraction_result.tables) > 0,
                'has_equations': len(extraction_result.equations) > 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return ProcessingResult(
                extraction=extraction_result,
                chunks=chunks,
                processing_metadata=processing_metadata,
                total_processing_time=total_time,
                extraction_time=extraction_time,
                chunking_time=chunking_time,
                embedding_time=embedding_time,
                success=True,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {e}")
            errors.append(str(e))
            
            # Return partial result with error
            return ProcessingResult(
                extraction=ExtractionResult(full_text=""),
                chunks=[],
                processing_metadata={},
                total_processing_time=time.time() - start_time,
                extraction_time=0,
                chunking_time=0,
                embedding_time=0,
                success=False,
                errors=errors,
                warnings=warnings
            )
    
    def _extract_content(
        self,
        pdf_path: Path,
        latex_path: Optional[Path] = None
    ) -> ExtractionResult:
        """
        Extract content from PDF and optional LaTeX source.
        
        Combines Docling extraction with LaTeX processing when available,
        preserving maximum semantic information from the source documents.
        """
        start_time = time.time()
        
        # Extract from PDF using Docling
        docling_result = self.docling_extractor.extract(str(pdf_path))
        
        # Process LaTeX if available
        latex_source = None
        has_latex = False
        if latex_path and latex_path.exists() and self.latex_extractor:
            try:
                with open(latex_path, 'r', encoding='utf-8') as f:
                    latex_source = f.read()
                has_latex = True
                
                # Extract equations from LaTeX
                latex_equations = self.latex_extractor.extract_equations(latex_source)
                # Merge with Docling equations (LaTeX takes precedence)
                if latex_equations:
                    docling_result['equations'] = latex_equations
            except Exception as e:
                logger.warning(f"Failed to process LaTeX source: {e}")
        
        # Build extraction result
        return ExtractionResult(
            full_text=docling_result.get('text', ''),
            tables=docling_result.get('tables', []),
            equations=docling_result.get('equations', []),
            images=docling_result.get('images', []),
            figures=docling_result.get('figures', []),
            metadata=docling_result.get('metadata', {}),
            latex_source=latex_source,
            has_latex=has_latex,
            extraction_time=time.time() - start_time,
            extractor_version=docling_result.get('version', 'unknown')
        )
    
    def _create_late_chunks(self, text: str) -> List[ChunkWithEmbedding]:
        """
        Create chunks using late chunking strategy.
        
        Processes the entire document first to preserve context,
        then creates chunks that maintain awareness of their surroundings.
        This prevents zero-propagation in the WHAT dimension by ensuring
        each chunk contains contextual information.
        """
        # Use Jina's late chunking capability
        chunks_with_embeddings = self.embedder.embed_with_late_chunking(text)
        return chunks_with_embeddings
    
    def _create_traditional_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Create traditional chunks (for backward compatibility).
        
        Chunks text first, then embeds each chunk independently.
        This is less optimal than late chunking but maintains compatibility
        with existing systems.
        """
        # Validate chunking parameters
        chunk_size = self.config.chunk_size_tokens
        overlap = self.config.chunk_overlap_tokens
        
        if chunk_size <= 0:
            raise ValueError(f"chunk_size_tokens must be positive, got {chunk_size}")
        
        if overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap_tokens ({overlap}) must be less than "
                f"chunk_size_tokens ({chunk_size})"
            )
        
        if overlap < 0:
            overlap = 0  # Clamp to 0 if negative
        
        # Calculate step size
        step = chunk_size - overlap
        if step <= 0:
            raise ValueError(
                f"Invalid chunking parameters: step size would be {step}. "
                f"Ensure chunk_size_tokens > chunk_overlap_tokens."
            )
        
        # Simple token-based chunking
        chunks = []
        tokens = text.split()  # Simplified - real implementation would use tokenizer
        
        if not tokens:
            return []  # Return empty list for empty text
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = ' '.join(chunk_tokens)
            
            chunks.append({
                'text': chunk_text,
                'start_token': i,
                'end_token': min(i + chunk_size, len(tokens)),
                'chunk_index': len(chunks)
            })
        
        return chunks
    
    def _embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[ChunkWithEmbedding]:
        """
        Generate embeddings for traditional chunks.
        
        Used when not using late chunking strategy.
        """
        embedded_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Generate embedding for chunk
            # JinaV4Embedder expects a list and returns array of embeddings
            embeddings = self.embedder.embed_texts([chunk['text']], batch_size=1)
            embedding = embeddings[0] if len(embeddings) > 0 else None
            
            # Normalize embedding to consistent numpy array format
            if embedding is not None:
                # Convert to numpy array and ensure it's 1D float32
                embedding_array = np.asarray(embedding, dtype=np.float32)
                if embedding_array.ndim != 1:
                    embedding_array = embedding_array.reshape(-1)
            else:
                # Create zero embedding if something went wrong
                embedding_array = np.zeros(self.config.embedding_dim, dtype=np.float32)
            
            # Create ChunkWithEmbedding object
            embedded_chunk = ChunkWithEmbedding(
                text=chunk['text'],
                embedding=embedding_array,
                start_char=chunk.get('start_char', 0),
                end_char=chunk.get('end_char', len(chunk['text'])),
                start_token=chunk.get('start_token', 0),
                end_token=chunk.get('end_token', 0),
                chunk_index=i,
                total_chunks=len(chunks),
                context_window_used=len(chunk['text'].split())
            )
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def process_batch(
        self,
        document_paths: List[Tuple[Path, Optional[Path]]],
        document_ids: Optional[List[str]] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple documents in batch.
        
        Args:
            document_paths: List of (pdf_path, latex_path) tuples
            document_ids: Optional list of document identifiers
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        for i, (pdf_path, latex_path) in enumerate(document_paths):
            # Safely get document ID if provided
            doc_id = document_ids[i] if document_ids and i < len(document_ids) else None
            result = self.process_document(pdf_path, latex_path, doc_id)
            results.append(result)
        
        return results