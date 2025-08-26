"""
ArangoDB ACID-compliant processor for ArXiv papers.

This is a SIMPLIFICATION of our existing pipeline!
- We ALREADY extract equations, tables, images with Docling
- We ALREADY have this data in PostgreSQL  
- We're just MOVING it to ArangoDB for single-database simplicity

Key Pattern: Reserve → Compute → Commit → Release
"""

import os
import time
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from arango import ArangoClient
from arango.exceptions import DocumentInsertError, TransactionAbortError

from core.framework.embedders import JinaV4Embedder
from core.framework.extractors.docling_extractor import DoclingExtractor

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a paper"""
    paper_id: str
    success: bool
    processing_time: float
    num_chunks: int = 0
    num_equations: int = 0
    num_tables: int = 0
    num_images: int = 0
    error: Optional[str] = None


class ArangoACIDProcessor:
    """
    Single-database ACID processor using ArangoDB native transactions.
    
    KEY INSIGHT: This is a SIMPLIFICATION of our existing pipeline!
    - We ALREADY extract equations, tables, images with Docling
    - We ALREADY have this data in PostgreSQL  
    - We're just MOVING it to ArangoDB for single-database simplicity
    
    This refactor REMOVES complexity, doesn't add it!
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processor with configuration
        
        Note: Password must be provided in config, not read from environment.
        """
        # Validate password is provided
        password = config.get('password')
        if not password:
            raise ValueError("ArangoDB password must be provided in config['password']")
        
        # Connect to ArangoDB
        self.arango = ArangoClient(hosts=config.get('arango_host', 'http://192.168.1.69:8529'))
        self.db = self.arango.db(
            config.get('database', 'academy_store'),
            username=config.get('username', 'root'),
            password=password
        )
        
        # These components ALREADY EXIST in our pipeline!
        embedder_config = config.get('embedder_config', {})
        self.jina_embedder = JinaV4Embedder(
            device=embedder_config.get('device', 'cuda'),
            use_fp16=embedder_config.get('use_fp16', True),
            chunk_size_tokens=embedder_config.get('chunk_size_tokens', 1000),
            chunk_overlap_tokens=embedder_config.get('chunk_overlap_tokens', 200)
        )
        
        extractor_config = config.get('extractor_config', {})
        self.docling = DoclingExtractor(
            use_ocr=extractor_config.get('use_ocr', False),
            extract_tables=extractor_config.get('extract_tables', True),
            use_fallback=extractor_config.get('use_fallback', False)
        )
        
        # Chunking parameters
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        
        # Initialize collections and indexes
        self._ensure_collections()
        self._setup_indexes()
    
    def _ensure_collections(self):
        """
        Create collections if they don't exist.
        
        NOTE: We already extract equations, tables, and images!
        This just moves them from PostgreSQL to ArangoDB.
        """
        collections = [
            ('arxiv_papers', {'sync': False}),
            ('arxiv_chunks', {'sync': False}),
            ('arxiv_embeddings', {'sync': False}),
            ('arxiv_equations', {'sync': False}),    # Already extracting!
            ('arxiv_tables', {'sync': False}),       # Already extracting!
            ('arxiv_images', {'sync': False}),       # Already extracting!
            ('arxiv_locks', {'sync': True})          # Locks need immediate consistency
        ]
        
        for name, options in collections:
            if not self.db.has_collection(name):
                logger.info(f"Creating collection: {name}")
                # For single-node development (not sharded)
                self.db.create_collection(
                    name,
                    **options
                )
    
    def _setup_indexes(self):
        """Setup required indexes"""
        # TTL for automatic lock cleanup
        locks_collection = self.db.collection('arxiv_locks')
        
        # Check if TTL index already exists
        existing_indexes = locks_collection.indexes()
        has_ttl = any(idx.get('type') == 'ttl' for idx in existing_indexes)
        
        if not has_ttl:
            locks_collection.add_ttl_index(
                fields=['expiresAt'],
                expiry_time=0  # Delete immediately after expiry
            )
            logger.info("Created TTL index on locks collection")
        
        # Unique constraint for locking (primary key already provides this)
        # No need to add another unique index on _key
    
    def process_paper(self, paper_id: str, pdf_path: str) -> ProcessingResult:
        """
        Process a paper with full ACID compliance.
        Pattern: Reserve → Compute → Commit → Release
        """
        start_time = time.time()
        
        # 1. RESERVE
        if not self._acquire_lock(paper_id):
            logger.info(f"Paper {paper_id} already locked")
            return ProcessingResult(
                paper_id=paper_id,
                success=False,
                processing_time=time.time() - start_time,
                error="Already locked by another worker"
            )
        
        try:
            # 2. COMPUTE (outside transaction!)
            start_compute = time.time()
            
            # THIS IS THE EXACT SAME EXTRACTION WE DO NOW!
            # Just moving the storage from PostgreSQL to ArangoDB
            
            # Extract from PDF (EXISTING CODE!)
            logger.info(f"Extracting PDF: {pdf_path}")
            doc = self.docling.extract(pdf_path)
            
            metadata = {
                'title': doc.get('title', ''),
                'abstract': doc.get('abstract', ''),
                'authors': doc.get('authors', []),
                'categories': doc.get('categories', [])
            }
            
            # Extract structures (ALREADY IN OUR PIPELINE!)
            equations = doc.get('equations', [])  # LaTeX formulas - already extracting!
            tables = doc.get('tables', [])        # Structured tables - already extracting!
            images = doc.get('images', [])        # Figure metadata - already extracting!
            
            # Use late chunking for better context (EXISTING CODE!)
            text = doc.get('full_text', '') or doc.get('text', '')
            if not text:
                logger.warning(f"No text content extracted from PDF {pdf_path} - file may be corrupt or image-only")
            else:
                logger.info(f"Extracted {len(text)} characters of text")
            
            # Generate embeddings with late chunking
            logger.info(f"Generating embeddings with late chunking for paper {paper_id}")
            # embed_batch_with_late_chunking expects a list of texts
            try:
                chunk_embeddings_list = self.jina_embedder.embed_batch_with_late_chunking([text])
                chunk_embeddings = chunk_embeddings_list[0] if chunk_embeddings_list else []
            except Exception as e:
                logger.error(f"Embedding generation failed for paper {paper_id}: {type(e).__name__}: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise RuntimeError(f"Embedding generation failed for paper {paper_id}") from e
            
            logger.info(f"Generated {len(chunk_embeddings)} chunks with embeddings")
            
            compute_time = time.time() - start_compute
            logger.info(f"Computed {paper_id} in {compute_time:.2f}s")
            
            # 3. COMMIT (fast transaction)
            start_txn = time.time()
            
            # Begin stream transaction
            txn_db = self.db.begin_transaction(
                write=['arxiv_papers', 'arxiv_chunks', 'arxiv_embeddings', 'arxiv_equations', 'arxiv_tables', 'arxiv_images'],
                read=[],
                exclusive=[],  # No exclusive locks needed
                sync=True,
                allow_implicit=False,
                lock_timeout=5
            )
            
            try:
                # All inserts are idempotent via deterministic keys
                
                # Insert paper (SAME METADATA AS BEFORE!)
                sanitized_id = self._sanitize_id(paper_id)
                txn_db.collection('arxiv_papers').insert({
                    '_key': sanitized_id,
                    'arxiv_id': paper_id,
                    'pdf_path': pdf_path,
                    'pdf_hash': self._hash_file(pdf_path),
                    'status': 'PROCESSED',
                    'processing_date': datetime.now().isoformat(),
                    'num_chunks': len(chunk_embeddings),
                    'num_equations': len(equations),  # We track these counts already!
                    'num_tables': len(tables),
                    'num_images': len(images),
                    **metadata
                }, overwrite=True)
                
                # Insert chunks with embeddings (EXISTING LOGIC!)
                for i, chunk_with_emb in enumerate(chunk_embeddings):
                    # Insert chunk text
                    txn_db.collection('arxiv_chunks').insert({
                        '_key': f"{sanitized_id}_chunk_{i}",
                        'paper_id': sanitized_id,
                        'text': chunk_with_emb.text,
                        'chunk_index': i,
                        'total_chunks': len(chunk_embeddings),
                        'start_char': chunk_with_emb.start_char,
                        'end_char': chunk_with_emb.end_char,
                        'start_token': chunk_with_emb.start_token,
                        'end_token': chunk_with_emb.end_token,
                        'context_window_used': chunk_with_emb.context_window_used
                    }, overwrite=True)
                    
                    # Insert embedding
                    txn_db.collection('arxiv_embeddings').insert({
                        '_key': f"{sanitized_id}_chunk_{i}_emb",
                        'paper_id': sanitized_id,
                        'chunk_id': f"{sanitized_id}_chunk_{i}",
                        'vector': chunk_with_emb.embedding.tolist(),
                        'model': 'jina-v4',
                        'embedding_date': datetime.now().isoformat()
                    }, overwrite=True)
                
                # Insert equations (JUST MOVING FROM POSTGRES TO ARANGO!)
                for i, eq in enumerate(equations):
                    txn_db.collection('arxiv_equations').insert({
                        '_key': f"{sanitized_id}_eq_{i}",
                        'paper_id': sanitized_id,
                        'equation_index': i,
                        'latex': eq.get('latex', ''),
                        'label': eq.get('label', ''),
                        'type': eq.get('type', 'display'),
                        'page_number': eq.get('page', 0)
                    }, overwrite=True)
                
                # Insert tables (JUST MOVING FROM POSTGRES TO ARANGO!)
                for i, tbl in enumerate(tables):
                    txn_db.collection('arxiv_tables').insert({
                        '_key': f"{sanitized_id}_table_{i}",
                        'paper_id': sanitized_id,
                        'table_index': i,
                        'caption': tbl.get('caption', ''),
                        'label': tbl.get('label', ''),
                        'headers': tbl.get('headers', []),
                        'data_rows': tbl.get('rows', []),
                        'latex': tbl.get('latex_source', ''),
                        'page_number': tbl.get('page', 0)
                    }, overwrite=True)
                
                # Insert images (JUST MOVING FROM POSTGRES TO ARANGO!)
                for i, img in enumerate(images):
                    txn_db.collection('arxiv_images').insert({
                        '_key': f"{sanitized_id}_img_{i}",
                        'paper_id': sanitized_id,
                        'image_index': i,
                        'caption': img.get('caption', ''),
                        'label': img.get('label', ''),
                        'page_number': img.get('page', 0),
                        'bbox': img.get('bbox', {}),
                        'type': img.get('type', 'figure')
                    }, overwrite=True)
                
                # Commit transaction
                txn_db.commit_transaction()
                
                txn_time = time.time() - start_txn
                logger.info(f"Committed {paper_id} in {txn_time:.3f}s")
                
                # 4. RELEASE
                self._release_lock(paper_id)
                
                return ProcessingResult(
                    paper_id=paper_id,
                    success=True,
                    processing_time=time.time() - start_time,
                    num_chunks=len(chunk_embeddings),
                    num_equations=len(equations),
                    num_tables=len(tables),
                    num_images=len(images)
                )
                
            except Exception as e:
                txn_db.abort_transaction()
                raise
            
        except Exception as e:
            logger.error(f"Paper processing failed for {paper_id}: {type(e).__name__}: {e}")
            self._release_lock(paper_id)
            return ProcessingResult(
                paper_id=paper_id,
                success=False,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def _acquire_lock(self, paper_id: str, timeout_minutes: int = 10) -> bool:
        """Acquire lock using unique constraint"""
        try:
            sanitized_id = self._sanitize_id(paper_id)
            self.db.collection('arxiv_locks').insert({
                '_key': sanitized_id,
                'paper_id': paper_id,
                'worker_id': os.getpid(),
                'acquired_at': datetime.now().isoformat(),
                'expiresAt': int((datetime.now() + timedelta(minutes=timeout_minutes)).timestamp())
            })
            logger.debug(f"Acquired lock for {paper_id}")
            return True
        except DocumentInsertError:
            logger.debug(f"Lock already held for {paper_id}")
            return False  # Already locked
    
    def _release_lock(self, paper_id: str):
        """Release lock (idempotent)"""
        try:
            sanitized_id = self._sanitize_id(paper_id)
            self.db.collection('arxiv_locks').delete(sanitized_id)
            logger.debug(f"Released lock for {paper_id}")
        except Exception as e:
            # Lock may have expired or been deleted - this is OK
            logger.debug(f"Lock release for {paper_id} failed (may have expired): {e}")
    
    def _sanitize_id(self, paper_id: str) -> str:
        """Sanitize arxiv_id for use as ArangoDB _key"""
        # Replace dots and slashes with underscores
        return paper_id.replace('.', '_').replace('/', '_')
    
    def _hash_file(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text into overlapping segments.
        This is simplified - in production we'd use the framework chunker.
        """
        chunks = []
        text_length = len(text)
        
        if text_length <= self.chunk_size:
            # Single chunk for short text
            chunks.append({
                'text': text,
                'start_char': 0,
                'end_char': text_length
            })
        else:
            # Overlapping chunks
            start = 0
            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunks.append({
                    'text': text[start:end],
                    'start_char': start,
                    'end_char': end
                })
                start += self.chunk_size - self.chunk_overlap
        
        return chunks