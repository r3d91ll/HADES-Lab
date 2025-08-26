#!/usr/bin/env python3
"""
ArXiv Manager
=============

ArXiv-specific document management that uses the generic DocumentProcessor
for expensive operations while handling ArXiv-specific metadata, validation,
and storage schemas.

This manager serves as a boundary object (Actor-Network Theory) between
ArXiv's external world of versioned papers, categories, and LaTeX sources,
and our internal processing framework. It translates ArXiv-specific concerns
into generic processing tasks while preserving source-specific metadata.
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.processors.document_processor import (
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult
)
from tools.arxiv.pipelines.arango_db_manager import ArangoDBManager

logger = logging.getLogger(__name__)


@dataclass
class ArXivPaperInfo:
    """
    ArXiv-specific paper information.
    
    Contains all ArXiv-specific metadata that distinguishes these papers
    from other sources. This represents the unique characteristics that
    make ArXiv papers special in our multi-source architecture.
    """
    arxiv_id: str
    version: str
    pdf_path: Path
    latex_path: Optional[Path]
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    submission_date: str
    update_date: Optional[str]
    comments: Optional[str]
    journal_ref: Optional[str]
    doi: Optional[str]
    
    @property
    def sanitized_id(self) -> str:
        """Get sanitized ID for database storage."""
        return self.arxiv_id.replace('.', '_').replace('/', '_')
    
    @property
    def has_latex(self) -> bool:
        """Check if LaTeX source is available."""
        return self.latex_path is not None and self.latex_path.exists()


class ArXivValidator:
    """
    Validates and parses ArXiv-specific identifiers and paths.
    
    Handles the complex ArXiv ID formats and directory structures,
    translating them into standardized paths for processing.
    """
    
    # ArXiv ID patterns
    ARXIV_ID_PATTERN = re.compile(r'^(\d{4})\.(\d{4,5})(v\d+)?$')
    OLD_ARXIV_ID_PATTERN = re.compile(r'^([a-z\-\.]+)\/(\d{7})(v\d+)?$')
    
    # Base paths for ArXiv data
    PDF_BASE_PATH = Path('/bulk-store/arxiv-data/pdf')
    LATEX_BASE_PATH = Path('/bulk-store/arxiv-data/src')
    METADATA_PATH = Path('/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json')
    
    @classmethod
    def validate_arxiv_id(cls, arxiv_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate ArXiv ID format.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Remove any version suffix for validation
        base_id = re.sub(r'v\d+$', '', arxiv_id)
        
        # Check new format (YYMM.NNNNN)
        if cls.ARXIV_ID_PATTERN.match(arxiv_id):
            return True, None
        
        # Check old format (category/YYMMNNN)
        if cls.OLD_ARXIV_ID_PATTERN.match(arxiv_id):
            return True, None
        
        return False, f"Invalid ArXiv ID format: {arxiv_id}"
    
    @classmethod
    def get_pdf_path(cls, arxiv_id: str) -> Path:
        """
        Get PDF path for an ArXiv ID.
        
        ArXiv PDFs are organized by YYMM/arxiv_id.pdf
        """
        # Extract base ID without version
        base_id = re.sub(r'v\d+$', '', arxiv_id)
        
        # Handle new format (YYMM.NNNNN)
        match = cls.ARXIV_ID_PATTERN.match(base_id)
        if match:
            yymm = match.group(1)
            return cls.PDF_BASE_PATH / yymm / f"{base_id}.pdf"
        
        # Handle old format (category/YYMMNNN)
        match = cls.OLD_ARXIV_ID_PATTERN.match(base_id)
        if match:
            category = match.group(1)
            number = match.group(2)
            yymm = number[:4]
            return cls.PDF_BASE_PATH / yymm / f"{base_id.replace('/', '_')}.pdf"
        
        raise ValueError(f"Cannot determine PDF path for: {arxiv_id}")
    
    @classmethod
    def get_latex_path(cls, arxiv_id: str) -> Optional[Path]:
        """
        Get LaTeX source path for an ArXiv ID if it exists.
        
        LaTeX sources are optional and follow similar structure to PDFs.
        """
        # Extract base ID without version
        base_id = re.sub(r'v\d+$', '', arxiv_id)
        
        # Handle new format (YYMM.NNNNN)
        match = cls.ARXIV_ID_PATTERN.match(base_id)
        if match:
            yymm = match.group(1)
            latex_dir = cls.LATEX_BASE_PATH / yymm / base_id
            if latex_dir.exists():
                # Look for main .tex file
                tex_files = list(latex_dir.glob('*.tex'))
                if tex_files:
                    # Prefer main.tex or paper.tex if they exist
                    for preferred in ['main.tex', 'paper.tex', f'{base_id}.tex']:
                        preferred_path = latex_dir / preferred
                        if preferred_path.exists():
                            return preferred_path
                    # Otherwise return first .tex file
                    return tex_files[0]
        
        return None
    
    @classmethod
    def parse_arxiv_id(cls, arxiv_id: str) -> Dict[str, str]:
        """
        Parse ArXiv ID into components.
        
        Returns dict with 'base_id', 'version', 'year_month', etc.
        """
        components = {}
        
        # Extract version if present
        version_match = re.search(r'(v\d+)$', arxiv_id)
        if version_match:
            components['version'] = version_match.group(1)
            components['base_id'] = arxiv_id[:-len(version_match.group(1))]
        else:
            components['version'] = 'v1'
            components['base_id'] = arxiv_id
        
        # Parse base ID format
        match = cls.ARXIV_ID_PATTERN.match(components['base_id'])
        if match:
            components['year_month'] = match.group(1)
            components['number'] = match.group(2)
            components['format'] = 'new'
        else:
            match = cls.OLD_ARXIV_ID_PATTERN.match(components['base_id'])
            if match:
                components['category'] = match.group(1)
                components['number'] = match.group(2)
                components['year_month'] = match.group(2)[:4]
                components['format'] = 'old'
        
        return components


class ArXivManager:
    """
    ArXiv-specific document management.
    
    Handles ArXiv metadata, validation, and storage while delegating
    expensive processing to the generic DocumentProcessor. This manager
    preserves all ArXiv-specific characteristics while benefiting from
    shared processing infrastructure.
    """
    
    def __init__(
        self,
        processing_config: Optional[ProcessingConfig] = None,
        arango_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ArXiv manager.
        
        Args:
            processing_config: Configuration for document processing
            arango_config: Configuration for ArangoDB connection
        """
        # Initialize generic processor
        self.processor = DocumentProcessor(processing_config)
        
        # Initialize database manager if config provided
        self.db_manager = ArangoDBManager(arango_config) if arango_config else None
        
        # Initialize validator
        self.validator = ArXivValidator()
        
        # Load metadata if available
        self.metadata_cache = {}
        self._load_metadata_cache()
        
        logger.info("Initialized ArXivManager")
    
    def _load_metadata_cache(self):
        """Load ArXiv metadata from JSON snapshot."""
        metadata_path = ArXivValidator.METADATA_PATH
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    # ArXiv metadata is typically in JSON Lines format
                    for line in f:
                        if line.strip():
                            paper = json.loads(line)
                            arxiv_id = paper.get('id', '')
                            if arxiv_id:
                                self.metadata_cache[arxiv_id] = paper
                logger.info(f"Loaded metadata for {len(self.metadata_cache)} papers")
            except Exception as e:
                logger.warning(f"Failed to load metadata cache: {e}")
    
    def get_paper_info(self, arxiv_id: str) -> ArXivPaperInfo:
        """
        Get complete ArXiv paper information.
        
        Combines filesystem paths with metadata to create complete
        paper information for processing.
        """
        # Validate ID
        is_valid, error = self.validator.validate_arxiv_id(arxiv_id)
        if not is_valid:
            raise ValueError(error)
        
        # Parse ID components
        id_components = self.validator.parse_arxiv_id(arxiv_id)
        
        # Get paths
        pdf_path = self.validator.get_pdf_path(arxiv_id)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        latex_path = self.validator.get_latex_path(arxiv_id)
        
        # Get metadata from cache or use defaults
        metadata = self.metadata_cache.get(id_components['base_id'], {})
        
        return ArXivPaperInfo(
            arxiv_id=arxiv_id,
            version=id_components['version'],
            pdf_path=pdf_path,
            latex_path=latex_path,
            title=metadata.get('title', f'ArXiv Paper {arxiv_id}'),
            authors=metadata.get('authors', []).split(', ') if isinstance(metadata.get('authors'), str) else metadata.get('authors', []),
            abstract=metadata.get('abstract', ''),
            categories=metadata.get('categories', '').split() if isinstance(metadata.get('categories'), str) else metadata.get('categories', []),
            submission_date=metadata.get('created', ''),
            update_date=metadata.get('updated'),
            comments=metadata.get('comments'),
            journal_ref=metadata.get('journal_ref'),
            doi=metadata.get('doi')
        )
    
    async def process_arxiv_paper(
        self,
        arxiv_id: str,
        store_in_db: bool = True
    ) -> ProcessingResult:
        """
        Process an ArXiv paper with ArXiv-specific handling.
        
        This method bridges ArXiv-specific concerns with generic processing,
        ensuring all ArXiv metadata is preserved while benefiting from
        shared processing infrastructure.
        
        Args:
            arxiv_id: ArXiv paper identifier
            store_in_db: Whether to store results in database
            
        Returns:
            ProcessingResult with ArXiv-specific metadata attached
        """
        logger.info(f"Processing ArXiv paper: {arxiv_id}")
        
        # Step 1: Get ArXiv-specific information
        paper_info = self.get_paper_info(arxiv_id)
        
        # Step 2: Use generic processor for expensive operations
        processing_result = self.processor.process_document(
            pdf_path=paper_info.pdf_path,
            latex_path=paper_info.latex_path,
            document_id=arxiv_id
        )
        
        # Step 3: Attach ArXiv-specific metadata
        processing_result.processing_metadata.update({
            'source': 'arxiv',
            'arxiv_id': arxiv_id,
            'version': paper_info.version,
            'categories': paper_info.categories,
            'has_latex_source': paper_info.has_latex,
            'submission_date': paper_info.submission_date
        })
        
        # Step 4: Store in ArXiv-specific collections if requested
        if store_in_db and self.db_manager and processing_result.success:
            await self._store_arxiv_result(paper_info, processing_result)
        
        return processing_result
    
    async def _store_arxiv_result(
        self,
        paper_info: ArXivPaperInfo,
        result: ProcessingResult
    ):
        """
        Store processing results in ArXiv-specific database collections.
        
        Maintains separate collections for ArXiv papers to preserve
        source-specific metadata and enable specialized queries.
        """
        if not self.db_manager:
            logger.warning("No database manager configured, skipping storage")
            return
        
        try:
            # Prepare ArXiv-specific document
            arxiv_doc = {
                '_key': paper_info.sanitized_id,
                'arxiv_id': paper_info.arxiv_id,
                'version': paper_info.version,
                'title': paper_info.title,
                'authors': paper_info.authors,
                'abstract': paper_info.abstract,
                'categories': paper_info.categories,
                'submission_date': paper_info.submission_date,
                'update_date': paper_info.update_date,
                'comments': paper_info.comments,
                'journal_ref': paper_info.journal_ref,
                'doi': paper_info.doi,
                'has_latex': paper_info.has_latex,
                'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                'processing_metadata': result.processing_metadata,
                'num_chunks': len(result.chunks),
                'num_tables': len(result.extraction.tables),
                'num_equations': len(result.extraction.equations),
                'num_images': len(result.extraction.images),
                'status': 'PROCESSED'
            }
            
            # Store main paper document
            await self.db_manager.insert_document('arxiv_papers', arxiv_doc)
            
            # Store chunks with embeddings
            for chunk in result.chunks:
                chunk_doc = {
                    '_key': f"{paper_info.sanitized_id}_chunk_{chunk.chunk_index}",
                    'paper_id': paper_info.sanitized_id,
                    'arxiv_id': paper_info.arxiv_id,
                    'chunk_index': chunk.chunk_index,
                    'text': chunk.text,
                    'embedding': chunk.embedding.tolist(),
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'context_window_used': chunk.context_window_used
                }
                await self.db_manager.insert_document('arxiv_embeddings', chunk_doc)
            
            # Store structures if present
            if result.extraction.tables or result.extraction.equations or result.extraction.images:
                structures_doc = {
                    '_key': paper_info.sanitized_id,
                    'arxiv_id': paper_info.arxiv_id,
                    'tables': result.extraction.tables,
                    'equations': result.extraction.equations,
                    'images': result.extraction.images,
                    'figures': result.extraction.figures
                }
                await self.db_manager.insert_document('arxiv_structures', structures_doc)
            
            logger.info(f"Stored ArXiv paper {paper_info.arxiv_id} in database")
            
        except Exception as e:
            logger.error(f"Failed to store ArXiv paper {paper_info.arxiv_id}: {e}")
            raise
    
    def process_batch(
        self,
        arxiv_ids: List[str],
        store_in_db: bool = True
    ) -> List[ProcessingResult]:
        """
        Process multiple ArXiv papers in batch.
        
        Args:
            arxiv_ids: List of ArXiv IDs to process
            store_in_db: Whether to store results in database
            
        Returns:
            List of ProcessingResults
        """
        results = []
        
        for arxiv_id in arxiv_ids:
            try:
                result = self.process_arxiv_paper(arxiv_id, store_in_db)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {arxiv_id}: {e}")
                # Create failed result
                failed_result = ProcessingResult(
                    extraction=None,
                    chunks=[],
                    processing_metadata={'arxiv_id': arxiv_id, 'error': str(e)},
                    total_processing_time=0,
                    extraction_time=0,
                    chunking_time=0,
                    embedding_time=0,
                    success=False,
                    errors=[str(e)]
                )
                results.append(failed_result)
        
        return results