#!/usr/bin/env python3
"""
ArXiv Document Manager
======================

ArXiv-specific document preparation and management.
Handles the ArXiv-specific logic:
- Finding PDFs in YYMM directory structure
- Matching LaTeX sources
- Extracting ArXiv IDs
- Preparing DocumentTask objects for the generic processor
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.processors.generic_document_processor import DocumentTask

logger = logging.getLogger(__name__)


class ArXivDocumentManager:
    """
    Manages ArXiv-specific document preparation.
    
    This class knows about ArXiv's directory structure and naming conventions,
    but delegates actual processing to the generic document processor.
    """
    
    def __init__(self, pdf_base_dir: str = "/bulk-store/arxiv-data/pdf"):
        """
        Initialize ArXiv manager.
        
        Args:
            pdf_base_dir: Base directory containing ArXiv PDFs in YYMM subdirectories
        """
        self.pdf_base_dir = Path(pdf_base_dir)
        if not self.pdf_base_dir.exists():
            raise ValueError(f"PDF base directory does not exist: {pdf_base_dir}")
        
        logger.info(f"Initialized ArXivDocumentManager with base dir: {pdf_base_dir}")
    
    def prepare_documents_from_ids(self, arxiv_ids: List[str]) -> List[DocumentTask]:
        """
        Prepare DocumentTask objects from ArXiv IDs.
        
        Args:
            arxiv_ids: List of ArXiv IDs (e.g., ["2301.12345", "2302.54321"])
            
        Returns:
            List of DocumentTask objects ready for processing
        """
        tasks = []
        
        for arxiv_id in arxiv_ids:
            task = self._prepare_single_document(arxiv_id)
            if task:
                tasks.append(task)
            else:
                logger.warning(f"Could not prepare document for {arxiv_id}")
        
        logger.info(f"Prepared {len(tasks)} documents from {len(arxiv_ids)} ArXiv IDs")
        return tasks
    
    def prepare_documents_from_directory(self, 
                                        year_month: str, 
                                        limit: Optional[int] = None) -> List[DocumentTask]:
        """
        Prepare documents from a specific YYMM directory.
        
        Args:
            year_month: Directory name (e.g., "2301" for Jan 2023)
            limit: Maximum number of documents to prepare
            
        Returns:
            List of DocumentTask objects
        """
        dir_path = self.pdf_base_dir / year_month
        if not dir_path.exists():
            logger.error(f"Directory does not exist: {dir_path}")
            return []
        
        pdf_files = sorted(dir_path.glob("*.pdf"))
        if limit:
            pdf_files = pdf_files[:limit]
        
        tasks = []
        for pdf_path in pdf_files:
            arxiv_id = f"{year_month}.{pdf_path.stem.split('.')[-1]}"
            task = self._prepare_from_path(pdf_path, arxiv_id)
            if task:
                tasks.append(task)
        
        logger.info(f"Prepared {len(tasks)} documents from {dir_path}")
        return tasks
    
    def prepare_recent_documents(self, count: int = 100) -> List[DocumentTask]:
        """
        Prepare the most recent documents.
        
        Args:
            count: Number of documents to prepare
            
        Returns:
            List of DocumentTask objects
        """
        tasks = []
        
        # Get all YYMM directories, sorted in reverse (most recent first)
        yymm_dirs = sorted([d for d in self.pdf_base_dir.iterdir() if d.is_dir()], 
                          reverse=True)
        
        for yymm_dir in yymm_dirs:
            if len(tasks) >= count:
                break
            
            # Get PDFs from this directory
            pdf_files = sorted(yymm_dir.glob("*.pdf"), reverse=True)
            
            for pdf_path in pdf_files:
                if len(tasks) >= count:
                    break
                
                # Extract ArXiv ID
                year_month = yymm_dir.name
                paper_id = pdf_path.stem
                if '.' in paper_id:
                    arxiv_id = paper_id  # Already has format YYMM.NNNNN
                else:
                    arxiv_id = f"{year_month}.{paper_id}"
                
                task = self._prepare_from_path(pdf_path, arxiv_id)
                if task:
                    tasks.append(task)
        
        logger.info(f"Prepared {len(tasks)} recent documents")
        return tasks
    
    def _prepare_single_document(self, arxiv_id: str) -> Optional[DocumentTask]:
        """Prepare a single document from ArXiv ID."""
        # Parse ArXiv ID to get YYMM directory
        parts = arxiv_id.split('.')
        if len(parts) != 2:
            logger.error(f"Invalid ArXiv ID format: {arxiv_id}")
            return None
        
        year_month = parts[0]
        
        # Construct PDF path
        pdf_path = self.pdf_base_dir / year_month / f"{arxiv_id}.pdf"
        
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            return None
        
        return self._prepare_from_path(pdf_path, arxiv_id)
    
    def _prepare_from_path(self, pdf_path: Path, arxiv_id: str) -> Optional[DocumentTask]:
        """Prepare a DocumentTask from a PDF path."""
        if not pdf_path.exists():
            return None
        
        # Check for LaTeX source
        latex_path = pdf_path.with_suffix('.tex')
        if not latex_path.exists():
            # Try .tar.gz (compressed LaTeX)
            latex_path = pdf_path.with_suffix('.tar.gz')
            if not latex_path.exists():
                latex_path = None
        
        # Extract metadata specific to ArXiv
        metadata = {
            'source': 'arxiv',
            'arxiv_id': arxiv_id,
            'year_month': arxiv_id.split('.')[0] if '.' in arxiv_id else None,
            'has_latex': latex_path is not None
        }
        
        return DocumentTask(
            document_id=arxiv_id,
            pdf_path=str(pdf_path),
            latex_path=str(latex_path) if latex_path else None,
            metadata=metadata
        )
    
    def validate_arxiv_id(self, arxiv_id: str) -> bool:
        """
        Validate ArXiv ID format.
        
        Args:
            arxiv_id: ID to validate
            
        Returns:
            True if valid ArXiv ID format
        """
        # Basic format: YYMM.NNNNN or YYMM.NNNNNN
        parts = arxiv_id.split('.')
        if len(parts) != 2:
            return False
        
        year_month, paper_num = parts
        
        # Check year_month is 4 digits
        if not year_month.isdigit() or len(year_month) != 4:
            return False
        
        # Check paper number is 4-6 digits (or with version like 12345v2)
        paper_num_base = paper_num.rstrip('v0123456789')
        if not paper_num_base.isdigit() or not (4 <= len(paper_num_base) <= 6):
            return False
        
        return True