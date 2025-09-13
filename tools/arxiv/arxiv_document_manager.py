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
from core.workflows.workflow_pdf_batch import DocumentTask

logger = logging.getLogger(__name__)


class ArXivDocumentManager:
    """
    Manages ArXiv-specific document preparation.
    
    This class knows about ArXiv's directory structure and naming conventions,
    but delegates actual processing to the generic document processor.
    """
    
    def __init__(self, pdf_base_dir: str = "/bulk-store/arxiv-data/pdf"):
        """
        Create an ArXivDocumentManager bound to a base PDF directory.
        
        Parameters:
            pdf_base_dir (str): Path to the root directory containing ArXiv PDFs organized by YYMM subdirectories (default: "/bulk-store/arxiv-data/pdf").
        
        Raises:
            ValueError: If the provided `pdf_base_dir` does not exist.
        """
        self.pdf_base_dir = Path(pdf_base_dir)
        if not self.pdf_base_dir.exists():
            raise ValueError(f"PDF base directory does not exist: {pdf_base_dir}")
        
        logger.info(f"Initialized ArXivDocumentManager with base dir: {pdf_base_dir}")
    
    def prepare_documents_from_ids(self, arxiv_ids: List[str]) -> List[DocumentTask]:
        """
        Create DocumentTask objects for the given list of arXiv IDs.
        
        For each ID in arxiv_ids attempts to prepare a DocumentTask via the internal _prepare_single_document helper; IDs that are invalid or whose PDF cannot be located are skipped (a warning is logged for each). The returned list contains only successfully prepared DocumentTask objects in the same order as the input IDs when successful.
        
        Parameters:
            arxiv_ids (List[str]): Iterable of arXiv identifiers (expected format like "YYMM.NNNNN" or "YYMM.NNNNNN", optionally including a version suffix).
        
        Returns:
            List[DocumentTask]: DocumentTask instances for the successfully prepared documents.
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
                                        Prepare DocumentTask objects for PDFs located in a specific ArXiv YYMM directory.
                                        
                                        Searches the manager's PDF base directory under the given year_month (e.g., "2301"), finds up to `limit` PDF files, and builds DocumentTask objects for each PDF. The ArXiv ID passed to the task is constructed as "{year_month}.{paper_id}" where `paper_id` is derived from the PDF filename stem.
                                        
                                        Parameters:
                                            year_month (str): Two-digit year and month string in YYMM format (e.g., "2301").
                                            limit (Optional[int]): Maximum number of PDFs to process from the directory. If None, all PDFs are processed.
                                        
                                        Returns:
                                            List[DocumentTask]: A list of prepared DocumentTask objects. Returns an empty list if the directory does not exist or no tasks could be prepared.
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
        Return DocumentTask objects for the most recent ArXiv PDFs found under the manager's base directory.
        
        Scans YYMM subdirectories in descending (newest-first) order and, within each, scans PDF files newest-first. For each PDF, derives an arXiv identifier (uses the PDF filename if it already contains a dot; otherwise prefixes with the containing YYMM directory name), builds a DocumentTask via _prepare_from_path, and collects tasks until `count` tasks have been prepared or no more PDFs are available.
        
        Parameters:
            count (int): Maximum number of recent documents to prepare (default 100).
        
        Returns:
            List[DocumentTask]: Prepared DocumentTask objects, up to `count`.
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
        """
        Prepare a DocumentTask for a single arXiv identifier.
        
        Given an arXiv ID of the form "YYMM.NNNNN" (or with a version suffix, e.g. "YYMM.NNNNNv1"), this locates the corresponding PDF under the manager's base directory (pdf_base_dir/YYMM/<arxiv_id>.pdf) and, if found, delegates construction of the DocumentTask to _prepare_from_path.
        
        Parameters:
            arxiv_id (str): ArXiv identifier expected to contain a single dot separating the YYMM prefix from the paper number.
        
        Returns:
            Optional[DocumentTask]: A DocumentTask for the PDF when the ID is valid and the PDF exists; otherwise None.
        """
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
        """
        Create a DocumentTask for an arXiv PDF and optional LaTeX source.
        
        Builds metadata for the document and returns a DocumentTask containing the PDF path,
        an optional LaTeX path (looks for a same-name `.tex` or `.tar.gz`), and ArXiv-specific metadata.
        If the provided pdf_path does not exist, returns None.
        
        Parameters:
            pdf_path (Path): Path to the PDF file to wrap. Must exist; otherwise the function returns None.
            arxiv_id (str): ArXiv identifier for the document (e.g., "2101.01234" or with version suffix "2101.01234v2").
                The function uses the portion before the first '.' as the year_month metadata when available.
        
        Returns:
            Optional[DocumentTask]: A DocumentTask populated with document_id (arxiv_id), pdf_path (string),
            latex_path (string or None), and metadata including source, arxiv_id, year_month, and has_latex.
        """
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
        Validate whether a string is a properly formatted ArXiv identifier.
        
        A valid ArXiv ID must have two parts separated by a single dot: `YYMM` and a paper number.
        - `YYMM`: exactly four digits (year and month).
        - paper number: 4–6 digits, optionally followed by a version suffix such as `v1`, `v2`, etc.
        
        Parameters:
            arxiv_id (str): The ArXiv identifier to validate (e.g., "2305.12345" or "2305.12345v2").
        
        Returns:
            bool: True if `arxiv_id` matches the expected ArXiv ID format, otherwise False.
        """
        # Basic format: YYMM.NNNNN or YYMM.NNNNNN
        parts = arxiv_id.split('.')
        if len(parts) != 2:
            return False
        
        year_month, paper_num = parts
        
        # Check year_month is 4 digits
        if not year_month.isdigit() or len(year_month) != 4:
            return False
        
        # Check paper number is 4-6 digits, optionally followed by version like v2
        base, _, version = paper_num.partition('v')
        if not base.isdigit() or not (4 <= len(base) <= 6):
            return False
        # If there's a version, validate it's numeric
        if version and not version.isdigit():
            return False
        
        return True