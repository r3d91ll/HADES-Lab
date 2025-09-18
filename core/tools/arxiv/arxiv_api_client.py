#!/usr/bin/env python3
"""
ArXiv API Client
================

A robust client for interacting with the ArXiv API to fetch paper metadata,
check availability, and download papers. Handles rate limiting, error recovery,
and provides a clean interface for the ArXiv Lifecycle Manager.

Following Actor-Network Theory principles, this client serves as a mediator
between our internal systems and the ArXiv network, translating between
different representational forms while maintaining data integrity.
"""

import os
import re
import time
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from urllib.parse import quote
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ArXivMetadata:
    """
    Structured representation of ArXiv paper metadata.
    
    This class embodies the boundary object concept from ANT - maintaining
    coherence across different institutional contexts (ArXiv, PostgreSQL, 
    ArangoDB) while being flexible enough to adapt to local needs.
    """
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    primary_category: str
    published: datetime
    updated: datetime
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    license: Optional[str] = None
    has_pdf: bool = True
    has_latex: bool = False
    pdf_url: str = ""
    latex_url: str = ""
    
    def __post_init__(self):
        """Generate PDF URL if not provided"""
        if not self.pdf_url:
            self.pdf_url = f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"


@dataclass
class DownloadResult:
    """Result of a paper download operation"""
    success: bool
    arxiv_id: str
    pdf_path: Optional[Path] = None
    latex_path: Optional[Path] = None
    metadata: Optional[ArXivMetadata] = None
    error_message: Optional[str] = None
    file_size_bytes: int = 0


class ArXivAPIClient:
    """
    Client for ArXiv API operations with comprehensive error handling.
    
    This client implements the obligatory passage point pattern - all
    interactions with ArXiv must flow through this standardized interface,
    ensuring consistency and reliability in our data acquisition process.
    """
    
    def __init__(self, 
                 rate_limit_delay: float = 3.0,
                 max_retries: int = 3,
                 timeout: int = 30):
        """
        Initialize the ArXiv API client.
        
        Args:
            rate_limit_delay: Seconds to wait between API calls
            max_retries: Maximum retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_request_time = 0
        
        # ArXiv API endpoints
        self.api_base_url = "http://export.arxiv.org/api/query"
        self.pdf_base_url = "https://arxiv.org/pdf"
        self.latex_base_url = "https://arxiv.org/e-print"
        
        logger.info(f"Initialized ArXiv API client with {rate_limit_delay}s rate limit")
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        """Make HTTP request with retries and error handling"""
        self._enforce_rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Request attempt {attempt + 1}: {url}")
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = (2 ** attempt) * self.rate_limit_delay
                    time.sleep(delay)
                else:
                    raise
        
        raise requests.exceptions.RequestException("Max retries exceeded")
    
    def validate_arxiv_id(self, arxiv_id: str) -> bool:
        """
        Validate ArXiv ID format.
        
        Supports both old and new formats:
        - New: YYMM.NNNNN[vN] (e.g., 2508.21038, 1234.5678v2)
        - Old: subject-class/YYMMnnn (e.g., cs.AI/0601001)
        """
        # New format: YYMM.NNNNN[vN]
        new_format = re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', arxiv_id)
        if new_format:
            return True
        
        # Old format: subject-class/YYMMnnn
        old_format = re.match(r'^[a-z-]+(\.[A-Z]{2})?/\d{7}$', arxiv_id)
        if old_format:
            return True
        
        return False
    
    def get_paper_metadata(self, arxiv_id: str) -> Optional[ArXivMetadata]:
        """
        Fetch paper metadata from ArXiv API.
        
        Args:
            arxiv_id: ArXiv paper identifier
            
        Returns:
            ArXivMetadata object or None if not found
        """
        if not self.validate_arxiv_id(arxiv_id):
            logger.error(f"Invalid ArXiv ID format: {arxiv_id}")
            return None
        
        try:
            # Query ArXiv API
            params = {
                'id_list': arxiv_id,
                'max_results': 1
            }
            
            response = self._make_request(self.api_base_url, params)
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Check if we got results
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            if not entries:
                logger.warning(f"No entry found for ArXiv ID: {arxiv_id}")
                return None
            
            entry = entries[0]
            
            # Extract metadata
            metadata = self._parse_entry(entry)
            logger.info(f"Retrieved metadata for {arxiv_id}: {metadata.title[:50]}...")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to fetch metadata for {arxiv_id}: {e}")
            return None
    
    def _parse_entry(self, entry: ET.Element) -> ArXivMetadata:
        """Parse XML entry into ArXivMetadata object"""
        
        # Extract basic fields
        id_url = entry.find('.//{http://www.w3.org/2005/Atom}id').text
        arxiv_id = id_url.split('/')[-1]  # Extract ID from URL
        
        title = entry.find('.//{http://www.w3.org/2005/Atom}title').text.strip()
        title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
        
        abstract = entry.find('.//{http://www.w3.org/2005/Atom}summary').text.strip()
        abstract = re.sub(r'\s+', ' ', abstract)  # Normalize whitespace
        
        # Extract authors
        authors = []
        for author in entry.findall('.//{http://www.w3.org/2005/Atom}author'):
            name = author.find('.//{http://www.w3.org/2005/Atom}name').text
            authors.append(name.strip())
        
        # Extract categories
        categories = []
        primary_category = None
        
        for category in entry.findall('.//{http://arxiv.org/schemas/atom}primary_category'):
            primary_category = category.get('term')
        
        for category in entry.findall('.//{http://arxiv.org/schemas/atom}category'):
            categories.append(category.get('term'))
        
        if not primary_category and categories:
            primary_category = categories[0]
        
        # Extract dates
        published_str = entry.find('.//{http://www.w3.org/2005/Atom}published').text
        updated_str = entry.find('.//{http://www.w3.org/2005/Atom}updated').text
        
        published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
        updated = datetime.fromisoformat(updated_str.replace('Z', '+00:00'))
        
        # Extract optional fields
        doi = None
        journal_ref = None
        
        doi_elem = entry.find('.//{http://arxiv.org/schemas/atom}doi')
        if doi_elem is not None:
            doi = doi_elem.text
        
        journal_elem = entry.find('.//{http://arxiv.org/schemas/atom}journal_ref')
        if journal_elem is not None:
            journal_ref = journal_elem.text
        
        # Check for LaTeX availability (we'll verify this in download)
        has_latex = self._check_latex_availability(arxiv_id)
        
        return ArXivMetadata(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            categories=categories,
            primary_category=primary_category,
            published=published,
            updated=updated,
            doi=doi,
            journal_ref=journal_ref,
            has_latex=has_latex
        )
    
    def _check_latex_availability(self, arxiv_id: str) -> bool:
        """Check if LaTeX source is available for a paper"""
        try:
            latex_url = f"{self.latex_base_url}/{arxiv_id}"
            
            # Make a HEAD request to check availability
            self._enforce_rate_limit()
            response = requests.head(latex_url, timeout=10)
            
            # LaTeX is available if we get 200
            return response.status_code == 200
            
        except Exception:
            # If check fails, assume no LaTeX
            return False
    
    def download_paper(self, 
                      arxiv_id: str, 
                      pdf_dir: Path, 
                      latex_dir: Optional[Path] = None,
                      force: bool = False) -> DownloadResult:
        """
        Download paper PDF and optionally LaTeX source.
        
        Args:
            arxiv_id: ArXiv paper identifier
            pdf_dir: Directory to save PDF files
            latex_dir: Directory to save LaTeX files (optional)
            force: Force download even if files exist
            
        Returns:
            DownloadResult with success status and file paths
        """
        if not self.validate_arxiv_id(arxiv_id):
            return DownloadResult(
                success=False,
                arxiv_id=arxiv_id,
                error_message=f"Invalid ArXiv ID format: {arxiv_id}"
            )
        
        # Get metadata first
        metadata = self.get_paper_metadata(arxiv_id)
        if not metadata:
            return DownloadResult(
                success=False,
                arxiv_id=arxiv_id,
                error_message="Failed to fetch paper metadata"
            )
        
        # Determine file paths using YYMM structure
        year_month = self._extract_year_month(arxiv_id)
        
        pdf_subdir = pdf_dir / year_month
        pdf_subdir.mkdir(parents=True, exist_ok=True)
        
        pdf_filename = f"{arxiv_id.replace('/', '_')}.pdf"
        pdf_path = pdf_subdir / pdf_filename
        
        latex_path = None
        if latex_dir and metadata.has_latex:
            latex_subdir = latex_dir / year_month
            latex_subdir.mkdir(parents=True, exist_ok=True)
            latex_filename = f"{arxiv_id.replace('/', '_')}.tar.gz"
            latex_path = latex_subdir / latex_filename
        
        # Check if files already exist
        if not force:
            if pdf_path.exists():
                if not latex_path or latex_path.exists():
                    logger.info(f"Paper {arxiv_id} already downloaded")
                    return DownloadResult(
                        success=True,
                        arxiv_id=arxiv_id,
                        pdf_path=pdf_path,
                        latex_path=latex_path,
                        metadata=metadata,
                        file_size_bytes=pdf_path.stat().st_size
                    )
        
        # Download PDF
        try:
            logger.info(f"Downloading PDF for {arxiv_id}")
            pdf_url = f"{self.pdf_base_url}/{arxiv_id}.pdf"
            
            response = self._make_request(pdf_url)
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            file_size = pdf_path.stat().st_size
            logger.info(f"Downloaded PDF: {pdf_path} ({file_size:,} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to download PDF for {arxiv_id}: {e}")
            return DownloadResult(
                success=False,
                arxiv_id=arxiv_id,
                error_message=f"PDF download failed: {str(e)}"
            )
        
        # Download LaTeX if available and requested
        if latex_path and metadata.has_latex:
            try:
                logger.info(f"Downloading LaTeX for {arxiv_id}")
                latex_url = f"{self.latex_base_url}/{arxiv_id}"
                
                response = self._make_request(latex_url)
                
                with open(latex_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded LaTeX: {latex_path}")
                
            except Exception as e:
                logger.warning(f"Failed to download LaTeX for {arxiv_id}: {e}")
                # Don't fail the entire operation if LaTeX fails
                latex_path = None
        
        return DownloadResult(
            success=True,
            arxiv_id=arxiv_id,
            pdf_path=pdf_path,
            latex_path=latex_path,
            metadata=metadata,
            file_size_bytes=file_size
        )
    
    def _extract_year_month(self, arxiv_id: str) -> str:
        """Extract year-month for directory organization"""
        if '.' in arxiv_id:
            # New format: YYMM.NNNNN
            return arxiv_id.split('.')[0]
        elif '/' in arxiv_id:
            # Old format: subject-class/YYMMnnn
            paper_id = arxiv_id.split('/', 1)[1]
            return paper_id[:4] if len(paper_id) >= 4 else '0000'
        else:
            return '0000'
    
    def batch_get_metadata(self, arxiv_ids: List[str]) -> Dict[str, Optional[ArXivMetadata]]:
        """
        Fetch metadata for multiple papers efficiently.
        
        Args:
            arxiv_ids: List of ArXiv IDs
            
        Returns:
            Dictionary mapping ArXiv ID to metadata (None if failed)
        """
        results: Dict[str, Optional[ArXivMetadata]] = {}
        
        # Process in batches to respect API limits
        batch_size = 10
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i:i + batch_size]
            
            try:
                # Query multiple papers at once
                params = {
                    'id_list': ','.join(batch),
                    'max_results': len(batch)
                }
                
                response = self._make_request(self.api_base_url, params)
                root = ET.fromstring(response.content)
                
                # Parse all entries
                entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
                
                for entry in entries:
                    try:
                        metadata = self._parse_entry(entry)
                        results[metadata.arxiv_id] = metadata
                    except Exception as e:
                        logger.error(f"Failed to parse entry: {e}")
                        
                # Mark missing papers as None
                for arxiv_id in batch:
                    if arxiv_id not in results:
                        results[arxiv_id] = None
                        logger.warning(f"No metadata found for {arxiv_id}")
                        
            except Exception as e:
                logger.error(f"Batch metadata fetch failed for batch {i//batch_size + 1}: {e}")
                # Mark entire batch as failed
                for arxiv_id in batch:
                    results[arxiv_id] = None
        
        logger.info(f"Fetched metadata for {len([r for r in results.values() if r])} out of {len(arxiv_ids)} papers")
        return results


# Convenience functions for common operations
def quick_fetch_metadata(arxiv_id: str) -> Optional[ArXivMetadata]:
    """Quick metadata fetch for a single paper"""
    client = ArXivAPIClient()
    return client.get_paper_metadata(arxiv_id)


def quick_download_paper(arxiv_id: str, 
                        pdf_dir: str = "/bulk-store/arxiv-data/pdf",
                        include_latex: bool = True) -> DownloadResult:
    """Quick download for a single paper"""
    client = ArXivAPIClient()
    pdf_path = Path(pdf_dir)
    latex_path = Path(pdf_dir.replace('/pdf', '/latex')) if include_latex else None
    
    return client.download_paper(arxiv_id, pdf_path, latex_path)


if __name__ == "__main__":
    # Simple CLI for testing
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python arxiv_api_client.py <arxiv_id>")
        sys.exit(1)
    
    arxiv_id = sys.argv[1]
    
    logging.basicConfig(level=logging.INFO)
    
    print(f"Testing ArXiv API client with paper: {arxiv_id}")
    
    # Test metadata fetch
    metadata = quick_fetch_metadata(arxiv_id)
    if metadata:
        print(f"\nTitle: {metadata.title}")
        print(f"Authors: {', '.join(metadata.authors)}")
        print(f"Categories: {', '.join(metadata.categories)}")
        print(f"Published: {metadata.published}")
        print(f"Has LaTeX: {metadata.has_latex}")
    else:
        print("Failed to fetch metadata")
