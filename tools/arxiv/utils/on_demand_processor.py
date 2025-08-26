"""
On-demand processor for ArXiv papers.

Process papers as they're discovered during research, building focused collections
around specific topics. This is the recommended approach for research workflows.

Key Pattern: Check Local → Download if Needed → Process → Store → Return
"""

import os
import sqlite3
import logging
import requests
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import quote

from arango import ArangoClient
try:
    # Try relative import first (when used as module)
    from ..pipelines.arango_acid_processor import ArangoACIDProcessor, ProcessingResult
except ImportError:
    # Fall back to adding path for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "pipelines"))
    from arango_acid_processor import ArangoACIDProcessor, ProcessingResult

logger = logging.getLogger(__name__)


class OnDemandProcessor:
    """
    Process papers on-demand based on research needs.
    Check local → Download if needed → Process → Cache
    
    This approach is ideal for research because:
    1. You don't know what papers you need until you start researching
    2. No wasted processing on papers you'll never query
    3. Fast experimentation - change strategies without reprocessing everything
    4. Progressive enhancement - start with 10 papers, expand as needed
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the on-demand processor"""
        config = config or {}
        
        # SQLite for lightweight metadata and tracking
        db_path = config.get('sqlite_db', '/bulk-store/arxiv-cache.db')
        self.metadata_db = sqlite3.connect(db_path)
        self._init_sqlite_schema()
        
        # ArangoDB for actual processed content
        arango_config = config.get('arango', {})
        
        # Validate password is provided
        password = arango_config.get('password')
        if not password:
            raise ValueError("ArangoDB password must be provided in config['arango']['password']")
        
        self.arango = ArangoClient(hosts=arango_config.get('host', ['http://192.168.1.69:8529']))
        self.db = self.arango.db(
            arango_config.get('database', 'academy_store'),
            username=arango_config.get('username', 'root'),
            password=password
        )
        
        # ACID processor for actual processing
        self.processor = ArangoACIDProcessor(config)
        
        # Local cache directories
        cache_root = Path(config.get('cache_root', '/bulk-store/arxiv-data'))
        self.pdf_cache_dir = cache_root / 'pdf'
        self.latex_cache_dir = cache_root / 'latex'
        
        # Ensure directories exist
        self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
        self.latex_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized on-demand processor with cache at {cache_root}")
    
    def _init_sqlite_schema(self):
        """
        Initialize SQLite tracking schema.
        
        Role Clarification: SQLite serves ONLY as a lightweight cache index 
        and download tracker. All actual paper metadata, content, and embeddings 
        go in ArangoDB. SQLite prevents redundant downloads and tracks processing status.
        """
        cursor = self.metadata_db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_tracking (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT,  -- For quick local search only
                pdf_path TEXT,  -- Local cache location
                latex_path TEXT,  -- Local cache location
                download_date TIMESTAMP,
                process_date TIMESTAMP,
                in_arango BOOLEAN DEFAULT 0,  -- Flag: fully processed in ArangoDB?
                processing_status TEXT,  -- 'downloaded', 'processing', 'complete', 'failed'
                error_message TEXT,
                file_size_mb REAL,
                num_chunks INTEGER,  -- Quick stats, not authoritative
                processing_time_seconds REAL
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON paper_tracking(title)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON paper_tracking(processing_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_process_date ON paper_tracking(process_date)")
        
        self.metadata_db.commit()
        logger.debug("SQLite tracking schema initialized")
    
    def process_papers(
        self,
        identifiers: List[str],
        force_reprocess: bool = False
    ) -> Dict[str, str]:
        """
        Process papers by title or arxiv_id.
        
        Args:
            identifiers: List of arxiv_ids or paper titles
            force_reprocess: Force reprocessing even if already in ArangoDB
            
        Returns:
            Dictionary mapping identifier to status
        """
        results = {}
        
        for identifier in identifiers:
            logger.info(f"Processing: {identifier}")
            
            # 1. Check if already processed in ArangoDB
            if not force_reprocess and self._is_in_arango(identifier):
                results[identifier] = "already_processed"
                logger.info(f"Paper {identifier} already in ArangoDB")
                continue
            
            # 2. Check local PDF cache or resolve identifier
            pdf_path, arxiv_id = self._find_or_download_pdf(identifier)
            
            if not pdf_path:
                results[identifier] = "not_found"
                logger.error(f"Could not find or download paper: {identifier}")
                continue
            
            # 3. Process and store in ArangoDB (with ACID)
            logger.info(f"Processing {arxiv_id} from {pdf_path}")
            processing_result = self.processor.process_paper(arxiv_id, str(pdf_path))
            
            if processing_result.success:
                results[identifier] = "processed"
                # 4. Update SQLite tracking
                self._update_tracking(
                    arxiv_id,
                    str(pdf_path),
                    processing_result
                )
                logger.info(f"Successfully processed {arxiv_id}")
            else:
                results[identifier] = f"failed: {processing_result.error}"
                self._update_tracking(
                    arxiv_id,
                    str(pdf_path),
                    processing_result
                )
                logger.error(f"Failed to process {arxiv_id}: {processing_result.error}")
        
        return results
    
    def _is_in_arango(self, identifier: str) -> bool:
        """Check if paper is already processed in ArangoDB"""
        try:
            # Try as arxiv_id first
            sanitized_id = identifier.replace('.', '_').replace('/', '_')
            
            papers_collection = self.db.collection('arxiv_papers')
            if papers_collection.has(sanitized_id):
                paper = papers_collection.get(sanitized_id)
                return paper.get('status') == 'PROCESSED'
            
            # Try searching by title if not an arxiv_id
            if '.' not in identifier and '/' not in identifier:
                # This is likely a title, search for it
                cursor = self.db.aql.execute("""
                    FOR paper IN arxiv_papers
                    FILTER LOWER(paper.title) == LOWER(@title)
                    LIMIT 1
                    RETURN paper
                """, bind_vars={'title': identifier})
                
                papers = list(cursor)
                if papers:
                    return papers[0].get('status') == 'PROCESSED'
            
            return False
        except Exception as e:
            logger.error(f"Error checking ArangoDB: {e}")
            return False
    
    def _find_or_download_pdf(self, identifier: str) -> Tuple[Optional[Path], Optional[str]]:
        """
        Find local PDF or download from ArXiv.
        
        Returns:
            Tuple of (pdf_path, arxiv_id) or (None, None) if not found
        """
        # Check if it's already an arxiv_id
        if self._looks_like_arxiv_id(identifier):
            arxiv_id = identifier
        else:
            # Try to resolve title to arxiv_id
            arxiv_id = self._resolve_identifier(identifier)
            if not arxiv_id:
                return None, None
        
        # Check local cache
        pdf_path = self._find_local_pdf(arxiv_id)
        if pdf_path and pdf_path.exists():
            logger.info(f"Found cached PDF: {pdf_path}")
            return pdf_path, arxiv_id
        
        # Download from ArXiv
        pdf_path = self._download_pdf(arxiv_id)
        if pdf_path:
            logger.info(f"Downloaded PDF to: {pdf_path}")
            return pdf_path, arxiv_id
        
        return None, None
    
    def _looks_like_arxiv_id(self, identifier: str) -> bool:
        """Check if string looks like an arxiv_id"""
        # Modern format: YYMM.NNNNN or YYMM.NNNNNN
        if '.' in identifier and identifier.replace('.', '').replace('/', '').isalnum():
            return True
        # Old format: category/YYMMNNN
        if '/' in identifier:
            return True
        return False
    
    def _find_local_pdf(self, arxiv_id: str) -> Optional[Path]:
        """Find PDF in local cache"""
        # Try different path patterns
        # Modern papers: YYMM/arxiv_id.pdf
        if '.' in arxiv_id:
            year_month = arxiv_id.split('.')[0]
            if len(year_month) == 4:  # YYMM format
                pdf_path = self.pdf_cache_dir / year_month / f"{arxiv_id}.pdf"
                if pdf_path.exists():
                    return pdf_path
        
        # Old format papers: YYMM/category_arxiv_id.pdf
        if '/' in arxiv_id:
            category, paper_id = arxiv_id.split('/', 1)
            year_month = paper_id[:4] if len(paper_id) >= 4 else '0000'
            pdf_name = f"{category}_{paper_id}.pdf".replace('/', '_')
            pdf_path = self.pdf_cache_dir / year_month / pdf_name
            if pdf_path.exists():
                return pdf_path
        
        # Check SQLite for cached path
        cursor = self.metadata_db.cursor()
        cursor.execute(
            "SELECT pdf_path FROM paper_tracking WHERE arxiv_id = ?",
            (arxiv_id,)
        )
        result = cursor.fetchone()
        if result and result[0]:
            pdf_path = Path(result[0])
            if pdf_path.exists():
                return pdf_path
        
        return None
    
    def _resolve_identifier(self, identifier: str) -> Optional[str]:
        """
        Resolve a title or partial identifier to an arxiv_id.
        
        Uses ArXiv API to search for the paper.
        """
        try:
            import xml.etree.ElementTree as ET
            
            # Search ArXiv API
            search_query = quote(identifier)
            url = f"http://export.arxiv.org/api/query?search_query=ti:{search_query}&max_results=1"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse the XML response properly
            root = ET.fromstring(response.text)
            
            # Define namespace
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Find the first entry
            entry = root.find('atom:entry', namespaces)
            if entry is not None:
                # Get the id element
                id_elem = entry.find('atom:id', namespaces)
                if id_elem is not None and id_elem.text:
                    # Extract arxiv_id from URL - handle various URL formats
                    arxiv_url = id_elem.text
                    
                    # Try different URL patterns
                    if 'arxiv.org/abs/' in arxiv_url:
                        arxiv_id = arxiv_url.split('arxiv.org/abs/')[-1]
                    elif 'arxiv.org/pdf/' in arxiv_url:
                        arxiv_id = arxiv_url.split('arxiv.org/pdf/')[-1]
                        # Remove .pdf extension if present
                        if arxiv_id.endswith('.pdf'):
                            arxiv_id = arxiv_id[:-4]
                    else:
                        # Fallback: take the last part after any slash
                        arxiv_id = arxiv_url.split('/')[-1]
                    
                    # Remove version suffix if present (e.g., v1, v2)
                    # Only remove if it's at the end and follows the pattern vN
                    arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
                    
                    logger.info(f"Resolved '{identifier}' to arxiv_id: {arxiv_id}")
                    return arxiv_id
            
            logger.warning(f"Could not resolve '{identifier}' to arxiv_id")
            return None
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error for identifier '{identifier}': {e}")
            return None
        except Exception as e:
            logger.error(f"Error resolving identifier '{identifier}': {e}")
            return None
    
    def _download_pdf(self, arxiv_id: str) -> Optional[Path]:
        """Download PDF from ArXiv"""
        try:
            # Construct download URL
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Determine cache path
            if '.' in arxiv_id:
                year_month = arxiv_id.split('.')[0]
            else:
                # Old format - extract year_month from paper ID
                if '/' in arxiv_id:
                    _, paper_id = arxiv_id.split('/', 1)
                    year_month = paper_id[:4] if len(paper_id) >= 4 else '0000'
                else:
                    year_month = '0000'
            
            # Create directory
            pdf_dir = self.pdf_cache_dir / year_month
            pdf_dir.mkdir(parents=True, exist_ok=True)
            
            # Download PDF
            pdf_name = f"{arxiv_id}.pdf".replace('/', '_')
            pdf_path = pdf_dir / pdf_name
            
            logger.info(f"Downloading {url} to {pdf_path}")
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Write to file
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Record in SQLite
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            cursor = self.metadata_db.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO paper_tracking 
                (arxiv_id, pdf_path, download_date, processing_status, file_size_mb)
                VALUES (?, ?, ?, ?, ?)
            """, (
                arxiv_id,
                str(pdf_path),
                datetime.now().isoformat(),
                'downloaded',
                file_size_mb
            ))
            self.metadata_db.commit()
            
            logger.info(f"Downloaded {arxiv_id} ({file_size_mb:.2f} MB)")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error downloading {arxiv_id}: {e}")
            return None
    
    def _update_tracking(
        self,
        arxiv_id: str,
        pdf_path: str,
        result: ProcessingResult
    ):
        """Update SQLite tracking after processing"""
        cursor = self.metadata_db.cursor()
        
        if result.success:
            cursor.execute("""
                UPDATE paper_tracking
                SET process_date = ?,
                    in_arango = 1,
                    processing_status = 'complete',
                    num_chunks = ?,
                    processing_time_seconds = ?,
                    error_message = NULL
                WHERE arxiv_id = ?
            """, (
                datetime.now().isoformat(),
                result.num_chunks,
                result.processing_time,
                arxiv_id
            ))
        else:
            cursor.execute("""
                UPDATE paper_tracking
                SET process_date = ?,
                    processing_status = 'failed',
                    error_message = ?,
                    processing_time_seconds = ?
                WHERE arxiv_id = ?
            """, (
                datetime.now().isoformat(),
                result.error,
                result.processing_time,
                arxiv_id
            ))
        
        self.metadata_db.commit()
    
    def expand_from_citations(
        self,
        seed_paper: str,
        max_citations: int = 10
    ) -> Dict[str, str]:
        """
        Expand collection based on citations from a paper.
        
        This is a key research workflow - start with one paper and
        progressively build a collection around it.
        
        Args:
            seed_paper: ArXiv ID or title of seed paper
            max_citations: Maximum number of citations to process
            
        Returns:
            Processing results for cited papers
        """
        # First ensure seed paper is processed
        seed_results = self.process_papers([seed_paper])
        
        if seed_results.get(seed_paper) not in ['processed', 'already_processed']:
            logger.error(f"Could not process seed paper: {seed_paper}")
            return {}
        
        # Get citations from the processed paper
        # TODO: Implement citation extraction
        logger.warning(
            f"Citation extraction not yet implemented for {seed_paper}. "
            "This feature requires:"
        )
        logger.warning("  1. PDF text extraction to find bibliography section")
        logger.warning("  2. Citation parsing (e.g., using GROBID or similar)")
        logger.warning("  3. ArXiv ID resolution for each citation")
        logger.warning("  4. Batch processing of resolved papers")
        
        # Implementation stub - returns empty dict until feature is complete
        # When implemented, this should:
        # 1. Query ArangoDB for the paper's extracted citations
        # 2. Resolve citation strings to arxiv_ids using ArXiv API
        # 3. Filter to max_citations most relevant papers
        # 4. Call self.process_papers() on the citation list
        
        return {}
    
    def find_similar_papers(
        self,
        paper: str,
        limit: int = 20
    ) -> List[str]:
        """
        Find papers similar to a given paper using embeddings.
        
        Args:
            paper: ArXiv ID or title
            limit: Maximum number of similar papers to find
            
        Returns:
            List of similar paper IDs
        """
        # This would use ArangoDB's vector similarity search
        # Placeholder for now
        logger.info(f"Finding papers similar to {paper}")
        return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed papers"""
        cursor = self.metadata_db.cursor()
        
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN in_arango = 1 THEN 1 ELSE 0 END) as processed,
                SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END) as failed,
                AVG(processing_time_seconds) as avg_time,
                SUM(file_size_mb) as total_size_mb
            FROM paper_tracking
        """)
        
        stats = cursor.fetchone()
        
        return {
            'total_tracked': stats[0] or 0,
            'processed': stats[1] or 0,
            'failed': stats[2] or 0,
            'avg_processing_time': stats[3] or 0,
            'total_cache_size_mb': stats[4] or 0
        }