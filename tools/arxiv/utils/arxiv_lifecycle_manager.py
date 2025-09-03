#!/usr/bin/env python3
"""
ArXiv Lifecycle Manager
======================

A unified system for managing the complete lifecycle of ArXiv papers from
discovery through to HiRAG integration. This class serves as the single
entry point for all paper ingestion operations, coordinating between:

- ArXiv API for metadata and downloads
- PostgreSQL for metadata storage  
- Local filesystem for PDF/LaTeX storage
- ACID pipeline for processing
- HiRAG system for entity extraction and clustering

Following Actor-Network Theory principles, this manager serves as the central
obligatory passage point through which all papers must flow to become part
of our knowledge system.
"""

import os
import sys
import asyncio
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from arxiv_api_client import ArXivAPIClient, ArXivMetadata, DownloadResult

logger = logging.getLogger(__name__)


class PaperStatus(Enum):
    """Status of a paper in the lifecycle"""
    NOT_FOUND = "not_found"
    METADATA_ONLY = "metadata_only"
    DOWNLOADED = "downloaded"  
    PROCESSING = "processing"
    PROCESSED = "processed"
    HIRAG_INTEGRATED = "hirag_integrated"
    ERROR = "error"


@dataclass
class LifecycleResult:
    """Complete result of paper lifecycle processing"""
    arxiv_id: str
    status: PaperStatus
    metadata_fetched: bool = False
    pdf_downloaded: bool = False
    latex_downloaded: bool = False
    processed: bool = False
    hirag_updated: bool = False
    error_message: Optional[str] = None
    pdf_path: Optional[Path] = None
    latex_path: Optional[Path] = None
    processing_time_seconds: float = 0.0
    database_updated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['status'] = self.status.value
        result['pdf_path'] = str(self.pdf_path) if self.pdf_path else None
        result['latex_path'] = str(self.latex_path) if self.latex_path else None
        return result


class ArXivLifecycleManager:
    """
    Unified manager for ArXiv paper lifecycle operations.
    
    This class implements the complete paper ingestion workflow:
    1. Check PostgreSQL for existing metadata
    2. Fetch from ArXiv API if missing
    3. Download PDF and LaTeX files
    4. Update PostgreSQL database
    5. Process through ACID pipeline
    6. Update HiRAG system
    
    All operations are idempotent and can be safely repeated.
    """
    
    def __init__(self,
                 pdf_storage_dir: str = "/bulk-store/arxiv-data/pdf",
                 latex_storage_dir: str = "/bulk-store/arxiv-data/latex",
                 postgresql_config: Optional[Dict[str, str]] = None,
                 acid_pipeline_config: str = None):
        """
        Initialize the lifecycle manager.
        
        Args:
            pdf_storage_dir: Directory for PDF storage
            latex_storage_dir: Directory for LaTeX storage
            postgresql_config: PostgreSQL connection configuration
            acid_pipeline_config: Path to ACID pipeline config
        """
        self.pdf_storage_dir = Path(pdf_storage_dir)
        self.latex_storage_dir = Path(latex_storage_dir)
        
        # Ensure storage directories exist
        self.pdf_storage_dir.mkdir(parents=True, exist_ok=True)
        self.latex_storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ArXiv API client
        self.arxiv_client = ArXivAPIClient()
        
        # PostgreSQL configuration
        self.pg_config = postgresql_config or {
            'host': 'localhost',
            'database': 'arxiv',
            'user': 'postgres',
            'password': os.getenv('PGPASSWORD', '')
        }
        
        # ACID pipeline configuration
        self.acid_config = acid_pipeline_config or str(
            Path(__file__).parent.parent / "configs" / "acid_pipeline_phased.yaml"
        )
        
        logger.info("Initialized ArXiv Lifecycle Manager")
        logger.info(f"PDF storage: {self.pdf_storage_dir}")
        logger.info(f"LaTeX storage: {self.latex_storage_dir}")
        logger.info(f"ACID config: {self.acid_config}")
    
    def get_database_connection(self):
        """Get PostgreSQL database connection"""
        try:
            conn = psycopg2.connect(**self.pg_config)
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def check_paper_status(self, arxiv_id: str) -> Tuple[PaperStatus, Dict[str, Any]]:
        """
        Check the current status of a paper in our system.
        
        Args:
            arxiv_id: ArXiv paper identifier
            
        Returns:
            Tuple of (status, details_dict)
        """
        details = {
            'in_postgresql': False,
            'pdf_exists': False,
            'latex_exists': False,
            'processed': False,
            'hirag_integrated': False
        }
        
        try:
            # Check PostgreSQL - handle version numbers
            with self.get_database_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # First try exact match
                    cur.execute(
                        "SELECT * FROM papers WHERE arxiv_id = %s",
                        (arxiv_id,)
                    )
                    row = cur.fetchone()
                    
                    # If not found, try with version pattern (e.g., 2508.21038 -> 2508.21038v*)
                    if not row:
                        cur.execute(
                            "SELECT * FROM papers WHERE arxiv_id LIKE %s ORDER BY arxiv_id DESC LIMIT 1",
                            (f"{arxiv_id}v%",)
                        )
                        row = cur.fetchone()
                    
                    if row:
                        details['in_postgresql'] = True
                        details['has_pdf'] = row.get('has_pdf', False)
                        details['has_latex'] = row.get('has_latex', False)
                        details['pdf_path'] = row.get('pdf_path')
                        details['latex_path'] = row.get('latex_path')
            
            # Check filesystem
            year_month = self._extract_year_month(arxiv_id)
            
            pdf_path = self.pdf_storage_dir / year_month / f"{arxiv_id.replace('/', '_')}.pdf"
            if pdf_path.exists():
                details['pdf_exists'] = True
                details['pdf_path'] = str(pdf_path)
            
            latex_path = self.latex_storage_dir / year_month / f"{arxiv_id.replace('/', '_')}.tar.gz"
            if latex_path.exists():
                details['latex_exists'] = True
                details['latex_path'] = str(latex_path)
            
            # Check if processed (simplified - could query ArangoDB)
            details['processed'] = details.get('has_pdf', False) and details['pdf_exists']
            
            # Determine overall status
            if not details['in_postgresql']:
                return PaperStatus.NOT_FOUND, details
            elif not details['pdf_exists']:
                return PaperStatus.METADATA_ONLY, details
            elif not details['processed']:
                return PaperStatus.DOWNLOADED, details
            else:
                return PaperStatus.PROCESSED, details
                
        except Exception as e:
            logger.error(f"Error checking status for {arxiv_id}: {e}")
            details['error'] = str(e)
            return PaperStatus.ERROR, details
    
    def process_paper(self, arxiv_id: str, force: bool = False) -> LifecycleResult:
        """
        Process a paper through the complete lifecycle.
        
        Args:
            arxiv_id: ArXiv paper identifier
            force: Force reprocessing even if already complete
            
        Returns:
            LifecycleResult with complete processing information
        """
        start_time = datetime.now()
        result = LifecycleResult(arxiv_id=arxiv_id, status=PaperStatus.NOT_FOUND)
        
        try:
            logger.info(f"Starting lifecycle processing for {arxiv_id}")
            
            # Step 1: Check current status
            current_status, details = self.check_paper_status(arxiv_id)
            
            if current_status == PaperStatus.PROCESSED and not force:
                logger.info(f"Paper {arxiv_id} already processed, skipping")
                result.status = PaperStatus.PROCESSED
                result.processed = True
                result.pdf_path = Path(details.get('pdf_path', '')) if details.get('pdf_path') else None
                result.latex_path = Path(details.get('latex_path', '')) if details.get('latex_path') else None
                return result
            
            # Step 2: Fetch metadata if not in database
            metadata = None
            if not details['in_postgresql'] or force:
                logger.info(f"Fetching metadata for {arxiv_id}")
                metadata = self.arxiv_client.get_paper_metadata(arxiv_id)
                
                if not metadata:
                    result.error_message = "Failed to fetch metadata from ArXiv API"
                    result.status = PaperStatus.ERROR
                    return result
                
                result.metadata_fetched = True
                
                # Update PostgreSQL
                self._update_paper_metadata(metadata)
                result.database_updated = True
            
            # Step 3: Download files if needed
            if not details['pdf_exists'] or force:
                logger.info(f"Downloading files for {arxiv_id}")
                
                if not metadata:
                    # Re-fetch metadata if we didn't already
                    metadata = self.arxiv_client.get_paper_metadata(arxiv_id)
                
                download_result = self.arxiv_client.download_paper(
                    arxiv_id,
                    self.pdf_storage_dir,
                    self.latex_storage_dir,
                    force=force
                )
                
                if not download_result.success:
                    result.error_message = download_result.error_message
                    result.status = PaperStatus.ERROR
                    return result
                
                result.pdf_downloaded = True
                result.pdf_path = download_result.pdf_path
                
                if download_result.latex_path:
                    result.latex_downloaded = True
                    result.latex_path = download_result.latex_path
                
                # Update database with file paths
                self._update_paper_files(arxiv_id, download_result)
                result.database_updated = True
            
            # Step 4: Process through ACID pipeline
            if not details.get('processed', False) or force:
                logger.info(f"Processing {arxiv_id} through ACID pipeline")
                
                processing_success = self._run_acid_pipeline([arxiv_id])
                if processing_success:
                    result.processed = True
                else:
                    result.error_message = "ACID pipeline processing failed"
                    result.status = PaperStatus.ERROR
                    return result
            
            # Step 5: Update HiRAG system
            logger.info(f"Updating HiRAG system for {arxiv_id}")
            hirag_success = self._update_hirag_system([arxiv_id])
            if hirag_success:
                result.hirag_updated = True
                result.status = PaperStatus.HIRAG_INTEGRATED
            else:
                logger.warning(f"HiRAG update failed for {arxiv_id}, but paper is processed")
                result.status = PaperStatus.PROCESSED
            
            # Calculate processing time
            result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Successfully processed {arxiv_id} in {result.processing_time_seconds:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {arxiv_id}: {e}")
            result.error_message = str(e)
            result.status = PaperStatus.ERROR
            result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            return result
    
    def batch_process_papers(self, arxiv_ids: List[str], force: bool = False) -> Dict[str, LifecycleResult]:
        """
        Process multiple papers through the lifecycle.
        
        Args:
            arxiv_ids: List of ArXiv paper identifiers
            force: Force reprocessing even if already complete
            
        Returns:
            Dictionary mapping ArXiv ID to LifecycleResult
        """
        logger.info(f"Starting batch processing of {len(arxiv_ids)} papers")
        results = {}
        
        for arxiv_id in arxiv_ids:
            try:
                result = self.process_paper(arxiv_id, force=force)
                results[arxiv_id] = result
                
                # Log progress
                success_count = len([r for r in results.values() if r.status != PaperStatus.ERROR])
                logger.info(f"Progress: {len(results)}/{len(arxiv_ids)} papers processed, {success_count} successful")
                
            except Exception as e:
                logger.error(f"Failed to process {arxiv_id}: {e}")
                results[arxiv_id] = LifecycleResult(
                    arxiv_id=arxiv_id,
                    status=PaperStatus.ERROR,
                    error_message=str(e)
                )
        
        # Summary
        success_count = len([r for r in results.values() if r.status != PaperStatus.ERROR])
        logger.info(f"Batch processing complete: {success_count}/{len(arxiv_ids)} papers successful")
        
        return results
    
    def _extract_year_month(self, arxiv_id: str) -> str:
        """Extract year-month for directory organization"""
        if '.' in arxiv_id:
            return arxiv_id.split('.')[0]
        elif '/' in arxiv_id:
            paper_id = arxiv_id.split('/', 1)[1]
            return paper_id[:4] if len(paper_id) >= 4 else '0000'
        else:
            return '0000'
    
    def _update_paper_metadata(self, metadata: ArXivMetadata):
        """Update PostgreSQL with paper metadata"""
        try:
            with self.get_database_connection() as conn:
                with conn.cursor() as cur:
                    # Insert or update paper metadata
                    cur.execute("""
                        INSERT INTO papers (
                            arxiv_id, title, abstract, primary_category,
                            published_at, updated_at, doi, journal_ref,
                            has_pdf, has_latex, created_at, modified_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                        )
                        ON CONFLICT (arxiv_id) 
                        DO UPDATE SET
                            title = EXCLUDED.title,
                            abstract = EXCLUDED.abstract,
                            primary_category = EXCLUDED.primary_category,
                            published_at = EXCLUDED.published_at,
                            updated_at = EXCLUDED.updated_at,
                            doi = EXCLUDED.doi,
                            journal_ref = EXCLUDED.journal_ref,
                            has_latex = EXCLUDED.has_latex,
                            modified_at = CURRENT_TIMESTAMP
                    """, (
                        metadata.arxiv_id,
                        metadata.title,
                        metadata.abstract,
                        metadata.primary_category,
                        metadata.published,
                        metadata.updated,
                        metadata.doi,
                        metadata.journal_ref,
                        metadata.has_pdf,
                        metadata.has_latex
                    ))
                    
                    # Insert categories
                    cur.execute("DELETE FROM paper_categories WHERE arxiv_id = %s", (metadata.arxiv_id,))
                    for category in metadata.categories:
                        cur.execute(
                            "INSERT INTO paper_categories (arxiv_id, category) VALUES (%s, %s)",
                            (metadata.arxiv_id, category)
                        )
                    
                    conn.commit()
                    logger.info(f"Updated PostgreSQL metadata for {metadata.arxiv_id}")
                    
        except Exception as e:
            logger.error(f"Failed to update PostgreSQL metadata for {metadata.arxiv_id}: {e}")
            raise
    
    def _update_paper_files(self, arxiv_id: str, download_result: DownloadResult):
        """Update PostgreSQL with file paths"""
        try:
            with self.get_database_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE papers SET
                            pdf_path = %s,
                            latex_path = %s,
                            has_pdf = %s,
                            pdf_size_bytes = %s,
                            modified_at = CURRENT_TIMESTAMP
                        WHERE arxiv_id = %s
                    """, (
                        str(download_result.pdf_path) if download_result.pdf_path else None,
                        str(download_result.latex_path) if download_result.latex_path else None,
                        download_result.pdf_path is not None,
                        download_result.file_size_bytes,
                        arxiv_id
                    ))
                    
                    conn.commit()
                    logger.info(f"Updated PostgreSQL file paths for {arxiv_id}")
                    
        except Exception as e:
            logger.error(f"Failed to update PostgreSQL file paths for {arxiv_id}: {e}")
            raise
    
    def _run_acid_pipeline(self, arxiv_ids: List[str]) -> bool:
        """Run ACID pipeline for specific papers"""
        try:
            # Create a temporary list file
            temp_list_file = Path("/tmp") / f"lifecycle_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(temp_list_file, 'w') as f:
                for arxiv_id in arxiv_ids:
                    f.write(f"{arxiv_id}\n")
            
            # Run pipeline with the specific papers
            pipeline_script = Path(__file__).parent.parent / "scripts" / "run_pipeline_from_list.py"
            
            cmd = [
                "python", str(pipeline_script),
                str(len(arxiv_ids)),
                "--list", str(temp_list_file),
                "--config", self.acid_config
            ]
            
            # Add environment variables
            env = os.environ.copy()
            if 'ARANGO_PASSWORD' not in env:
                arango_password = os.getenv('ARANGO_PASSWORD')
                if arango_password:
                    env['ARANGO_PASSWORD'] = arango_password
            
            logger.info(f"Running ACID pipeline: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
            
            # Cleanup temp file
            temp_list_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                logger.info("ACID pipeline completed successfully")
                return True
            else:
                logger.error(f"ACID pipeline failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running ACID pipeline: {e}")
            return False
    
    def _update_hirag_system(self, arxiv_ids: List[str]) -> bool:
        """Update HiRAG system with new papers"""
        try:
            # Run the PostgreSQL metadata sync to update HiRAG
            hirag_sync_script = Path(__file__).parent.parent.parent / "hirag" / "postgres_metadata_sync.py"
            
            if not hirag_sync_script.exists():
                logger.warning("HiRAG sync script not found, skipping HiRAG update")
                return False
            
            cmd = ["python", str(hirag_sync_script), "--incremental"]
            
            # Add environment variables
            env = os.environ.copy()
            if 'ARANGO_PASSWORD' not in env:
                arango_password = os.getenv('ARANGO_PASSWORD')
                if arango_password:
                    env['ARANGO_PASSWORD'] = arango_password
            
            logger.info("Updating HiRAG system")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("HiRAG system updated successfully")
                return True
            else:
                logger.error(f"HiRAG update failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating HiRAG system: {e}")
            return False
    
    def generate_report(self, results: Dict[str, LifecycleResult]) -> str:
        """Generate a summary report of processing results"""
        total = len(results)
        successful = len([r for r in results.values() if r.status != PaperStatus.ERROR])
        
        status_counts = {}
        for result in results.values():
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        report = f"""
ArXiv Lifecycle Manager - Processing Report
==========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Total papers processed: {total}
- Successful: {successful}
- Failed: {total - successful}
- Success rate: {(successful/total*100):.1f}%

Status Breakdown:
"""
        
        for status, count in sorted(status_counts.items()):
            percentage = (count/total*100)
            report += f"- {status}: {count} ({percentage:.1f}%)\n"
        
        report += "\nFailed Papers:\n"
        for arxiv_id, result in results.items():
            if result.status == PaperStatus.ERROR:
                report += f"- {arxiv_id}: {result.error_message}\n"
        
        return report


# Convenience functions
def quick_process_paper(arxiv_id: str, force: bool = False) -> LifecycleResult:
    """Quick processing of a single paper"""
    manager = ArXivLifecycleManager()
    return manager.process_paper(arxiv_id, force=force)


def quick_check_status(arxiv_id: str) -> Tuple[PaperStatus, Dict[str, Any]]:
    """Quick status check for a single paper"""
    manager = ArXivLifecycleManager()
    return manager.check_paper_status(arxiv_id)


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="ArXiv Lifecycle Manager")
    parser.add_argument("arxiv_id", help="ArXiv paper ID to process")
    parser.add_argument("--status", action="store_true", help="Check status only")
    parser.add_argument("--force", action="store_true", help="Force reprocessing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = ArXivLifecycleManager()
    
    if args.status:
        # Status check only
        status, details = manager.check_paper_status(args.arxiv_id)
        print(f"Status: {status.value}")
        print(f"Details: {json.dumps(details, indent=2)}")
    else:
        # Full processing
        result = manager.process_paper(args.arxiv_id, force=args.force)
        print(f"Result: {json.dumps(result.to_dict(), indent=2)}")