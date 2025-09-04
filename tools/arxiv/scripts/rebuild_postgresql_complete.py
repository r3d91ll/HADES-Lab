#!/usr/bin/env python3
"""
Complete PostgreSQL ArXiv Database Rebuild Script
=================================================

Complete rebuild with proper schema and all ArXiv metadata fields.
"""

import os
import sys
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Optional, List
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/todd/olympus/HADES-Lab/tools/arxiv/logs/postgresql_rebuild_complete.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PostgreSQLRebuilder:
    def __init__(self):
        self.pg_config = {
            'host': 'localhost',
            'database': 'arxiv', 
            'user': 'postgres',
            'password': os.getenv('PGPASSWORD', '')
        }
        
        # File paths
        self.metadata_file = Path("/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json")
        self.pdf_dir = Path("/bulk-store/arxiv-data/pdf")
        self.latex_dir = Path("/bulk-store/arxiv-data/latex")
        
        # Stats
        self.stats = {
            'metadata_imported': 0,
            'pdfs_found': 0,
            'latex_found': 0,
            'signal_files': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info("PostgreSQL Rebuilder initialized with complete metadata support")
    
    def get_database_connection(self):
        """Get PostgreSQL database connection."""
        return psycopg2.connect(**self.pg_config)
    
    def parse_date(self, date_str: str) -> Optional[datetime.date]:
        """Parse various date formats from ArXiv data."""
        if not date_str:
            return None
        
        try:
            # Handle YYYY-MM-DD format
            if len(date_str) == 10 and '-' in date_str:
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Handle GMT format: 'Mon, 2 Apr 2007 19:18:42 GMT'
            if 'GMT' in date_str:
                # Extract just the date part
                date_part = date_str.split(' GMT')[0]
                return datetime.strptime(date_part, '%a, %d %b %Y %H:%M:%S').date()
            
            # Try other common formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d', '%m/%d/%Y']:
                try:
                    return datetime.strptime(date_str[:10], '%Y-%m-%d').date()
                except:
                    continue
        
        except Exception as e:
            logger.debug(f"Could not parse date: {date_str} - {e}")
        
        return None
    
    def extract_primary_category(self, categories: str) -> Optional[str]:
        """Extract primary category (first category) from categories string."""
        if not categories:
            return None
        return categories.split()[0] if categories.split() else None
    
    def import_metadata_from_snapshot(self):
        """Import all metadata from arxiv-metadata-oai-snapshot.json with complete fields."""
        logger.info("Starting complete metadata import...")
        
        if not self.metadata_file.exists():
            logger.error(f"Metadata file not found: {self.metadata_file}")
            return False
        
        batch_size = 1000
        batch = []
        processed = 0
        
        conn = self.get_database_connection()
        
        try:
            with open(self.metadata_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        paper = json.loads(line.strip())
                        
                        # Extract all fields with proper handling
                        arxiv_id = paper.get('id', '')
                        title = paper.get('title', '').replace('\n', ' ').strip()
                        abstract = paper.get('abstract', '').replace('\n', ' ').strip()
                        authors = paper.get('authors', '')
                        categories = paper.get('categories', '')
                        primary_category = self.extract_primary_category(categories)
                        comments = paper.get('comments', None)
                        journal_ref = paper.get('journal-ref', None)
                        doi = paper.get('doi', None)
                        report_number = paper.get('report-no', None)
                        license_info = paper.get('license', None)
                        
                        # Handle dates
                        update_date = self.parse_date(paper.get('update_date', ''))
                        
                        # Handle versions
                        versions = paper.get('versions', [])
                        versions_count = len(versions)
                        latest_version = versions[-1].get('version', None) if versions else None
                        submission_date = None
                        
                        if versions:
                            first_version_date = versions[0].get('created', '')
                            submission_date = self.parse_date(first_version_date)
                        
                        # Handle authors_parsed (store as JSON)
                        authors_parsed = json.dumps(paper.get('authors_parsed', [])) if paper.get('authors_parsed') else None
                        
                        # Calculate year/month for partitioning
                        year = submission_date.year if submission_date else None
                        month = submission_date.month if submission_date else None
                        yymm = f"{year % 100:02d}{month:02d}" if year and month else None
                        
                        # Build record tuple
                        record = (
                            arxiv_id, title, abstract, primary_category,
                            submission_date, update_date, year, month, yymm,
                            doi, license_info, journal_ref,
                            authors, categories, comments, report_number,
                            versions_count, latest_version, authors_parsed,
                            False, None, False, None  # PDF/LaTeX fields - set later
                        )
                        
                        batch.append(record)
                        
                        # Insert batch when full
                        if len(batch) >= batch_size:
                            self._insert_metadata_batch(conn, batch)
                            processed += len(batch)
                            batch = []
                            
                            if processed % 50000 == 0:
                                logger.info(f"Imported {processed:,} papers...")
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        self.stats['errors'] += 1
                    except Exception as e:
                        logger.error(f"Error processing line {line_num}: {e}")
                        self.stats['errors'] += 1
                
                # Insert final batch
                if batch:
                    self._insert_metadata_batch(conn, batch)
                    processed += len(batch)
                
                self.stats['metadata_imported'] = processed
                logger.info(f"Metadata import complete: {processed:,} papers imported")
                
        except Exception as e:
            logger.error(f"Fatal error during metadata import: {e}")
            return False
        
        finally:
            conn.close()
        
        return True
    
    def _insert_metadata_batch(self, conn, batch):
        """Insert a batch of metadata records."""
        insert_sql = """
            INSERT INTO papers (
                arxiv_id, title, abstract, primary_category,
                submission_date, update_date, year, month, yymm,
                doi, license, journal_ref,
                authors, categories, comments, report_number,
                versions_count, latest_version, authors_parsed,
                has_pdf, pdf_path, has_latex, latex_path
            ) VALUES %s
            ON CONFLICT (arxiv_id) DO UPDATE SET
                title = EXCLUDED.title,
                abstract = EXCLUDED.abstract,
                primary_category = EXCLUDED.primary_category,
                submission_date = EXCLUDED.submission_date,
                update_date = EXCLUDED.update_date,
                year = EXCLUDED.year,
                month = EXCLUDED.month,
                yymm = EXCLUDED.yymm,
                doi = EXCLUDED.doi,
                license = EXCLUDED.license,
                journal_ref = EXCLUDED.journal_ref,
                authors = EXCLUDED.authors,
                categories = EXCLUDED.categories,
                comments = EXCLUDED.comments,
                report_number = EXCLUDED.report_number,
                versions_count = EXCLUDED.versions_count,
                latest_version = EXCLUDED.latest_version,
                authors_parsed = EXCLUDED.authors_parsed
        """
        
        with conn:
            with conn.cursor() as cur:
                execute_values(cur, insert_sql, batch, template=None, page_size=100)
    
    def scan_and_update_pdfs(self):
        """Scan PDF directory and update database."""
        logger.info("Scanning PDFs and updating database...")
        
        if not self.pdf_dir.exists():
            logger.warning(f"PDF directory not found: {self.pdf_dir}")
            return False
        
        pdf_updates = []
        batch_size = 1000
        
        for year_month_dir in sorted(self.pdf_dir.iterdir()):
            if not year_month_dir.is_dir() or not year_month_dir.name.isdigit():
                continue
            
            year_month = year_month_dir.name
            logger.info(f"Scanning PDF directory: {year_month}")
            
            for pdf_file in year_month_dir.glob("*.pdf"):
                arxiv_id = pdf_file.stem.split('v')[0]  # Remove version
                pdf_path = str(pdf_file)
                file_size = pdf_file.stat().st_size
                
                pdf_updates.append((True, pdf_path, file_size, arxiv_id))
                
                if len(pdf_updates) >= batch_size:
                    self._update_pdf_batch(pdf_updates)
                    pdf_updates = []
        
        # Final batch
        if pdf_updates:
            self._update_pdf_batch(pdf_updates)
        
        logger.info(f"PDF scan complete: {self.stats['pdfs_found']:,} files processed")
        return True
    
    def _update_pdf_batch(self, pdf_updates):
        """Update PDF information for a batch."""
        conn = self.get_database_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    update_sql = """
                        UPDATE papers 
                        SET has_pdf = %s, pdf_path = %s, pdf_size_bytes = %s
                        WHERE arxiv_id = %s
                    """
                    cur.executemany(update_sql, pdf_updates)
                    self.stats['pdfs_found'] += len(pdf_updates)
                    
                    if self.stats['pdfs_found'] % 25000 == 0:
                        logger.info(f"Updated PDF info for {self.stats['pdfs_found']:,} papers")
        
        except Exception as e:
            logger.error(f"Error updating PDF batch: {e}")
            self.stats['errors'] += 1
        finally:
            conn.close()
    
    def parse_latex_filename(self, filename: str, directory: str) -> Optional[str]:
        """Parse LaTeX filename to extract ArXiv ID across all format changes."""
        
        if filename.endswith('.gz'):
            # LaTeX source available
            arxiv_id = filename[:-3]  # Remove .gz
            
            # Pre-2007: astro-ph0001001 -> astro-ph/0001.001
            if not arxiv_id[0].isdigit():
                # Extract category and numeric parts
                match = re.match(r'([a-z-]+)(\d{7})$', arxiv_id)
                if match:
                    category, numeric = match.groups()
                    yymm = numeric[:4]
                    nnn = numeric[4:]
                    # Convert to slash format: astro-ph/0001.001
                    arxiv_id = f"{category}/{yymm}.{nnn}"
            
            # 2007+: Already in correct format (YYMM.NNNN or YYMM.NNNNN)
            return arxiv_id
            
        elif filename.endswith('.pdf'):
            # No LaTeX available (ArXiv only has PDF)
            arxiv_id = filename[:-4]  # Remove .pdf
            
            # Same conversion logic for old format
            if not arxiv_id[0].isdigit():
                match = re.match(r'([a-z-]+)(\d{7})$', arxiv_id)
                if match:
                    category, numeric = match.groups()
                    yymm = numeric[:4]
                    nnn = numeric[4:]
                    arxiv_id = f"{category}/{yymm}.{nnn}"
            
            return arxiv_id
        
        return None

    def scan_and_update_latex(self):
        """Scan LaTeX directory and update database with proper format handling."""
        logger.info("Scanning LaTeX files and updating database...")
        
        if not self.latex_dir.exists():
            logger.warning(f"LaTeX directory not found: {self.latex_dir}")
            return False
        
        latex_updates = []
        signal_updates = []
        batch_size = 1000
        
        for year_month_dir in sorted(self.latex_dir.iterdir()):
            if not year_month_dir.is_dir():
                continue
            
            year_month = year_month_dir.name
            logger.info(f"Scanning LaTeX directory: {year_month}")
            
            # Process all files in directory
            for file in year_month_dir.iterdir():
                if not file.is_file():
                    continue
                
                filename = file.name
                arxiv_id = self.parse_latex_filename(filename, year_month)
                
                if not arxiv_id:
                    continue
                
                if filename.endswith('.gz'):
                    # LaTeX source available
                    latex_path = str(file)
                    latex_updates.append((True, latex_path, arxiv_id))
                    
                    if len(latex_updates) >= batch_size:
                        self._update_latex_batch(latex_updates, is_latex=True)
                        latex_updates = []
                
                elif filename.endswith('.pdf'):
                    # Signal file: no LaTeX available upstream
                    signal_updates.append((False, None, arxiv_id))
                    
                    if len(signal_updates) >= batch_size:
                        self._update_latex_batch(signal_updates, is_latex=False)
                        signal_updates = []
        
        # Final batches
        if latex_updates:
            self._update_latex_batch(latex_updates, is_latex=True)
        if signal_updates:
            self._update_latex_batch(signal_updates, is_latex=False)
        
        logger.info(f"LaTeX scan complete: {self.stats['latex_found']:,} LaTeX files, {self.stats['signal_files']:,} signal files")
        return True
    
    def _update_latex_batch(self, updates, is_latex: bool):
        """Update LaTeX information for a batch."""
        conn = self.get_database_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    update_sql = """
                        UPDATE papers 
                        SET has_latex = %s, latex_path = %s
                        WHERE arxiv_id = %s
                    """
                    cur.executemany(update_sql, updates)
                    
                    if is_latex:
                        self.stats['latex_found'] += len(updates)
                    else:
                        self.stats['signal_files'] += len(updates)
        
        except Exception as e:
            logger.error(f"Error updating LaTeX batch: {e}")
            self.stats['errors'] += 1
        finally:
            conn.close()
    
    def print_final_summary(self):
        """Print comprehensive rebuild summary."""
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time'] if self.stats['start_time'] else None
        
        print(f"\n{'='*80}")
        print("POSTGRESQL ARXIV DATABASE REBUILD COMPLETE")
        print(f"{'='*80}")
        print(f"Duration: {duration}")
        print(f"\nImport Results:")
        print(f"  Metadata imported: {self.stats['metadata_imported']:,} papers")
        print(f"  PDFs found: {self.stats['pdfs_found']:,} papers")
        print(f"  LaTeX found: {self.stats['latex_found']:,} papers")
        print(f"  Signal files (no LaTeX): {self.stats['signal_files']:,} papers")
        print(f"  Errors: {self.stats['errors']:,}")
        
        # Final database statistics
        try:
            conn = self.get_database_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM papers")
                    total = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM papers WHERE has_pdf = true")
                    with_pdf = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM papers WHERE has_latex = true")
                    with_latex = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM papers WHERE has_latex = false")
                    no_latex = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM papers WHERE categories LIKE '%cs.%'")
                    cs_papers = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM papers WHERE submission_date >= '2020-01-01'")
                    recent_papers = cur.fetchone()[0]
                    
                    print(f"\nFinal Database Status:")
                    print(f"  Total papers: {total:,}")
                    print(f"  Papers with PDF: {with_pdf:,} ({with_pdf/total*100:.1f}%)")
                    print(f"  Papers with LaTeX: {with_latex:,} ({with_latex/total*100:.1f}%)")
                    print(f"  Papers without LaTeX: {no_latex:,} ({no_latex/total*100:.1f}%)")
                    print(f"  Computer Science papers: {cs_papers:,} ({cs_papers/total*100:.1f}%)")
                    print(f"  Papers from 2020+: {recent_papers:,} ({recent_papers/total*100:.1f}%)")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting final stats: {e}")
    
    def run_complete_rebuild(self):
        """Run the complete rebuild process."""
        self.stats['start_time'] = datetime.now()
        logger.info("Starting complete PostgreSQL rebuild with full metadata...")
        
        success = True
        
        # Phase 1: Import complete metadata
        if not self.import_metadata_from_snapshot():
            logger.error("Metadata import failed")
            success = False
        
        # Phase 2: Scan PDFs
        if not self.scan_and_update_pdfs():
            logger.error("PDF scan failed")
            success = False
        
        # Phase 3: Scan LaTeX
        if not self.scan_and_update_latex():
            logger.error("LaTeX scan failed")
            success = False
        
        self.print_final_summary()
        return success


def main():
    if not os.getenv('PGPASSWORD'):
        print("❌ PGPASSWORD environment variable required")
        return 1
    
    rebuilder = PostgreSQLRebuilder()
    
    print("🔄 Starting complete PostgreSQL ArXiv rebuild...")
    print("   This includes all metadata fields from Kaggle dataset")
    print("   Estimated time: 45-90 minutes for 2.7M papers")
    print("   Monitor: tail -f /home/todd/olympus/HADES-Lab/tools/arxiv/logs/postgresql_rebuild_complete.log")
    
    if rebuilder.run_complete_rebuild():
        print("🎉 Complete rebuild successful!")
        return 0
    else:
        print("❌ Rebuild encountered errors - check logs")
        return 1


if __name__ == "__main__":
    exit(main())