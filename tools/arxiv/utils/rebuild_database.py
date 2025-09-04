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

# Setup logging with configurable path
def setup_logging():
    """Setup logging with configurable path"""
    # Use project-relative path or environment variable
    log_path = os.getenv('REBUILD_LOG_PATH')
    if not log_path:
        script_dir = Path(__file__).parent.resolve()
        log_dir = script_dir.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / 'postgresql_rebuild.log'
    
    # Ensure log directory exists
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return log_path

# Initialize logging
log_path = setup_logging()
logger = logging.getLogger(__name__)

class PostgreSQLRebuilder:
    def __init__(self):
        """
        Initialize the PostgreSQLRebuilder instance.
        
        Sets PostgreSQL connection configuration (reads password from the PGPASSWORD environment variable), default file system paths for the metadata JSONL snapshot, PDF archive, and LaTeX archive, and initializes run statistics counters (metadata_imported, pdfs_found, latex_found, signal_files, errors, start_time, end_time). Logs an initialization message.
        """
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
        """
        Return a new psycopg2 connection to the configured PostgreSQL database.
        
        Uses the rebuilder's stored pg_config to open the connection. The caller is
        responsible for closing the returned connection.
        """
        return psycopg2.connect(**self.pg_config)
    
    def parse_date(self, date_str: str) -> Optional[datetime.date]:
        """
        Parse an arXiv metadata date/time string and return a datetime.date or None.
        
        Accepts several common representations produced in arXiv metadata, including:
        - ISO dates: "YYYY-MM-DD"
        - RFC-style GMT timestamps: "Mon, 2 Apr 2007 19:18:42 GMT"
        - Some common variants such as "YYYY-MM-DD HH:MM:SS", "YYYY/MM/DD", and "MM/DD/YYYY"
        
        Parameters:
            date_str (str): Date/time string from metadata. Falsy values (None, empty string) return None.
        
        Returns:
            datetime.date | None: Parsed date on success; None if input is falsy or cannot be parsed.
        """
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
                    return datetime.strptime(date_str, fmt).date()
                except:
                    continue
        
        except Exception as e:
            logger.debug(f"Could not parse date: {date_str} - {e}")
        
        return None
    
    def extract_primary_category(self, categories: str) -> Optional[str]:
        """
        Return the first (primary) category token from a space-delimited categories string.
        
        If `categories` is falsy or contains no tokens, returns None. Example input: "cs.CL cs.AI".
        """
        if not categories:
            return None
        return categories.split()[0] if categories.split() else None
    
    def import_metadata_from_snapshot(self):
        """
        Import arXiv metadata from the JSONL snapshot (self.metadata_file) and upsert records into the papers table.
        
        Reads the line-delimited JSON snapshot, parses each paper, normalizes key fields (id, title, abstract, authors, categories, primary category, comments, journal_ref, doi, report_number, license), computes submission and update dates, derives version counts and latest version, serializes parsed authors, and computes partitioning fields (year, month, yymm) when submission date is available. Records are written to the database in batches via the internal bulk-upsert helper; PDF and LaTeX presence fields are left as placeholders for later phases.
        
        Side effects:
        - Upserts metadata rows into the papers table (calls self._insert_metadata_batch).
        - Updates self.stats['metadata_imported'] with the number of processed records and increments self.stats['errors'] for malformed lines or per-line processing failures.
        
        Returns:
            bool: True on successful completion; False if the metadata file is missing or a fatal error occurs during import.
        """
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
        """
        Insert or update a batch of paper metadata rows into the `papers` table.
        
        Performs a bulk upsert: inserts each record in `batch` and, on arxiv_id conflict,
        updates the existing row's metadata fields. The operation is executed inside a
        transaction using the provided database connection.
        
        Parameters:
            batch (Iterable[tuple]): Iterable of tuples whose values must match the
                following column order:
                (arxiv_id, title, abstract, primary_category,
                 submission_date, update_date, year, month, yymm,
                 doi, license, journal_ref,
                 authors, categories, comments, report_number,
                 versions_count, latest_version, authors_parsed,
                 has_pdf, pdf_path, has_latex, latex_path)
        
        Notes:
            - `conn` is expected to be a psycopg2 connection and is used as the
              transactional context; it is intentionally not documented as a parameter
              here because it represents the database client/service.
            - The function does not return a value; database errors propagate from the
              underlying DB driver.
        """
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
        """
        Scan the configured PDF directory tree and update the papers table with PDF presence, path, and size.
        
        Walks year-month subdirectories under self.pdf_dir, finds files ending with `.pdf`, derives the arXiv identifier by stripping version suffix (e.g. `1234.5678v2 -> 1234.5678`), and batches updates to the database via self._update_pdf_batch. Updates self.stats['pdfs_found'] as batches are applied.
        
        Returns:
            bool: True on successful scan and update, False if the configured PDF directory does not exist.
        """
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
                # Remove version suffix if present (e.g., '1234.5678v2' -> '1234.5678')
                arxiv_id = re.sub(r'v\d+$', '', pdf_file.stem)
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
        """
        Update a batch of papers' PDF presence and metadata in the database.
        
        Performs a batched UPDATE (via executemany) for each 4-tuple in pdf_updates, setting has_pdf, pdf_path,
        and pdf_size_bytes on the papers table matching arxiv_id.
        
        Parameters:
            pdf_updates (Iterable[tuple]): Iterable of 4-tuples in the order
                (has_pdf: bool, pdf_path: Optional[str], pdf_size_bytes: Optional[int], arxiv_id: str).
                Each tuple updates the corresponding paper identified by arxiv_id.
        
        Side effects:
            - Increments self.stats['pdfs_found'] by the number of updates attempted.
            - On exception, increments self.stats['errors'] and logs the error; exceptions are not re-raised.
            - Always closes the database connection before returning.
        """
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
        """
        Normalize and extract an arXiv identifier from a LaTeX-related filename.
        
        Parses filenames that end with ".gz" (LaTeX source) or ".pdf" (signal file) and returns a normalized arXiv ID suitable for database lookup. Recognizes both legacy pre-2007 concatenated category form (e.g., "astro-ph0001001") and modern forms (e.g., "YYMM.NNNN" or "category/YYMM.NNNN"). Legacy names are converted to the canonical "category/YYMM.NNN" form (e.g., "astro-ph0001001" -> "astro-ph/0001.001"). Filenames that do not end with ".gz" or ".pdf" or that cannot be parsed return None.
        
        Parameters:
            filename: Basename of the file (including extension) to parse.
            directory: Containing directory path (provided for context; this function does not use it for parsing).
        
        Returns:
            The normalized arXiv identifier string on success, or None if the filename cannot be interpreted as an arXiv ID.
        """
        
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
        """
        Scan the configured LaTeX data directory and update papers' LaTeX presence in the database.
        
        Scans year-month subdirectories under self.latex_dir and normalizes arXiv IDs using parse_latex_filename. Files ending in `.gz` are treated as LaTeX sources (sets has_latex=True and stores latex_path); files ending in `.pdf` are treated as signal files indicating no LaTeX source (sets has_latex=False). Updates are applied in batches via self._update_latex_batch and the method relies on those calls to increment self.stats counters ('latex_found' and 'signal_files').
        
        Returns:
            bool: True on successful completion; False if the configured LaTeX directory does not exist.
        """
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
        """
        Apply a batch of LaTeX presence/path updates to the papers table.
        
        Each update row must be a tuple (has_latex: bool, latex_path: Optional[str], arxiv_id: str).
        When is_latex is True the batch represents actual LaTeX source files (has_latex=True, latex_path set);
        when False the batch represents PDF-only signal files (has_latex=False, latex_path should be None).
        
        Parameters:
            updates (Iterable[tuple]): Iterable of (has_latex, latex_path, arxiv_id) tuples to apply.
            is_latex (bool): True if this batch contains real LaTeX sources (increments latex_found);
                             False if this batch contains signal files (increments signal_files).
        
        Side effects:
            - Updates `has_latex` and `latex_path` in the `papers` table for each arxiv_id in the batch.
            - Increments self.stats['latex_found'] or self.stats['signal_files'] by the number of updates.
            - On failure increments self.stats['errors'].
        """
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
        """
        Print a concise end-of-run summary for the rebuild and record its end time.
        
        Updates self.stats['end_time'], computes total duration from self.stats['start_time'], prints a human-readable summary of counts collected during the run (metadata imported, PDFs found, LaTeX found, signal files, errors), and queries the database for final aggregate counts (total papers, papers with PDF, with/without LaTeX, computer-science papers, and papers from 2020 onward) which it prints with percentages.
        
        Side effects:
        - Mutates self.stats by setting 'end_time'.
        - Writes summary output to stdout.
        - Opens a database connection to read final aggregate statistics; logs an error on exception but does not raise it.
        """
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
                    
                    # Guard against division by zero
                    if total > 0:
                        print(f"  Papers with PDF: {with_pdf:,} ({with_pdf/total*100:.1f}%)")
                        print(f"  Papers with LaTeX: {with_latex:,} ({with_latex/total*100:.1f}%)")
                        print(f"  Papers without LaTeX: {no_latex:,} ({no_latex/total*100:.1f}%)")
                        print(f"  Computer Science papers: {cs_papers:,} ({cs_papers/total*100:.1f}%)")
                        print(f"  Papers from 2020+: {recent_papers:,} ({recent_papers/total*100:.1f}%)")
                    else:
                        print(f"  Papers with PDF: {with_pdf:,} (N/A)")
                        print(f"  Papers with LaTeX: {with_latex:,} (N/A)")
                        print(f"  Papers without LaTeX: {no_latex:,} (N/A)")
                        print(f"  Computer Science papers: {cs_papers:,} (N/A)")
                        print(f"  Papers from 2020+: {recent_papers:,} (N/A)")
            
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
    """
    Entry point for the PostgreSQL ArXiv rebuild CLI.
    
    Checks that the PGPASSWORD environment variable is set, constructs a PostgreSQLRebuilder, prints startup guidance, runs the full rebuild (metadata import, PDF scan, LaTeX scan), and prints a final success/failure message.
    
    Returns:
        int: Process exit code — 0 on success, 1 on failure or when required environment is missing.
    """
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