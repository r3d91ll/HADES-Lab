#!/usr/bin/env python3
"""
PostgreSQL ArXiv Database Rebuild Script
========================================

Comprehensive rebuild of PostgreSQL arxiv database in proper order:
1. Import all metadata from arxiv-metadata-oai-snapshot.json (2.7M papers)
2. Scan local PDFs and update has_pdf/pdf_path flags
3. Scan local LaTeX and update has_latex/latex_path flags

Run this script in a separate terminal window as it will take time to complete.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

# Import PDF scanner for consistent ArXiv ID extraction
try:
    # Absolute import per guidelines
    from tools.arxiv.scripts.pdf_scanner import PDFScanner

    _id_extractor = PDFScanner().extract_arxiv_id_from_filename
except Exception:
    # Fallback: local extractor kept in sync with PDFScanner
    def _id_extractor(filename: str) -> str:
        """
        Extract an arXiv identifier from a PDF filename (fallback implementation).
        
        This removes a trailing ".pdf" and any version suffix like "v2". It then:
        - Converts legacy category-style names of the form "category-subcat-1234" or
          "category-1234" into "category/1234" when there are at least two hyphen-separated
          parts and the last part looks like the paper number.
        - Accepts modern numeric IDs matching "YYYY.NNNN" or "YYYY.NNNNN" and returns them unchanged.
        - For any other form, returns the cleaned base filename.
        
        Parameters:
            filename (str): The PDF filename (may include the ".pdf" extension and version suffix).
        
        Returns:
            str: The extracted arXiv identifier or the cleaned filename when no pattern matches.
        """
        base_name = filename.replace(".pdf", "")
        base_name = re.sub(r"v\d+$", "", base_name)

        if re.match(r"^[a-zA-Z-]+-\d+", base_name):
            parts = base_name.split("-")
            if len(parts) >= 3:
                category = "-".join(parts[:-1])
                paper_num = parts[-1]
                return f"{category}/{paper_num}"

        if re.match(r"^\d{4}\.\d{4,5}$", base_name):
            return base_name

        return base_name


def get_log_file_path() -> Path:
    """
    Return the absolute path to the rebuild log file.
    
    Looks for the ARXIV_POSTGRESQL_REBUILD_LOG environment variable and, if present, uses its value (expanded and resolved). If the environment variable is not set, returns the default ../logs/postgresql_rebuild.log path relative to the script. Ensures the log file's parent directory exists before returning.
    
    Returns:
        pathlib.Path: Resolved path to the log file.
    """
    # Check environment variable first
    env_log_path = os.getenv("ARXIV_POSTGRESQL_REBUILD_LOG")
    if env_log_path:
        log_path = Path(env_log_path).expanduser().resolve()
    else:
        # Default to relative path from script location
        script_dir = Path(__file__).parent
        log_path = script_dir / "../logs/postgresql_rebuild.log"
        log_path = log_path.resolve()

    # Ensure log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    return log_path


# Get configured log path
LOG_FILE_PATH = get_log_file_path()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class RebuildStats:
    metadata_imported: int = 0
    pdfs_found: int = 0
    latex_found: int = 0
    signal_files_found: int = 0
    errors: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None


class PostgreSQLRebuilder:
    def __init__(self):
        """
        Initialize the PostgreSQLRebuilder.
        
        Sets up database connection configuration (reads PGPASSWORD from the environment), initializes rebuild statistics, and records default absolute paths for the metadata file, PDF directory, and LaTeX directory. Logs the configured paths.
        """
        self.pg_config = {
            "host": "localhost",
            "database": "arxiv",
            "user": "postgres",
            "password": os.getenv("PGPASSWORD", ""),
        }
        self.stats = RebuildStats()

        # File paths
        self.metadata_file = Path("/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json")
        self.pdf_dir = Path("/bulk-store/arxiv-data/pdf")
        self.latex_dir = Path("/bulk-store/arxiv-data/latex")

        logger.info("PostgreSQL Rebuilder initialized")
        logger.info(f"Metadata file: {self.metadata_file}")
        logger.info(f"PDF directory: {self.pdf_dir}")
        logger.info(f"LaTeX directory: {self.latex_dir}")

    def get_database_connection(self):
        """
        Return an active psycopg2 PostgreSQL connection using the instance's pg_config.
        
        Raises:
            ConnectionError: If establishing the connection fails (wraps psycopg2 errors).
        """
        try:
            return psycopg2.connect(connect_timeout=10, **self.pg_config)
        except psycopg2.OperationalError as e:
            logger.error(
                f"PostgreSQL connection failed - database: {self.pg_config['database']}, host: {self.pg_config['host']}, user: {self.pg_config['user']}"
            )
            logger.error(f"OperationalError: {e}")
            raise ConnectionError(f"Failed to connect to PostgreSQL database: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected database connection error: {e}")
            raise ConnectionError(f"Database connection failed: {e}") from e

    def import_metadata_from_snapshot(self):
        """
        Import paper metadata from the arxiv-metadata-oai-snapshot.json file into the PostgreSQL `papers` table.
        
        Reads the snapshot file line-by-line, parses each JSON record into the expected paper fields (arXiv id, title, abstract, authors, categories, doi, journal-ref, comments, submission and update dates) and performs batched upserts into the database. Updates the instance RebuildStats (metadata_imported and errors) as it processes records.
        
        Returns:
            bool: True if the import completed and the transaction committed successfully; False if the metadata file is missing or a database/processing error occurred (the operation is rolled back on failure).
        """
        logger.info("Phase 1: Starting metadata import from snapshot...")

        if not self.metadata_file.exists():
            logger.error(f"Metadata file not found: {self.metadata_file}")
            return False

        conn = self.get_database_connection()
        batch_size = 1000
        batch = []

        try:
            with conn:
                with conn.cursor() as cur:
                    # Read and process the metadata file line by line
                    with open(self.metadata_file) as f:
                        for line_num, line in enumerate(f, 1):
                            if line_num % 50000 == 0:
                                logger.info(f"Processing line {line_num:,}...")

                            try:
                                paper = json.loads(line.strip())

                                # Extract key fields
                                arxiv_id = paper.get("id", "")
                                title = paper.get("title", "").replace("\n", " ").strip()
                                abstract = paper.get("abstract", "").replace("\n", " ").strip()
                                authors = paper.get("authors", "")
                                categories = paper.get("categories", "")
                                doi = paper.get("doi", None)
                                journal_ref = paper.get("journal-ref", None)
                                comments = paper.get("comments", None)

                                # Parse dates
                                submitted = paper.get("versions", [{}])[0].get("created", None)
                                updated = paper.get("update_date", submitted)

                                # Parse submission date
                                submission_date = None
                                update_date = None
                                if submitted:
                                    try:
                                        submission_date = datetime.strptime(submitted[:10], "%Y-%m-%d").date()
                                    except ValueError:
                                        pass

                                if updated:
                                    try:
                                        update_date = datetime.strptime(updated[:10], "%Y-%m-%d").date()
                                    except ValueError:
                                        pass

                                # Add to batch
                                batch.append(
                                    (
                                        arxiv_id,
                                        title,
                                        abstract,
                                        authors,
                                        categories,
                                        submission_date,
                                        update_date,
                                        doi,
                                        journal_ref,
                                        comments,
                                        False,
                                        False,
                                        None,
                                        None,  # has_pdf, has_latex, pdf_path, latex_path
                                    )
                                )

                                # Insert batch when full
                                if len(batch) >= batch_size:
                                    self._insert_paper_batch(cur, batch)
                                    self.stats.metadata_imported += len(batch)
                                    batch = []

                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON decode error at line {line_num}: {e}")
                                self.stats.errors += 1
                            except Exception as e:
                                logger.error(f"Error processing line {line_num}: {e}")
                                self.stats.errors += 1

                    # Insert final batch
                    if batch:
                        self._insert_paper_batch(cur, batch)
                        self.stats.metadata_imported += len(batch)

                    # Commit all changes
                    conn.commit()

        except Exception as e:
            logger.error(f"Database error during metadata import: {e}")
            conn.rollback()
            return False

        finally:
            conn.close()

        logger.info(f"Phase 1 complete: {self.stats.metadata_imported:,} papers imported")
        return True

    def _insert_paper_batch(self, cursor, batch):
        """
        Insert or update a batch of paper records into the `papers` table.
        
        Each item in `batch` should be an iterable/tuple of values in the following order:
        (arxiv_id, title, abstract, authors, categories, submission_date, update_date,
        doi, journal_ref, comments, has_pdf, has_latex, pdf_path, latex_path).
        
        Performs a bulk upsert: new rows are inserted and existing rows (matching arxiv_id)
        are updated for the textual/metadata fields (title, abstract, authors, categories,
        submission_date, update_date, doi, journal_ref, comments).
        """
        insert_sql = """
            INSERT INTO papers (
                arxiv_id, title, abstract, authors, categories,
                submission_date, update_date, doi, journal_ref, comments,
                has_pdf, has_latex, pdf_path, latex_path
            ) VALUES %s
            ON CONFLICT (arxiv_id) DO UPDATE SET
                title = EXCLUDED.title,
                abstract = EXCLUDED.abstract,
                authors = EXCLUDED.authors,
                categories = EXCLUDED.categories,
                submission_date = EXCLUDED.submission_date,
                update_date = EXCLUDED.update_date,
                doi = EXCLUDED.doi,
                journal_ref = EXCLUDED.journal_ref,
                comments = EXCLUDED.comments
        """

        execute_values(cursor, insert_sql, batch, template=None, page_size=100)

    def scan_and_update_pdfs(self):
        """
        Scan the local PDF archive, set has_pdf and pdf_path for found papers, and apply updates to the database in batches.
        
        This method walks numeric year-month subdirectories under self.pdf_dir, derives an arXiv identifier for each PDF using the module's ID extractor, and updates corresponding rows (has_pdf=True, pdf_path) in the database via batched calls to self._update_pdf_batch. Progress is accumulated in self.stats.pdfs_found.
        
        Returns:
            bool: True on successful completion; False if the configured PDF directory does not exist.
        """
        logger.info("Phase 2: Scanning local PDFs and updating database...")

        if not self.pdf_dir.exists():
            logger.error(f"PDF directory not found: {self.pdf_dir}")
            return False

        pdf_updates = []
        batch_size = 1000

        # Scan PDF directory structure
        for year_month_dir in self.pdf_dir.iterdir():
            if not year_month_dir.is_dir() or not year_month_dir.name.isdigit():
                continue

            year_month = year_month_dir.name
            logger.info(f"Scanning PDF directory: {year_month}")

            for pdf_file in year_month_dir.glob("*.pdf"):
                # Extract ArXiv ID from filename using consistent method
                arxiv_id = _id_extractor(pdf_file.name)
                pdf_path = str(pdf_file)

                pdf_updates.append((True, pdf_path, arxiv_id))

                # Process batch when full
                if len(pdf_updates) >= batch_size:
                    self._update_pdf_batch(pdf_updates)
                    pdf_updates = []

        # Process final batch
        if pdf_updates:
            self._update_pdf_batch(pdf_updates)

        logger.info(f"Phase 2 complete: {self.stats.pdfs_found:,} PDF files processed")
        return True

    def _update_pdf_batch(self, pdf_updates):
        """
        Update database rows for a batch of PDF presence/path updates.
        
        This performs a transactional UPDATE on the papers table for each entry in pdf_updates,
        increments the internal pdfs_found counter by the batch size, and closes the database
        connection when finished. Errors during execution are caught; on failure the method
        increments the internal error counter and does not re-raise.
        
        Parameters:
            pdf_updates (Iterable[tuple]): An iterable of 3-tuples matching the SQL parameters:
                (has_pdf: bool, pdf_path: str | None, arxiv_id: str).
        """
        conn = self.get_database_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    update_sql = """
                        UPDATE papers
                        SET has_pdf = %s, pdf_path = %s
                        WHERE arxiv_id = %s
                    """
                    cur.executemany(update_sql, pdf_updates)
                    self.stats.pdfs_found += len(pdf_updates)

                    if self.stats.pdfs_found % 10000 == 0:
                        logger.info(f"Updated PDF flags for {self.stats.pdfs_found:,} papers")

        except Exception as e:
            logger.error(f"Error updating PDF batch: {e}")
            self.stats.errors += 1

        finally:
            conn.close()

    def scan_and_update_latex(self):
        """
        Scan the local LaTeX archive tree and update papers' has_latex/latex_path flags in the database.
        
        Traverses year-month subdirectories (directory names composed of digits) under self.latex_dir. For each
        *.tar.gz file it treats the archive as available LaTeX for the corresponding arXiv ID (sets has_latex=True
        and records latex_path). For each *.pdf file found in these directories it treats the file as a signal that
        no LaTeX is available for that paper (sets has_latex=False and clears latex_path). ArXiv IDs are derived
        from the archive filename (strip version suffix) and from PDFs using the module's _id_extractor.
        
        Updates are applied in batches via self._update_latex_batch(latex_updates, is_latex), which performs the
        database writes and updates the RebuildStats counters (latex_found or signal_files_found). Returns False
        immediately if the configured latex_dir does not exist; otherwise returns True on completion.
        """
        logger.info("Phase 3: Scanning local LaTeX files and updating database...")

        if not self.latex_dir.exists():
            logger.error(f"LaTeX directory not found: {self.latex_dir}")
            return False

        latex_updates = []
        signal_updates = []
        batch_size = 1000

        # Scan LaTeX directory structure
        for year_month_dir in self.latex_dir.iterdir():
            if not year_month_dir.is_dir() or not year_month_dir.name.isdigit():
                continue

            year_month = year_month_dir.name
            logger.info(f"Scanning LaTeX directory: {year_month}")

            # Check for LaTeX archives (.tar.gz)
            for latex_file in year_month_dir.glob("*.tar.gz"):
                arxiv_id = latex_file.stem.replace(".tar", "").split("v")[0]
                latex_path = str(latex_file)
                latex_updates.append((True, latex_path, arxiv_id))

                if len(latex_updates) >= batch_size:
                    self._update_latex_batch(latex_updates, is_latex=True)
                    latex_updates = []

            # Check for signal files (.pdf in LaTeX directory = no LaTeX available)
            for signal_file in year_month_dir.glob("*.pdf"):
                arxiv_id = _id_extractor(signal_file.name)
                signal_updates.append((False, None, arxiv_id))

                if len(signal_updates) >= batch_size:
                    self._update_latex_batch(signal_updates, is_latex=False)
                    signal_updates = []

        # Process final batches
        if latex_updates:
            self._update_latex_batch(latex_updates, is_latex=True)
        if signal_updates:
            self._update_latex_batch(signal_updates, is_latex=False)

        logger.info(
            f"Phase 3 complete: {self.stats.latex_found:,} LaTeX files, {self.stats.signal_files_found:,} signal files"
        )
        return True

    def _update_latex_batch(self, latex_updates, is_latex: bool):
        """
        Update the database in a single transaction for a batch of LaTeX-related records.
        
        Parameters:
            latex_updates (Sequence[Tuple[bool, Optional[str], str]]): Sequence of parameter tuples for executemany(),
                each tuple is (has_latex, latex_path, arxiv_id).
            is_latex (bool): True when the batch represents found LaTeX archives (increments latex_found);
                False when the batch represents signal files indicating absence of LaTeX (increments signal_files_found).
        
        Side effects:
            - Executes UPDATE statements against the papers table to set has_latex and latex_path.
            - Increments the appropriate counter on self.stats.
            - Logs errors and increments self.stats.errors on failure.
            - Closes the database connection before returning.
        
        Returns:
            None
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
                    cur.executemany(update_sql, latex_updates)

                    if is_latex:
                        self.stats.latex_found += len(latex_updates)
                    else:
                        self.stats.signal_files_found += len(latex_updates)

        except Exception as e:
            logger.error(f"Error updating LaTeX batch: {e}")
            self.stats.errors += 1

        finally:
            conn.close()

    def print_final_summary(self):
        """
        Print a concise final summary of the rebuild and query final database statistics.
        
        Sets self.stats.end_time (and computes duration), prints a human-readable summary of
        counts collected during the run (metadata imported, PDFs/LaTeX found, signal files, errors),
        then queries the database for total paper counts and percentages for papers with PDF,
        with LaTeX, and confirmed with no LaTeX, printing those results. Any exceptions raised
        while querying the database are logged; the method does not raise on query failure.
        """
        self.stats.end_time = datetime.now()
        duration = self.stats.end_time - self.stats.start_time if self.stats.start_time else None

        print(f"\n{'='*80}")
        print("POSTGRESQL ARXIV DATABASE REBUILD COMPLETE")
        print(f"{'='*80}")
        print(f"Start time: {self.stats.start_time}")
        print(f"End time: {self.stats.end_time}")
        if duration:
            print(f"Total duration: {duration}")
        print("\nResults:")
        print(f"  Metadata imported: {self.stats.metadata_imported:,} papers")
        print(f"  PDFs found: {self.stats.pdfs_found:,} papers")
        print(f"  LaTeX found: {self.stats.latex_found:,} papers")
        print(f"  Signal files (no LaTeX): {self.stats.signal_files_found:,} papers")
        print(f"  Errors: {self.stats.errors:,}")

        # Query final database stats
        try:
            conn = self.get_database_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM papers")
                    total_papers = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) FROM papers WHERE has_pdf = true")
                    papers_with_pdf = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) FROM papers WHERE has_latex = true")
                    papers_with_latex = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) FROM papers WHERE has_latex = false")
                    papers_no_latex = cur.fetchone()[0]

                    print("\nFinal Database Status:")
                    print(f"  Total papers in database: {total_papers:,}")

                    # Avoid division by zero
                    if total_papers > 0:
                        pdf_percentage = papers_with_pdf / total_papers * 100
                        latex_percentage = papers_with_latex / total_papers * 100
                        no_latex_percentage = papers_no_latex / total_papers * 100
                    else:
                        pdf_percentage = latex_percentage = no_latex_percentage = 0.0

                    print(f"  Papers with PDF: {papers_with_pdf:,} ({pdf_percentage:.1f}%)")
                    print(f"  Papers with LaTeX: {papers_with_latex:,} ({latex_percentage:.1f}%)")
                    print(f"  Papers confirmed no LaTeX: {papers_no_latex:,} ({no_latex_percentage:.1f}%)")

            conn.close()

        except Exception as e:
            logger.error(f"Error querying final stats: {e}")

    def run_complete_rebuild(self):
        """
        Orchestrates the full three-phase PostgreSQL rebuild and returns success status.
        
        Runs in order:
        1. Import metadata from the snapshot (phase 1) — aborts the rebuild and returns False if this phase fails.
        2. Scan and update PDFs (phase 2) — logs an error on failure but continues to phase 3.
        3. Scan and update LaTeX sources and signal files (phase 3).
        
        Side effects:
        - Updates the database via the phase methods.
        - Updates self.stats.start_time (set before running) and final statistics via print_final_summary().
        - Emits log messages describing progress and errors.
        
        Returns:
            bool: True if the rebuild completed (phase 1 succeeded and phases 2/3 were attempted), False if phase 1 failed and the rebuild was aborted.
        """
        self.stats.start_time = datetime.now()
        logger.info("Starting complete PostgreSQL ArXiv database rebuild...")

        # Phase 1: Import metadata
        if not self.import_metadata_from_snapshot():
            logger.error("Phase 1 failed - aborting rebuild")
            return False

        # Phase 2: Scan PDFs
        if not self.scan_and_update_pdfs():
            logger.error("Phase 2 failed - continuing with Phase 3")

        # Phase 3: Scan LaTeX
        if not self.scan_and_update_latex():
            logger.error("Phase 3 failed")

        # Final summary
        self.print_final_summary()

        logger.info("PostgreSQL rebuild completed!")
        return True


def main():
    # Verify environment variables
    """
    Run the full three-phase PostgreSQL ArXiv database rebuild from the command line.
    
    Performs environment validation, constructs a PostgreSQLRebuilder, and runs the end-to-end
    rebuild (metadata import, PDF scan/update, LaTeX scan/update). Prints short progress
    hints to stdout and returns an exit code suitable for use from a shell.
    
    Returns:
        int: 0 on success; 1 on failure or if required environment variables are missing.
    """
    if not os.getenv("PGPASSWORD"):
        print("❌ PGPASSWORD environment variable is required")
        print("   Set it with: export PGPASSWORD='your-password'")
        return 1

    # Create rebuilder and run
    rebuilder = PostgreSQLRebuilder()

    print("🔄 Starting PostgreSQL ArXiv database rebuild...")
    print("   This will take significant time (30-60 minutes)")
    print(f"   Progress will be logged to: {LOG_FILE_PATH}")
    print(f"   You can monitor progress with: tail -f {LOG_FILE_PATH}")

    success = rebuilder.run_complete_rebuild()

    if success:
        print("\n🎉 PostgreSQL rebuild completed successfully!")
        print("   You can now create targeted paper lists from the database")
        return 0
    else:
        print("\n❌ PostgreSQL rebuild encountered errors")
        print("   Check the log file for details")
        return 1


if __name__ == "__main__":
    exit(main())
