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
        """Extract ArXiv ID from PDF filename (fallback implementation)."""
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
    """Get the log file path from environment variable or default location."""
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
        """Get PostgreSQL database connection with error handling."""
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
        Phase 1: Import all metadata from arxiv-metadata-oai-snapshot.json
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
        """Insert a batch of papers into PostgreSQL."""
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
        Phase 2: Scan local PDF directory and update has_pdf/pdf_path flags
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
        """Update PDF flags for a batch of papers."""
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
        Phase 3: Scan local LaTeX directory and update has_latex/latex_path flags
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
        """Update LaTeX flags for a batch of papers."""
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
        """Print final rebuild summary."""
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
        """Run the complete PostgreSQL rebuild process."""
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
