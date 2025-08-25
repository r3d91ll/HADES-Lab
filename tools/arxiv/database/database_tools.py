#!/usr/bin/env python3
"""
Database Tools Module for ArXiv Data Lake
==========================================

Consolidated production utilities for database operations.
Combines useful functions from coverage_analyzer, db_updater, and tar_processor.

Following Actor-Network Theory: These tools act as translators between
the PostgreSQL actant and various analysis/update operations, maintaining
the integrity of our information network.
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import psycopg2
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatabaseAnalyzer:
    """
    Analyzes coverage and health of the PostgreSQL data lake.
    
    In Information Reconstructionism terms, this measures the completeness
    of our information space - identifying "voids" where WHERE exists but
    WHAT is missing, preventing information from manifesting.
    """
    
    def __init__(self, db_connection):
        """Initialize with database connection."""
        self.conn = db_connection
        self.cur = self.conn.cursor()
    
    def get_coverage_report(self) -> Dict:
        """Generate comprehensive coverage report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall': self._get_overall_coverage(),
            'by_year': self._get_coverage_by_year(),
            'by_category': self._get_coverage_by_category(),
            'experiment_window': self._get_experiment_coverage(),
            'file_statistics': self._get_file_statistics()
        }
        return report
    
    def _get_overall_coverage(self) -> Dict:
        """Get overall coverage statistics."""
        self.cur.execute("""
            SELECT 
                COUNT(*) as total_papers,
                COUNT(pdf_path) as papers_with_pdf,
                COUNT(latex_path) as papers_with_latex,
                COUNT(*) FILTER (WHERE has_pdf = TRUE AND has_latex_source = TRUE) as both,
                COUNT(*) FILTER (WHERE has_pdf = TRUE OR has_latex_source = TRUE) as any_source,
                AVG(pdf_size_mb) FILTER (WHERE pdf_size_mb IS NOT NULL) as avg_pdf_size
            FROM arxiv_papers
        """)
        
        stats = self.cur.fetchone()
        total = stats[0] if stats[0] else 1
        
        return {
            'total_papers': stats[0],
            'papers_with_pdf': stats[1],
            'papers_with_latex': stats[2],
            'papers_with_both': stats[3],
            'papers_with_any': stats[4],
            'avg_pdf_size_mb': float(stats[5]) if stats[5] else 0,
            'pdf_coverage_percent': (stats[1] / total) * 100,
            'latex_coverage_percent': (stats[2] / total) * 100,
            'both_coverage_percent': (stats[3] / total) * 100,
            'any_coverage_percent': (stats[4] / total) * 100
        }
    
    def _get_coverage_by_year(self) -> List[Dict]:
        """Get coverage statistics by year."""
        self.cur.execute("""
            SELECT 
                EXTRACT(YEAR FROM v.created_date) as year,
                COUNT(DISTINCT p.id) as total_papers,
                COUNT(DISTINCT p.id) FILTER (WHERE p.has_pdf = TRUE) as with_pdf,
                COUNT(DISTINCT p.id) FILTER (WHERE p.has_latex_source = TRUE) as with_latex
            FROM arxiv_papers p
            JOIN arxiv_versions v ON p.id = v.paper_id AND v.version = 'v1'
            WHERE v.created_date IS NOT NULL
            GROUP BY year
            ORDER BY year DESC
            LIMIT 20
        """)
        
        results = []
        for row in self.cur.fetchall():
            year, total, pdf, latex = row
            results.append({
                'year': int(year) if year else None,
                'total_papers': total,
                'papers_with_pdf': pdf,
                'papers_with_latex': latex,
                'pdf_coverage_percent': (pdf / total * 100) if total else 0,
                'latex_coverage_percent': (latex / total * 100) if total else 0
            })
        
        return results
    
    def _get_coverage_by_category(self) -> List[Dict]:
        """Get coverage statistics by primary category."""
        self.cur.execute("""
            SELECT 
                split_part(categories, ' ', 1) as primary_category,
                COUNT(*) as total_papers,
                COUNT(*) FILTER (WHERE has_pdf = TRUE) as with_pdf,
                COUNT(*) FILTER (WHERE has_latex_source = TRUE) as with_latex
            FROM arxiv_papers
            GROUP BY primary_category
            ORDER BY total_papers DESC
            LIMIT 30
        """)
        
        results = []
        for row in self.cur.fetchall():
            cat, total, pdf, latex = row
            results.append({
                'category': cat,
                'total_papers': total,
                'papers_with_pdf': pdf,
                'papers_with_latex': latex,
                'pdf_coverage_percent': (pdf / total * 100) if total else 0,
                'latex_coverage_percent': (latex / total * 100) if total else 0
            })
        
        return results
    
    def _get_experiment_coverage(self) -> Dict:
        """Get coverage for experiment window (Dec 2012 - Aug 2016)."""
        self.cur.execute("""
            SELECT 
                COUNT(DISTINCT p.id) as total_papers,
                COUNT(DISTINCT p.id) FILTER (WHERE p.has_pdf = TRUE) as with_pdf,
                COUNT(DISTINCT p.id) FILTER (WHERE p.has_latex_source = TRUE) as with_latex,
                COUNT(DISTINCT p.id) FILTER (WHERE p.embeddings_created = TRUE) as with_embeddings
            FROM arxiv_papers p
            JOIN arxiv_versions v ON p.id = v.paper_id
            WHERE v.created_date >= '2012-12-01' AND v.created_date <= '2016-08-31'
        """)
        
        stats = self.cur.fetchone()
        total = stats[0] if stats[0] else 1
        
        return {
            'total_papers': stats[0],
            'papers_with_pdf': stats[1],
            'papers_with_latex': stats[2],
            'papers_with_embeddings': stats[3],
            'pdf_coverage_percent': (stats[1] / total) * 100,
            'latex_coverage_percent': (stats[2] / total) * 100,
            'embeddings_coverage_percent': (stats[3] / total) * 100
        }
    
    def _get_file_statistics(self) -> Dict:
        """Get file size and path statistics."""
        self.cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE pdf_size_mb < 1) as small_pdfs,
                COUNT(*) FILTER (WHERE pdf_size_mb >= 1 AND pdf_size_mb < 5) as medium_pdfs,
                COUNT(*) FILTER (WHERE pdf_size_mb >= 5 AND pdf_size_mb < 10) as large_pdfs,
                COUNT(*) FILTER (WHERE pdf_size_mb >= 10) as huge_pdfs,
                SUM(pdf_size_mb) as total_pdf_size_mb,
                MIN(pdf_size_mb) as min_pdf_size,
                MAX(pdf_size_mb) as max_pdf_size
            FROM arxiv_papers
            WHERE pdf_size_mb IS NOT NULL
        """)
        
        stats = self.cur.fetchone()
        
        return {
            'small_pdfs_under_1mb': stats[0] or 0,
            'medium_pdfs_1_5mb': stats[1] or 0,
            'large_pdfs_5_10mb': stats[2] or 0,
            'huge_pdfs_over_10mb': stats[3] or 0,
            'total_pdf_size_gb': (stats[4] / 1024) if stats[4] else 0,
            'min_pdf_size_mb': stats[5] or 0,
            'max_pdf_size_mb': stats[6] or 0
        }


class TarExtractor:
    """
    Handles extraction of new ArXiv tar files with checkpointing.
    
    Acts as an obligatory passage point (ANT) for new data entering
    our system from external sources.
    """
    
    def __init__(self,
                 pdf_tar_dir: str = "/bulk-store/arxiv-data/tars/pdfs",
                 latex_tar_dir: str = "/bulk-store/arxiv-data/tars/latex",
                 pdf_extract_dir: str = "/bulk-store/arxiv-data/pdf",
                 latex_extract_dir: str = "/bulk-store/arxiv-data/latex",
                 checkpoint_file: str = "/bulk-store/arxiv-data/.processed_tars.json"):
        """Initialize with configurable paths."""
        self.pdf_tar_dir = Path(pdf_tar_dir)
        self.latex_tar_dir = Path(latex_tar_dir)
        self.pdf_extract_dir = Path(pdf_extract_dir)
        self.latex_extract_dir = Path(latex_extract_dir)
        self.checkpoint_file = Path(checkpoint_file)
        self.processed_tars = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Set[str]:
        """Load checkpoint of processed tar files."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return set()
    
    def _save_checkpoint(self):
        """Save checkpoint of processed tar files."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(sorted(list(self.processed_tars)), f, indent=2)
        except Exception as e:
            logger.error(f"Could not save checkpoint: {e}")
    
    def find_new_tars(self) -> Tuple[List[Path], List[Path]]:
        """Find tar files that haven't been processed yet."""
        new_pdf_tars = []
        new_latex_tars = []
        
        # Check PDF tars
        if self.pdf_tar_dir.exists():
            for tar_file in sorted(self.pdf_tar_dir.glob("*.tar")):
                if tar_file.name not in self.processed_tars:
                    new_pdf_tars.append(tar_file)
        
        # Check LaTeX tars
        if self.latex_tar_dir.exists():
            for tar_file in sorted(self.latex_tar_dir.glob("*.tar")):
                if tar_file.name not in self.processed_tars:
                    new_latex_tars.append(tar_file)
        
        return new_pdf_tars, new_latex_tars
    
    def extract_tar(self, tar_path: Path, extract_dir: Path) -> bool:
        """Extract a single tar file."""
        try:
            logger.info(f"Extracting {tar_path.name} to {extract_dir}")
            
            # Create year-month directory
            yymm = tar_path.stem.replace('arXiv_pdf_', '').replace('arXiv_src_', '')
            target_dir = extract_dir / yymm
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract using tar command
            cmd = ['tar', '-xf', str(tar_path), '-C', str(target_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to extract {tar_path.name}: {result.stderr}")
                return False
            
            # Mark as processed
            self.processed_tars.add(tar_path.name)
            self._save_checkpoint()
            
            logger.info(f"✓ Successfully extracted {tar_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting {tar_path.name}: {e}")
            return False
    
    def process_new_tars(self, db_connection=None) -> Dict:
        """Process all new tar files and optionally update database."""
        new_pdfs, new_latex = self.find_new_tars()
        
        stats = {
            'new_pdf_tars': len(new_pdfs),
            'new_latex_tars': len(new_latex),
            'extracted_pdfs': 0,
            'extracted_latex': 0,
            'errors': []
        }
        
        logger.info(f"Found {len(new_pdfs)} new PDF tars and {len(new_latex)} new LaTeX tars")
        
        # Extract PDF tars
        for tar_path in tqdm(new_pdfs, desc="Extracting PDF tars"):
            if self.extract_tar(tar_path, self.pdf_extract_dir):
                stats['extracted_pdfs'] += 1
            else:
                stats['errors'].append(f"Failed to extract {tar_path.name}")
        
        # Extract LaTeX tars
        for tar_path in tqdm(new_latex, desc="Extracting LaTeX tars"):
            if self.extract_tar(tar_path, self.latex_extract_dir):
                stats['extracted_latex'] += 1
            else:
                stats['errors'].append(f"Failed to extract {tar_path.name}")
        
        # Update database if connection provided
        if db_connection and (stats['extracted_pdfs'] > 0 or stats['extracted_latex'] > 0):
            logger.info("Updating database with new file locations...")
            updater = FileLocationUpdater(db_connection)
            update_stats = updater.update_all_locations()
            stats.update(update_stats)
        
        return stats


class FileLocationUpdater:
    """
    Updates PostgreSQL with file locations after extraction.
    
    Manages the WHERE dimension of Information Reconstructionism.
    """
    
    def __init__(self, db_connection):
        """Initialize with database connection."""
        self.conn = db_connection
        self.cur = self.conn.cursor()
    
    def update_all_locations(self) -> Dict:
        """Update both PDF and LaTeX locations in database."""
        stats = {}
        
        # Update PDFs
        pdf_stats = self._update_pdf_locations()
        stats.update({'pdf_' + k: v for k, v in pdf_stats.items()})
        
        # Update LaTeX
        latex_stats = self._update_latex_locations()
        stats.update({'latex_' + k: v for k, v in latex_stats.items()})
        
        return stats
    
    def _update_pdf_locations(self) -> Dict:
        """Update PDF file locations."""
        pdf_base = Path("/bulk-store/arxiv-data/pdf")
        updated = 0
        errors = 0
        
        for pdf_file in pdf_base.glob("*/*.pdf"):
            try:
                # Extract arxiv_id
                filename = pdf_file.stem
                if '.' in filename:
                    arxiv_id = filename
                else:
                    arxiv_id = filename.replace('_', '/')
                
                # Update database
                rel_path = pdf_file.relative_to(pdf_base)
                size_mb = pdf_file.stat().st_size / (1024 * 1024)
                
                self.cur.execute("""
                    UPDATE arxiv_papers
                    SET pdf_path = %s, has_pdf = TRUE, pdf_size_mb = %s
                    WHERE id = %s AND pdf_path IS NULL
                """, (str(rel_path), size_mb, arxiv_id))
                
                if self.cur.rowcount > 0:
                    updated += 1
                
                if updated % 1000 == 0:
                    self.conn.commit()
                    
            except Exception as e:
                errors += 1
                logger.debug(f"Error updating PDF {pdf_file.name}: {e}")
        
        self.conn.commit()
        return {'updated': updated, 'errors': errors}
    
    def _update_latex_locations(self) -> Dict:
        """Update LaTeX source locations."""
        latex_base = Path("/bulk-store/arxiv-data/latex")
        updated = 0
        errors = 0
        
        for latex_file in latex_base.glob("*/*"):
            if not latex_file.is_file():
                continue
            
            try:
                # Extract arxiv_id
                filename = latex_file.stem
                is_latex = latex_file.suffix == '.gz'
                source_format = 'latex' if is_latex else 'pdf_only'
                
                if '.' in filename:
                    arxiv_id = filename
                else:
                    arxiv_id = filename.replace('_', '/')
                
                # Update database
                rel_path = latex_file.relative_to(latex_base)
                
                self.cur.execute("""
                    UPDATE arxiv_papers
                    SET latex_path = %s, has_latex_source = %s, source_format = %s
                    WHERE id = %s AND latex_path IS NULL
                """, (str(rel_path), is_latex, source_format, arxiv_id))
                
                if self.cur.rowcount > 0:
                    updated += 1
                
                if updated % 1000 == 0:
                    self.conn.commit()
                    
            except Exception as e:
                errors += 1
                logger.debug(f"Error updating LaTeX {latex_file.name}: {e}")
        
        self.conn.commit()
        return {'updated': updated, 'errors': errors}


class DatabaseMaintenance:
    """
    Database maintenance utilities extracted from cleanup scripts.
    
    Following Actor-Network Theory, these functions maintain the health
    of our database actants, ensuring smooth information flow.
    """
    
    @staticmethod
    def kill_hanging_transactions(pg_password: str, pg_host: str = 'localhost',
                                 pg_database: str = 'arxiv_datalake') -> bool:
        """
        Kill hanging PostgreSQL transactions that might block operations.
        
        Args:
            pg_password: PostgreSQL password
            pg_host: PostgreSQL host
            pg_database: Database name
            
        Returns:
            True if successful
        """
        try:
            # Connect to postgres database to see all connections
            conn = psycopg2.connect(
                host=pg_host,
                database='postgres',
                user='postgres',
                password=pg_password
            )
            cur = conn.cursor()
            
            # Find hanging connections to our database
            cur.execute("""
                SELECT pid, state, query, query_start, state_change
                FROM pg_stat_activity
                WHERE datname = %s 
                AND pid != pg_backend_pid()
                AND state IN ('idle in transaction', 'idle in transaction (aborted)')
                AND state_change < NOW() - INTERVAL '5 minutes'
            """, (pg_database,))
            
            hanging = cur.fetchall()
            
            if hanging:
                logger.info(f"Found {len(hanging)} hanging transactions")
                for pid, state, query, start, change in hanging:
                    try:
                        cur.execute("SELECT pg_terminate_backend(%s)", (pid,))
                        logger.info(f"Killed connection PID {pid}")
                    except Exception as e:
                        logger.error(f"Failed to kill PID {pid}: {e}")
            
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error managing PostgreSQL connections: {e}")
            return False
    
    @staticmethod
    def clear_arxiv_collections(arango_password: str, arango_host: str = 'localhost',
                               pg_password: Optional[str] = None,
                               update_postgres: bool = False) -> Dict:
        """
        Clear all arxiv_* collections in ArangoDB.
        
        WARNING: This deletes all data in arxiv_* collections!
        
        Args:
            arango_password: ArangoDB password
            arango_host: ArangoDB host
            pg_password: PostgreSQL password (if updating tracking)
            update_postgres: Whether to update PostgreSQL embeddings_created flag
            
        Returns:
            Statistics dictionary
        """
        try:
            from arango import ArangoClient
            
            client = ArangoClient(hosts=f'http://{arango_host}:8529')
            db = client.db('academy_store', username='root', password=arango_password)
            
            stats = {
                'collections_cleared': [],
                'total_documents': 0,
                'papers_updated_in_pg': 0
            }
            
            # Find all arxiv_* collections
            arxiv_collections = []
            for collection in db.collections():
                if collection['name'].startswith('arxiv_'):
                    arxiv_collections.append(collection['name'])
            
            if not arxiv_collections:
                logger.info("No arxiv_* collections found")
                return stats
            
            # Get list of papers before clearing (for PostgreSQL update)
            cleared_papers = set()
            if update_postgres and pg_password and 'arxiv_embeddings' in arxiv_collections:
                try:
                    cursor = db.aql.execute('FOR doc IN arxiv_embeddings RETURN doc.arxiv_id')
                    cleared_papers = set(cursor)
                except Exception as e:
                    logger.warning(f"Could not get paper list: {e}")
            
            # Clear collections
            for coll_name in arxiv_collections:
                try:
                    collection = db.collection(coll_name)
                    count = collection.count()
                    collection.truncate()
                    stats['collections_cleared'].append(coll_name)
                    stats['total_documents'] += count
                    logger.info(f"Cleared {coll_name}: {count} documents")
                except Exception as e:
                    logger.error(f"Failed to clear {coll_name}: {e}")
            
            # Update PostgreSQL if requested
            if update_postgres and pg_password and cleared_papers:
                try:
                    conn = psycopg2.connect(
                        host='localhost',
                        database='arxiv_datalake',
                        user='postgres',
                        password=pg_password
                    )
                    cur = conn.cursor()
                    
                    # Update papers to mark as not processed
                    for arxiv_id in cleared_papers:
                        cur.execute("""
                            UPDATE arxiv_papers 
                            SET embeddings_created = FALSE,
                                embeddings_date = NULL
                            WHERE id = %s
                        """, (arxiv_id,))
                        stats['papers_updated_in_pg'] += cur.rowcount
                    
                    conn.commit()
                    cur.close()
                    conn.close()
                    
                    logger.info(f"Updated {stats['papers_updated_in_pg']} papers in PostgreSQL")
                    
                except Exception as e:
                    logger.error(f"Failed to update PostgreSQL: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error clearing collections: {e}")
            return {'error': str(e)}


def generate_coverage_report(pg_password: str, database: str = "arxiv_datalake",
                            output_file: Optional[str] = None) -> Dict:
    """
    Generate comprehensive coverage report for the database.
    
    Args:
        pg_password: PostgreSQL password
        database: Database name
        output_file: Optional JSON file to save report
        
    Returns:
        Coverage report dictionary
    """
    conn = psycopg2.connect(
        host='localhost',
        database=database,
        user='postgres',
        password=pg_password
    )
    
    try:
        analyzer = DatabaseAnalyzer(conn)
        report = analyzer.get_coverage_report()
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Coverage report saved to {output_file}")
        
        return report
        
    finally:
        conn.close()


def process_new_tars(pg_password: str, database: str = "arxiv_datalake") -> Dict:
    """
    Process any new tar files and update database.
    
    Args:
        pg_password: PostgreSQL password
        database: Database name
        
    Returns:
        Processing statistics
    """
    conn = psycopg2.connect(
        host='localhost',
        database=database,
        user='postgres',
        password=pg_password
    )
    
    try:
        extractor = TarExtractor()
        stats = extractor.process_new_tars(db_connection=conn)
        return stats
        
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Database tools for ArXiv data lake')
    parser.add_argument('--password', required=True, help='PostgreSQL password')
    parser.add_argument('--database', default='arxiv_datalake', help='Database name')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Coverage report command
    coverage_parser = subparsers.add_parser('coverage', help='Generate coverage report')
    coverage_parser.add_argument('--output', help='Output JSON file')
    
    # Process tars command
    tars_parser = subparsers.add_parser('process-tars', help='Process new tar files')
    
    args = parser.parse_args()
    
    if args.command == 'coverage':
        report = generate_coverage_report(args.password, args.database, args.output)
        print(f"Coverage Report for {args.database}:")
        print(f"  Total papers: {report['overall']['total_papers']:,}")
        print(f"  PDF coverage: {report['overall']['pdf_coverage_percent']:.1f}%")
        print(f"  LaTeX coverage: {report['overall']['latex_coverage_percent']:.1f}%")
        
    elif args.command == 'process-tars':
        stats = process_new_tars(args.password, args.database)
        print(f"Processed {stats['extracted_pdfs']} PDF tars and {stats['extracted_latex']} LaTeX tars")
        if stats.get('pdf_updated'):
            print(f"Updated {stats['pdf_updated']} PDF locations")
        if stats.get('latex_updated'):
            print(f"Updated {stats['latex_updated']} LaTeX locations")