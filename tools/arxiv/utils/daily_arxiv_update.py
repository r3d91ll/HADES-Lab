#!/usr/bin/env python3
"""
Daily ArXiv Updater for PostgreSQL
===================================

Fetches latest ArXiv papers and updates PostgreSQL metadata.
Designed to run daily via cron to keep our data current.

Following Actor-Network Theory: This script acts as a translator between
the ArXiv API (external actant) and our PostgreSQL data lake (internal actant),
maintaining temporal continuity in our information network.

Implements: C = (W·R·H)/T · Ctx^α
- Maximizes R (relational completeness) by keeping metadata current
- Minimizes T (access time) by having data pre-fetched
"""

import os
import sys
import logging
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
import time
import argparse
import json

# Add HADES root to path for core utilities
hades_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hades_root))

from core.utils import PreflightChecker, BatchProcessor, StateManager

# Setup logging
log_dir = Path('/home/todd/olympus/HADES/logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'daily_arxiv_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ArXivDailyUpdater:
    """
    Updates PostgreSQL with latest ArXiv papers daily.
    
    This replaces the old ArangoDB updater to align with our new
    PostgreSQL-first architecture where PostgreSQL is the metadata
    source of truth.
    """
    
    def __init__(self, pg_password: str, pg_host: str = "localhost", 
                 pg_database: str = "arxiv_datalake"):
        """
        Initialize the updater with PostgreSQL connection.
        
        Args:
            pg_password: PostgreSQL password
            pg_host: PostgreSQL host
            pg_database: Database name
        """
        self.pg_host = pg_host
        self.pg_database = pg_database
        self.pg_password = pg_password
        self.base_url = "http://export.arxiv.org/api/query"
        
        # Run pre-flight checks
        if not self._run_preflight_checks():
            raise RuntimeError("Pre-flight checks failed")
        
        # Initialize database connection
        self._init_database()
        
        # Initialize state manager for tracking updates
        state_file = log_dir / 'daily_update_state.json'
        self.state_manager = StateManager(str(state_file), 'ArXivDailyUpdater')
        
        # Initialize batch processor for better error handling
        self.batch_processor = BatchProcessor(
            db_connection=self.conn,
            batch_size=100,
            use_savepoints=True
        )
    
    def _run_preflight_checks(self) -> bool:
        """
        Run pre-flight checks before starting updater.
        
        Returns:
            True if all checks pass
        """
        checker = PreflightChecker("ArXivDailyUpdater")
        
        # Check log directory exists
        checker.add_check(
            "log_directory",
            lambda: log_dir.exists() or self._create_log_dir(),
            "Log directory must exist",
            critical=False
        )
        
        # Check database connectivity
        checker.add_check(
            "postgresql",
            lambda: checker.check_database_connection(
                self.pg_host, self.pg_database,
                'postgres', self.pg_password
            ),
            "PostgreSQL connection must be available",
            critical=True
        )
        
        # Check network connectivity to ArXiv
        checker.add_check(
            "arxiv_api",
            lambda: self._check_arxiv_api(),
            "ArXiv API must be accessible",
            critical=True
        )
        
        # Check disk space for logs
        checker.add_check(
            "disk_space",
            lambda: checker.check_disk_space(log_dir, required_gb=0.5),
            "Need at least 500MB free disk space for logs",
            critical=False
        )
        
        # Run all checks
        results = checker.run_all()
        
        if not results['all_passed']:
            logger.error("Pre-flight checks failed:")
            for check in results['failed']:
                logger.error(f"  ✗ {check['name']}: {check.get('error', 'Failed')}")
        
        return results['all_passed']
    
    def _create_log_dir(self) -> bool:
        """Create log directory if it doesn't exist."""
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create log directory: {e}")
            return False
    
    def _check_arxiv_api(self) -> bool:
        """Check ArXiv API connectivity."""
        try:
            # Test with a simple query
            params = {
                'search_query': 'all:electron',
                'max_results': 1
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"ArXiv API check failed: {e}")
            return False
    
    def _init_database(self):
        """Initialize PostgreSQL connection."""
        try:
            self.conn = psycopg2.connect(
                host=self.pg_host,
                database=self.pg_database,
                user='postgres',
                password=self.pg_password
            )
            self.cur = self.conn.cursor()
            logger.info(f"Connected to PostgreSQL {self.pg_database} at {self.pg_host}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def fetch_papers_for_date(self, date: datetime, max_results: int = 5000) -> List[Dict]:
        """
        Fetch papers submitted on a specific date from ArXiv API.
        
        Args:
            date: Date to fetch papers for
            max_results: Maximum papers to fetch (ArXiv typically has 500-1000/day)
            
        Returns:
            List of paper dictionaries
        """
        # Format date for ArXiv API (YYYYMMDD)
        date_str = date.strftime("%Y%m%d")
        next_date_str = (date + timedelta(days=1)).strftime("%Y%m%d")
        
        # Build query for papers submitted on this date
        query = f"submittedDate:[{date_str}0000 TO {next_date_str}0000]"
        
        papers = []
        start = 0
        batch_size = 100  # ArXiv API limit per request
        
        while start < max_results:
            params = {
                'search_query': query,
                'start': start,
                'max_results': min(batch_size, max_results - start),
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            try:
                logger.info(f"Fetching papers {start} to {start + batch_size}...")
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.text)
                
                # Extract namespace
                ns = {'atom': 'http://www.w3.org/2005/Atom',
                      'arxiv': 'http://arxiv.org/schemas/atom'}
                
                # Find all entries
                entries = root.findall('atom:entry', ns)
                
                if not entries:
                    logger.info(f"No more papers found for {date_str}")
                    break
                
                for entry in entries:
                    paper = self._parse_entry(entry, ns)
                    if paper:
                        papers.append(paper)
                
                logger.info(f"Fetched {len(entries)} papers (total: {len(papers)})")
                start += batch_size
                
                # Rate limiting - ArXiv allows 3 requests per second
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching papers: {e}")
                break
        
        return papers
    
    def _parse_entry(self, entry, ns) -> Optional[Dict]:
        """
        Parse a single ArXiv entry into our format.
        
        Returns:
            Paper dictionary or None if parsing fails
        """
        try:
            # Extract ID from URL
            id_url = entry.find('atom:id', ns).text
            # Remove version suffix to get base ID
            arxiv_id = id_url.split('/abs/')[-1]
            base_id = arxiv_id.split('v')[0]  # Remove version
            
            # Extract version number
            version = 'v1'  # Default
            if 'v' in arxiv_id:
                version = 'v' + arxiv_id.split('v')[-1]
            
            # Extract other fields
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            abstract = entry.find('atom:summary', ns).text.strip()
            
            # Authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns).text
                authors.append(name)
            
            # Categories
            categories = []
            primary_category = entry.find('arxiv:primary_category', ns)
            if primary_category is not None:
                categories.append(primary_category.get('term'))
            
            for category in entry.findall('atom:category', ns):
                cat_term = category.get('term')
                if cat_term and cat_term not in categories:
                    categories.append(cat_term)
            
            # Dates
            published = entry.find('atom:published', ns).text
            updated = entry.find('atom:updated', ns).text
            
            # DOI if available
            doi = None
            doi_elem = entry.find('arxiv:doi', ns)
            if doi_elem is not None:
                doi = doi_elem.text
            
            # Comment if available
            comment = None
            comment_elem = entry.find('arxiv:comment', ns)
            if comment_elem is not None:
                comment = comment_elem.text
            
            # Journal reference if available
            journal_ref = None
            journal_elem = entry.find('arxiv:journal_ref', ns)
            if journal_elem is not None:
                journal_ref = journal_elem.text
            
            return {
                'arxiv_id': base_id,
                'version': version,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'categories': categories,
                'published_date': published,
                'updated_date': updated,
                'doi': doi,
                'comment': comment,
                'journal_ref': journal_ref,
                'pdf_url': f"https://arxiv.org/pdf/{base_id}.pdf"
            }
            
        except Exception as e:
            logger.error(f"Error parsing entry: {e}")
            return None
    
    def update_database(self, papers: List[Dict]) -> Dict:
        """
        Update PostgreSQL with new papers.
        
        Uses the same schema as our bulk import but handles daily incremental updates.
        Now uses BatchProcessor for better error isolation.
        
        Returns:
            Statistics about the update
        """
        # Define processing function for batch processor
        def process_paper(paper: Dict) -> bool:
            """Process a single paper with all its relations."""
            try:
                # Check if paper exists
                self.cur.execute("SELECT id FROM arxiv_papers WHERE id = %s", (paper['arxiv_id'],))
                existing = self.cur.fetchone()
                
                if existing:
                    # Update existing paper
                    self.cur.execute("""
                        UPDATE arxiv_papers 
                        SET title = %s, abstract = %s, categories = %s,
                            doi = %s, comments = %s, journal_ref = %s,
                            update_date = CURRENT_DATE
                        WHERE id = %s
                    """, (
                        paper['title'], paper['abstract'], paper['categories'],
                        paper['doi'], paper['comment'], paper['journal_ref'],
                        paper['arxiv_id']
                    ))
                    self.batch_processor.stats['updated_papers'] = \
                        self.batch_processor.stats.get('updated_papers', 0) + 1
                    logger.debug(f"Updated paper {paper['arxiv_id']}")
                else:
                    # Insert new paper
                    self.cur.execute("""
                        INSERT INTO arxiv_papers 
                        (id, title, abstract, categories, 
                         doi, comments, journal_ref, update_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
                    """, (
                        paper['arxiv_id'], paper['title'], paper['abstract'],
                        paper['categories'], paper['doi'], 
                        paper['comment'], paper['journal_ref']
                    ))
                    self.batch_processor.stats['new_papers'] = \
                        self.batch_processor.stats.get('new_papers', 0) + 1
                    logger.debug(f"Added new paper {paper['arxiv_id']}")
                
                # Handle version
                self.cur.execute("""
                    SELECT version FROM arxiv_versions 
                    WHERE paper_id = %s AND version = %s
                """, (paper['arxiv_id'], paper['version']))
                
                if not self.cur.fetchone():
                    # Parse date once
                    created_date = datetime.fromisoformat(paper['published_date'].replace('Z', '+00:00'))
                    created_timestamp = created_date  # Reuse the same parsed datetime object
                    
                    # Insert new version
                    self.cur.execute("""
                        INSERT INTO arxiv_versions 
                        (paper_id, version, created_date, created_timestamp)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (paper_id, version) DO NOTHING
                    """, (
                        paper['arxiv_id'], paper['version'],
                        created_date.date(), created_timestamp
                    ))
                    self.batch_processor.stats['new_versions'] = \
                        self.batch_processor.stats.get('new_versions', 0) + 1
                
                # Handle authors
                for position, author_name in enumerate(paper['authors'], start=1):
                    # Insert author if new
                    self.cur.execute("""
                        INSERT INTO arxiv_authors (author_name)
                        VALUES (%s)
                        ON CONFLICT (author_name) DO NOTHING
                        RETURNING id
                    """, (author_name,))
                    
                    # Get author ID
                    author_result = self.cur.fetchone()
                    if author_result:
                        author_id = author_result[0]
                        self.batch_processor.stats['new_authors'] = \
                            self.batch_processor.stats.get('new_authors', 0) + 1
                    else:
                        self.cur.execute("SELECT id FROM arxiv_authors WHERE author_name = %s", (author_name,))
                        author_id = self.cur.fetchone()[0]
                    
                    # Link author to paper with position
                    self.cur.execute("""
                        INSERT INTO arxiv_paper_authors (paper_id, author_id, author_position)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (paper_id, author_id) DO UPDATE
                        SET author_position = EXCLUDED.author_position
                    """, (paper['arxiv_id'], author_id, position))
                
                return True
                
            except Exception as e:
                logger.error(f"Error processing {paper.get('arxiv_id', 'unknown')}: {e}")
                return False
        
        # Reset batch processor stats for this run
        self.batch_processor.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'batches': 0,
            'errors': [],
            'new_papers': 0,
            'new_versions': 0,
            'updated_papers': 0,
            'new_authors': 0
        }
        
        # Process papers using batch processor
        batch_stats = self.batch_processor.process_items(
            papers,
            process_paper,
            commit_interval=50  # Commit every 50 successful papers
        )
        
        # Return stats in expected format
        return {
            'new_papers': self.batch_processor.stats.get('new_papers', 0),
            'new_versions': self.batch_processor.stats.get('new_versions', 0),
            'updated_papers': self.batch_processor.stats.get('updated_papers', 0),
            'new_authors': self.batch_processor.stats.get('new_authors', 0),
            'errors': batch_stats['failed']
        }
    
    def run_daily_update(self, days_back: int = 1, catch_up: bool = False):
        """
        Run the daily update process.
        
        Args:
            days_back: Number of days to look back (default 1 for yesterday)
            catch_up: If True, process all days from days_back to yesterday
        """
        logger.info("=" * 60)
        logger.info("Starting ArXiv daily update for PostgreSQL")
        logger.info("=" * 60)
        
        # Determine dates to process
        if catch_up:
            dates_to_process = []
            for i in range(days_back, 0, -1):
                dates_to_process.append(datetime.utcnow() - timedelta(days=i))
            logger.info(f"Catch-up mode: Processing {len(dates_to_process)} days")
        else:
            dates_to_process = [datetime.utcnow() - timedelta(days=days_back)]
        
        total_stats = {
            'new_papers': 0,
            'new_versions': 0,
            'updated_papers': 0,
            'new_authors': 0,
            'errors': 0,
            'total_fetched': 0
        }
        
        for target_date in dates_to_process:
            logger.info(f"\nFetching papers for {target_date.strftime('%Y-%m-%d')}")
            
            # Fetch papers
            papers = self.fetch_papers_for_date(target_date)
            logger.info(f"Fetched {len(papers)} papers from ArXiv")
            total_stats['total_fetched'] += len(papers)
            
            if papers:
                # Update database
                stats = self.update_database(papers)
                
                # Accumulate stats
                for key in stats:
                    if key in total_stats:
                        total_stats[key] += stats[key]
                
                logger.info(f"Update for {target_date.strftime('%Y-%m-%d')}:")
                logger.info(f"  New papers: {stats['new_papers']}")
                logger.info(f"  New versions: {stats['new_versions']}")
                logger.info(f"  Updated papers: {stats['updated_papers']}")
                logger.info(f"  New authors: {stats['new_authors']}")
                logger.info(f"  Errors: {stats['errors']}")
            else:
                logger.info(f"No papers found for {target_date.strftime('%Y-%m-%d')}")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("DAILY UPDATE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total papers fetched: {total_stats['total_fetched']}")
        logger.info(f"New papers added: {total_stats['new_papers']}")
        logger.info(f"New versions added: {total_stats['new_versions']}")
        logger.info(f"Papers updated: {total_stats['updated_papers']}")
        logger.info(f"New authors added: {total_stats['new_authors']}")
        logger.info(f"Errors: {total_stats['errors']}")
        
        # Save update log
        self._save_update_log(total_stats, dates_to_process)
        
        logger.info("=" * 60)
    
    def _save_update_log(self, stats: Dict, dates: List[datetime]):
        """Save update statistics using StateManager."""
        # Update state manager with latest run info
        self.state_manager.set_checkpoint('last_run_time', datetime.utcnow().isoformat())
        self.state_manager.set_checkpoint('last_dates_processed', [d.isoformat() for d in dates])
        self.state_manager.set_checkpoint('last_run_stats', stats)
        
        # Keep history of last 30 runs
        run_history = self.state_manager.get_checkpoint('run_history', [])
        run_entry = {
            'run_time': datetime.utcnow().isoformat(),
            'dates_processed': [d.isoformat() for d in dates],
            'statistics': stats
        }
        run_history.append(run_entry)
        run_history = run_history[-30:]  # Keep only last 30
        self.state_manager.set_checkpoint('run_history', run_history)
        
        # Update cumulative statistics
        cumulative = self.state_manager.get_checkpoint('cumulative_stats', {
            'total_papers': 0,
            'total_versions': 0,
            'total_authors': 0,
            'total_errors': 0,
            'total_runs': 0
        })
        cumulative['total_papers'] += stats.get('new_papers', 0) + stats.get('updated_papers', 0)
        cumulative['total_versions'] += stats.get('new_versions', 0)
        cumulative['total_authors'] += stats.get('new_authors', 0)
        cumulative['total_errors'] += stats.get('errors', 0)
        cumulative['total_runs'] += 1
        self.state_manager.set_checkpoint('cumulative_stats', cumulative)
        
        # Save state atomically
        if self.state_manager.save():
            logger.info(f"Update statistics saved via StateManager")
        else:
            logger.error("Failed to save update statistics")
    
    def verify_update(self, date: datetime) -> Dict:
        """
        Verify that papers from a specific date are in the database.
        
        Returns:
            Verification statistics
        """
        logger.info(f"Verifying papers for {date.strftime('%Y-%m-%d')}")
        
        # Query for papers created on this date
        self.cur.execute("""
            SELECT COUNT(*) FROM arxiv_papers p
            JOIN arxiv_versions v ON p.id = v.paper_id
            WHERE DATE(v.created_date) = %s
        """, (date.date(),))
        
        count = self.cur.fetchone()[0]
        
        logger.info(f"Found {count} papers for {date.strftime('%Y-%m-%d')} in database")
        
        return {'date': date.isoformat(), 'paper_count': count}
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - clean up database connection."""
        self.close()
        return False
    
    def close(self):
        """Close database connections and clean up resources."""
        if hasattr(self, 'cur') and self.cur:
            try:
                self.cur.close()
            except:
                pass
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
            except:
                pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Daily ArXiv updater for PostgreSQL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update with yesterday's papers (default)
  %(prog)s --pg-password "$PGPASSWORD"
  
  # Catch up on last 7 days
  %(prog)s --pg-password "$PGPASSWORD" --days-back 7 --catch-up
  
  # Verify papers from specific date
  %(prog)s --pg-password "$PGPASSWORD" --verify-date 2024-01-15
        """
    )
    
    parser.add_argument('--pg-password', required=True,
                       help='PostgreSQL password (or set PGPASSWORD)')
    parser.add_argument('--pg-host', default='localhost',
                       help='PostgreSQL host')
    parser.add_argument('--pg-database', default='arxiv_datalake',
                       help='Database name')
    parser.add_argument('--days-back', type=int, default=1,
                       help='Number of days to look back (default: 1 for yesterday)')
    parser.add_argument('--catch-up', action='store_true',
                       help='Process all days from days-back to yesterday')
    parser.add_argument('--verify-date', type=str,
                       help='Verify papers for specific date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Handle password from environment if not provided
    if not args.pg_password:
        args.pg_password = os.environ.get('PGPASSWORD')
        if not args.pg_password:
            logger.error("PostgreSQL password required (use --pg-password or set PGPASSWORD)")
            sys.exit(1)
    
    try:
        with ArXivDailyUpdater(
            pg_password=args.pg_password,
            pg_host=args.pg_host,
            pg_database=args.pg_database
        ) as updater:
            if args.verify_date:
                # Verification mode
                verify_date = datetime.strptime(args.verify_date, '%Y-%m-%d')
                updater.verify_update(verify_date)
            else:
                # Update mode
                updater.run_daily_update(
                    days_back=args.days_back,
                    catch_up=args.catch_up
                )
            
    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()