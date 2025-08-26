"""
Daily updater for ArXiv SQLite database.

Fetches new papers from ArXiv API and updates the local SQLite cache.
Can be run as a cron job to keep the database current.
"""

import os
import sqlite3
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time
import json
import re

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ArxivDailyUpdater:
    """Update SQLite database with latest papers from ArXiv API"""
    
    def __init__(self, sqlite_path: str = '/bulk-store/arxiv-cache.db'):
        """Initialize updater"""
        self.sqlite_path = Path(sqlite_path)
        
        if not self.sqlite_path.exists():
            raise FileNotFoundError(
                f"SQLite database not found: {self.sqlite_path}\n"
                f"Run import_arxiv_to_sqlite.py first to create the database"
            )
        
        self.conn = sqlite3.connect(str(self.sqlite_path))
        self.base_url = "http://export.arxiv.org/api/query"
        
        logger.info(f"Connected to SQLite: {self.sqlite_path}")
    
    def get_last_update_date(self) -> Optional[datetime]:
        """Get the date of the last successful update"""
        cursor = self.conn.cursor()
        
        # Check update log
        cursor.execute("""
            SELECT MAX(update_date) 
            FROM update_log 
            WHERE success = 1 AND source = 'api_update'
        """)
        
        result = cursor.fetchone()
        if result and result[0]:
            return datetime.fromisoformat(result[0])
        
        # If no update log, get the most recent paper date
        cursor.execute("""
            SELECT MAX(created_date) 
            FROM papers 
            WHERE created_date IS NOT NULL
        """)
        
        result = cursor.fetchone()
        if result and result[0]:
            date_str = result[0]
            # Try different date formats
            date_formats = [
                "%a, %d %b %Y %H:%M:%S %Z",  # RFC 2822 format
                "%a, %d %b %Y %H:%M:%S GMT",  # Explicit GMT
                "%Y-%m-%dT%H:%M:%S",          # ISO format without timezone
                "%Y-%m-%dT%H:%M:%S.%f",       # ISO format with microseconds
                "%Y-%m-%dT%H:%M:%SZ",         # ISO format with Z
                "%Y-%m-%dT%H:%M:%S.%fZ",      # ISO format with microseconds and Z
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # Last resort: try fromisoformat (handles many ISO variants)
            try:
                # Remove timezone suffix if present for compatibility
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                return datetime.fromisoformat(date_str)
            except:
                logger.warning(f"Could not parse date: {date_str}")
        
        # Default to 7 days ago if no date found
        return datetime.now() - timedelta(days=7)
    
    def _validate_category(self, category: str) -> bool:
        """
        Validate ArXiv category format.
        Examples: cs.AI, math.CO, astro-ph.GA, q-bio.GN
        """
        pattern = r'^[a-z-]+(\.[A-Z]{2,4})?$'
        return bool(re.match(pattern, category))
    
    def fetch_papers_since(
        self,
        since_date: datetime,
        max_results: int = 10000,
        categories: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Fetch papers submitted since a given date.
        
        Args:
            since_date: Fetch papers submitted after this date
            max_results: Maximum number of papers to fetch
            categories: Optional list of categories to filter (e.g., ['cs.AI', 'cs.LG'])
        """
        papers = []
        start = 0
        batch_size = 100  # ArXiv API limit
        
        # Validate max_results
        if max_results <= 0:
            logger.warning(f"Invalid max_results: {max_results}, using default 10000")
            max_results = 10000
        
        # Format date for ArXiv query (handle edge cases)
        try:
            date_str = since_date.strftime("%Y%m%d%H%M%S")
        except (AttributeError, ValueError) as e:
            logger.error(f"Invalid date format: {since_date}, error: {e}")
            # Default to 7 days ago
            since_date = datetime.now() - timedelta(days=7)
            date_str = since_date.strftime("%Y%m%d%H%M%S")
        
        # Build query
        query_parts = [f"submittedDate:[{date_str} TO 999999999999]"]
        
        if categories:
            # Validate and filter categories
            valid_categories = []
            for cat in categories:
                if self._validate_category(cat):
                    valid_categories.append(cat)
                else:
                    logger.warning(f"Invalid category format: {cat}, skipping")
            
            if valid_categories:
                cat_query = " OR ".join(f"cat:{cat}" for cat in valid_categories)
                query_parts.append(f"({cat_query})")
            else:
                logger.warning("No valid categories provided, fetching all categories")
        
        query = " AND ".join(query_parts)
        
        logger.info(f"Fetching papers since {since_date.isoformat()}")
        if categories:
            logger.info(f"Categories: {', '.join(categories)}")
        
        while start < max_results:
            params = {
                'search_query': query,
                'start': start,
                'max_results': min(batch_size, max_results - start),
                'sortBy': 'submittedDate',
                'sortOrder': 'ascending'
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                
                # Define namespaces
                ns = {
                    'atom': 'http://www.w3.org/2005/Atom',
                    'arxiv': 'http://arxiv.org/schemas/atom'
                }
                
                # Extract entries
                entries = root.findall('atom:entry', ns)
                
                if not entries:
                    logger.info(f"No more papers found (fetched {len(papers)} total)")
                    break
                
                for entry in entries:
                    paper = self._parse_entry(entry, ns)
                    if paper:
                        papers.append(paper)
                
                logger.info(f"Fetched {len(papers)} papers so far...")
                
                start += batch_size
                time.sleep(3)  # Be nice to the API
                
            except Exception as e:
                logger.error(f"Error fetching papers: {e}")
                break
        
        return papers
    
    def _validate_arxiv_id(self, arxiv_id: str) -> bool:
        """
        Validate ArXiv ID format.
        
        Valid formats:
        - Modern: YYMM.NNNNN (e.g., "1234.5678", "2310.08560")
        - Old: category/YYMMNNN (e.g., "cs.AI/0612345", "math.CO/9901234")
        - With version: Any of above with vN suffix (e.g., "1234.5678v2")
        """
        # Remove version suffix for validation
        base_id = re.sub(r'v\d+$', '', arxiv_id)
        
        # Modern format: YYMM.NNNNN or YYMM.NNNN
        modern_pattern = r'^\d{4}\.\d{4,5}$'
        if re.match(modern_pattern, base_id):
            return True
        
        # Old format: category/YYMMNNN
        old_pattern = r'^[a-z-]+(\.[A-Z]{2})?/\d{7}$'
        if re.match(old_pattern, base_id):
            return True
        
        return False
    
    def _parse_entry(self, entry, namespaces) -> Optional[Dict]:
        """Parse a single entry from ArXiv API response with validation"""
        try:
            # Extract ArXiv ID from the id URL
            id_elem = entry.find('atom:id', namespaces)
            if id_elem is None:
                logger.warning("Entry missing ID element")
                return None
            
            arxiv_url = id_elem.text
            if not arxiv_url:
                logger.warning("Entry has empty ID URL")
                return None
            
            # Extract ID from URL (handles both /abs/ and /pdf/ URLs)
            if '/abs/' in arxiv_url:
                arxiv_id = arxiv_url.split('/abs/')[-1]
            elif '/pdf/' in arxiv_url:
                arxiv_id = arxiv_url.split('/pdf/')[-1].replace('.pdf', '')
            else:
                logger.warning(f"Unexpected URL format: {arxiv_url}")
                return None
            
            # Validate the ID format
            if not self._validate_arxiv_id(arxiv_id):
                logger.warning(f"Invalid ArXiv ID format: {arxiv_id}")
                return None
            
            # Remove version suffix if present (e.g., "1234.5678v2" -> "1234.5678")
            version_match = re.match(r'^(.+?)v\d+$', arxiv_id)
            if version_match:
                base_id = version_match.group(1)
            else:
                base_id = arxiv_id
            
            # Extract other fields
            paper = {
                'arxiv_id': base_id,
                'title': entry.find('atom:title', namespaces).text.strip().replace('\n', ' '),
                'abstract': entry.find('atom:summary', namespaces).text.strip(),
                'authors': [],
                'categories': [],
                'primary_category': None,
                'doi': None,
                'journal_ref': None,
                'comments': None,
                'published': None,
                'updated': None
            }
            
            # Authors
            for author in entry.findall('atom:author', namespaces):
                name = author.find('atom:name', namespaces)
                if name is not None:
                    paper['authors'].append(name.text)
            
            # Categories
            for category in entry.findall('atom:category', namespaces):
                cat_term = category.get('term')
                if cat_term:
                    paper['categories'].append(cat_term)
            
            # Primary category
            primary_cat = entry.find('arxiv:primary_category', namespaces)
            if primary_cat is not None:
                paper['primary_category'] = primary_cat.get('term')
            elif paper['categories']:
                paper['primary_category'] = paper['categories'][0]
            
            # DOI
            doi_elem = entry.find('arxiv:doi', namespaces)
            if doi_elem is not None:
                paper['doi'] = doi_elem.text
            
            # Journal ref
            journal_elem = entry.find('arxiv:journal_ref', namespaces)
            if journal_elem is not None:
                paper['journal_ref'] = journal_elem.text
            
            # Comments
            comment_elem = entry.find('arxiv:comment', namespaces)
            if comment_elem is not None:
                paper['comments'] = comment_elem.text
            
            # Dates
            published = entry.find('atom:published', namespaces)
            if published is not None:
                paper['published'] = published.text
            
            updated = entry.find('atom:updated', namespaces)
            if updated is not None:
                paper['updated'] = updated.text
            
            return paper
            
        except Exception as e:
            logger.error(f"Error parsing entry: {e}")
            return None
    
    def update_database(self, papers: List[Dict]) -> Dict[str, int]:
        """Update SQLite database with new papers"""
        cursor = self.conn.cursor()
        
        added = 0
        updated = 0
        skipped = 0
        
        for paper in papers:
            try:
                # Check if paper exists
                cursor.execute(
                    "SELECT arxiv_id, updated_date FROM papers WHERE arxiv_id = ?",
                    (paper['arxiv_id'],)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update if newer version
                    if paper.get('updated') and (not existing[1] or paper['updated'] > existing[1]):
                        cursor.execute("""
                            UPDATE papers 
                            SET title = ?, abstract = ?, updated_date = ?,
                                categories = ?, primary_category = ?,
                                doi = ?, journal_ref = ?, comments = ?
                            WHERE arxiv_id = ?
                        """, (
                            paper['title'],
                            paper['abstract'],
                            paper['updated'],
                            json.dumps(paper['categories']),
                            paper['primary_category'],
                            paper['doi'],
                            paper['journal_ref'],
                            paper['comments'],
                            paper['arxiv_id']
                        ))
                        updated += 1
                    else:
                        skipped += 1
                else:
                    # Check for local PDF
                    pdf_path = self._find_pdf_path(paper['arxiv_id'])
                    pdf_status = 'found' if pdf_path else 'not_checked'
                    
                    # Insert new paper
                    cursor.execute("""
                        INSERT INTO papers (
                            arxiv_id, title, abstract, authors, categories,
                            primary_category, doi, journal_ref, comments,
                            created_date, updated_date, pdf_path, pdf_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        paper['arxiv_id'],
                        paper['title'],
                        paper['abstract'],
                        json.dumps(paper['authors']),
                        json.dumps(paper['categories']),
                        paper['primary_category'],
                        paper['doi'],
                        paper['journal_ref'],
                        paper['comments'],
                        paper['published'],
                        paper['updated'],
                        pdf_path,
                        pdf_status
                    ))
                    added += 1
                    
            except Exception as e:
                logger.error(f"Error updating paper {paper['arxiv_id']}: {e}")
        
        self.conn.commit()
        
        return {'added': added, 'updated': updated, 'skipped': skipped}
    
    def _find_pdf_path(self, arxiv_id: str) -> Optional[str]:
        """Check if PDF exists locally
        
        Args:
            arxiv_id: ArXiv ID (e.g., '1234.5678' or 'cs.AI/0612345')
            
        Returns:
            Path to PDF if found, None otherwise
        """
        pdf_base = Path('/bulk-store/arxiv-data/pdf')
        
        # Modern format: YYMM.NNNNN (e.g., "1234.5678")
        if '.' in arxiv_id and '/' not in arxiv_id:
            parts = arxiv_id.split('.')
            if len(parts) == 2 and len(parts[0]) == 4 and parts[0].isdigit():
                year_month = parts[0]
                pdf_path = pdf_base / year_month / f"{arxiv_id}.pdf"
                if pdf_path.exists():
                    return str(pdf_path)
        
        # Old format: category/YYMMNNN (e.g., "cs.AI/0612345")
        elif '/' in arxiv_id:
            category, paper_id = arxiv_id.split('/', 1)
            if len(paper_id) >= 7 and paper_id[:7].isdigit():
                year_month = paper_id[:4]
                # Replace slashes and dots in filename
                pdf_name = f"{category}_{paper_id}.pdf".replace('/', '_').replace('.', '_')
                pdf_path = pdf_base / year_month / pdf_name
                if pdf_path.exists():
                    return str(pdf_path)
                
                # Try alternative naming convention
                pdf_name_alt = f"{arxiv_id.replace('/', '_')}.pdf"
                pdf_path_alt = pdf_base / year_month / pdf_name_alt
                if pdf_path_alt.exists():
                    return str(pdf_path_alt)
        
        return None
    
    def log_update(self, stats: Dict[str, int], source: str = 'api_update'):
        """Log update statistics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO update_log (
                update_date, papers_added, papers_updated, source, success
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            stats['added'],
            stats['updated'],
            source,
            True
        ))
        self.conn.commit()
    
    def run_update(
        self,
        days_back: Optional[int] = None,
        categories: Optional[List[str]] = None
    ):
        """
        Run the daily update process.
        
        Args:
            days_back: Override how many days back to fetch (None = since last update)
            categories: Optional list of categories to focus on
        """
        logger.info("="*60)
        logger.info("ARXIV DAILY UPDATE")
        logger.info("="*60)
        
        # Determine start date
        if days_back:
            since_date = datetime.now() - timedelta(days=days_back)
            logger.info(f"Fetching papers from last {days_back} days")
        else:
            since_date = self.get_last_update_date()
            logger.info(f"Fetching papers since last update: {since_date.isoformat()}")
        
        # Fetch new papers
        papers = self.fetch_papers_since(since_date, categories=categories)
        
        if not papers:
            logger.info("No new papers found")
            return
        
        logger.info(f"Found {len(papers)} papers to process")
        
        # Update database
        stats = self.update_database(papers)
        
        # Log the update
        self.log_update(stats)
        
        # Report results
        logger.info("="*60)
        logger.info("UPDATE COMPLETE")
        logger.info("="*60)
        logger.info(f"Papers added: {stats['added']}")
        logger.info(f"Papers updated: {stats['updated']}")
        logger.info(f"Papers skipped: {stats['skipped']}")
        
        # Get current totals
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM papers")
        total = cursor.fetchone()[0]
        logger.info(f"Total papers in database: {total:,}")


def main():
    """Main function for cron job or manual execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update ArXiv SQLite database')
    parser.add_argument('--sqlite-path', default='/bulk-store/arxiv-cache.db',
                      help='Path to SQLite database')
    parser.add_argument('--days-back', type=int, default=None,
                      help='Fetch papers from N days back (default: since last update)')
    parser.add_argument('--categories', nargs='+',
                      help='Categories to fetch (e.g., cs.AI cs.LG math.CO)')
    
    args = parser.parse_args()
    
    try:
        updater = ArxivDailyUpdater(args.sqlite_path)
        updater.run_update(
            days_back=args.days_back,
            categories=args.categories
        )
    except Exception as e:
        logger.error(f"Update failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())