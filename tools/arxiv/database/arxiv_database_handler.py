#!/usr/bin/env python3
"""
ArXiv Database Handler
======================

Centralized handler for all ArXiv database operations.
Ensures consistent handling of papers, versions, authors, and relationships.

Following Actor-Network Theory: This handler acts as the obligatory passage point
for all database writes, ensuring data integrity across the network.
"""

import json
import logging
import psycopg2
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)


class ArXivDatabaseHandler:
    """
    Unified handler for ArXiv database operations.
    
    Ensures consistent handling of:
    - Papers (main metadata)
    - Versions (v1, v2, etc.)
    - Authors (normalized)
    - Paper-Author relationships
    """
    
    def __init__(self, connection):
        """
        Initialize with existing database connection.
        
        Args:
            connection: psycopg2 connection object
        """
        self.conn = connection
        self.cur = connection.cursor()
        self.stats = {
            'papers_inserted': 0,
            'papers_updated': 0,
            'versions_created': 0,
            'authors_created': 0,
            'relationships_created': 0,
            'errors': []
        }
    
    def process_paper(self, paper_data: Dict, source: str = 'snapshot') -> bool:
        """
        Process a complete paper with all its relationships.
        
        This is the main entry point that handles:
        1. Paper metadata
        2. All versions
        3. All authors and relationships
        
        Args:
            paper_data: Complete paper dictionary from JSON
            source: Source of the data ('snapshot', 'api', etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create savepoint for atomic paper processing
            self.cur.execute("SAVEPOINT paper_process")
            
            # Extract and validate arxiv_id
            arxiv_id = paper_data.get('id')
            if not arxiv_id:
                raise ValueError("Missing paper ID")
            
            # Process main paper record
            paper_exists = self._upsert_paper(arxiv_id, paper_data, source)
            
            # Process all versions
            versions = paper_data.get('versions', [])
            if versions:
                for version_info in versions:
                    self._process_version(arxiv_id, version_info)
            else:
                # If no versions provided, create a default v1 with update_date
                default_version = {
                    'version': 'v1',
                    'created': paper_data.get('update_date')
                }
                if default_version['created']:
                    self._process_version(arxiv_id, default_version)
            
            # Process authors
            authors_parsed = paper_data.get('authors_parsed', [])
            if authors_parsed:
                self._process_authors(arxiv_id, authors_parsed)
            
            # Release savepoint on success
            self.cur.execute("RELEASE SAVEPOINT paper_process")
            
            if paper_exists:
                self.stats['papers_updated'] += 1
            else:
                self.stats['papers_inserted'] += 1
            
            return True
            
        except Exception as e:
            # Rollback to savepoint on error
            self.cur.execute("ROLLBACK TO SAVEPOINT paper_process")
            error_msg = f"Paper {paper_data.get('id', 'unknown')}: {str(e)}"
            logger.debug(f"Failed to process: {error_msg}")
            self.stats['errors'].append(error_msg[:200])
            return False
    
    def _upsert_paper(self, arxiv_id: str, paper_data: Dict, source: str) -> bool:
        """
        Insert or update paper metadata.
        
        Returns:
            True if paper already existed, False if newly inserted
        """
        # Check if paper exists
        self.cur.execute("SELECT id FROM arxiv_papers WHERE id = %s", (arxiv_id,))
        exists = self.cur.fetchone() is not None
        
        # Clean and prepare data
        title = paper_data.get('title', '').replace('\n', ' ').strip()
        abstract = paper_data.get('abstract', '').replace('\n', ' ').strip()
        
        # Handle categories (could be string or list)
        categories = paper_data.get('categories', '')
        if isinstance(categories, list):
            categories = ' '.join(categories)
        
        # Parse update_date if present
        update_date = None
        if paper_data.get('update_date'):
            try:
                if 'T' in paper_data['update_date']:
                    update_date = datetime.fromisoformat(
                        paper_data['update_date'].replace('Z', '+00:00')
                    ).date()
                else:
                    update_date = parsedate_to_datetime(paper_data['update_date']).date()
            except:
                pass
        
        if exists:
            # Update existing paper
            self.cur.execute("""
                UPDATE arxiv_papers 
                SET title = %s, abstract = %s, categories = %s,
                    submitter = %s, comments = %s, journal_ref = %s,
                    doi = %s, report_no = %s, license = %s,
                    versions = %s, update_date = %s, authors_parsed = %s,
                    import_source = %s
                WHERE id = %s
            """, (
                title, abstract, categories,
                paper_data.get('submitter'),
                paper_data.get('comments'),
                paper_data.get('journal-ref') or paper_data.get('journal_ref'),
                paper_data.get('doi'),
                paper_data.get('report-no') or paper_data.get('report_no'),
                paper_data.get('license'),
                json.dumps(paper_data.get('versions', [])),
                update_date,
                json.dumps(paper_data.get('authors_parsed', [])),
                source,
                arxiv_id
            ))
        else:
            # Insert new paper
            self.cur.execute("""
                INSERT INTO arxiv_papers (
                    id, submitter, title, comments, journal_ref, doi,
                    report_no, categories, license, abstract, versions,
                    update_date, authors_parsed, import_source
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                arxiv_id,
                paper_data.get('submitter'),
                title,
                paper_data.get('comments'),
                paper_data.get('journal-ref') or paper_data.get('journal_ref'),
                paper_data.get('doi'),
                paper_data.get('report-no') or paper_data.get('report_no'),
                categories,
                paper_data.get('license'),
                abstract,
                json.dumps(paper_data.get('versions', [])),
                update_date,
                json.dumps(paper_data.get('authors_parsed', [])),
                source
            ))
        
        return exists
    
    def _process_version(self, paper_id: str, version_info: Dict) -> bool:
        """
        Process a single version entry.
        
        Args:
            paper_id: ArXiv ID
            version_info: Version dictionary with 'version', 'created', etc.
            
        Returns:
            True if successful
        """
        try:
            version = version_info.get('version', 'v1')
            created = version_info.get('created')
            
            created_date = None
            created_timestamp = None
            
            if created:
                try:
                    # Parse different date formats
                    if 'T' in created:
                        # ISO format
                        created_timestamp = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    else:
                        # Email date format like "Mon, 2 Apr 2007 20:00:00 GMT"
                        created_timestamp = parsedate_to_datetime(created)
                    
                    created_date = created_timestamp.date()
                except Exception as e:
                    logger.debug(f"Could not parse date '{created}': {e}")
            
            # Extract size if present
            size_kb = version_info.get('size_kb') or version_info.get('size')
            if size_kb and isinstance(size_kb, str):
                # Remove 'kb' suffix if present
                size_kb = int(size_kb.replace('kb', '').strip())
            elif not size_kb:
                size_kb = None
            
            # Insert version (ignore duplicates)
            self.cur.execute("""
                INSERT INTO arxiv_versions (paper_id, version, created_date, created_timestamp, size_kb)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (paper_id, version) DO UPDATE
                SET created_date = EXCLUDED.created_date,
                    created_timestamp = EXCLUDED.created_timestamp,
                    size_kb = EXCLUDED.size_kb
            """, (paper_id, version, created_date, created_timestamp, size_kb))
            
            if self.cur.rowcount > 0:
                self.stats['versions_created'] += 1
            
            return True
            
        except Exception as e:
            logger.debug(f"Failed to process version for {paper_id}: {e}")
            return False
    
    def _process_authors(self, paper_id: str, authors_parsed: List) -> int:
        """
        Process authors and create relationships.
        
        Args:
            paper_id: ArXiv ID
            authors_parsed: List of parsed author names
            
        Returns:
            Number of relationships created
        """
        relationships = 0
        
        for position, author_parts in enumerate(authors_parsed, start=1):
            try:
                # Reconstruct author name from parts
                if author_parts:
                    # Format: [last, first middle]
                    if len(author_parts) >= 2:
                        author_name = f"{author_parts[1]} {author_parts[0]}"
                    else:
                        author_name = ' '.join(author_parts)
                    
                    author_name = author_name.strip()
                    
                    if author_name:
                        # Insert author (ignore if exists)
                        self.cur.execute("""
                            INSERT INTO arxiv_authors (author_name)
                            VALUES (%s)
                            ON CONFLICT (author_name) DO NOTHING
                            RETURNING id
                        """, (author_name,))
                        
                        result = self.cur.fetchone()
                        if result:
                            author_id = result[0]
                            self.stats['authors_created'] += 1
                        else:
                            # Get existing author ID
                            self.cur.execute(
                                "SELECT id FROM arxiv_authors WHERE author_name = %s",
                                (author_name,)
                            )
                            author_id = self.cur.fetchone()[0]
                        
                        # Create paper-author relationship
                        self.cur.execute("""
                            INSERT INTO arxiv_paper_authors (paper_id, author_id, author_position)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (paper_id, author_id) DO UPDATE
                            SET author_position = EXCLUDED.author_position
                        """, (paper_id, author_id, position))
                        
                        if self.cur.rowcount > 0:
                            relationships += 1
                            self.stats['relationships_created'] += 1
                            
            except Exception as e:
                logger.debug(f"Failed to process author {author_parts} for {paper_id}: {e}")
        
        return relationships
    
    def process_batch(self, papers: List[Dict], commit_interval: int = 100) -> Dict:
        """
        Process a batch of papers with periodic commits.
        
        Args:
            papers: List of paper dictionaries
            commit_interval: Commit after N successful papers
            
        Returns:
            Processing statistics
        """
        successful = 0
        failed = 0
        
        for i, paper in enumerate(papers):
            if self.process_paper(paper):
                successful += 1
                
                # Periodic commit
                if successful % commit_interval == 0:
                    self.conn.commit()
                    logger.debug(f"Committed after {successful} papers")
            else:
                failed += 1
        
        # Final commit
        self.conn.commit()
        
        return {
            'successful': successful,
            'failed': failed,
            'stats': self.stats
        }
    
    def get_stats(self) -> Dict:
        """Get current processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'papers_inserted': 0,
            'papers_updated': 0,
            'versions_created': 0,
            'authors_created': 0,
            'relationships_created': 0,
            'errors': []
        }