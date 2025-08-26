"""
Setup SQLite cache database for on-demand processing.

Imports ArXiv metadata from PostgreSQL into lightweight SQLite cache
for tracking and quick lookups.
"""

import os
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import psycopg2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SQLiteCacheSetup:
    """Setup and populate SQLite cache from PostgreSQL"""
    
    def __init__(
        self,
        sqlite_path: str = '/bulk-store/arxiv-cache.db',
        pg_password: Optional[str] = None
    ):
        """Initialize setup"""
        self.sqlite_path = sqlite_path
        
        # Get password - require explicit parameter or environment variable
        password = pg_password or os.environ.get('PGPASSWORD')
        if not password:
            raise ValueError("PostgreSQL password must be provided via pg_password parameter or PGPASSWORD environment variable")
        
        # PostgreSQL connection
        self.pg_conn = psycopg2.connect(
            host='localhost',
            database='arxiv_datalake',
            user='postgres',
            password=password
        )
        
        # SQLite connection
        self.sqlite_conn = sqlite3.connect(sqlite_path)
        
        logger.info(f"Connected to PostgreSQL and SQLite at {sqlite_path}")
    
    def create_schema(self):
        """Create SQLite schema for paper tracking"""
        cursor = self.sqlite_conn.cursor()
        
        # Main tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_tracking (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,  -- Store abstract for quick search
                categories TEXT,  -- JSON array as text
                authors TEXT,  -- JSON array as text
                created_date TEXT,
                pdf_path TEXT,
                latex_path TEXT,
                download_date TIMESTAMP,
                process_date TIMESTAMP,
                in_arango BOOLEAN DEFAULT 0,
                processing_status TEXT DEFAULT 'not_started',
                error_message TEXT,
                file_size_mb REAL,
                num_chunks INTEGER,
                processing_time_seconds REAL
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON paper_tracking(title)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON paper_tracking(processing_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_process_date ON paper_tracking(process_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_date ON paper_tracking(created_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_in_arango ON paper_tracking(in_arango)")
        
        # Statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS import_stats (
                import_date TIMESTAMP PRIMARY KEY,
                total_papers INTEGER,
                papers_with_pdf INTEGER,
                papers_with_latex INTEGER,
                import_source TEXT
            )
        """)
        
        self.sqlite_conn.commit()
        logger.info("SQLite schema created")
    
    def import_from_postgresql(self, batch_size: int = 10000, limit: Optional[int] = None):
        """
        Import paper metadata from PostgreSQL to SQLite.
        
        Args:
            batch_size: Number of papers to process per batch
            limit: Maximum number of papers to import (None for all)
        """
        logger.info("Starting import from PostgreSQL...")
        
        pg_cursor = self.pg_conn.cursor()
        sqlite_cursor = self.sqlite_conn.cursor()
        
        # Count total papers
        pg_cursor.execute("SELECT COUNT(*) FROM arxiv_papers")
        total_count = pg_cursor.fetchone()[0]
        logger.info(f"Total papers in PostgreSQL: {total_count:,}")
        
        # Import in batches
        offset = 0
        imported = 0
        papers_with_pdf = 0
        papers_with_latex = 0
        
        while True:
            # Fetch batch from PostgreSQL
            query = """
                SELECT 
                    p.arxiv_id,
                    p.title,
                    p.abstract,
                    p.categories,
                    v.created_date,
                    array_agg(DISTINCT a.name) as authors
                FROM arxiv_papers p
                LEFT JOIN arxiv_versions v ON p.id = v.paper_id AND v.version = 'v1'
                LEFT JOIN arxiv_paper_authors pa ON p.id = pa.paper_id
                LEFT JOIN arxiv_authors a ON pa.author_id = a.id
                GROUP BY p.id, p.arxiv_id, p.title, p.abstract, p.categories, v.created_date
                ORDER BY p.id
                LIMIT %s OFFSET %s
            """
            
            pg_cursor.execute(query, (batch_size, offset))
            papers = pg_cursor.fetchall()
            
            if not papers:
                break
            
            # Insert into SQLite
            for paper in papers:
                arxiv_id, title, abstract, categories, created_date, authors = paper
                
                # Check for local PDF
                pdf_path = self._find_pdf_path(arxiv_id)
                if pdf_path:
                    papers_with_pdf += 1
                
                # Convert arrays to JSON strings
                import json
                categories_json = json.dumps(categories) if categories else '[]'
                authors_json = json.dumps(authors) if authors else '[]'
                
                sqlite_cursor.execute("""
                    INSERT OR REPLACE INTO paper_tracking 
                    (arxiv_id, title, abstract, categories, authors, created_date, pdf_path, processing_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    arxiv_id,
                    title,
                    abstract,
                    categories_json,
                    authors_json,
                    created_date.isoformat() if created_date else None,
                    pdf_path,
                    'has_pdf' if pdf_path else 'no_pdf'
                ))
                
                imported += 1
            
            # Commit batch
            self.sqlite_conn.commit()
            offset += batch_size
            
            logger.info(f"Imported {imported:,}/{total_count:,} papers ({imported*100/total_count:.1f}%)")
            
            if limit and imported >= limit:
                logger.info(f"Reached limit of {limit} papers")
                break
        
        # Record import statistics
        sqlite_cursor.execute("""
            INSERT INTO import_stats (import_date, total_papers, papers_with_pdf, papers_with_latex, import_source)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            imported,
            papers_with_pdf,
            papers_with_latex,
            'postgresql_arxiv_datalake'
        ))
        
        self.sqlite_conn.commit()
        pg_cursor.close()
        
        logger.info(f"\nImport complete!")
        logger.info(f"  Total imported: {imported:,}")
        logger.info(f"  Papers with PDF: {papers_with_pdf:,} ({papers_with_pdf*100/imported:.1f}%)")
        
        return imported
    
    def _find_pdf_path(self, arxiv_id: str) -> Optional[str]:
        """Check if PDF exists locally"""
        pdf_base = Path('/bulk-store/arxiv-data/pdf')
        
        # Try different path patterns
        if '.' in arxiv_id:
            # Modern format: YYMM.NNNNN
            year_month = arxiv_id.split('.')[0]
            if len(year_month) == 4:
                pdf_path = pdf_base / year_month / f"{arxiv_id}.pdf"
                if pdf_path.exists():
                    return str(pdf_path)
        
        elif '/' in arxiv_id:
            # Old format: category/YYMMNNN
            category, paper_id = arxiv_id.split('/', 1)
            year_month = paper_id[:4] if len(paper_id) >= 4 else '0000'
            pdf_name = f"{category}_{paper_id}.pdf".replace('/', '_')
            pdf_path = pdf_base / year_month / pdf_name
            if pdf_path.exists():
                return str(pdf_path)
        
        return None
    
    def get_statistics(self):
        """Get cache statistics"""
        cursor = self.sqlite_conn.cursor()
        
        stats = {}
        
        # Total papers
        cursor.execute("SELECT COUNT(*) FROM paper_tracking")
        stats['total_papers'] = cursor.fetchone()[0]
        
        # Papers by status
        cursor.execute("""
            SELECT processing_status, COUNT(*) 
            FROM paper_tracking 
            GROUP BY processing_status
        """)
        stats['by_status'] = dict(cursor.fetchall())
        
        # Papers with PDFs
        cursor.execute("SELECT COUNT(*) FROM paper_tracking WHERE pdf_path IS NOT NULL")
        stats['with_pdf'] = cursor.fetchone()[0]
        
        # Papers processed
        cursor.execute("SELECT COUNT(*) FROM paper_tracking WHERE in_arango = 1")
        stats['in_arango'] = cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Close database connections"""
        self.pg_conn.close()
        self.sqlite_conn.close()


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup SQLite cache for ACID pipeline')
    parser.add_argument('--sqlite-path', default='/bulk-store/arxiv-cache.db',
                      help='Path to SQLite database')
    parser.add_argument('--pg-password', default=None,
                      help='PostgreSQL password')
    parser.add_argument('--batch-size', type=int, default=10000,
                      help='Batch size for import')
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit number of papers to import (for testing)')
    parser.add_argument('--stats-only', action='store_true',
                      help='Only show statistics, don\'t import')
    
    args = parser.parse_args()
    
    setup = SQLiteCacheSetup(args.sqlite_path, args.pg_password)
    
    if args.stats_only:
        stats = setup.get_statistics()
        print("\nSQLite Cache Statistics:")
        print(f"  Total papers: {stats['total_papers']:,}")
        print(f"  With PDFs: {stats['with_pdf']:,}")
        print(f"  In ArangoDB: {stats['in_arango']:,}")
        print("\nBy status:")
        for status, count in stats.get('by_status', {}).items():
            print(f"    {status}: {count:,}")
    else:
        # Create schema
        setup.create_schema()
        
        # Import data
        imported = setup.import_from_postgresql(
            batch_size=args.batch_size,
            limit=args.limit
        )
        
        # Show final statistics
        stats = setup.get_statistics()
        print("\nFinal Statistics:")
        print(f"  Total papers: {stats['total_papers']:,}")
        print(f"  With PDFs: {stats['with_pdf']:,}")
    
    setup.close()


if __name__ == "__main__":
    main()