#!/usr/bin/env python3
"""
Unified Database Verification and Sync Status Tool
===================================================

Combines PostgreSQL schema verification with bidirectional sync status checking.
Provides comprehensive database health and sync monitoring in one tool.

Usage:
    python verify_database_sync.py --password YOUR_PASSWORD [options]
"""

import os
import sys
import psycopg2
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from arango import ArangoClient
import json


class DatabaseSyncVerifier:
    """
    Unified verifier for PostgreSQL schema and ArangoDB sync status.
    
    Following Actor-Network Theory: This verifier acts as the auditor
    ensuring proper translation between PostgreSQL and ArangoDB actants.
    """
    
    def __init__(self, pg_password: str, arango_password: Optional[str] = None,
                 pg_host: str = 'localhost', pg_database: str = 'arxiv_datalake',
                 arango_host: str = 'localhost', arango_port: int = 8529):
        """
        Initialize verifier with database connections.
        
        Args:
            pg_password: PostgreSQL password
            arango_password: ArangoDB password (optional for sync checks)
            pg_host: PostgreSQL host
            pg_database: PostgreSQL database name
            arango_host: ArangoDB host
            arango_port: ArangoDB port
        """
        self.pg_password = pg_password
        self.arango_password = arango_password
        self.pg_host = pg_host
        self.pg_database = pg_database
        self.arango_host = arango_host
        self.arango_port = arango_port
        
        # Connect to PostgreSQL
        self.pg_conn = psycopg2.connect(
            host=pg_host,
            database=pg_database,
            user='postgres',
            password=pg_password
        )
        self.pg_cur = self.pg_conn.cursor()
        
        # Connect to ArangoDB if password provided
        self.arango_db = None
        if arango_password:
            try:
                client = ArangoClient(hosts=f'http://{arango_host}:{arango_port}')
                self.arango_db = client.db(
                    'academy_store',
                    username='root',
                    password=arango_password
                )
            except Exception as e:
                print(f"⚠️  Could not connect to ArangoDB: {e}")
                print("   Sync verification will be skipped")
    
    def verify_schema(self) -> Dict:
        """
        Verify PostgreSQL schema completeness.
        
        Returns:
            Dictionary of verification results
        """
        results = {
            'tables': {},
            'columns': {},
            'indexes': {},
            'issues': []
        }
        
        print("=" * 60)
        print("POSTGRESQL SCHEMA VERIFICATION")
        print("=" * 60)
        
        # Required tables
        required_tables = [
            'arxiv_papers',
            'arxiv_versions',
            'arxiv_authors',
            'arxiv_paper_authors'
        ]
        
        print("\n1. TABLE VERIFICATION:")
        print("-" * 40)
        
        for table in required_tables:
            self.pg_cur.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = %s
            """, (table,))
            
            exists = self.pg_cur.fetchone()[0] > 0
            results['tables'][table] = exists
            
            status = "✓" if exists else "✗"
            print(f"  {status} {table}")
            
            if not exists:
                results['issues'].append(f"Missing table: {table}")
        
        # Critical columns for embedding tracking
        print("\n2. EMBEDDING TRACKING COLUMNS:")
        print("-" * 40)
        
        embedding_columns = {
            'embeddings_created': 'BOOLEAN',
            'embeddings_created_at': 'TIMESTAMP',
            'embeddings_updated_at': 'TIMESTAMP',
            'arango_sync_status': 'VARCHAR',
            'arango_document_id': 'VARCHAR'
        }
        
        self.pg_cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'arxiv_papers'
        """)
        
        existing_columns = {row[0]: row[1] for row in self.pg_cur.fetchall()}
        
        for col, expected_type in embedding_columns.items():
            exists = col in existing_columns
            results['columns'][col] = exists
            
            status = "✓" if exists else "✗"
            print(f"  {status} {col} ({expected_type})")
            
            if not exists:
                results['issues'].append(f"Missing column: {col}")
        
        # PDF and LaTeX tracking columns
        print("\n3. FILE TRACKING COLUMNS:")
        print("-" * 40)
        
        file_columns = {
            'has_pdf': 'BOOLEAN',
            'has_latex_source': 'BOOLEAN',
            'pdf_path': 'TEXT',
            'latex_path': 'TEXT',
            'pdf_size_mb': 'REAL',
            'source_format': 'VARCHAR'
        }
        
        for col, expected_type in file_columns.items():
            exists = col in existing_columns
            status = "✓" if exists else "✗"
            print(f"  {status} {col} ({expected_type})")
            
            if not exists:
                results['issues'].append(f"Missing column: {col}")
        
        # Check indexes
        print("\n4. INDEX VERIFICATION:")
        print("-" * 40)
        
        self.pg_cur.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'arxiv_papers'
        """)
        
        indexes = [row[0] for row in self.pg_cur.fetchall()]
        
        required_indexes = [
            'idx_arxiv_papers_embeddings_created',
            'idx_arxiv_papers_has_pdf',
            'idx_arxiv_papers_has_latex'
        ]
        
        for idx in required_indexes:
            exists = idx in indexes
            results['indexes'][idx] = exists
            status = "✓" if exists else "✗"
            print(f"  {status} {idx}")
            
            if not exists:
                results['issues'].append(f"Missing index: {idx}")
        
        return results
    
    def verify_sync_status(self) -> Dict:
        """
        Verify bidirectional sync status between PostgreSQL and ArangoDB.
        
        Returns:
            Dictionary of sync statistics
        """
        if not self.arango_db:
            print("\n⚠️  ArangoDB not connected - skipping sync verification")
            return {}
        
        print("\n" + "=" * 60)
        print("BIDIRECTIONAL SYNC VERIFICATION")
        print("=" * 60)
        
        stats = {}
        
        # 1. Papers with embeddings in PostgreSQL
        self.pg_cur.execute("""
            SELECT COUNT(*) FROM arxiv_papers 
            WHERE embeddings_created = TRUE
        """)
        pg_with_embeddings = self.pg_cur.fetchone()[0]
        stats['pg_with_embeddings'] = pg_with_embeddings
        
        # 2. Papers pending sync to ArangoDB
        self.pg_cur.execute("""
            SELECT COUNT(*) FROM arxiv_papers 
            WHERE embeddings_created = TRUE 
            AND arango_sync_status = 'pending'
        """)
        pending_to_arango = self.pg_cur.fetchone()[0]
        stats['pending_to_arango'] = pending_to_arango
        
        # 3. Papers synced to ArangoDB
        self.pg_cur.execute("""
            SELECT COUNT(*) FROM arxiv_papers 
            WHERE embeddings_created = TRUE 
            AND arango_sync_status = 'synced'
        """)
        synced_to_arango = self.pg_cur.fetchone()[0]
        stats['synced_to_arango'] = synced_to_arango
        
        # 4. Check ArangoDB collections
        try:
            # Count documents in ArangoDB
            arxiv_metadata = self.arango_db.collection('arxiv_metadata')
            arxiv_chunks = self.arango_db.collection('arxiv_chunks')
            
            arango_metadata_count = arxiv_metadata.count()
            arango_chunks_count = arxiv_chunks.count()
            
            stats['arango_metadata_count'] = arango_metadata_count
            stats['arango_chunks_count'] = arango_chunks_count
            
            # Find papers in ArangoDB not marked as synced in PostgreSQL
            query = """
            FOR doc IN arxiv_metadata
                RETURN doc.arxiv_id
            """
            arango_ids = set(self.arango_db.aql.execute(query))
            
            # Get PostgreSQL IDs marked as synced
            self.pg_cur.execute("""
                SELECT id FROM arxiv_papers 
                WHERE arango_sync_status = 'synced'
            """)
            pg_synced_ids = set(row[0] for row in self.pg_cur.fetchall())
            
            # Find discrepancies
            in_arango_not_marked = arango_ids - pg_synced_ids
            marked_not_in_arango = pg_synced_ids - arango_ids
            
            stats['in_arango_not_marked'] = len(in_arango_not_marked)
            stats['marked_not_in_arango'] = len(marked_not_in_arango)
            
        except Exception as e:
            print(f"  ⚠️  Error checking ArangoDB: {e}")
            stats['arango_error'] = str(e)
        
        # Display results
        print("\n1. POSTGRESQL STATUS:")
        print("-" * 40)
        print(f"  Papers with embeddings: {pg_with_embeddings:,}")
        print(f"  Pending sync to ArangoDB: {pending_to_arango:,}")
        print(f"  Marked as synced: {synced_to_arango:,}")
        
        if 'arango_metadata_count' in stats:
            print("\n2. ARANGODB STATUS:")
            print("-" * 40)
            print(f"  Documents in arxiv_metadata: {stats['arango_metadata_count']:,}")
            print(f"  Documents in arxiv_chunks: {stats['arango_chunks_count']:,}")
            
            print("\n3. SYNC DISCREPANCIES:")
            print("-" * 40)
            
            if stats['in_arango_not_marked'] > 0:
                print(f"  ⚠️  In ArangoDB but not marked: {stats['in_arango_not_marked']}")
            else:
                print(f"  ✓ All ArangoDB docs properly marked")
            
            if stats['marked_not_in_arango'] > 0:
                print(f"  ⚠️  Marked synced but not in ArangoDB: {stats['marked_not_in_arango']}")
            else:
                print(f"  ✓ All marked docs exist in ArangoDB")
        
        return stats
    
    def verify_data_quality(self) -> Dict:
        """
        Verify data quality and completeness.
        
        Returns:
            Dictionary of quality metrics
        """
        print("\n" + "=" * 60)
        print("DATA QUALITY VERIFICATION")
        print("=" * 60)
        
        quality = {}
        
        # Total papers
        self.pg_cur.execute("SELECT COUNT(*) FROM arxiv_papers")
        total_papers = self.pg_cur.fetchone()[0]
        quality['total_papers'] = total_papers
        
        # Papers with PDFs
        self.pg_cur.execute("SELECT COUNT(*) FROM arxiv_papers WHERE has_pdf = TRUE")
        papers_with_pdf = self.pg_cur.fetchone()[0]
        quality['papers_with_pdf'] = papers_with_pdf
        quality['pdf_percentage'] = (papers_with_pdf / total_papers * 100) if total_papers > 0 else 0
        
        # Papers with LaTeX
        self.pg_cur.execute("SELECT COUNT(*) FROM arxiv_papers WHERE has_latex_source = TRUE")
        papers_with_latex = self.pg_cur.fetchone()[0]
        quality['papers_with_latex'] = papers_with_latex
        quality['latex_percentage'] = (papers_with_latex / total_papers * 100) if total_papers > 0 else 0
        
        # Check for orphaned records
        self.pg_cur.execute("""
            SELECT COUNT(*) FROM arxiv_versions v
            WHERE NOT EXISTS (
                SELECT 1 FROM arxiv_papers p WHERE p.id = v.paper_id
            )
        """)
        orphaned_versions = self.pg_cur.fetchone()[0]
        quality['orphaned_versions'] = orphaned_versions
        
        # Check for papers without authors
        self.pg_cur.execute("""
            SELECT COUNT(*) FROM arxiv_papers p
            WHERE NOT EXISTS (
                SELECT 1 FROM arxiv_paper_authors pa WHERE pa.paper_id = p.id
            )
        """)
        papers_without_authors = self.pg_cur.fetchone()[0]
        quality['papers_without_authors'] = papers_without_authors
        
        # Display results
        print("\n1. DATA COMPLETENESS:")
        print("-" * 40)
        print(f"  Total papers: {total_papers:,}")
        print(f"  Papers with PDF: {papers_with_pdf:,} ({quality['pdf_percentage']:.1f}%)")
        print(f"  Papers with LaTeX: {papers_with_latex:,} ({quality['latex_percentage']:.1f}%)")
        
        print("\n2. DATA INTEGRITY:")
        print("-" * 40)
        
        if orphaned_versions > 0:
            print(f"  ⚠️  Orphaned versions: {orphaned_versions}")
        else:
            print(f"  ✓ No orphaned versions")
        
        if papers_without_authors > 0:
            print(f"  ⚠️  Papers without authors: {papers_without_authors}")
        else:
            print(f"  ✓ All papers have authors")
        
        # Experiment window
        self.pg_cur.execute("""
            SELECT COUNT(DISTINCT p.id)
            FROM arxiv_papers p
            JOIN arxiv_versions v ON p.id = v.paper_id
            WHERE v.created_date >= '2012-12-01' 
            AND v.created_date <= '2016-08-31'
        """)
        experiment_papers = self.pg_cur.fetchone()[0]
        quality['experiment_papers'] = experiment_papers
        
        print("\n3. EXPERIMENT WINDOW (Dec 2012 - Aug 2016):")
        print("-" * 40)
        print(f"  Papers in window: {experiment_papers:,}")
        
        return quality
    
    def generate_fix_script(self, schema_results: Dict) -> Optional[str]:
        """
        Generate SQL script to fix schema issues.
        
        Args:
            schema_results: Results from verify_schema()
            
        Returns:
            SQL script or None if no issues
        """
        if not schema_results.get('issues'):
            return None
        
        sql_commands = []
        sql_commands.append("-- SQL script to fix schema issues")
        sql_commands.append(f"-- Generated: {datetime.now().isoformat()}")
        sql_commands.append("")
        
        # Check for missing embedding columns
        missing_cols = []
        for col in ['embeddings_created', 'embeddings_created_at', 
                   'embeddings_updated_at', 'arango_sync_status', 'arango_document_id']:
            if not schema_results['columns'].get(col):
                missing_cols.append(col)
        
        if missing_cols:
            sql_commands.append("-- Add missing embedding tracking columns")
            sql_commands.append("ALTER TABLE arxiv_papers")
            
            col_defs = []
            if 'embeddings_created' in missing_cols:
                col_defs.append("  ADD COLUMN IF NOT EXISTS embeddings_created BOOLEAN DEFAULT FALSE")
            if 'embeddings_created_at' in missing_cols:
                col_defs.append("  ADD COLUMN IF NOT EXISTS embeddings_created_at TIMESTAMP")
            if 'embeddings_updated_at' in missing_cols:
                col_defs.append("  ADD COLUMN IF NOT EXISTS embeddings_updated_at TIMESTAMP")
            if 'arango_sync_status' in missing_cols:
                col_defs.append("  ADD COLUMN IF NOT EXISTS arango_sync_status VARCHAR(20) DEFAULT 'pending'")
            if 'arango_document_id' in missing_cols:
                col_defs.append("  ADD COLUMN IF NOT EXISTS arango_document_id VARCHAR(255)")
            
            sql_commands.append(",\n".join(col_defs) + ";")
            sql_commands.append("")
        
        # Check for missing indexes
        missing_indexes = []
        for idx in ['idx_arxiv_papers_embeddings_created']:
            if not schema_results['indexes'].get(idx):
                missing_indexes.append(idx)
        
        if missing_indexes:
            sql_commands.append("-- Add missing indexes")
            if 'idx_arxiv_papers_embeddings_created' in missing_indexes:
                sql_commands.append("CREATE INDEX IF NOT EXISTS idx_arxiv_papers_embeddings_created")
                sql_commands.append("  ON arxiv_papers(embeddings_created);")
            sql_commands.append("")
        
        return "\n".join(sql_commands)
    
    def run_comprehensive_check(self, output_format: str = 'text') -> Dict:
        """
        Run all verification checks and output results.
        
        Args:
            output_format: 'text' or 'json'
            
        Returns:
            Complete verification results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'database': self.pg_database,
            'schema': {},
            'sync': {},
            'quality': {},
            'summary': {}
        }
        
        # Run all checks
        results['schema'] = self.verify_schema()
        results['sync'] = self.verify_sync_status()
        results['quality'] = self.verify_data_quality()
        
        # Generate summary
        schema_ok = len(results['schema'].get('issues', [])) == 0
        sync_ok = (results['sync'].get('pending_to_arango', 0) == 0 and
                  results['sync'].get('in_arango_not_marked', 0) == 0 and
                  results['sync'].get('marked_not_in_arango', 0) == 0)
        quality_ok = (results['quality'].get('orphaned_versions', 0) == 0 and
                     results['quality'].get('papers_without_authors', 0) == 0)
        
        results['summary'] = {
            'schema_ok': schema_ok,
            'sync_ok': sync_ok,
            'quality_ok': quality_ok,
            'overall_health': 'GOOD' if (schema_ok and sync_ok and quality_ok) else 'ISSUES FOUND'
        }
        
        # Generate fix script if needed
        if not schema_ok:
            fix_script = self.generate_fix_script(results['schema'])
            if fix_script:
                print("\n" + "=" * 60)
                print("RECOMMENDED FIXES")
                print("=" * 60)
                print("\nSave the following SQL to fix schema issues:")
                print("-" * 40)
                print(fix_script)
                
                # Save to file
                fix_file = Path('fix_schema.sql')
                with open(fix_file, 'w') as f:
                    f.write(fix_script)
                print(f"\n✓ Fix script saved to: {fix_file}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        
        if results['summary']['overall_health'] == 'GOOD':
            print("\n✅ All checks passed - database is healthy!")
        else:
            print("\n⚠️  Issues found:")
            if not schema_ok:
                print(f"  - Schema issues: {len(results['schema']['issues'])}")
            if not sync_ok:
                print(f"  - Sync issues detected")
            if not quality_ok:
                print(f"  - Data quality issues")
        
        if output_format == 'json':
            # Save JSON report
            report_file = Path(f'db_verification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📊 Full report saved to: {report_file}")
        
        return results
    
    def __del__(self):
        """Clean up database connections."""
        try:
            # Use getattr to safely retrieve attributes during shutdown
            pg_cur = getattr(self, 'pg_cur', None)
            if pg_cur:
                try:
                    pg_cur.close()
                except Exception:
                    pass  # Ignore errors during cleanup
                self.pg_cur = None
            
            pg_conn = getattr(self, 'pg_conn', None)
            if pg_conn:
                try:
                    pg_conn.close()
                except Exception:
                    pass  # Ignore errors during cleanup
                self.pg_conn = None
        except (AttributeError, BaseException):
            # Catch any possible errors during interpreter shutdown
            pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Unified database verification and sync status tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic schema verification (PostgreSQL only)
  %(prog)s --password "$PGPASSWORD"
  
  # Full verification with ArangoDB sync status
  %(prog)s --password "$PGPASSWORD" --arango-password "$ARANGO_PASSWORD"
  
  # Output JSON report
  %(prog)s --password "$PGPASSWORD" --arango-password "$ARANGO_PASSWORD" --json
  
  # Check specific database
  %(prog)s --password "$PGPASSWORD" --database arxiv_datalake_20250118_120000
        """
    )
    
    parser.add_argument('--password', required=True,
                       help='PostgreSQL password')
    parser.add_argument('--arango-password',
                       help='ArangoDB password (for sync verification)')
    parser.add_argument('--database', default='arxiv_datalake',
                       help='PostgreSQL database name')
    parser.add_argument('--pg-host', default='localhost',
                       help='PostgreSQL host')
    parser.add_argument('--arango-host', default='localhost',
                       help='ArangoDB host')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    # Handle environment variables
    if not args.password:
        args.password = os.environ.get('PGPASSWORD')
    if not args.arango_password:
        args.arango_password = os.environ.get('ARANGO_PASSWORD')
    
    try:
        verifier = DatabaseSyncVerifier(
            pg_password=args.password,
            arango_password=args.arango_password,
            pg_host=args.pg_host,
            pg_database=args.database,
            arango_host=args.arango_host
        )
        
        output_format = 'json' if args.json else 'text'
        results = verifier.run_comprehensive_check(output_format)
        
        # Exit with appropriate code
        sys.exit(0 if results['summary']['overall_health'] == 'GOOD' else 1)
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()