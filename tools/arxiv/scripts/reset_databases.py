#!/usr/bin/env python3
"""
Clean Database Reset Script for Overnight Run
============================================

Comprehensive database cleanup script to prepare for overnight verification run.
Clears both PostgreSQL processing flags and ArangoDB collections for a fresh start.

Following Actor-Network Theory: This script resets the state of database actants
to ensure clean baseline for conveyance measurement C = (W·R·H)/T · Ctx^α.
"""

import os
import sys
import psycopg2
import argparse
import logging
from datetime import datetime
from pathlib import Path
from arango import ArangoClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_postgresql(pg_password: str, dry_run: bool = False):
    """
    Reset PostgreSQL processing flags for fresh run.
    
    Args:
        pg_password: PostgreSQL password
        dry_run: If True, only show what would be done
    """
    logger.info("=" * 60)
    logger.info("CLEANING POSTGRESQL DATABASE")
    logger.info("=" * 60)
    
    try:
        # Connect to PostgreSQL (Avernus database)
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='Avernus',  # Updated to use Avernus
            user='postgres',
            password=pg_password
        )
        cur = conn.cursor()
        
        # Check current processing status
        cur.execute("""
            SELECT 
                COUNT(*) as total_papers,
                COUNT(CASE WHEN arango_processed = true THEN 1 END) as processed_papers,
                COUNT(CASE WHEN embeddings_created = true THEN 1 END) as embeddings_created
            FROM arxiv_papers
        """)
        
        stats = cur.fetchone()
        logger.info(f"Current PostgreSQL status:")
        logger.info(f"  Total papers: {stats[0]:,}")
        logger.info(f"  ArangoDB processed: {stats[1]:,}")
        logger.info(f"  Embeddings created: {stats[2]:,}")
        
        if dry_run:
            logger.info("DRY RUN: Would reset all processing flags to false")
        else:
            # Reset all processing flags (only columns that exist)
            logger.info("Resetting all processing flags...")
            
            cur.execute("""
                UPDATE arxiv_papers 
                SET 
                    arango_processed = false,
                    embeddings_created = false,
                    embeddings_created_at = NULL,
                    embeddings_updated_at = NULL,
                    chunk_count = NULL,
                    embedding_model = NULL,
                    embedding_date = NULL,
                    arango_sync_status = NULL,
                    has_pdf = false,
                    has_latex_source = false
                WHERE arango_processed = true OR embeddings_created = true
            """)
            
            affected = cur.rowcount
            conn.commit()
            
            logger.info(f"✓ Reset processing flags for {affected:,} papers")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"PostgreSQL cleanup failed: {e}")
        raise


def clean_arangodb(arango_password: str, dry_run: bool = False):
    """
    Clear ArangoDB collections for fresh run.
    
    Args:
        arango_password: ArangoDB password
        dry_run: If True, only show what would be done
    """
    logger.info("=" * 60)
    logger.info("CLEANING ARANGODB COLLECTIONS")
    logger.info("=" * 60)
    
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts='http://192.168.1.69:8529')
        db = client.db('academy_store', username='root', password=arango_password)
        
        # Collections to clean (unified approach)
        collections_to_clean = [
            'arxiv_unified_embeddings',  # Combined PDF+LaTeX embeddings
            'arxiv_structures'           # Extracted equations, tables, etc.
        ]
        
        for collection_name in collections_to_clean:
            if db.has_collection(collection_name):
                count = db.collection(collection_name).count()
                logger.info(f"Collection {collection_name}: {count:,} documents")
                
                if dry_run:
                    logger.info(f"  DRY RUN: Would truncate {collection_name}")
                else:
                    if count > 0:
                        db.collection(collection_name).truncate()
                        logger.info(f"  ✓ Truncated {collection_name}")
                    else:
                        logger.info(f"  - {collection_name} already empty")
            else:
                logger.info(f"Collection {collection_name}: NOT FOUND")
                if not dry_run:
                    # Create missing collection
                    db.create_collection(collection_name)
                    logger.info(f"  ✓ Created {collection_name}")
        
        logger.info("ArangoDB cleanup completed")
        
    except Exception as e:
        logger.error(f"ArangoDB cleanup failed: {e}")
        raise


def clean_checkpoint_files(dry_run: bool = False):
    """
    Remove checkpoint files for fresh start.
    
    Args:
        dry_run: If True, only show what would be done
    """
    logger.info("=" * 60)
    logger.info("CLEANING CHECKPOINT FILES")
    logger.info("=" * 60)
    
    checkpoint_files = [
        'hybrid_checkpoint.json',
        'arxiv_pipeline_checkpoint.json',
        'pipeline_checkpoint.json',
        'unified_checkpoint.json'  # Add unified pipeline checkpoint
    ]
    
    for filename in checkpoint_files:
        filepath = Path(filename)
        if filepath.exists():
            if dry_run:
                logger.info(f"DRY RUN: Would remove {filename}")
            else:
                filepath.unlink()
                logger.info(f"✓ Removed {filename}")
        else:
            logger.info(f"- {filename} not found")


def verify_clean_state(pg_password: str, arango_password: str):
    """
    Verify that databases are in clean state for fresh run.
    
    Args:
        pg_password: PostgreSQL password
        arango_password: ArangoDB password
    """
    logger.info("=" * 60)
    logger.info("VERIFYING CLEAN STATE")
    logger.info("=" * 60)
    
    # Check PostgreSQL
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='arxiv_datalake',
            user='postgres',
            password=pg_password
        )
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                COUNT(CASE WHEN arango_processed = true THEN 1 END) as still_processed,
                COUNT(CASE WHEN pdf_embedded = true THEN 1 END) as still_embedded
            FROM arxiv_papers
        """)
        
        pg_check = cur.fetchone()
        if pg_check[0] == 0 and pg_check[1] == 0:
            logger.info("✓ PostgreSQL: All processing flags reset")
        else:
            logger.error(f"✗ PostgreSQL: {pg_check[0]} still processed, {pg_check[1]} still embedded")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"PostgreSQL verification failed: {e}")
    
    # Check ArangoDB
    try:
        client = ArangoClient(hosts='http://192.168.1.69:8529')
        db = client.db('academy_store', username='root', password=arango_password)
        
        total_docs = 0
        for collection_name in ['arxiv_pdf_embeddings', 'arxiv_latex_embeddings', 'arxiv_structures']:
            if db.has_collection(collection_name):
                count = db.collection(collection_name).count()
                total_docs += count
        
        if total_docs == 0:
            logger.info("✓ ArangoDB: All collections empty")
        else:
            logger.error(f"✗ ArangoDB: {total_docs} documents still remain")
            
    except Exception as e:
        logger.error(f"ArangoDB verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Clean databases for overnight verification run'
    )
    parser.add_argument('--pg-password', 
                       help='PostgreSQL password (or use PGPASSWORD env var)')
    parser.add_argument('--arango-password',
                       help='ArangoDB password (or use ARANGO_PASSWORD env var)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--skip-postgresql', action='store_true',
                       help='Skip PostgreSQL cleanup')
    parser.add_argument('--skip-arangodb', action='store_true',
                       help='Skip ArangoDB cleanup')
    parser.add_argument('--skip-checkpoints', action='store_true',
                       help='Skip checkpoint file cleanup')
    
    args = parser.parse_args()
    
    # Get passwords
    pg_password = args.pg_password or os.environ.get('PGPASSWORD')
    if not pg_password and not args.skip_postgresql:
        logger.error("PostgreSQL password required (use --pg-password or PGPASSWORD env var)")
        sys.exit(1)
    
    arango_password = args.arango_password or os.environ.get('ARANGO_PASSWORD')
    if not arango_password and not args.skip_arangodb:
        logger.error("ArangoDB password required (use --arango-password or ARANGO_PASSWORD env var)")
        sys.exit(1)
    
    logger.info("DATABASE CLEANUP FOR OVERNIGHT RUN")
    logger.info(f"Dry run mode: {args.dry_run}")
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Clean PostgreSQL
        if not args.skip_postgresql:
            clean_postgresql(pg_password, args.dry_run)
        
        # Clean ArangoDB
        if not args.skip_arangodb:
            clean_arangodb(arango_password, args.dry_run)
        
        # Clean checkpoint files
        if not args.skip_checkpoints:
            clean_checkpoint_files(args.dry_run)
        
        # Verify clean state
        if not args.dry_run:
            verify_clean_state(pg_password, arango_password)
        
        logger.info("=" * 60)
        if args.dry_run:
            logger.info("DRY RUN COMPLETED - No changes made")
        else:
            logger.info("DATABASE CLEANUP COMPLETED SUCCESSFULLY")
            logger.info("Ready for overnight verification run!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()