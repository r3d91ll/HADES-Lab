#!/usr/bin/env python3
"""
PostgreSQL ArXiv Database Rebuild Script - Fixed Version
========================================================

Fixed version with better error handling and transaction management.
"""

import os
import sys
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Optional
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PostgreSQLRebuilder:
    def __init__(self):
        self.pg_config = {
            'host': 'localhost',
            'database': 'arxiv', 
            'user': 'postgres',
            'password': os.getenv('PGPASSWORD', '')
        }
        self.metadata_file = Path("/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json")
        
    def get_database_connection(self):
        """Get PostgreSQL database connection."""
        return psycopg2.connect(**self.pg_config)
    
    def test_single_insert(self):
        """Test inserting a single record to identify the issue."""
        logger.info("Testing single record insert...")
        
        # Read first few lines to test
        with open(self.metadata_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > 5:  # Test first 5 lines
                    break
                    
                try:
                    paper = json.loads(line.strip())
                    logger.info(f"Line {line_num}: {paper.get('id', 'NO_ID')}")
                    
                    # Try to insert this single record
                    conn = self.get_database_connection()
                    try:
                        with conn:
                            with conn.cursor() as cur:
                                # Simple insert with minimal fields
                                arxiv_id = paper.get('id', '')
                                title = paper.get('title', '').replace('\n', ' ').strip()[:500]  # Truncate title
                                
                                logger.info(f"Attempting insert: {arxiv_id} - {title[:50]}...")
                                
                                cur.execute("""
                                    INSERT INTO papers (arxiv_id, title, has_pdf, has_latex) 
                                    VALUES (%s, %s, %s, %s)
                                    ON CONFLICT (arxiv_id) DO NOTHING
                                """, (arxiv_id, title, False, False))
                                
                                logger.info(f"✅ Successfully inserted line {line_num}")
                    
                    except Exception as e:
                        logger.error(f"❌ Failed to insert line {line_num}: {e}")
                        logger.error(f"   ArXiv ID: {arxiv_id}")
                        logger.error(f"   Title: {title}")
                        return False
                    
                    finally:
                        conn.close()
                
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at line {line_num}: {e}")
                    return False
        
        logger.info("✅ Single insert test completed successfully")
        return True
    
    def check_database_schema(self):
        """Check database schema and constraints."""
        logger.info("Checking database schema...")
        
        conn = self.get_database_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    # Check table structure
                    cur.execute("""
                        SELECT column_name, data_type, character_maximum_length, is_nullable
                        FROM information_schema.columns 
                        WHERE table_name = 'papers'
                        ORDER BY ordinal_position
                    """)
                    
                    columns = cur.fetchall()
                    logger.info("Papers table schema:")
                    for col in columns:
                        logger.info(f"  {col[0]}: {col[1]} (max_length: {col[2]}, nullable: {col[3]})")
                    
                    # Check constraints
                    cur.execute("""
                        SELECT constraint_name, constraint_type 
                        FROM information_schema.table_constraints 
                        WHERE table_name = 'papers'
                    """)
                    
                    constraints = cur.fetchall()
                    logger.info("Table constraints:")
                    for constraint in constraints:
                        logger.info(f"  {constraint[0]}: {constraint[1]}")
        
        except Exception as e:
            logger.error(f"Error checking schema: {e}")
            return False
        
        finally:
            conn.close()
        
        return True
    
    def analyze_metadata_sample(self):
        """Analyze a sample of metadata to understand the data structure."""
        logger.info("Analyzing metadata sample...")
        
        with open(self.metadata_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > 3:
                    break
                
                try:
                    paper = json.loads(line.strip())
                    
                    logger.info(f"\n--- Sample {line_num} ---")
                    logger.info(f"ID: {paper.get('id', 'MISSING')}")
                    logger.info(f"Title length: {len(paper.get('title', ''))}")
                    logger.info(f"Abstract length: {len(paper.get('abstract', ''))}")
                    logger.info(f"Authors type: {type(paper.get('authors', ''))}")
                    logger.info(f"Categories: {paper.get('categories', 'MISSING')}")
                    
                    # Check for any unusual characters or data
                    title = paper.get('title', '')
                    if len(title) > 500:
                        logger.warning(f"Very long title: {len(title)} characters")
                    
                    # Check version info
                    versions = paper.get('versions', [])
                    if versions:
                        logger.info(f"First version date: {versions[0].get('created', 'MISSING')}")
                
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at line {line_num}: {e}")
                    logger.error(f"Line content: {line[:200]}...")


def main():
    # Verify environment variables
    if not os.getenv('PGPASSWORD'):
        print("❌ PGPASSWORD environment variable is required")
        return 1
    
    rebuilder = PostgreSQLRebuilder()
    
    print("🔍 Diagnosing PostgreSQL import issue...")
    
    # Run diagnostics
    print("\n1. Checking database schema...")
    rebuilder.check_database_schema()
    
    print("\n2. Analyzing metadata sample...")
    rebuilder.analyze_metadata_sample()
    
    print("\n3. Testing single record insert...")
    if rebuilder.test_single_insert():
        print("✅ Single insert works - the issue might be with batch processing or specific records")
    else:
        print("❌ Single insert failed - there's a fundamental schema or data issue")
    
    return 0


if __name__ == "__main__":
    exit(main())