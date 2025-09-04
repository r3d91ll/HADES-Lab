#!/usr/bin/env python3
"""
ArXiv Metadata Analysis
======================

Analyze the Kaggle ArXiv metadata JSON file to understand all available fields
and create a proper PostgreSQL schema.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_arxiv_metadata(metadata_file: str = "/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json"):
    """Analyze the structure of ArXiv metadata file."""
    
    logger.info(f"Analyzing ArXiv metadata file: {metadata_file}")
    
    # Track all fields found and their types
    all_fields = defaultdict(set)
    sample_records = []
    
    with open(metadata_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > 1000:  # Analyze first 1000 records
                break
            
            try:
                paper = json.loads(line.strip())
                
                # Save first few complete records
                if line_num <= 3:
                    sample_records.append(paper)
                
                # Track field types
                for key, value in paper.items():
                    all_fields[key].add(type(value).__name__)
                    
                    # Handle nested structures
                    if key == 'versions' and isinstance(value, list) and value:
                        version_fields = set(value[0].keys()) if value[0] else set()
                        all_fields['versions_fields'].update(version_fields)
            
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error at line {line_num}: {e}")
    
    # Print analysis
    print(f"\n{'='*60}")
    print("ARXIV METADATA ANALYSIS")
    print(f"{'='*60}")
    print(f"Analyzed {line_num-1} records")
    print(f"\nAll fields found:")
    
    for field, types in sorted(all_fields.items()):
        types_str = ', '.join(types)
        print(f"  {field:<20} : {types_str}")
    
    # Show sample records
    print(f"\n{'='*60}")
    print("SAMPLE RECORDS")
    print(f"{'='*60}")
    
    for i, record in enumerate(sample_records, 1):
        print(f"\n--- Sample {i}: {record.get('id', 'NO_ID')} ---")
        for key, value in record.items():
            if key == 'versions' and isinstance(value, list):
                print(f"  {key}: [{len(value)} versions]")
                if value:
                    print(f"    First version: {value[0]}")
            elif key in ['title', 'abstract'] and isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")


def generate_postgresql_schema():
    """Generate PostgreSQL schema based on ArXiv metadata structure."""
    
    schema_sql = """
-- Enhanced PostgreSQL schema for ArXiv papers with complete metadata
-- Generated based on Kaggle ArXiv metadata analysis

-- Drop existing table (if you want to recreate)
-- DROP TABLE IF EXISTS papers CASCADE;

-- Add missing columns to existing papers table
ALTER TABLE papers 
ADD COLUMN IF NOT EXISTS authors TEXT,
ADD COLUMN IF NOT EXISTS categories TEXT,
ADD COLUMN IF NOT EXISTS comments TEXT,
ADD COLUMN IF NOT EXISTS report_number TEXT,
ADD COLUMN IF NOT EXISTS msc_class TEXT,
ADD COLUMN IF NOT EXISTS acm_class TEXT,
ADD COLUMN IF NOT EXISTS submission_date DATE,
ADD COLUMN IF NOT EXISTS update_date DATE,
ADD COLUMN IF NOT EXISTS versions_count INTEGER,
ADD COLUMN IF NOT EXISTS latest_version TEXT,
ADD COLUMN IF NOT EXISTS authors_parsed JSONB;

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers USING GIN(to_tsvector('english', categories));
CREATE INDEX IF NOT EXISTS idx_papers_authors ON papers USING GIN(to_tsvector('english', authors));
CREATE INDEX IF NOT EXISTS idx_papers_submission_date ON papers(submission_date);
CREATE INDEX IF NOT EXISTS idx_papers_update_date ON papers(update_date);
CREATE INDEX IF NOT EXISTS idx_papers_primary_category ON papers(primary_category);

-- Add comments
COMMENT ON COLUMN papers.authors IS 'Comma-separated list of paper authors';
COMMENT ON COLUMN papers.categories IS 'Space-separated ArXiv categories';
COMMENT ON COLUMN papers.comments IS 'Paper comments from ArXiv';
COMMENT ON COLUMN papers.submission_date IS 'Original paper submission date';
COMMENT ON COLUMN papers.update_date IS 'Last update date';
COMMENT ON COLUMN papers.versions_count IS 'Number of paper versions';
COMMENT ON COLUMN papers.latest_version IS 'Latest version identifier';
COMMENT ON COLUMN papers.authors_parsed IS 'Structured author information as JSON';
"""
    
    print(f"\n{'='*60}")
    print("PROPOSED POSTGRESQL SCHEMA UPDATE")
    print(f"{'='*60}")
    print(schema_sql)
    
    # Save to file
    schema_file = "/home/todd/olympus/HADES-Lab/tools/arxiv/db/add_arxiv_metadata_columns.sql"
    Path(schema_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(schema_file, 'w') as f:
        f.write(schema_sql)
    
    print(f"\nSchema update saved to: {schema_file}")


if __name__ == "__main__":
    analyze_arxiv_metadata()
    generate_postgresql_schema()