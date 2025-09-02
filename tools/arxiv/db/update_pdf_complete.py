#!/usr/bin/env python3
"""
Complete PDF update - sets both has_pdf flags and file sizes
This combines artifact scanning and size updating in one efficient pass
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple
import argparse

import structlog
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.arxiv.db.config import load_config
from tools.arxiv.db.pg import get_connection

logger = structlog.get_logger()


def scan_all_pdfs(base_path: str = "/bulk-store/arxiv-data/pdf") -> Dict[str, Tuple[str, int]]:
    """
    Scan all PDFs and return mapping of arxiv_id to (path, size).
    
    Returns:
        Dict mapping arxiv_id to tuple of (file_path, size_in_bytes)
    """
    pdf_data = {}
    base = Path(base_path)
    
    logger.info(f"Scanning PDFs in {base_path}")
    
    # Get all PDF files
    pdf_files = list(base.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_path in tqdm(pdf_files, desc="Scanning PDFs"):
        try:
            # Extract arxiv_id from path
            # File format: /bulk-store/arxiv-data/pdf/YYMM/YYMM.NNNNN.pdf
            # Database format: YYMM.NNNNN (with leading zeros)
            # The filename itself is the arxiv_id!
            arxiv_id = pdf_path.stem  # e.g., "2401.00001"
            
            # Get file size
            size = pdf_path.stat().st_size
            
            pdf_data[arxiv_id] = (str(pdf_path), size)
            
        except Exception as e:
            logger.debug(f"Error processing {pdf_path}: {e}")
            continue
    
    return pdf_data


def estimate_complexity(size_bytes: int) -> str:
    """Estimate processing complexity based on file size."""
    mb = size_bytes / (1024 * 1024)
    
    if mb < 1:
        return 'simple'
    elif mb < 5:
        return 'moderate'
    else:
        return 'complex'


def update_database_efficiently(pdf_data: Dict[str, Tuple[str, int]], batch_size: int = 10000):
    """
    Update database with PDF information using efficient batch updates.
    
    Args:
        pdf_data: Dictionary mapping arxiv_id to (path, size)
        batch_size: Number of records to update per batch
    """
    config = load_config('tools/arxiv/configs/db.yaml')
    
    updated = 0
    not_found = 0
    
    with get_connection(config.postgres) as conn:
        cur = conn.cursor()
        
        # Process in batches for efficiency
        items = list(pdf_data.items())
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        logger.info(f"Updating {len(items)} records in {total_batches} batches")
        
        for i in tqdm(range(0, len(items), batch_size), desc="Updating database"):
            batch = items[i:i + batch_size]
            
            # Build batch update using VALUES list
            values = []
            for arxiv_id, (path, size) in batch:
                complexity = estimate_complexity(size)
                values.append(f"('{arxiv_id}', true, {size}, '{path}', '{complexity}')")
            
            if values:
                # Use UPDATE FROM VALUES for efficient batch update
                update_sql = f"""
                    UPDATE papers p
                    SET has_pdf = v.has_pdf,
                        pdf_size_bytes = v.pdf_size_bytes,
                        pdf_path = v.pdf_path,
                        processing_complexity = v.processing_complexity
                    FROM (VALUES {','.join(values)}) 
                        AS v(arxiv_id, has_pdf, pdf_size_bytes, pdf_path, processing_complexity)
                    WHERE p.arxiv_id = v.arxiv_id
                """
                
                cur.execute(update_sql)
                updated += cur.rowcount
                not_found += len(batch) - cur.rowcount
                
                # Commit every batch
                conn.commit()
        
        logger.info(f"Update complete: {updated} updated, {not_found} not found in database")
        
        # Get final statistics
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN has_pdf THEN 1 END) as with_pdf,
                COUNT(pdf_size_bytes) as with_size,
                AVG(pdf_size_bytes)/1024/1024 as avg_mb,
                MIN(pdf_size_bytes)/1024/1024 as min_mb,
                MAX(pdf_size_bytes)/1024/1024 as max_mb,
                SUM(pdf_size_bytes)/1024/1024/1024 as total_gb
            FROM papers
            WHERE has_pdf = true
        """)
        
        stats = cur.fetchone()
        
        print("\n" + "=" * 60)
        print("DATABASE UPDATE COMPLETE")
        print("=" * 60)
        print(f"Total papers in DB: {stats[0]:,}")
        print(f"Papers with PDF: {stats[1]:,}")
        print(f"Papers with size: {stats[2]:,}")
        if stats[3]:
            print(f"Average PDF size: {stats[3]:.2f} MB")
            print(f"Min PDF size: {stats[4]:.2f} MB")
            print(f"Max PDF size: {stats[5]:.2f} MB")
            print(f"Total storage: {stats[6]:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Complete PDF update for ArXiv database")
    parser.add_argument(
        "--pdf-dir",
        default="/bulk-store/arxiv-data/pdf",
        help="Base directory for PDF files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for database updates"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ['POSTGRES_USER'] = 'postgres'
    os.environ['PGPASSWORD'] = '1luv93ngu1n$'
    
    # Scan PDFs
    print(f"Scanning PDFs in {args.pdf_dir}...")
    pdf_data = scan_all_pdfs(args.pdf_dir)
    
    if args.limit:
        pdf_data = dict(list(pdf_data.items())[:args.limit])
        print(f"Limited to {args.limit} files for testing")
    
    print(f"Found {len(pdf_data)} PDF files")
    
    # Calculate some statistics
    if pdf_data:
        sizes = [size for _, size in pdf_data.values()]
        avg_size = sum(sizes) / len(sizes)
        total_size = sum(sizes)
        
        print(f"Average size: {avg_size/1024/1024:.2f} MB")
        print(f"Total size: {total_size/1024/1024/1024:.2f} GB")
    
    # Update database
    print("\nUpdating database...")
    update_database_efficiently(pdf_data, args.batch_size)
    
    return 0


if __name__ == "__main__":
    exit(main())