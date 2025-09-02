"""
Update PDF Size Information in Database

This script scans the actual PDF files and updates the database with
real file sizes for accurate compute cost assessment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import structlog
from tqdm import tqdm

from core.framework.logging import LogManager

# Support running as script or module
try:
    from .config import ArxivDBConfig, load_config
    from .pg import get_connection
    from .utils import normalize_arxiv_id
except Exception:
    from tools.arxiv.db.config import ArxivDBConfig, load_config  # type: ignore
    from tools.arxiv.db.pg import get_connection  # type: ignore
    from tools.arxiv.db.utils import normalize_arxiv_id  # type: ignore

logger = structlog.get_logger()


def scan_pdf_sizes(base_path: str = "/bulk-store/arxiv-data/pdf") -> Dict[str, int]:
    """
    Scan PDF files and get their sizes.
    
    Args:
        base_path: Base directory for PDF files
        
    Returns:
        Dictionary mapping arxiv_id to file size in bytes
    """
    pdf_sizes = {}
    base = Path(base_path)
    
    if not base.exists():
        logger.error(f"PDF directory does not exist: {base_path}")
        return pdf_sizes
    
    logger.info(f"Scanning PDF files in {base_path}")
    
    # Scan all PDF files
    pdf_files = list(base.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_path in tqdm(pdf_files, desc="Scanning PDFs"):
        try:
            # Extract arxiv_id from filename
            # Format: YYMM/NNNN.pdf or YYMM/NNNNN.pdf
            stem = pdf_path.stem
            parent = pdf_path.parent.name
            
            # Construct arxiv_id
            if parent.isdigit() and len(parent) == 4:  # YYMM format
                arxiv_id = f"{parent}.{stem}"
            else:
                # Try to extract from stem directly
                arxiv_id = stem
            
            # Normalize the ID
            arxiv_id = normalize_arxiv_id(arxiv_id)
            
            # Get file size
            size = pdf_path.stat().st_size
            pdf_sizes[arxiv_id] = size
            
        except Exception as e:
            logger.warning(f"Error processing {pdf_path}: {e}")
            continue
    
    return pdf_sizes


def estimate_complexity(size_bytes: int) -> str:
    """
    Estimate processing complexity based on file size.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Complexity level: 'simple', 'moderate', or 'complex'
    """
    mb = size_bytes / (1024 * 1024)
    
    if mb < 1:
        return 'simple'
    elif mb < 5:
        return 'moderate'
    else:
        return 'complex'


def update_database(
    config: ArxivDBConfig,
    pdf_sizes: Dict[str, int],
    batch_size: int = 1000
) -> Tuple[int, int]:
    """
    Update database with PDF sizes.
    
    Args:
        config: Database configuration
        pdf_sizes: Dictionary of arxiv_id to size
        batch_size: Number of records to update in each batch
        
    Returns:
        Tuple of (updated_count, failed_count)
    """
    updated = 0
    failed = 0
    
    with get_connection(config.postgres) as conn:
        cur = conn.cursor()
        
        # Process in batches
        items = list(pdf_sizes.items())
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        logger.info(f"Updating {len(items)} records in {total_batches} batches")
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            try:
                # Build batch update query
                for arxiv_id, size in batch:
                    complexity = estimate_complexity(size)
                    
                    cur.execute("""
                        UPDATE papers 
                        SET pdf_size_bytes = %s,
                            processing_complexity = %s,
                            pdf_path = %s
                        WHERE arxiv_id = %s AND has_pdf = true
                    """, (size, complexity, f"/bulk-store/arxiv-data/pdf/{arxiv_id[:4]}/{arxiv_id}.pdf", arxiv_id))
                    
                    if cur.rowcount > 0:
                        updated += 1
                    else:
                        failed += 1
                
                conn.commit()
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Progress: {i + len(batch)}/{len(items)} records processed")
                    
            except Exception as e:
                logger.error(f"Error updating batch: {e}")
                conn.rollback()
                failed += len(batch)
        
        cur.close()
    
    return updated, failed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update PDF sizes in database")
    parser.add_argument(
        "--config",
        default="tools/arxiv/configs/db.yaml",
        help="Database configuration file"
    )
    parser.add_argument(
        "--pdf-dir",
        default="/bulk-store/arxiv-data/pdf",
        help="Base directory for PDF files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for database updates"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    LogManager.setup(log_level="INFO")
    
    # Load configuration
    config = load_config(args.config)
    
    # Scan PDF sizes
    pdf_sizes = scan_pdf_sizes(args.pdf_dir)
    
    if args.limit:
        pdf_sizes = dict(list(pdf_sizes.items())[:args.limit])
        logger.info(f"Limited to {args.limit} files for testing")
    
    if not pdf_sizes:
        logger.error("No PDF files found")
        return 1
    
    logger.info(f"Found sizes for {len(pdf_sizes)} PDF files")
    
    # Calculate statistics
    sizes = list(pdf_sizes.values())
    avg_size = sum(sizes) / len(sizes)
    max_size = max(sizes)
    min_size = min(sizes)
    
    logger.info(f"Size statistics:")
    logger.info(f"  Average: {avg_size / (1024*1024):.2f} MB")
    logger.info(f"  Min: {min_size / (1024*1024):.2f} MB")
    logger.info(f"  Max: {max_size / (1024*1024):.2f} MB")
    
    # Update database
    updated, failed = update_database(config, pdf_sizes, args.batch_size)
    
    logger.info(f"Update complete: {updated} updated, {failed} failed")
    
    # Show final statistics
    with get_connection(config.postgres) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(pdf_size_bytes) as with_size,
                AVG(pdf_size_bytes) as avg_size,
                MIN(pdf_size_bytes) as min_size,
                MAX(pdf_size_bytes) as max_size
            FROM papers
            WHERE has_pdf = true
        """)
        stats = cur.fetchone()
        cur.close()
    
    if stats:
        logger.info("Database statistics after update:")
        logger.info(f"  Total PDFs: {stats[0]}")
        logger.info(f"  With size data: {stats[1]}")
        if stats[2]:
            logger.info(f"  Average size: {stats[2] / (1024*1024):.2f} MB")
            logger.info(f"  Min size: {stats[3] / (1024*1024):.2f} MB")
            logger.info(f"  Max size: {stats[4] / (1024*1024):.2f} MB")
    
    return 0


if __name__ == "__main__":
    exit(main())