#!/usr/bin/env python3
"""
Normalize external paper metadata to consistent format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_metadata(data: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """
    Normalize metadata to standard format.

    Standard format:
    {
        "paper_id": str,
        "source": "external",
        "source_file": str,
        "source_path": str,
        "metadata": {
            "title": str,
            "authors": List[str],
            "year": int,
            "journal": str,
            "abstract": str,
            ...
        },
        "bibliography": List[Dict],
        "extraction_stats": Dict
    }
    """

    # Check if already in standard format
    if 'paper_id' in data and 'metadata' in data:
        logger.info(f"  {filename}: Already in standard format")
        return data

    # Convert from direct format
    logger.info(f"  {filename}: Converting from direct format")

    # Extract authors
    authors = data.get('authors', '')
    if isinstance(authors, str):
        # Split by comma if string
        authors = [a.strip() for a in authors.split(',') if a.strip()]

    # Create paper_id if missing
    paper_id = data.get('id', '')
    if not paper_id:
        # Generate from filename
        paper_id = f"external_{Path(filename).stem}"

    # Build normalized structure
    normalized = {
        "paper_id": paper_id,
        "source": "external",
        "source_file": filename.replace('.metadata.json', '.pdf'),
        "source_path": f"/bulk-store/random_pdfs/{filename.replace('.metadata.json', '.pdf')}",
        "metadata": {
            "title": data.get('title', ''),
            "authors": authors,
            "year": data.get('year'),
            "journal": data.get('journal_ref', data.get('journal', '')),
            "volume": data.get('volume', ''),
            "pages": data.get('pages', ''),
            "doi": data.get('doi', ''),
            "abstract": data.get('abstract', ''),
            "keywords": data.get('categories', '').split() if data.get('categories') else [],
            "venue_type": data.get('venue_type', ''),
            "publisher": data.get('publisher', ''),
            "pdf_urls": data.get('pdf_urls', []),
            "submission_date": data.get('submission_date', ''),
            "update_date": data.get('update_date', '')
        },
        "bibliography": data.get('bibliography', []),
        "extraction_stats": data.get('extraction_stats', {
            "normalized": True,
            "original_format": "direct"
        })
    }

    return normalized

def process_directory(directory: Path, backup: bool = True):
    """Process all metadata files in directory."""
    metadata_files = []

    # Find all metadata files (excluding symlinks)
    for f in directory.glob("*.metadata.json"):
        if not f.is_symlink() and not f.name.endswith('.pdf.metadata.json'):
            metadata_files.append(f)

    logger.info(f"Found {len(metadata_files)} metadata files to process")

    updated_count = 0

    for metadata_file in sorted(metadata_files):
        try:
            # Read current metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Normalize
            normalized = normalize_metadata(data, metadata_file.name)

            # Check if changed
            if normalized != data:
                # Backup original if requested
                if backup:
                    backup_file = metadata_file.with_suffix('.metadata.original.json')
                    if not backup_file.exists():
                        with open(backup_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2)
                        logger.info(f"    Backed up to {backup_file.name}")

                # Write normalized
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(normalized, f, indent=2)

                updated_count += 1
                logger.info(f"    ✓ Updated {metadata_file.name}")

        except Exception as e:
            logger.error(f"  ✗ Error processing {metadata_file.name}: {e}")

    logger.info(f"\nUpdated {updated_count}/{len(metadata_files)} files")

    # Verify consistency
    logger.info("\n=== Verification ===")
    standard_count = 0
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'paper_id' in data and 'metadata' in data:
                standard_count += 1
        except:
            pass

    logger.info(f"Files in standard format: {standard_count}/{len(metadata_files)}")

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Normalize external paper metadata')
    parser.add_argument('--directory', type=str,
                       default='/bulk-store/random_pdfs',
                       help='Directory containing metadata files')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup files')

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return 1

    process_directory(directory, backup=not args.no_backup)
    return 0

if __name__ == "__main__":
    exit(main())
