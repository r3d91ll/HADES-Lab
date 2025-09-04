#!/usr/bin/env python3
"""
Merge ArXiv Paper Lists
=======================

Utility to merge multiple ArXiv paper lists while removing duplicates.
Useful for combining collections from different search sessions.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Set, List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_id_files(*id_files, output_dir: str = "../../../data/arxiv_collections") -> Path:
    """
    Merge multiple ID text files into a single file.
    
    Args:
        *id_files: Paths to ID files to merge
        output_dir: Output directory for merged file
        
    Returns:
        Path to merged file
    
    Raises:
        ValueError: If no valid ID files provided or no valid IDs found
    """
    # Validate at least one input file provided
    if not id_files:
        raise ValueError("At least one ID file must be provided")
    
    output_dir = Path(output_dir)
    
    # Validate output directory is writable
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Cannot create output directory {output_dir}: {e}")
        raise ValueError(f"Output directory not writable: {output_dir}") from e
    
    all_ids = set()
    
    for id_file in id_files:
        id_path = Path(id_file)
        # Validate file exists and is readable
        if not id_path.is_file():
            logger.warning(f"Skipping non-file entry: {id_path}")
            continue
            
        try:
            logger.info(f"Loading IDs from {id_path}")
            with open(id_path, 'r', encoding='utf-8') as f:
                for line in f:
                    arxiv_id = line.strip()
                    if arxiv_id:
                        all_ids.add(arxiv_id)
            logger.info(f"  Loaded {len(all_ids)} unique IDs so far")
        except IOError as e:
            logger.error(f"Error reading {id_path}: {e}")
            continue
    
    # Validate we have at least some IDs
    if not all_ids:
        raise ValueError("No valid IDs found in any input files")
    
    # Sort IDs for consistent ordering
    sorted_ids = sorted(all_ids)
    
    # Save merged list
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"arxiv_ids_merged_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        for arxiv_id in sorted_ids:
            f.write(f"{arxiv_id}\n")
    
    logger.info(f"Saved {len(sorted_ids)} unique IDs to {output_file}")
    return output_file


def merge_json_collections(*json_files, output_dir: str = "../../../data/arxiv_collections") -> Path:
    """
    Merge multiple JSON collection files.
    
    Args:
        *json_files: Paths to JSON collection files
        output_dir: Output directory for merged file
        
    Returns:
        Path to merged file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_papers = {}
    all_ids = set()
    paper_details = {}
    
    for json_file in json_files:
        json_path = Path(json_file)
        if json_path.exists():
            logger.info(f"Loading collection from {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            if 'papers' in data:
                for topic, papers in data['papers'].items():
                    if topic not in all_papers:
                        all_papers[topic] = []
                    
                    for paper in papers:
                        if paper['arxiv_id'] not in all_ids:
                            all_papers[topic].append(paper)
                            all_ids.add(paper['arxiv_id'])
                            paper_details[paper['arxiv_id']] = paper
                
                logger.info(f"  Loaded {len(all_ids)} unique papers so far")
        else:
            logger.warning(f"File not found: {json_path}")
    
    # Create merged collection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"arxiv_merged_collection_{timestamp}.json"
    
    stats = {
        'collection_date': datetime.now().isoformat(),
        'total_papers': len(all_ids),
        'topics': {topic: len(papers) for topic, papers in all_papers.items()},
        'papers': all_papers
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved merged collection to {output_file}")
    
    # Also save ID list
    id_file = output_dir / f"arxiv_ids_from_merge_{timestamp}.txt"
    sorted_ids = sorted(all_ids)
    
    with open(id_file, 'w') as f:
        for arxiv_id in sorted_ids:
            f.write(f"{arxiv_id}\n")
    
    logger.info(f"Saved {len(sorted_ids)} IDs to {id_file}")
    
    return output_file


def print_collection_stats(json_file: str):
    """Print statistics about a collection."""
    json_path = Path(json_file)
    if not json_path.exists():
        logger.error(f"File not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*60)
    print("COLLECTION STATISTICS")
    print("="*60)
    
    if 'total_papers' in data:
        print(f"Total papers: {data['total_papers']}")
    
    if 'topics' in data:
        print("\nPapers by topic:")
        for topic, count in sorted(data['topics'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {topic:30s}: {count:5d}")
    
    if 'collection_date' in data:
        print(f"\nCollection date: {data['collection_date']}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge ArXiv paper lists")
    parser.add_argument('--id-files', nargs='+', help='ID text files to merge')
    parser.add_argument('--json-files', nargs='+', help='JSON collection files to merge')
    parser.add_argument('--output-dir', default='../../../data/arxiv_collections', 
                       help='Output directory')
    parser.add_argument('--stats', help='Print statistics for a collection file')
    
    args = parser.parse_args()
    
    if args.stats:
        print_collection_stats(args.stats)
    elif args.id_files:
        merged_file = merge_id_files(*args.id_files, output_dir=args.output_dir)
        print(f"\n✅ Merged ID file: {merged_file}")
    elif args.json_files:
        merged_file = merge_json_collections(*args.json_files, output_dir=args.output_dir)
        print(f"\n✅ Merged collection: {merged_file}")
    else:
        print("Please specify --id-files, --json-files, or --stats")
        parser.print_help()


if __name__ == "__main__":
    main()