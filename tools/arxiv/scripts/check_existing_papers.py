#!/usr/bin/env python3
"""
Check Existing Papers in ArangoDB
==================================

Check which papers from a list are already processed in ArangoDB.
"""

import os
import sys
from pathlib import Path
from typing import List, Set, Dict
import json
from datetime import datetime

# Add proper paths for imports
project_root = Path(__file__).parent.parent.parent.parent  # Goes up to HADES-Lab
sys.path.insert(0, str(project_root))

# Import ArangoDBManager from the correct location
from tools.arxiv.pipelines.arango_db_manager import ArangoDBManager


def check_existing_papers(paper_list_file: str = None) -> Dict:
    """
    Check which papers from a list are already in ArangoDB.
    
    Args:
        paper_list_file: Path to file with ArXiv IDs (one per line)
        
    Returns:
        Dictionary with statistics
    """
    # Get ArangoDB password
    arango_password = os.getenv('ARANGO_PASSWORD')
    if not arango_password:
        print("❌ ARANGO_PASSWORD not set")
        print("Please set: export ARANGO_PASSWORD='your-password'")
        return {}
    
    # Find paper list if not specified
    if paper_list_file is None:
        data_dir = Path("data/arxiv_collections")
        if data_dir.exists():
            id_files = sorted(data_dir.glob("arxiv_ids*.txt"))
            if id_files:
                paper_list_file = str(id_files[-1])
                print(f"📂 Using latest paper list: {id_files[-1].name}")
            else:
                print("❌ No paper lists found")
                return {}
        else:
            print("❌ Data directory not found")
            return {}
    
    # Load paper IDs
    list_path = Path(paper_list_file)
    if not list_path.exists():
        print(f"❌ Paper list not found: {list_path}")
        return {}
    
    with open(list_path, 'r') as f:
        paper_ids = [line.strip() for line in f if line.strip()]
    
    print(f"📄 Loaded {len(paper_ids):,} paper IDs from {list_path.name}")
    
    # Connect to ArangoDB
    config = {
        'host': os.getenv('ARANGO_HOST', 'http://192.168.1.69:8529'),
        'database': 'academy_store',
        'username': 'root',
        'password': arango_password
    }
    
    try:
        db_manager = ArangoDBManager(config)
        db = db_manager.db
        
        # Check each collection
        collections_to_check = ['arxiv_papers', 'arxiv_embeddings', 'arxiv_chunks']
        
        print("\n🔍 Checking ArangoDB collections...")
        print("=" * 60)
        
        # Get existing papers from database
        existing_in_db = set()
        failed_in_db = set()
        
        if db.has_collection('arxiv_papers'):
            papers_collection = db.collection('arxiv_papers')
            
            # Check each paper
            for paper_id in paper_ids:
                # ArangoDB keys use underscore instead of dot
                clean_id = paper_id.replace('.', '_')
                
                if papers_collection.has(clean_id):
                    doc = papers_collection.get(clean_id)
                    status = doc.get('status', 'unknown')
                    
                    if status == 'PROCESSED':
                        existing_in_db.add(paper_id)
                    elif status == 'FAILED':
                        failed_in_db.add(paper_id)
        
        # Calculate statistics
        total_papers = len(paper_ids)
        already_processed = len(existing_in_db)
        already_failed = len(failed_in_db)
        not_processed = total_papers - already_processed - already_failed
        
        # Print results
        print(f"📊 Database Status:")
        print(f"  Total papers in list:     {total_papers:,}")
        print(f"  ✅ Already processed:     {already_processed:,} ({already_processed/total_papers*100:.1f}%)")
        print(f"  ❌ Failed previously:     {already_failed:,} ({already_failed/total_papers*100:.1f}%)")
        print(f"  ⏳ Not yet processed:     {not_processed:,} ({not_processed/total_papers*100:.1f}%)")
        
        # Check collection sizes
        print(f"\n📦 Collection Statistics:")
        for coll_name in collections_to_check:
            if db.has_collection(coll_name):
                coll = db.collection(coll_name)
                count = coll.count()
                print(f"  {coll_name:20s}: {count:,} documents")
            else:
                print(f"  {coll_name:20s}: Collection doesn't exist")
        
        # Show sample of processed papers
        if already_processed > 0:
            print(f"\n📝 Sample of already processed papers (first 10):")
            for i, paper_id in enumerate(list(existing_in_db)[:10]):
                print(f"  {i+1:2d}. {paper_id}")
            if already_processed > 10:
                print(f"  ... and {already_processed - 10:,} more")
        
        # Create list of unprocessed papers
        unprocessed_ids = [pid for pid in paper_ids if pid not in existing_in_db and pid not in failed_in_db]
        
        if unprocessed_ids:
            # Save unprocessed list
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unprocessed_file = Path(f"data/arxiv_collections/arxiv_ids_unprocessed_{timestamp}.txt")
            unprocessed_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(unprocessed_file, 'w') as f:
                for paper_id in unprocessed_ids:
                    f.write(f"{paper_id}\n")
            
            print(f"\n💾 Saved unprocessed papers to:")
            print(f"   {unprocessed_file}")
            print(f"\n🚀 To process only new papers, use:")
            print(f"   python run_pipeline_from_list.py 100 --list {unprocessed_file}")
        
        return {
            'total': total_papers,
            'processed': already_processed,
            'failed': already_failed,
            'unprocessed': not_processed,
            'existing_ids': existing_in_db,
            'failed_ids': failed_in_db,
            'unprocessed_ids': unprocessed_ids
        }
        
    except Exception as e:
        print(f"❌ Error connecting to ArangoDB: {e}")
        return {}


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check which ArXiv papers are already in ArangoDB")
    parser.add_argument('--list', type=str, help='Paper ID list file (default: latest)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    results = check_existing_papers(args.list)
    
    if results and args.verbose:
        print("\n" + "="*60)
        print("Detailed Results")
        print("="*60)
        print(json.dumps({
            'total': results['total'],
            'processed': results['processed'],
            'failed': results['failed'],
            'unprocessed': results['unprocessed']
        }, indent=2))


if __name__ == "__main__":
    main()