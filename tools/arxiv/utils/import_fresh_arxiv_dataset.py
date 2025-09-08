#!/usr/bin/env python3
"""
Import fresh ArXiv dataset from Kaggle (August 2024 version) directly to ArangoDB.

This script:
1. Downloads the latest ArXiv dataset using kagglehub
2. Imports it directly into ArangoDB
3. Shows statistics about the data
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import kagglehub
from tqdm import tqdm
from arango import ArangoClient

def download_dataset() -> Path:
    """Download the latest ArXiv dataset from Kaggle."""
    print("="*80)
    print("Downloading latest ArXiv dataset from Kaggle...")
    print("="*80)
    
    # Download latest version
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    
    print(f"✓ Dataset downloaded to: {path}")
    
    # List files in the dataset
    dataset_path = Path(path)
    files = list(dataset_path.glob("*"))
    
    print("\nDataset contents:")
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024) if file.is_file() else 0
        print(f"  - {file.name}: {size_mb:.2f} MB")
    
    return dataset_path


def analyze_dataset(dataset_path: Path) -> Dict[str, Any]:
    """Analyze the dataset to understand its structure and date range."""
    print("\n" + "="*80)
    print("Analyzing dataset...")
    print("="*80)
    
    # Find the JSON file
    json_files = list(dataset_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No JSON file found in dataset")
    
    json_file = json_files[0]
    print(f"Reading {json_file.name}...")
    
    stats = {
        'total_papers': 0,
        'earliest_date': None,
        'latest_date': None,
        'categories': set(),
        'years': {},
        'sample_papers': []
    }
    
    with open(json_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Analyzing papers")):
            try:
                paper = json.loads(line)
                stats['total_papers'] += 1
                
                # Get date from versions
                if paper.get('versions'):
                    for version in paper['versions']:
                        date_str = version.get('created')
                        if date_str:
                            # Parse date (format: "Mon, 2 Apr 2007 19:18:42 GMT")
                            try:
                                date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
                                year = date.year
                                
                                # Track years
                                stats['years'][year] = stats['years'].get(year, 0) + 1
                                
                                # Track earliest/latest
                                if not stats['earliest_date'] or date < stats['earliest_date']:
                                    stats['earliest_date'] = date
                                if not stats['latest_date'] or date > stats['latest_date']:
                                    stats['latest_date'] = date
                            except:
                                pass
                
                # Track categories
                if paper.get('categories'):
                    cats = paper['categories'].split() if isinstance(paper['categories'], str) else paper['categories']
                    stats['categories'].update(cats)
                
                # Save sample of recent papers
                if stats['total_papers'] <= 10:
                    stats['sample_papers'].append({
                        'id': paper.get('id'),
                        'title': paper.get('title'),
                        'categories': paper.get('categories')
                    })
                
                # Also save some 2024 papers
                arxiv_id = paper.get('id', '')
                if arxiv_id.startswith('24'):
                    if len([p for p in stats['sample_papers'] if p['id'].startswith('24')]) < 5:
                        stats['sample_papers'].append({
                            'id': arxiv_id,
                            'title': paper.get('title'),
                            'categories': paper.get('categories')
                        })
                
            except json.JSONDecodeError:
                continue
            
            # Show progress every 100k papers
            if (line_num + 1) % 100000 == 0:
                print(f"  Processed {line_num + 1:,} papers...")
    
    return stats


def import_to_arangodb(dataset_path: Path, limit: int = None):
    """Import dataset directly into ArangoDB."""
    print("\n" + "="*80)
    print("Importing to ArangoDB...")
    print("="*80)
    
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('academy_store', username='root', password=os.environ.get('ARANGO_PASSWORD'))
    
    # Ensure collection exists
    if not db.has_collection('arxiv_papers'):
        db.create_collection('arxiv_papers')
        print("Created arxiv_papers collection")
    
    papers_collection = db.collection('arxiv_papers')
    
    # Clear existing data (optional)
    response = input("\nClear existing arxiv_papers data? (y/n): ")
    if response.lower() == 'y':
        papers_collection.truncate()
        print("Cleared existing data")
    
    # Import data
    json_file = list(dataset_path.glob("*.json"))[0]
    
    imported = 0
    skipped = 0
    papers_2024 = 0
    
    with open(json_file, 'r', encoding='utf-8') as f:
        batch = []
        for line in tqdm(f, desc="Importing to ArangoDB"):
            if limit and imported >= limit:
                break
                
            try:
                paper = json.loads(line)
                
                # Prepare data for ArangoDB
                arxiv_id = paper.get('id', '').replace('.', '_').replace('/', '_')
                
                # Track 2024 papers
                if arxiv_id.startswith('24'):
                    papers_2024 += 1
                
                doc = {
                    '_key': arxiv_id,
                    'arxiv_id': paper.get('id'),
                    'title': paper.get('title'),
                    'authors': paper.get('authors'),
                    'categories': paper.get('categories').split() if isinstance(paper.get('categories'), str) else paper.get('categories'),
                    'abstract': paper.get('abstract'),
                    'update_date': paper.get('update_date'),
                    'doi': paper.get('doi'),
                    'journal_ref': paper.get('journal-ref'),
                    'comments': paper.get('comments'),
                    'report_no': paper.get('report-no'),
                    'license': paper.get('license'),
                    'versions': paper.get('versions', []),
                    'submitter': paper.get('submitter'),
                    'status': 'NOT_PROCESSED'  # For tracking which papers have been embedded
                }
                
                batch.append(doc)
                
                # Insert in batches
                if len(batch) >= 1000:
                    try:
                        papers_collection.insert_many(batch, overwrite=True)
                        imported += len(batch)
                        batch = []
                    except Exception as e:
                        print(f"Batch insert error: {e}")
                        # Try individual inserts for this batch
                        for doc in batch:
                            try:
                                papers_collection.insert(doc, overwrite=True)
                                imported += 1
                            except:
                                skipped += 1
                        batch = []
                    
            except Exception as e:
                skipped += 1
                if skipped < 10:
                    print(f"Error importing paper: {e}")
        
        # Insert remaining batch
        if batch:
            try:
                papers_collection.insert_many(batch, overwrite=True)
                imported += len(batch)
            except:
                for doc in batch:
                    try:
                        papers_collection.insert(doc, overwrite=True)
                        imported += 1
                    except:
                        skipped += 1
    
    print(f"\n✓ Imported {imported:,} papers to ArangoDB")
    print(f"  Found {papers_2024:,} papers from 2024")
    print(f"  Skipped {skipped:,} papers due to errors")
    
    # Show some stats
    total_count = papers_collection.count()
    
    print(f"\nDatabase statistics:")
    print(f"  Total papers in collection: {total_count:,}")


def prepare_arangodb():
    """Prepare ArangoDB collections for the fresh data."""
    print("\n" + "="*80)
    print("Preparing ArangoDB collections...")
    print("="*80)
    
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('academy_store', username='root', password=os.environ.get('ARANGO_PASSWORD'))
    
    # Collections to create/clear
    collections = [
        'arxiv_papers',
        'arxiv_chunks', 
        'arxiv_embeddings',
        'arxiv_equations',
        'arxiv_tables',
        'arxiv_images',
        'arxiv_structures'
    ]
    
    response = input("\nClear existing ArangoDB collections? (y/n): ")
    
    for coll_name in collections:
        if db.has_collection(coll_name):
            if response.lower() == 'y':
                db.collection(coll_name).truncate()
                print(f"  ✓ Cleared {coll_name}")
            else:
                count = db.collection(coll_name).count()
                print(f"  → {coll_name} exists with {count:,} documents")
        else:
            db.create_collection(coll_name)
            print(f"  ✓ Created {coll_name}")
    
    print("\n✓ ArangoDB collections ready")


def show_summary(stats: Dict[str, Any]):
    """Show summary of the imported data."""
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    
    print(f"\nTotal papers: {stats['total_papers']:,}")
    
    if stats['earliest_date'] and stats['latest_date']:
        print(f"Date range: {stats['earliest_date'].strftime('%Y-%m-%d')} to {stats['latest_date'].strftime('%Y-%m-%d')}")
    
    print(f"Categories: {len(stats['categories'])}")
    
    # Show papers by year
    print("\nPapers by year:")
    recent_years = sorted([y for y in stats['years'].keys() if y >= 2020])
    for year in recent_years[-5:]:
        count = stats['years'].get(year, 0)
        print(f"  {year}: {count:,}")
    
    # Show 2024 papers
    papers_2024 = stats['years'].get(2024, 0)
    if papers_2024 > 0:
        print(f"\n🎉 Found {papers_2024:,} papers from 2024!")
        
        # Show sample 2024 papers
        papers_2024_sample = [p for p in stats['sample_papers'] if p['id'].startswith('24')]
        if papers_2024_sample:
            print("\nSample 2024 papers:")
            for paper in papers_2024_sample[:5]:
                print(f"  [{paper['id']}] {paper['title'][:60]}...")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. The ArXiv metadata is now imported
2. Run the ACID pipeline to process papers:
   cd ../pipelines/
   python arxiv_pipeline.py --config ../configs/acid_pipeline_phased.yaml --count 100
   
3. Test the embedding pipeline with 2024 papers
4. Search for unified intelligence architectures in the fresh data
    """)


def main():
    """Main entry point."""
    print("ArXiv Dataset Fresh Import to ArangoDB")
    print("="*80)
    
    # Check environment
    if not os.environ.get('ARANGO_PASSWORD'):
        print("Error: ARANGO_PASSWORD environment variable not set")
        print("Please set: export ARANGO_PASSWORD='your-arango-password'")
        sys.exit(1)
    
    # Download dataset
    dataset_path = download_dataset()
    
    # Analyze dataset
    stats = analyze_dataset(dataset_path)
    
    # Prepare ArangoDB collections
    prepare_arangodb()
    
    # Import to ArangoDB
    import_to_arangodb(dataset_path)
    
    # Show summary
    show_summary(stats)


if __name__ == "__main__":
    main()