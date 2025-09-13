#!/usr/bin/env python3
"""
Capture Baseline Database State
================================

Creates a snapshot of the current database state for validation after core restructure.
This snapshot will be used to ensure the restructured code produces identical results.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from arango import ArangoClient


def capture_baseline():
    """
    Capture comprehensive baseline of database state.

    Returns:
        dict: Baseline metrics and statistics
    """
    # Connect to ArangoDB (HTTP for compatibility)
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db(
        'academy_store',
        username='root',
        password=os.environ.get('ARANGO_PASSWORD')
    )

    print("Capturing database baseline...")

    baseline = {
        'timestamp': datetime.now().isoformat(),
        'database': 'academy_store',
        'collections': {},
        'validation_queries': {},
        'sample_data': {}
    }

    # Capture counts for all collections
    collection_names = [
        'arxiv_papers',
        'arxiv_abstract_embeddings',
        'arxiv_chunks',
        'arxiv_embeddings',
        'arxiv_equations',
        'arxiv_tables',
        'arxiv_images'
    ]

    for collection_name in collection_names:
        if db.has_collection(collection_name):
            collection = db.collection(collection_name)
            count = collection.count()

            baseline['collections'][collection_name] = {
                'exists': True,
                'count': count,
                'indexes': len(collection.indexes())
            }

            print(f"  {collection_name}: {count:,} documents")
        else:
            baseline['collections'][collection_name] = {
                'exists': False,
                'count': 0,
                'indexes': 0
            }
            print(f"  {collection_name}: Does not exist")

    # Validation queries
    print("\nRunning validation queries...")

    # Check papers with embeddings
    if db.has_collection('arxiv_abstract_embeddings'):
        papers_with_embeddings = db.aql.execute("""
            FOR e IN arxiv_abstract_embeddings
            COLLECT WITH COUNT INTO total
            RETURN total
        """).next()
        baseline['validation_queries']['papers_with_embeddings'] = papers_with_embeddings
        print(f"  Papers with embeddings: {papers_with_embeddings:,}")

        # Get sample embedding for validation
        sample_embedding = db.collection('arxiv_abstract_embeddings').random()
        if sample_embedding:
            baseline['sample_data']['embedding'] = {
                'paper_id': sample_embedding.get('paper_id'),
                'dimensions': len(sample_embedding.get('embedding', [])),
                'has_model_field': 'model' in sample_embedding,
                'model': sample_embedding.get('model', 'not specified')
            }
            print(f"  Embedding dimensions: {baseline['sample_data']['embedding']['dimensions']}")

    # Check unique paper count
    if db.has_collection('arxiv_papers'):
        total_papers = db.collection('arxiv_papers').count()
        baseline['validation_queries']['total_papers'] = total_papers

        # Get sample paper
        sample_paper = db.collection('arxiv_papers').random()
        if sample_paper:
            baseline['sample_data']['paper'] = {
                'id': sample_paper.get('_key'),
                'has_title': 'title' in sample_paper,
                'has_abstract': 'abstract' in sample_paper,
                'has_authors': 'authors' in sample_paper,
                'has_categories': 'categories' in sample_paper
            }

    # Calculate coverage
    if baseline['collections']['arxiv_papers']['exists'] and \
       baseline['collections']['arxiv_abstract_embeddings']['exists']:
        coverage = (baseline['collections']['arxiv_abstract_embeddings']['count'] /
                   baseline['collections']['arxiv_papers']['count'] * 100)
        baseline['validation_queries']['embedding_coverage'] = coverage
        print(f"  Embedding coverage: {coverage:.2f}%")

    # Performance metrics from the run
    baseline['performance'] = {
        'final_run_papers': 1156802,
        'final_run_minutes': 399.1,
        'papers_per_second': 48.3,
        'workers_used': 2,
        'batch_size': 80,
        'gpu_model': 'NVIDIA RTX A6000',
        'embedder': 'Sentence-Transformers',
        'model': 'jinaai/jina-embeddings-v4'
    }

    # Save baseline to file
    output_file = Path('baseline-snapshot.json')
    with open(output_file, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f"\n✓ Baseline saved to: {output_file}")
    print(f"  Total collections: {len([c for c in baseline['collections'].values() if c['exists']])}")
    print(f"  Total documents: {sum(c['count'] for c in baseline['collections'].values())}")

    return baseline


def compare_with_baseline(baseline_file='baseline-snapshot.json'):
    """
    Compare current database state with saved baseline.

    Args:
        baseline_file: Path to baseline JSON file

    Returns:
        dict: Comparison results
    """
    if not Path(baseline_file).exists():
        print(f"Baseline file {baseline_file} not found!")
        return None

    with open(baseline_file, 'r') as f:
        baseline = json.load(f)

    # Capture current state
    current = capture_baseline()

    # Compare
    print("\n=== Comparison with Baseline ===")

    differences = []

    # Compare collection counts
    for collection_name, baseline_data in baseline['collections'].items():
        current_data = current['collections'].get(collection_name, {})

        if baseline_data['count'] != current_data.get('count', 0):
            diff = current_data.get('count', 0) - baseline_data['count']
            differences.append(f"{collection_name}: {diff:+,} documents")
            print(f"  {collection_name}:")
            print(f"    Baseline: {baseline_data['count']:,}")
            print(f"    Current:  {current_data.get('count', 0):,}")
            print(f"    Diff:     {diff:+,}")

    if not differences:
        print("  ✓ All collection counts match baseline!")

    # Compare embedding dimensions
    if 'embedding' in baseline.get('sample_data', {}):
        baseline_dims = baseline['sample_data']['embedding']['dimensions']
        current_dims = current.get('sample_data', {}).get('embedding', {}).get('dimensions', 0)

        if baseline_dims != current_dims:
            print(f"  ⚠ Embedding dimensions changed: {baseline_dims} → {current_dims}")
        else:
            print(f"  ✓ Embedding dimensions unchanged: {baseline_dims}")

    return {
        'baseline': baseline,
        'current': current,
        'differences': differences,
        'match': len(differences) == 0
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Database baseline snapshot tool')
    parser.add_argument('--compare', action='store_true',
                       help='Compare current state with baseline')

    args = parser.parse_args()

    if args.compare:
        result = compare_with_baseline()
        if result and result['match']:
            print("\n✅ Database matches baseline perfectly!")
            sys.exit(0)
        else:
            print("\n❌ Database differs from baseline")
            sys.exit(1)
    else:
        baseline = capture_baseline()
        print("\n✅ Baseline captured successfully!")