#!/usr/bin/env python3
"""
Database Status Checker for ArXiv Processor
Verifies Jina v4 deployment and database integrity.

Usage:
    python utils/check_db_status.py [--detailed]
"""

import os
import sys
from arango import ArangoClient
import argparse
from datetime import datetime
from typing import Dict, Any

def get_db_connection():
    """Establish database connection."""
    client = ArangoClient(hosts='http://192.168.1.69:8529')
    db = client.db(
        'academy_store',
        username='root',
        password=os.environ.get('ARANGO_PASSWORD')
    )
    return db

def check_chunks(db, detailed: bool = False) -> Dict[str, Any]:
    """Check chunks collection and late chunking status."""
    results = {}
    
    # Get chunking method breakdown
    chunks_summary = list(db.aql.execute("""
        FOR chunk IN arxiv_chunks
        COLLECT method = chunk.embedding_method WITH COUNT INTO count
        RETURN {method: method, count: count}
    """))
    
    total_chunks = sum(c['count'] for c in chunks_summary)
    late_chunks = sum(c['count'] for c in chunks_summary if c['method'] == 'late_chunking')
    traditional_chunks = total_chunks - late_chunks
    
    results['total'] = total_chunks
    results['late_chunking'] = late_chunks
    results['traditional'] = traditional_chunks
    results['late_percentage'] = late_chunks * 100 // total_chunks if total_chunks else 0
    
    # Get context window stats for late chunks
    if late_chunks > 0:
        # Get chunks with context window field
        chunks_with_context = list(db.aql.execute("""
            FOR chunk IN arxiv_chunks
            FILTER chunk.embedding_method == 'late_chunking' 
            AND chunk.context_window_tokens > 0
            RETURN chunk.context_window_tokens
        """))
        
        if chunks_with_context:
            results['avg_context'] = int(sum(chunks_with_context) / len(chunks_with_context))
            results['min_context'] = int(min(chunks_with_context))
            results['max_context'] = int(max(chunks_with_context))
    
    # Check embedding dimensions
    dim_check = list(db.aql.execute(
        'FOR c IN arxiv_chunks FILTER c.embedding != null LIMIT 1 RETURN LENGTH(c.embedding)'
    ))
    results['embedding_dim'] = dim_check[0] if dim_check else 0
    
    return results

def check_tables(db) -> Dict[str, Any]:
    """Check tables collection."""
    results = {}
    results['total'] = db.collection('arxiv_tables').count()
    
    if results['total'] > 0:
        # Check for structure preservation
        sample = list(db.aql.execute("FOR t IN arxiv_tables LIMIT 1 RETURN t"))[0]
        results['has_headers'] = bool('headers' in sample and sample['headers'])
        results['has_data_rows'] = bool('data_rows' in sample and sample['data_rows'])
        results['has_latex'] = bool('latex' in sample and sample['latex'])
        
        # Check embedding dimensions
        results['embedding_dim'] = len(sample.get('embedding', []))
    
    return results

def check_images(db) -> Dict[str, Any]:
    """Check images collection."""
    results = {}
    results['total'] = db.collection('arxiv_images').count()
    
    if results['total'] > 0:
        # Sample check
        sample = list(db.aql.execute("FOR i IN arxiv_images LIMIT 1 RETURN i"))[0]
        results['has_caption'] = bool('caption' in sample and sample['caption'])
        results['has_bbox'] = bool('bbox' in sample and sample['bbox'])
        results['embedding_dim'] = len(sample.get('embedding', []))
    
    return results

def check_equations(db) -> Dict[str, Any]:
    """Check equations collection."""
    results = {}
    results['total'] = db.collection('arxiv_equations').count()
    
    if results['total'] > 0:
        # Sample check
        sample = list(db.aql.execute("FOR e IN arxiv_equations LIMIT 1 RETURN e"))[0]
        results['has_latex'] = bool('latex' in sample and sample['latex'])
        results['has_label'] = 'label' in sample
        results['embedding_dim'] = len(sample.get('embedding', []))
    
    return results

def check_metadata(db) -> Dict[str, Any]:
    """Check metadata collection."""
    results = {}
    results['total'] = db.collection('arxiv_metadata').count()
    
    if results['total'] > 0:
        # Check processing status
        processed = list(db.aql.execute("""
            FOR doc IN arxiv_metadata
            FILTER doc.full_text_processed == true
            COLLECT WITH COUNT INTO count
            RETURN count
        """))[0]
        results['processed'] = processed
        
        # Get recent processing
        recent = list(db.aql.execute("""
            FOR doc IN arxiv_metadata
            FILTER doc.processing_date != null
            SORT doc.processing_date DESC
            LIMIT 1
            RETURN {
                arxiv_id: doc.arxiv_id,
                date: doc.processing_date,
                method: doc.chunking_method
            }
        """))
        
        if recent:
            results['most_recent'] = recent[0]
    
    return results

def print_status_report(results: Dict[str, Any], detailed: bool = False):
    """Print formatted status report."""
    print("=" * 80)
    print("📊 DATABASE STATUS REPORT")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()[:19]}")
    
    # Metadata Status
    meta = results['metadata']
    print(f"\n📚 METADATA:")
    print(f"  • Total papers: {meta['total']}")
    print(f"  • Fully processed: {meta.get('processed', 0)}")
    if 'most_recent' in meta:
        recent = meta['most_recent']
        print(f"  • Most recent: {recent['arxiv_id']} ({recent['date'][:19]})")
    
    # Chunks Status
    chunks = results['chunks']
    print(f"\n📄 CHUNKS:")
    print(f"  • Total chunks: {chunks['total']}")
    if chunks['total'] > 0:
        print(f"  • Late chunking: {chunks['late_chunking']} ({chunks['late_percentage']}%)")
        print(f"  • Traditional: {chunks['traditional']} ({100-chunks['late_percentage']}%)")
        if 'avg_context' in chunks:
            print(f"  • Avg context window: {chunks['avg_context']} tokens")
            if detailed:
                print(f"    Min/Max: {chunks['min_context']}/{chunks['max_context']} tokens")
        print(f"  • Embedding dims: {chunks['embedding_dim']}")
    
    # Tables Status
    tables = results['tables']
    print(f"\n📊 TABLES:")
    print(f"  • Total tables: {tables['total']}")
    if tables['total'] > 0:
        status_symbols = {
            'has_headers': '✅' if tables.get('has_headers') else '❌',
            'has_data_rows': '✅' if tables.get('has_data_rows') else '❌',
            'has_latex': '✅' if tables.get('has_latex') else '❌'
        }
        print(f"  • Structure: Headers {status_symbols['has_headers']}, "
              f"Data {status_symbols['has_data_rows']}, "
              f"LaTeX {status_symbols['has_latex']}")
        print(f"  • Embedding dims: {tables.get('embedding_dim', 0)}")
    
    # Equations Status
    equations = results['equations']
    print(f"\n🔢 EQUATIONS:")
    print(f"  • Total equations: {equations['total']}")
    if equations['total'] > 0:
        status_symbols = {
            'has_latex': '✅' if equations.get('has_latex') else '❌',
            'has_label': '✅' if equations.get('has_label') else '❌'
        }
        print(f"  • LaTeX {status_symbols['has_latex']}, Labels {status_symbols['has_label']}")
        print(f"  • Embedding dims: {equations.get('embedding_dim', 0)}")
    
    # Images Status
    images = results['images']
    print(f"\n🖼️  IMAGES:")
    print(f"  • Total images: {images['total']}")
    if images['total'] > 0:
        status_symbols = {
            'has_caption': '✅' if images.get('has_caption') else '❌',
            'has_bbox': '✅' if images.get('has_bbox') else '❌'
        }
        print(f"  • Captions {status_symbols['has_caption']}, BBox {status_symbols['has_bbox']}")
        print(f"  • Embedding dims: {images.get('embedding_dim', 0)}")
    
    # Overall Status
    print("\n" + "=" * 80)
    print("🎯 JINA V4 DEPLOYMENT STATUS:")
    
    # Check if all embeddings are 2048-dimensional
    all_2048 = all(
        results[key].get('embedding_dim') == 2048
        for key in ['chunks', 'tables', 'equations', 'images']
        if results[key].get('embedding_dim')
    )
    
    # Check late chunking
    late_chunking_active = chunks.get('late_percentage', 0) > 90
    
    status_items = []
    status_items.append(f"{'✅' if all_2048 else '❌'} 2048-dim embeddings")
    status_items.append(f"{'✅' if late_chunking_active else '⚠️'} Late chunking ({chunks.get('late_percentage', 0)}%)")
    status_items.append(f"{'✅' if chunks.get('avg_context', 0) > 1000 else '⚠️'} Context windows (~{chunks.get('avg_context', 0)} tokens)")
    
    for item in status_items:
        print(f"  {item}")
    
    # Final verdict
    print("\n" + "=" * 80)
    if all_2048 and late_chunking_active:
        print("✅ SYSTEM STATUS: FULLY OPERATIONAL")
    elif all_2048:
        print("⚠️  SYSTEM STATUS: OPERATIONAL (Late chunking below 90%)")
    else:
        print("❌ SYSTEM STATUS: ISSUES DETECTED")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Check ArXiv database status and Jina v4 deployment'
    )
    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Show detailed statistics'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON (for programmatic use)'
    )
    args = parser.parse_args()
    
    try:
        db = get_db_connection()
        
        # Collect all status information
        results = {
            'metadata': check_metadata(db),
            'chunks': check_chunks(db, args.detailed),
            'tables': check_tables(db),
            'equations': check_equations(db),
            'images': check_images(db),
            'timestamp': datetime.now().isoformat()
        }
        
        if args.json:
            import json
            print(json.dumps(results, indent=2))
        else:
            print_status_report(results, args.detailed)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error checking database status: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())