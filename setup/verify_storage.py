#!/usr/bin/env python3
"""
Verify that records are being stored properly in the database.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path (parent of setup/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database.database_factory import DatabaseFactory

def verify_recent_records():
    """Check recently stored records."""
    db = DatabaseFactory.get_arango_memory_service(
        database='academy_store'
    )

    print("Database Storage Verification")
    print("=" * 60)

    # Get counts
    papers_count = db.execute_query("RETURN LENGTH(arxiv_papers)")[0]
    embeddings_count = db.execute_query("RETURN LENGTH(arxiv_embeddings)")[0]
    structures_count = db.execute_query("RETURN LENGTH(arxiv_structures)")[0]

    print(f"\nCollection Counts:")
    print(f"  arxiv_papers:              {papers_count:,}")
    print(f"  arxiv_embeddings:          {embeddings_count:,}")
    print(f"  arxiv_structures:          {structures_count:,}")

    # Check if counts are aligned (they should be close)
    print(f"\nConsistency Check:")
    if papers_count:
        avg_embeddings = embeddings_count / papers_count
        print(f"  Average embeddings per paper: {avg_embeddings:.2f}")
    else:
        print("  ⚠️  No papers found")

    # Get a recent record to verify structure
    print(f"\nSample Recent Record:")
    try:
        # Get most recent from metadata
        recent_docs = db.execute_query('''
            FOR doc IN arxiv_papers
                SORT doc.processing_timestamp DESC
                LIMIT 1
                RETURN doc
        ''')
        recent_meta = recent_docs[0] if recent_docs else None

        if recent_meta:
            arxiv_id = recent_meta.get('arxiv_id')
            print(f"  ArXiv ID: {arxiv_id}")
            print(f"  Title: {recent_meta.get('title', 'N/A')[:80]}...")
            print(f"  Processed: {recent_meta.get('processed_at', 'N/A')}")

            # Check for corresponding embedding
            embedding_result = db.execute_query('''
                FOR doc IN arxiv_embeddings
                    FILTER doc.arxiv_id == @id
                    LIMIT 1
                    RETURN doc
            ''', bind_vars={'id': arxiv_id})

            embedding = embedding_result[0] if embedding_result else None
            if embedding:
                vector = embedding.get('vector', [])
                dim = len(vector) if isinstance(vector, list) else embedding.get('embedding_dim', 'N/A')
                print(f"  ✅ Has embedding (dim: {dim})")
            else:
                print(f"  ❌ No embedding found")

            # Check for chunks
            chunk_count = db.execute_query('''
                RETURN LENGTH(
                    FOR doc IN arxiv_embeddings
                        FILTER doc.arxiv_id == @id
                        RETURN 1
                )
            ''', bind_vars={'id': arxiv_id})[0]
            print(f"  ✅ Has {chunk_count} embeddings")

    except Exception as e:
        print(f"  Error checking recent record: {e}")

    # Check processing rate over last minute
    print(f"\nRecent Processing Activity:")
    try:
        one_min_ago = (datetime.now() - timedelta(minutes=1)).isoformat()
        cursor = db.execute_query('''
            FOR doc IN arxiv_papers
                FILTER doc.processing_timestamp >= @time
                COLLECT WITH COUNT INTO count
                RETURN count
        ''', bind_vars={'time': one_min_ago})
        recent_count = cursor[0] if cursor else 0
        print(f"  Records in last minute: {recent_count}")
        print(f"  Rate: ~{recent_count:.0f} records/minute")

    except Exception as e:
        print(f"  Could not check recent activity: {e}")

if __name__ == "__main__":
    verify_recent_records()
