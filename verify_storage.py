#!/usr/bin/env python3
"""
Verify that records are being stored properly in the database.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.database.database_factory import DatabaseFactory

def verify_recent_records():
    """Check recently stored records."""
    db = DatabaseFactory.get_arango(
        database='academy_store',
        username='root',
        use_unix=True
    )

    print("Database Storage Verification")
    print("=" * 60)

    # Get counts
    metadata_count = db.collection('arxiv_metadata').count()
    chunks_count = db.collection('arxiv_abstract_chunks').count()
    embeddings_count = db.collection('arxiv_abstract_embeddings').count()

    print(f"\nCollection Counts:")
    print(f"  arxiv_metadata:            {metadata_count:,}")
    print(f"  arxiv_abstract_chunks:     {chunks_count:,}")
    print(f"  arxiv_abstract_embeddings: {embeddings_count:,}")

    # Check if counts are aligned (they should be close)
    print(f"\nConsistency Check:")
    if abs(metadata_count - embeddings_count) <= 10:
        print(f"  ✅ Counts are aligned (diff: {abs(metadata_count - embeddings_count)})")
    else:
        print(f"  ⚠️  Counts differ by {abs(metadata_count - embeddings_count)}")

    # Get a recent record to verify structure
    print(f"\nSample Recent Record:")
    try:
        # Get most recent from metadata
        cursor = db.aql.execute('''
            FOR doc IN arxiv_metadata
                SORT doc.processed_at DESC
                LIMIT 1
                RETURN doc
        ''')
        recent_meta = next(cursor, None)

        if recent_meta:
            arxiv_id = recent_meta.get('arxiv_id')
            print(f"  ArXiv ID: {arxiv_id}")
            print(f"  Title: {recent_meta.get('title', 'N/A')[:80]}...")
            print(f"  Processed: {recent_meta.get('processed_at', 'N/A')}")

            # Check for corresponding embedding
            cursor = db.aql.execute('''
                FOR doc IN arxiv_abstract_embeddings
                    FILTER doc.arxiv_id == @id
                    LIMIT 1
                    RETURN doc
            ''', bind_vars={'id': arxiv_id})

            embedding = next(cursor, None)
            if embedding:
                print(f"  ✅ Has embedding (dim: {embedding.get('embedding_dim', 'N/A')})")
            else:
                print(f"  ❌ No embedding found")

            # Check for chunks
            cursor = db.aql.execute('''
                FOR doc IN arxiv_abstract_chunks
                    FILTER doc.arxiv_id == @id
                    RETURN doc
            ''', bind_vars={'id': arxiv_id})

            chunks = list(cursor)
            print(f"  ✅ Has {len(chunks)} chunks")

    except Exception as e:
        print(f"  Error checking recent record: {e}")

    # Check processing rate over last minute
    print(f"\nRecent Processing Activity:")
    try:
        one_min_ago = (datetime.now() - timedelta(minutes=1)).isoformat()
        cursor = db.aql.execute('''
            FOR doc IN arxiv_metadata
                FILTER doc.processed_at >= @time
                COLLECT WITH COUNT INTO count
                RETURN count
        ''', bind_vars={'time': one_min_ago})

        recent_count = next(cursor, 0)
        print(f"  Records in last minute: {recent_count}")
        print(f"  Rate: ~{recent_count:.0f} records/minute")

    except Exception as e:
        print(f"  Could not check recent activity: {e}")

if __name__ == "__main__":
    verify_recent_records()