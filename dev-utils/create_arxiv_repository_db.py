#!/usr/bin/env python3
"""
Create new ArXiv Repository database for testing sorted workflow.
This keeps the existing arxiv_repository_new database intact.
"""

import os
import sys
from arango import ArangoClient

def create_arxiv_repository_db():
    """Create new database and collections for sorted workflow testing."""

    # Get credentials
    password = os.environ.get('ARANGO_PASSWORD')
    if not password:
        print("ERROR: ARANGO_PASSWORD environment variable not set")
        sys.exit(1)

    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')
    sys_db = client.db('_system', username='root', password=password)

    # Database name for sorted workflow
    db_name = 'arxiv_repository'

    # Check if database exists
    if sys_db.has_database(db_name):
        print(f"Database '{db_name}' already exists")
        response = input("Do you want to drop and recreate it? (y/n): ")
        if response.lower() != 'y':
            print("Keeping existing database")
            return
        else:
            sys_db.delete_database(db_name)
            print(f"Dropped existing database '{db_name}'")

    # Create new database
    sys_db.create_database(db_name)
    print(f"Created database '{db_name}'")

    # Connect to new database
    db = client.db(db_name, username='root', password=password)

    # Create collections
    collections = [
        'arxiv_papers',           # Main metadata collection
        'arxiv_chunks',           # Chunk storage
        'arxiv_abstract_embeddings',  # Embeddings
        'arxiv_processing_order'  # Size-based processing order
    ]

    for collection_name in collections:
        if not db.has_collection(collection_name):
            db.create_collection(collection_name)
            print(f"  Created collection: {collection_name}")

    # Create indexes for efficient querying
    papers_col = db['arxiv_papers']

    # Index on arxiv_id for fast lookups
    papers_col.add_hash_index(fields=['arxiv_id'], unique=True)
    print("  Created index on arxiv_id")

    # Index on processing position for sorted workflow
    papers_col.add_skiplist_index(fields=['size_order_position'])
    print("  Created index on size_order_position")

    # Index on token count for analytics
    papers_col.add_skiplist_index(fields=['token_count'])
    print("  Created index on token_count")

    # Index on processed_at for resume functionality
    papers_col.add_skiplist_index(fields=['processed_at'])
    print("  Created index on processed_at")

    # Create index on processing order collection
    order_col = db['arxiv_processing_order']
    order_col.add_skiplist_index(fields=['position'])
    order_col.add_hash_index(fields=['arxiv_id'], unique=True)
    print("  Created indexes on arxiv_processing_order")

    print(f"\n✅ Database '{db_name}' is ready for sorted workflow testing!")
    print("\nTo use this database, run:")
    print(f"python -m core.workflows.workflow_arxiv_sorted --database {db_name} --drop-collections --max-records 10000")

if __name__ == "__main__":
    create_arxiv_repository_db()