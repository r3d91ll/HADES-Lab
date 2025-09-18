#!/usr/bin/env python3
from arango import ArangoClient
import os
import sys

# Todd's Law #2: Reliability with controlled failure modes
password = os.environ.get("ARANGO_PASSWORD", "")
if not password:
    print("ERROR: ARANGO_PASSWORD environment variable not set", file=sys.stderr)
    print("Set it with: export ARANGO_PASSWORD='your_password'", file=sys.stderr)
    sys.exit(1)

try:
    client = ArangoClient(hosts="http://localhost:8529")
    db = client.db("arxiv_repository", username="root", password=password)
    # Test connection
    db.aql.execute("RETURN 1").next()
except Exception as e:
    print(f"ERROR: Failed to connect to ArangoDB: {e}", file=sys.stderr)
    print("Ensure ArangoDB is running and accessible at localhost:8529", file=sys.stderr)
    sys.exit(1)

# Get counts
metadata_count = db.collection('arxiv_metadata').count()
embeddings_count = db.collection('arxiv_abstract_embeddings').count()
chunks_count = db.collection('arxiv_abstract_chunks').count()

print("="*60)
print("FINAL DATABASE STATE")
print("="*60)
print(f"Metadata: {metadata_count:,}")
print(f"Embeddings: {embeddings_count:,}")
print(f"Chunks: {chunks_count:,}")

# Get last processed
print("\n" + "="*60)
print("LAST PROCESSED RECORDS")
print("="*60)
cursor = db.aql.execute("""
    FOR e IN arxiv_abstract_embeddings
        SORT e.created_at DESC
        LIMIT 5
        FOR doc IN arxiv_metadata
            FILTER doc.arxiv_id == e.arxiv_id
            RETURN {id: doc.arxiv_id, length: doc.abstract_length}
""")

for rec in cursor:
    print(f"ID: {rec['id']:20} | Length: {rec['length']:4}")

# Find next unprocessed
print("\n" + "="*60)
print("NEXT UNPROCESSED RECORDS")
print("="*60)

# Get processed IDs
cursor = db.aql.execute("FOR e IN arxiv_abstract_embeddings RETURN DISTINCT e.arxiv_id")
processed = set(cursor)

# Find unprocessed in size order
cursor = db.aql.execute("""
    FOR doc IN arxiv_metadata
        FILTER doc.abstract != null AND doc.abstract_length >= 655
        SORT doc.abstract_length ASC
        LIMIT 1000
        RETURN {id: doc.arxiv_id, length: doc.abstract_length}
""")

count = 0
for rec in cursor:
    if rec['id'] not in processed:
        count += 1
        print(f"{count:2}. ID: {rec['id']:20} | Length: {rec['length']:4}")
        if count >= 10:
            break

print("\n" + "="*60)
print("RESUME EXPECTATION")
print("="*60)
print("When resumed WITHOUT --drop-collections:")
print("1. Workflow will detect embeddings_count > 0")
print("2. Will use recovery query to find unprocessed records")
print("3. Will maintain size-sorted order (abstract_length ASC)")
print("4. Should start processing abstracts of length 655+")