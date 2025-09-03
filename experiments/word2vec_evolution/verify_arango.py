#!/usr/bin/env python3
"""
Verify our papers are in ArangoDB.
"""

from arango import ArangoClient

# Connect
client = ArangoClient(hosts='http://192.168.1.69:8529')
db = client.db('academy_store', username='root', password='1luv93ngu1n$')

# Check our papers
papers_coll = db.collection('arxiv_papers')
chunks_coll = db.collection('arxiv_chunks')
embeddings_coll = db.collection('arxiv_embeddings')

our_papers = ['1301.3781', '1405.4053', '1803.09473']

print("=" * 80)
print("WORD2VEC EVOLUTION PAPERS IN ARANGODB")
print("=" * 80)

for arxiv_id in our_papers:
    key = arxiv_id.replace('.', '_')
    
    # Get paper
    paper = papers_coll.get({'_key': key})
    if paper:
        print(f"\n✓ {arxiv_id}: {paper.get('title', 'N/A')[:60]}")
        print(f"  Status: {paper.get('status', 'N/A')}")
        print(f"  Chunks: {paper.get('num_chunks', 'N/A')}")
        
        # Count chunks and embeddings
        chunks = list(chunks_coll.find({'paper_id': arxiv_id}))
        embeddings = list(embeddings_coll.find({'paper_id': arxiv_id}))
        
        print(f"  Verified chunks in DB: {len(chunks)}")
        print(f"  Verified embeddings in DB: {len(embeddings)}")
        
        # Check paper embedding
        if 'paper_embedding' in paper:
            print(f"  Paper-level embedding: {len(paper['paper_embedding'])} dimensions")
    else:
        print(f"\n✗ {arxiv_id}: NOT FOUND")

print("\n" + "=" * 80)
print("Summary:")
print(f"Total papers in collection: {papers_coll.count()}")
print(f"Total chunks in collection: {chunks_coll.count()}")
print(f"Total embeddings in collection: {embeddings_coll.count()}")