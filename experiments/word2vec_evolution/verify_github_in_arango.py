#!/usr/bin/env python3
"""
Verify GitHub repositories are in ArangoDB.
"""

from arango import ArangoClient

# Connect
client = ArangoClient(hosts='http://192.168.1.69:8529')
db = client.db('academy_store', username='root', password='1luv93ngu1n$')

# Check collections
repos_coll = db.collection('github_repositories') if db.has_collection('github_repositories') else None
papers_coll = db.collection('github_papers') if db.has_collection('github_papers') else None
chunks_coll = db.collection('github_chunks') if db.has_collection('github_chunks') else None
embeddings_coll = db.collection('github_embeddings') if db.has_collection('github_embeddings') else None

our_repos = [
    ('dav/word2vec', 'word2vec', '1301.3781'),
    ('bnosac/doc2vec', 'doc2vec', '1405.4053'),
    ('tech-srl/code2vec', 'code2vec', '1803.09473')
]

print("=" * 80)
print("GITHUB REPOSITORIES IN ARANGODB")
print("=" * 80)

if not repos_coll:
    print("✗ github_repositories collection not found!")
else:
    for repo_path, name, paper_id in our_repos:
        # Create the key format used by the pipeline
        repo_key = repo_path.replace('/', '_')
        
        # Get repository
        repo = repos_coll.get({'_key': repo_key})
        if repo:
            print(f"\n✓ {name} ({repo_path})")
            print(f"  Repository: {repo.get('full_name', 'N/A')}")
            print(f"  URL: {repo.get('url', 'N/A')}")
            print(f"  Clone URL: {repo.get('clone_url', 'N/A')}")
            print(f"  Associated paper: {paper_id}")
            
            # Count files, chunks, embeddings
            if papers_coll:
                files = list(papers_coll.find({'repository_id': repo['_id']}))
                print(f"  Files in repository: {len(files)}")
                
                # Show some file examples
                if files:
                    print("  Sample files:")
                    for f in files[:3]:
                        print(f"    - {f.get('path', 'N/A')}")
            
            if chunks_coll:
                # Count chunks for this repo's files
                chunk_count = 0
                if papers_coll:
                    for f in papers_coll.find({'repository_id': repo['_id']}):
                        chunk_count += chunks_coll.count({'file_id': f['_id']})
                print(f"  Total chunks: {chunk_count}")
            
            if embeddings_coll:
                # Count embeddings similarly
                embedding_count = 0
                if chunks_coll and papers_coll:
                    for f in papers_coll.find({'repository_id': repo['_id']}):
                        for c in chunks_coll.find({'file_id': f['_id']}):
                            embedding_count += embeddings_coll.count({'chunk_id': c['_id']})
                print(f"  Total embeddings: {embedding_count}")
        else:
            print(f"\n✗ {name} ({repo_path}): NOT FOUND")
            # Try to find it with different key format
            print(f"  Trying alternate key formats...")
            for doc in repos_coll.find({'full_name': repo_path}):
                print(f"  Found with _id: {doc['_id']}")

print("\n" + "=" * 80)
print("Collection Summary:")
if repos_coll:
    print(f"Total repositories: {repos_coll.count()}")
if papers_coll:
    print(f"Total files: {papers_coll.count()}")
if chunks_coll:
    print(f"Total chunks: {chunks_coll.count()}")
if embeddings_coll:
    print(f"Total embeddings: {embeddings_coll.count()}")