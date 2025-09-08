#!/usr/bin/env python3
"""
Build Graph from Existing Data.

Creates edges between:
1. Papers and their chunks
2. Similar papers (based on embeddings)
3. Papers and code (where we can infer connections)
4. Citation networks (if available)

This prepares the graph for GraphSAGE processing.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from arango import ArangoClient

sys.path.append(str(Path(__file__).parent.parent.parent.parent))


class GraphBuilder:
    """Builds graph edges from existing node data."""
    
    def __init__(self, db_config: Dict):
        """Initialize with database configuration."""
        self.client = ArangoClient(hosts=db_config.get('host', 'http://localhost:8529'))
        self.db = self.client.db(
            db_config['database'],
            username=db_config['username'],
            password=db_config.get('password', os.environ.get('ARANGO_PASSWORD'))
        )
        
        # Edge collections
        self.edge_collections = {
            'paper_has_chunk': None,
            'chunk_similarity': None,
            'paper_similarity': None,
            'paper_cites_paper': None,
            'code_implements_paper': None
        }
        
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure all edge collections exist."""
        for collection_name in self.edge_collections.keys():
            if not self.db.has_collection(collection_name):
                self.db.create_collection(collection_name, edge=True)
                print(f"Created edge collection: {collection_name}")
            self.edge_collections[collection_name] = self.db.collection(collection_name)
    
    def build_paper_chunk_edges(self):
        """Create edges between papers and their chunks."""
        print("\nBuilding paper-chunk edges...")
        
        # Get all chunks with paper_id
        cursor = self.db.aql.execute("""
            FOR chunk IN arxiv_chunks
            FILTER chunk.paper_id != null
            RETURN {
                chunk_id: chunk._id,
                paper_id: chunk.paper_id
            }
        """)
        
        edges = []
        for doc in cursor:
            # Create edge from paper to chunk
            edge = {
                '_from': f"arxiv_papers/{doc['paper_id']}",
                '_to': doc['chunk_id'],
                'type': 'has_chunk'
            }
            edges.append(edge)
        
        if edges:
            # Insert in batches
            batch_size = 1000
            total_inserted = 0
            
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                try:
                    self.edge_collections['paper_has_chunk'].insert_many(batch, overwrite=True)
                    total_inserted += len(batch)
                except Exception as e:
                    print(f"Error inserting batch: {e}")
            
            print(f"Created {total_inserted} paper-chunk edges")
        else:
            print("No paper-chunk relationships found")
    
    def build_embedding_similarity_edges(self, similarity_threshold: float = 0.8):
        """Create edges between similar embeddings."""
        print(f"\nBuilding similarity edges (threshold={similarity_threshold})...")
        
        # Get all embeddings
        cursor = self.db.aql.execute("""
            FOR embed IN arxiv_embeddings
            RETURN {
                id: embed._id,
                key: embed._key,
                paper_id: embed.paper_id,
                chunk_id: embed.chunk_id,
                vector: embed.vector
            }
        """)
        
        embeddings = list(cursor)
        
        if not embeddings:
            print("No embeddings found")
            return
        
        print(f"Processing {len(embeddings)} embeddings...")
        
        # Convert to numpy array
        vectors = np.array([e['vector'] for e in embeddings])
        
        # Compute pairwise similarities (in batches for memory efficiency)
        batch_size = 100
        edges = []
        
        for i in tqdm(range(0, len(embeddings), batch_size)):
            batch_end = min(i + batch_size, len(embeddings))
            batch_vectors = vectors[i:batch_end]
            
            # Compute similarities with all other vectors
            similarities = cosine_similarity(batch_vectors, vectors)
            
            # Find high similarity pairs
            for local_idx in range(batch_end - i):
                global_idx = i + local_idx
                
                # Get top similar items (excluding self)
                sim_scores = similarities[local_idx]
                sim_scores[global_idx] = -1  # Exclude self
                
                # Find indices above threshold
                similar_indices = np.where(sim_scores >= similarity_threshold)[0]
                
                for sim_idx in similar_indices[:10]:  # Limit to top 10
                    if sim_idx != global_idx:
                        edge = {
                            '_from': embeddings[global_idx]['id'],
                            '_to': embeddings[sim_idx]['id'],
                            'similarity': float(sim_scores[sim_idx]),
                            'type': 'semantic_similarity'
                        }
                        edges.append(edge)
        
        if edges:
            # Insert edges
            print(f"\nInserting {len(edges)} similarity edges...")
            
            batch_size = 1000
            total_inserted = 0
            
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                try:
                    self.edge_collections['chunk_similarity'].insert_many(batch, overwrite=True)
                    total_inserted += len(batch)
                except Exception as e:
                    print(f"Error inserting batch: {e}")
            
            print(f"Created {total_inserted} similarity edges")
        else:
            print("No similar embeddings found above threshold")
    
    def build_code_paper_edges(self):
        """Create edges between code and papers based on heuristics."""
        print("\nBuilding code-paper edges...")
        
        # Simple heuristic: Connect papers about specific topics to relevant code
        # This is a placeholder - in production, use more sophisticated matching
        
        connections = [
            # Known connections (examples)
            ('1706.03762', 'transformers'),  # Attention is All You Need -> Transformers library
            ('1810.04805', 'bert'),  # BERT paper -> BERT implementations
        ]
        
        edges = []
        for paper_id, code_pattern in connections:
            # Find matching code files
            cursor = self.db.aql.execute("""
                FOR code IN github_papers
                FILTER CONTAINS(LOWER(code.path), @pattern)
                RETURN code._id
            """, bind_vars={'pattern': code_pattern})
            
            code_files = list(cursor)
            
            for code_id in code_files:
                edge = {
                    '_from': f'arxiv_papers/{paper_id}',
                    '_to': code_id,
                    'type': 'implements',
                    'confidence': 0.8
                }
                edges.append(edge)
        
        if edges:
            try:
                self.edge_collections['code_implements_paper'].insert_many(edges, overwrite=True)
                print(f"Created {len(edges)} code-paper edges")
            except Exception as e:
                print(f"Error creating code-paper edges: {e}")
        else:
            print("No code-paper connections found")
    
    def build_citation_edges(self):
        """Build citation edges between papers."""
        print("\nBuilding citation edges...")
        
        # Check if we have citation data
        cursor = self.db.aql.execute("""
            FOR paper IN arxiv_papers
            FILTER paper.references != null OR paper.citations != null
            LIMIT 100
            RETURN {
                id: paper.arxiv_id,
                refs: paper.references,
                cites: paper.citations
            }
        """)
        
        papers_with_citations = list(cursor)
        
        if not papers_with_citations:
            print("No citation data found in papers")
            return
        
        edges = []
        for paper in papers_with_citations:
            # Process references
            if paper.get('refs'):
                for ref_id in paper['refs']:
                    edge = {
                        '_from': f"arxiv_papers/{paper['id']}",
                        '_to': f"arxiv_papers/{ref_id}",
                        'type': 'cites'
                    }
                    edges.append(edge)
        
        if edges:
            try:
                self.edge_collections['paper_cites_paper'].insert_many(edges, overwrite=True)
                print(f"Created {len(edges)} citation edges")
            except Exception as e:
                print(f"Error creating citation edges: {e}")
        else:
            print("No citations to process")
    
    def get_statistics(self):
        """Get statistics about the built graph."""
        print("\n" + "="*60)
        print("GRAPH STATISTICS")
        print("="*60)
        
        # Node counts
        node_collections = ['arxiv_papers', 'arxiv_chunks', 'arxiv_embeddings', 
                          'github_repositories', 'github_papers']
        
        print("\nNodes:")
        total_nodes = 0
        for coll_name in node_collections:
            try:
                count = self.db.collection(coll_name).count()
                print(f"  {coll_name}: {count:,}")
                total_nodes += count
            except:
                pass
        print(f"  Total: {total_nodes:,}")
        
        # Edge counts
        print("\nEdges:")
        total_edges = 0
        for coll_name, collection in self.edge_collections.items():
            if collection:
                try:
                    count = collection.count()
                    print(f"  {coll_name}: {count:,}")
                    total_edges += count
                except:
                    pass
        print(f"  Total: {total_edges:,}")
        
        # Sample connections
        print("\nSample connections:")
        
        # Get a paper with chunks
        cursor = self.db.aql.execute("""
            FOR edge IN paper_has_chunk
            LIMIT 1
            RETURN {
                paper: edge._from,
                chunk: edge._to
            }
        """)
        
        samples = list(cursor)
        if samples:
            print(f"  Paper->Chunk: {samples[0]['paper']} -> {samples[0]['chunk']}")
        
        # Get a similarity edge
        cursor = self.db.aql.execute("""
            FOR edge IN chunk_similarity
            LIMIT 1
            RETURN {
                from: edge._from,
                to: edge._to,
                similarity: edge.similarity
            }
        """)
        
        samples = list(cursor)
        if samples:
            print(f"  Similarity: {samples[0]['from']} <-> {samples[0]['to']} (score: {samples[0]['similarity']:.3f})")
    
    def build_all(self):
        """Build all graph edges."""
        print("\n" + "="*60)
        print("BUILDING COMPLETE GRAPH")
        print("="*60)
        
        start_time = time.time()
        
        # Build all edge types
        self.build_paper_chunk_edges()
        self.build_embedding_similarity_edges(similarity_threshold=0.75)
        self.build_code_paper_edges()
        self.build_citation_edges()
        
        elapsed = time.time() - start_time
        
        # Show statistics
        self.get_statistics()
        
        print(f"\nGraph building completed in {elapsed:.2f} seconds")


def main():
    """Main entry point."""
    db_config = {
        'host': 'http://localhost:8529',
        'database': 'academy_store',
        'username': 'root',
        'password': os.environ.get('ARANGO_PASSWORD')
    }
    
    if not db_config['password']:
        print("Error: ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
    
    builder = GraphBuilder(db_config)
    builder.build_all()
    
    print("\n✅ Graph building complete! Ready for GraphSAGE processing.")


if __name__ == "__main__":
    main()