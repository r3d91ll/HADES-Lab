"""
Build Metadata-Based Graph for All ArXiv Papers.

Creates edges based on metadata without requiring full document processing:
1. Co-authorship networks
2. Category/domain connections  
3. Citation networks (if available)
4. Temporal proximity
5. Abstract similarity (if embeddings available)

This provides uniform graph structure across all 2.8M papers.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
from tqdm import tqdm
from arango import ArangoClient
import time

sys.path.append(str(Path(__file__).parent.parent.parent.parent))


class MetadataGraphBuilder:
    """Builds graph edges from paper metadata."""
    
    def __init__(self, db_config: Dict):
        """Initialize with database configuration."""
        self.client = ArangoClient(hosts=db_config.get('host', 'http://localhost:8529'))
        self.db = self.client.db(
            db_config['database'],
            username=db_config['username'],
            password=db_config.get('password', os.environ.get('ARANGO_PASSWORD'))
        )
        
        # Edge collections for metadata graph
        self.edge_collections = {
            'coauthorship': None,
            'shared_category': None,
            'temporal_proximity': None,
            'abstract_similarity': None,
            'citation_network': None
        }
        
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure all edge collections exist."""
        for collection_name in self.edge_collections.keys():
            if not self.db.has_collection(collection_name):
                self.db.create_collection(collection_name, edge=True)
                print(f"Created edge collection: {collection_name}")
            self.edge_collections[collection_name] = self.db.collection(collection_name)
    
    def build_coauthorship_edges(self, sample_size: int = None):
        """Create edges between papers with shared authors."""
        print("\nBuilding co-authorship edges...")
        
        # Build author to papers index
        query = """
            FOR p IN arxiv_papers
            FILTER p.authors != null AND LENGTH(p.authors) > 0
            LIMIT @limit
            RETURN {
                paper_id: p._key,
                authors: p.authors
            }
        """
        
        bind_vars = {'limit': sample_size if sample_size else 1000000}
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        
        # Build inverted index: author -> papers
        author_papers = defaultdict(set)
        paper_authors = {}
        
        for doc in tqdm(cursor, desc="Indexing authors"):
            paper_id = doc['paper_id']
            paper_authors[paper_id] = set(doc['authors'])
            
            for author in doc['authors']:
                author_papers[author].add(paper_id)
        
        print(f"Found {len(author_papers)} unique authors across {len(paper_authors)} papers")
        
        # Create edges between papers with shared authors
        edges = []
        processed_pairs = set()
        
        for author, papers in tqdm(author_papers.items(), desc="Creating co-authorship edges"):
            if len(papers) < 2:
                continue
                
            papers_list = list(papers)
            for i in range(len(papers_list)):
                for j in range(i + 1, len(papers_list)):
                    pair = tuple(sorted([papers_list[i], papers_list[j]]))
                    
                    if pair not in processed_pairs:
                        processed_pairs.add(pair)
                        
                        # Calculate collaboration strength
                        shared = len(paper_authors[papers_list[i]] & paper_authors[papers_list[j]])
                        total = len(paper_authors[papers_list[i]] | paper_authors[papers_list[j]])
                        strength = shared / total if total > 0 else 0
                        
                        edge = {
                            '_from': f'arxiv_papers/{papers_list[i]}',
                            '_to': f'arxiv_papers/{papers_list[j]}',
                            'weight': strength,
                            'shared_authors': shared,
                            'type': 'coauthorship'
                        }
                        edges.append(edge)
        
        # Insert edges in batches
        if edges:
            batch_size = 1000
            total_inserted = 0
            
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                try:
                    self.edge_collections['coauthorship'].insert_many(batch, overwrite=True)
                    total_inserted += len(batch)
                except Exception as e:
                    print(f"Error inserting batch: {e}")
            
            print(f"Created {total_inserted} co-authorship edges")
        else:
            print("No co-authorship edges found")
    
    def build_category_edges(self, sample_size: int = None):
        """Create edges between papers in same categories."""
        print("\nBuilding category edges...")
        
        # Build category to papers index
        query = """
            FOR p IN arxiv_papers
            FILTER p.categories != null AND LENGTH(p.categories) > 0
            LIMIT @limit
            RETURN {
                paper_id: p._key,
                categories: p.categories
            }
        """
        
        bind_vars = {'limit': sample_size if sample_size else 1000000}
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        
        # Build inverted index: category -> papers
        category_papers = defaultdict(set)
        paper_categories = {}
        
        for doc in tqdm(cursor, desc="Indexing categories"):
            paper_id = doc['paper_id']
            paper_categories[paper_id] = set(doc['categories'])
            
            for category in doc['categories']:
                # Also index parent category (e.g., 'cs' from 'cs.LG')
                category_papers[category].add(paper_id)
                if '.' in category:
                    parent = category.split('.')[0]
                    category_papers[parent].add(paper_id)
        
        print(f"Found {len(category_papers)} unique categories across {len(paper_categories)} papers")
        
        # Create edges between papers with shared categories (sample for efficiency)
        edges = []
        processed_pairs = set()
        
        for category, papers in tqdm(category_papers.items(), desc="Creating category edges"):
            if len(papers) < 2 or len(papers) > 10000:  # Skip huge categories
                continue
            
            # Sample if too many papers
            papers_list = list(papers)
            if len(papers_list) > 100:
                import random
                papers_list = random.sample(papers_list, 100)
            
            for i in range(len(papers_list)):
                for j in range(i + 1, min(i + 10, len(papers_list))):  # Limit connections
                    pair = tuple(sorted([papers_list[i], papers_list[j]]))
                    
                    if pair not in processed_pairs:
                        processed_pairs.add(pair)
                        
                        # Calculate category similarity
                        if papers_list[i] in paper_categories and papers_list[j] in paper_categories:
                            shared = len(paper_categories[papers_list[i]] & paper_categories[papers_list[j]])
                            total = len(paper_categories[papers_list[i]] | paper_categories[papers_list[j]])
                            similarity = shared / total if total > 0 else 0
                            
                            edge = {
                                '_from': f'arxiv_papers/{papers_list[i]}',
                                '_to': f'arxiv_papers/{papers_list[j]}',
                                'weight': similarity,
                                'shared_categories': shared,
                                'category': category,
                                'type': 'same_field'
                            }
                            edges.append(edge)
        
        # Insert edges in batches
        if edges:
            batch_size = 1000
            total_inserted = 0
            
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                try:
                    self.edge_collections['shared_category'].insert_many(batch, overwrite=True)
                    total_inserted += len(batch)
                except Exception as e:
                    print(f"Error inserting batch: {e}")
            
            print(f"Created {total_inserted} category edges")
        else:
            print("No category edges found")
    
    def build_temporal_edges(self, sample_size: int = None):
        """Create edges between papers published close in time."""
        print("\nBuilding temporal proximity edges...")
        
        # Get papers with dates
        query = """
            FOR p IN arxiv_papers
            FILTER p.update_date != null
            SORT p.update_date
            LIMIT @limit
            RETURN {
                paper_id: p._key,
                date: p.update_date,
                categories: p.categories
            }
        """
        
        bind_vars = {'limit': sample_size if sample_size else 10000}
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        
        papers = list(cursor)
        print(f"Processing {len(papers)} papers for temporal edges")
        
        edges = []
        window_days = 7  # Papers within a week
        
        for i in tqdm(range(len(papers)), desc="Creating temporal edges"):
            paper_i = papers[i]
            
            # Look ahead for papers within time window
            j = i + 1
            while j < len(papers) and j < i + 50:  # Limit lookahead
                paper_j = papers[j]
                
                # Check if in same category (for relevance)
                if paper_i.get('categories') and paper_j.get('categories'):
                    shared_cats = set(paper_i['categories']) & set(paper_j['categories'])
                    
                    if shared_cats:
                        edge = {
                            '_from': f"arxiv_papers/{paper_i['paper_id']}",
                            '_to': f"arxiv_papers/{paper_j['paper_id']}",
                            'type': 'temporal_proximity',
                            'shared_categories': list(shared_cats)
                        }
                        edges.append(edge)
                
                j += 1
        
        # Insert edges
        if edges:
            batch_size = 1000
            total_inserted = 0
            
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                try:
                    self.edge_collections['temporal_proximity'].insert_many(batch, overwrite=True)
                    total_inserted += len(batch)
                except Exception as e:
                    print(f"Error inserting batch: {e}")
            
            print(f"Created {total_inserted} temporal edges")
    
    def get_statistics(self):
        """Get statistics about the built graph."""
        print("\n" + "="*60)
        print("METADATA GRAPH STATISTICS")
        print("="*60)
        
        # Node count
        papers_count = self.db.collection('arxiv_papers').count()
        print(f"\nNodes: {papers_count:,} papers")
        
        # Edge counts
        print("\nEdges:")
        total_edges = 0
        for coll_name, collection in self.edge_collections.items():
            if collection:
                try:
                    count = collection.count()
                    print(f"  {coll_name}: {count:,}")
                    total_edges += count
                except Exception:
                    pass
        print(f"  Total: {total_edges:,}")
        
        # Sample connections
        print("\nSample connections:")
        
        # Get a paper with many connections
        cursor = self.db.aql.execute("""
            FOR p IN arxiv_papers
            FILTER LENGTH(p.authors) > 5
            LIMIT 1
            RETURN p._key
        """)
        
        sample_papers = list(cursor)
        if sample_papers:
            paper_id = sample_papers[0]
            
            # Count connections
            for edge_type in ['coauthorship', 'shared_category']:
                cursor = self.db.aql.execute(f"""
                    FOR e IN {edge_type}
                    FILTER e._from == 'arxiv_papers/{paper_id}' OR e._to == 'arxiv_papers/{paper_id}'
                    RETURN e
                """, bind_vars={'paper_id': paper_id})
                
                edges = list(cursor)
                if edges:
                    print(f"  Paper {paper_id} has {len(edges)} {edge_type} connections")
    
    def build_all(self, sample_size: int = 10000):
        """Build all metadata-based edges."""
        print("\n" + "="*60)
        print("BUILDING METADATA GRAPH")
        print("="*60)
        print(f"Sample size: {sample_size:,} papers")
        
        start_time = time.time()
        
        # Build different edge types
        self.build_coauthorship_edges(sample_size)
        self.build_category_edges(sample_size)
        self.build_temporal_edges(sample_size)
        
        elapsed = time.time() - start_time
        
        # Show statistics
        self.get_statistics()
        
        print(f"\nMetadata graph building completed in {elapsed:.2f} seconds")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build metadata graph")
    parser.add_argument('--sample', type=int, default=10000,
                       help='Number of papers to process (default: 10000)')
    args = parser.parse_args()
    
    db_config = {
        'host': 'http://localhost:8529',
        'database': 'academy_store',
        'username': 'root',
        'password': os.environ.get('ARANGO_PASSWORD')
    }
    
    if not db_config['password']:
        print("Error: ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
    
    builder = MetadataGraphBuilder(db_config)
    builder.build_all(sample_size=args.sample)
    
    print("\n✅ Metadata graph building complete!")


if __name__ == "__main__":
    main()