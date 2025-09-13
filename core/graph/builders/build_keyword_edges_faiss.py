#!/usr/bin/env python3
"""
Use Faiss for efficient large-scale similarity search on 2.8M papers.
Faiss is designed for billion-scale similarity search.
"""

import os
import sys
import time
import logging
import numpy as np
import faiss
from tqdm import tqdm
from pathlib import Path
import click
from arango import ArangoClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaissKeywordEdgeBuilder:
    """Build keyword edges using Faiss for efficient similarity search."""
    
    def __init__(self):
        # Connect to database
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
    
    def build_edges_faiss(
        self,
        threshold: float = 0.65,
        top_k: int = 50,
        sample_size: int = None
    ):
        """
        Build keyword edges using Faiss.
        
        Args:
            threshold: Minimum similarity threshold
            top_k: Number of nearest neighbors to find
            sample_size: If set, only process this many papers (for testing)
        """
        logger.info("="*70)
        logger.info("FAISS KEYWORD EDGE BUILDING")
        logger.info("="*70)
        
        # Load papers with keywords
        logger.info("Loading papers with keywords...")
        query = "FOR p IN arxiv_papers FILTER p.keywords != null RETURN {_key: p._key, categories: p.categories}"
        papers = list(self.db.aql.execute(query))
        
        if sample_size:
            papers = papers[:sample_size]
        
        logger.info(f"Processing {len(papers):,} papers")
        
        # Load embeddings efficiently using batch queries
        logger.info("Loading keyword embeddings in batches...")
        embeddings = []
        paper_ids = []
        categories = []
        
        # Process in chunks for efficiency
        chunk_size = 50000  # Larger chunks for batch query
        
        for i in tqdm(range(0, len(papers), chunk_size), desc="Loading chunks"):
            chunk = papers[i:i+chunk_size]
            keys = [p['_key'] for p in chunk]
            
            # Batch fetch embeddings using AQL with direct key list
            query = """
            FOR p IN @keys
            LET e = DOCUMENT('arxiv_embeddings', p)
            FILTER e != null AND e.keyword_embedding != null
            RETURN {key: p, embedding: e.keyword_embedding}
            """
            
            results = list(self.db.aql.execute(query, bind_vars={'keys': keys}))
            
            # Process results
            key_to_paper = {p['_key']: p for p in chunk}
            for r in results:
                embeddings.append(r['embedding'])
                paper_ids.append(r['key'])
                paper = key_to_paper.get(r['key'])
                if paper:
                    categories.append(paper.get('categories', ['unknown'])[0][:30])
        
        if not embeddings:
            logger.error("No embeddings found!")
            return
        
        embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"Loaded {len(embeddings):,} embeddings")
        
        # Normalize for cosine similarity
        logger.info("Normalizing embeddings...")
        faiss.normalize_L2(embeddings)
        
        # Build Faiss index
        logger.info("Building Faiss index...")
        dimension = embeddings.shape[1]
        
        # Use IVF index for large-scale search
        # Adjust nlist based on dataset size (sqrt(n) is a good heuristic)
        nlist = int(np.sqrt(len(embeddings)))
        nlist = min(nlist, 2048)  # Reduced cap for better memory usage
        
        # Create index with PQ quantization for memory efficiency
        logger.info(f"Creating index with {nlist} clusters...")
        quantizer = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization
        
        # Use IVFPQ for memory efficiency on large datasets
        if len(embeddings) > 100000:
            # PQ parameters: dimension must be divisible by m
            m = 64  # Number of subquantizers
            nbits = 8  # Bits per subquantizer
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train index
        logger.info(f"Training index...")
        # Sample training data if dataset is huge
        if len(embeddings) > 1000000:
            train_sample = embeddings[::10]  # Use every 10th vector for training
            index.train(train_sample)
        else:
            index.train(embeddings)
        
        # Add vectors
        logger.info("Adding vectors to index...")
        index.add(embeddings)
        
        # Set search parameters
        index.nprobe = min(32, nlist // 8)  # More conservative search
        
        # Clear existing keyword edges
        logger.info("Clearing existing keyword edges...")
        keyword_coll = self.db.collection('keyword_similarity')
        keyword_coll.truncate()
        
        # Search for similar papers
        logger.info(f"Finding top-{top_k} similar papers...")
        
        # Process in smaller batches to manage memory
        batch_size = 5000  # Reduced batch size
        all_edges = []
        total_edges = 0
        cross_category_edges = 0
        
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Finding similarities"):
            batch_end = min(i + batch_size, len(embeddings))
            batch = embeddings[i:batch_end]
            
            # Search
            similarities, indices = index.search(batch, min(top_k + 1, 101))  # Cap at 101 for memory
            
            # Process results
            for j in range(batch_end - i):
                paper_idx = i + j
                paper_id = paper_ids[paper_idx]
                paper_cat = categories[paper_idx]
                
                for k in range(1, top_k + 1):  # Skip first (self)
                    if indices[j, k] == -1:  # No more neighbors
                        break
                    
                    similarity = similarities[j, k]
                    if similarity < threshold:
                        continue
                    
                    target_idx = indices[j, k]
                    target_id = paper_ids[target_idx]
                    target_cat = categories[target_idx]
                    
                    # Create edge
                    is_cross = paper_cat != target_cat
                    weight = float(similarity) * (1.2 if is_cross else 1.0)
                    
                    all_edges.append({
                        '_from': f'arxiv_papers/{paper_id}',
                        '_to': f'arxiv_papers/{target_id}',
                        'weight': weight,
                        'similarity': float(similarity),
                        'from_category': paper_cat[:30],  # Truncate long category names
                        'to_category': target_cat[:30],
                        'cross_category': is_cross
                    })
                    
                    if is_cross:
                        cross_category_edges += 1
                    
                    # Insert batch
                    if len(all_edges) >= 50000:
                        keyword_coll.insert_many(all_edges)
                        total_edges += len(all_edges)
                        all_edges = []
        
        # Insert remaining
        if all_edges:
            keyword_coll.insert_many(all_edges)
            total_edges += len(all_edges)
        
        # Report results
        logger.info("="*70)
        logger.info("RESULTS")
        logger.info("="*70)
        logger.info(f"Total keyword edges: {total_edges:,}")
        logger.info(f"Cross-category edges: {cross_category_edges:,} ({cross_category_edges/total_edges*100:.1f}%)")
        
        # Sample interdisciplinary connections
        if cross_category_edges > 0:
            logger.info("\nTop interdisciplinary connections:")
            query = """
            FOR e IN keyword_similarity
            FILTER e.cross_category == true
            SORT e.weight DESC
            LIMIT 10
            RETURN {
                from: e.from_category,
                to: e.to_category,
                similarity: e.similarity
            }
            """
            for conn in self.db.aql.execute(query):
                logger.info(f"  {conn['from']:20s} <-> {conn['to']:20s} (sim: {conn['similarity']:.3f})")


@click.command()
@click.option('--threshold', default=0.65, help='Similarity threshold')
@click.option('--top-k', default=50, help='Top K neighbors')
@click.option('--sample', default=None, type=int, help='Sample size for testing')
def main(threshold: float, top_k: int, sample: int):
    """Build keyword edges using Faiss."""
    builder = FaissKeywordEdgeBuilder()
    builder.build_edges_faiss(threshold, top_k, sample)


if __name__ == "__main__":
    main()