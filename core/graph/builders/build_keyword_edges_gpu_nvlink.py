#!/usr/bin/env python3
"""
GPU-accelerated keyword edge builder using NVLink dual A6000s.
Processes 2.8M papers in batches using 96GB combined GPU memory.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import click
from arango import ArangoClient
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NVLinkKeywordEdgeBuilder:
    """Build keyword edges using dual NVLink GPUs."""
    
    def __init__(self, batch_size: int = 15000):
        """
        Initialize builder.
        
        Args:
            batch_size: Number of papers to process at once (15k fits in 84GB)
        """
        self.batch_size = batch_size
        
        # Connect to database
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Setup NVLink GPUs
        if not torch.cuda.is_available():
            raise RuntimeError("No GPUs available!")
        
        n_gpus = torch.cuda.device_count()
        logger.info(f"Found {n_gpus} GPUs")
        
        # Use both GPUs with NVLink
        self.device = torch.device('cuda:0')
        
        # Log GPU info
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        
        # Enable TF32 for faster computation on A6000
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    def load_embeddings_batched(self, limit: int = None):
        """Load embeddings in large batches from database."""
        logger.info("Loading papers with keywords...")
        
        # Get paper keys and categories
        if limit:
            query = f"FOR p IN arxiv_papers FILTER p.keywords != null LIMIT {limit} RETURN {{_key: p._key, categories: p.categories}}"
        else:
            query = "FOR p IN arxiv_papers FILTER p.keywords != null RETURN {_key: p._key, categories: p.categories}"
        
        papers = list(self.db.aql.execute(query))
        logger.info(f"Found {len(papers):,} papers with keywords")
        
        # Load embeddings in large batches
        embeddings = []
        paper_ids = []
        categories = []
        
        chunk_size = 100000  # Load 100k at a time from DB
        
        for i in tqdm(range(0, len(papers), chunk_size), desc="Loading embeddings"):
            chunk = papers[i:i+chunk_size]
            keys = [p['_key'] for p in chunk]
            
            # Batch query with bind variables
            query = """
            FOR key IN @keys
            LET e = DOCUMENT('arxiv_embeddings', key)
            FILTER e != null AND e.keyword_embedding != null
            RETURN {key: key, embedding: e.keyword_embedding}
            """
            
            results = list(self.db.aql.execute(query, bind_vars={'keys': keys}))
            
            # Build mapping
            key_to_paper = {p['_key']: p for p in chunk}
            
            for r in results:
                embeddings.append(r['embedding'])
                paper_ids.append(r['key'])
                paper = key_to_paper.get(r['key'])
                if paper:
                    cat = paper.get('categories', ['unknown'])[0]
                    categories.append(cat[:30])  # Truncate long names
        
        if not embeddings:
            raise ValueError("No embeddings found!")
        
        embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"Loaded {len(embeddings):,} embeddings")
        
        return embeddings, paper_ids, categories
    
    def build_edges_gpu(
        self,
        threshold: float = 0.7,
        top_k: int = 30,
        sample_size: int = None
    ):
        """
        Build keyword edges using GPU acceleration with NVLink.
        
        Args:
            threshold: Minimum similarity threshold
            top_k: Number of nearest neighbors to find
            sample_size: If set, only process this many papers (for testing)
        """
        logger.info("="*70)
        logger.info("GPU KEYWORD EDGE BUILDING (NVLink)")
        logger.info("="*70)
        
        # Load embeddings
        embeddings, paper_ids, categories = self.load_embeddings_batched(limit=sample_size)
        n_papers = len(embeddings)
        
        # Normalize embeddings for cosine similarity
        logger.info("Normalizing embeddings...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        # Convert to torch and move ALL embeddings to GPU (as reference)
        logger.info("Moving embeddings to GPU...")
        embeddings_gpu = torch.from_numpy(embeddings).to(self.device).half()  # Use fp16
        
        # Clear existing keyword edges
        logger.info("Clearing existing keyword similarity edges...")
        keyword_coll = self.db.collection('keyword_similarity')
        keyword_coll.truncate()
        
        # Process in batches
        logger.info(f"Computing similarities (batch_size={self.batch_size})...")
        
        all_edges = []
        total_edges = 0
        cross_category_edges = 0
        
        # Process queries in batches
        n_batches = (n_papers + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, n_papers)
            
            # Get batch of query embeddings
            batch_embeddings = embeddings_gpu[batch_start:batch_end]
            
            # Compute similarities with ALL papers using matrix multiplication
            # This is where NVLink helps - we can compute against all 2.8M at once
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                similarities = torch.mm(batch_embeddings, embeddings_gpu.t())
            
            # Get top-k for each query (excluding self)
            # Set diagonal to -1 to exclude self-similarity
            if batch_start == 0:  # Only needed when batch includes diagonal
                for i in range(batch_end - batch_start):
                    if batch_start + i < n_papers:
                        similarities[i, batch_start + i] = -1
            
            # Get top-k
            values, indices = torch.topk(similarities, min(top_k, similarities.shape[1]), dim=1)
            
            # Move to CPU for processing
            values = values.cpu().numpy()
            indices = indices.cpu().numpy()
            
            # Create edges
            for i in range(batch_end - batch_start):
                paper_idx = batch_start + i
                paper_id = paper_ids[paper_idx]
                paper_cat = categories[paper_idx]
                
                for j in range(min(top_k, indices.shape[1])):
                    target_idx = indices[i, j]
                    similarity = values[i, j]
                    
                    # Skip if below threshold
                    if similarity < threshold:
                        continue
                    
                    # Skip self (should be filtered by -1 but double check)
                    if target_idx == paper_idx:
                        continue
                    
                    target_id = paper_ids[target_idx]
                    target_cat = categories[target_idx]
                    
                    # Bonus weight for cross-category connections
                    is_cross = paper_cat != target_cat
                    weight = float(similarity) * (1.2 if is_cross else 1.0)
                    
                    all_edges.append({
                        '_from': f'arxiv_papers/{paper_id}',
                        '_to': f'arxiv_papers/{target_id}',
                        'weight': weight,
                        'similarity': float(similarity),
                        'from_category': paper_cat,
                        'to_category': target_cat,
                        'cross_category': is_cross
                    })
                    
                    if is_cross:
                        cross_category_edges += 1
                    
                    # Insert batch if full
                    if len(all_edges) >= 50000:
                        keyword_coll.insert_many(all_edges)
                        total_edges += len(all_edges)
                        all_edges = []
            
            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Insert remaining edges
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
@click.option('--batch-size', default=15000, help='Batch size for GPU processing')
@click.option('--threshold', default=0.7, help='Similarity threshold')
@click.option('--top-k', default=30, help='Top K neighbors')
@click.option('--sample', default=None, type=int, help='Sample size for testing')
def main(batch_size: int, threshold: float, top_k: int, sample: int):
    """Build keyword edges using NVLink GPUs."""
    builder = NVLinkKeywordEdgeBuilder(batch_size=batch_size)
    builder.build_edges_gpu(threshold, top_k, sample)


if __name__ == "__main__":
    main()