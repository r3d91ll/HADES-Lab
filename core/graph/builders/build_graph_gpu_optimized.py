#!/usr/bin/env python3
"""
GPU-optimized graph builder using PyTorch for massive speedup.
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from arango import ArangoClient
from pathlib import Path
import pickle
import click

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPUGraphBuilder:
    """Build keyword similarity edges using GPU acceleration."""
    
    def __init__(self):
        # Use Unix socket for better performance
        try:
            self.client = ArangoClient(hosts='unix:///tmp/arangodb.sock')
        except:
            # Fall back to TCP if socket not available
            self.client = ArangoClient(hosts='http://localhost:8529')
            
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available! This script requires CUDA.")
        
        self.device = torch.device('cuda:0')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def build_keyword_similarity_edges(self, threshold: float = 0.65, top_k: int = 100):
        """Build keyword similarity edges using efficient GPU processing.
        
        Args:
            threshold: Minimum similarity threshold
            top_k: Number of nearest neighbors to find per paper
        """
        logger.info("="*70)
        logger.info("GPU-OPTIMIZED KEYWORD SIMILARITY EDGES")
        logger.info("="*70)
        
        # Ensure collection exists
        if 'keyword_similarity' not in [c['name'] for c in self.db.collections()]:
            self.db.create_collection('keyword_similarity', edge=True)
        
        keyword_sim = self.db.collection('keyword_similarity')
        keyword_sim.truncate()
        
        start_time = time.time()
        
        # Load papers with keywords
        logger.info("Loading papers with keywords...")
        query = "FOR paper IN arxiv_papers FILTER paper.keywords != null RETURN paper"
        papers = list(self.db.aql.execute(query))
        logger.info(f"Found {len(papers):,} papers with keywords")
        
        # Load embeddings
        logger.info("Loading keyword embeddings...")
        embeddings = []
        paper_ids = []
        paper_categories = []
        
        for paper in tqdm(papers, desc="Loading embeddings"):
            embed_doc = self.db.collection('arxiv_embeddings').get(paper['_key'])
            if embed_doc and 'keyword_embedding' in embed_doc:
                embeddings.append(embed_doc['keyword_embedding'])
                paper_ids.append(paper['_key'])
                cat = paper['categories'][0] if 'categories' in paper and paper['categories'] else 'unknown'
                paper_categories.append(cat)
        
        if not embeddings:
            logger.warning("No keyword embeddings found")
            return
        
        embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"Loaded {len(embeddings):,} keyword embeddings")
        
        # Move to GPU and normalize
        logger.info("Moving embeddings to GPU and normalizing...")
        embeddings_gpu = torch.from_numpy(embeddings).to(self.device)
        embeddings_gpu = F.normalize(embeddings_gpu, p=2, dim=1)
        
        # Process in larger batches for better GPU utilization
        batch_size = 10000  # Much larger batch size
        edge_count = 0
        interdisciplinary_count = 0
        edges_buffer = []
        
        logger.info(f"Finding top {top_k} similar papers for each paper...")
        logger.info(f"Processing in batches of {batch_size:,}")
        
        for batch_start in tqdm(range(0, len(embeddings_gpu), batch_size), 
                                desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(embeddings_gpu))
            batch = embeddings_gpu[batch_start:batch_end]
            
            # Compute similarities for this batch with ALL embeddings
            # This is the key operation that should use GPU heavily
            with torch.cuda.amp.autocast():  # Use mixed precision for speed
                similarities = torch.mm(batch, embeddings_gpu.t())
            
            # Find top-k similar papers for each paper in batch
            # This also runs on GPU
            top_values, top_indices = torch.topk(similarities, 
                                                 min(top_k + 1, similarities.shape[1]), 
                                                 dim=1)
            
            # Move results to CPU for processing
            top_values = top_values.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            # Process results on CPU
            for i in range(batch.shape[0]):
                paper_idx = batch_start + i
                paper_id = paper_ids[paper_idx]
                paper_cat = paper_categories[paper_idx]
                
                # Skip self (usually the first match)
                for j in range(1, len(top_indices[i])):
                    neighbor_idx = top_indices[i][j]
                    similarity = top_values[i][j]
                    
                    # Skip if below threshold or self-connection
                    if similarity < threshold or neighbor_idx == paper_idx:
                        continue
                    
                    # Avoid duplicate edges (only create edge from lower to higher index)
                    if neighbor_idx < paper_idx:
                        continue
                    
                    neighbor_id = paper_ids[neighbor_idx]
                    neighbor_cat = paper_categories[neighbor_idx]
                    
                    edge = {
                        '_from': f"arxiv_papers/{paper_id}",
                        '_to': f"arxiv_papers/{neighbor_id}",
                        'weight': float(similarity),
                        'similarity': float(similarity)
                    }
                    
                    # Track interdisciplinary connections
                    if paper_cat != neighbor_cat:
                        interdisciplinary_count += 1
                    
                    edges_buffer.append(edge)
                    edge_count += 1
            
            # Batch insert to database
            if len(edges_buffer) >= 50000:
                keyword_sim.insert_many(edges_buffer, overwrite=True)
                edges_buffer = []
                logger.info(f"  Inserted {edge_count:,} edges so far...")
            
            # Clear GPU cache periodically
            if batch_start % 50000 == 0:
                torch.cuda.empty_cache()
        
        # Insert remaining edges
        if edges_buffer:
            keyword_sim.insert_many(edges_buffer, overwrite=True)
        
        elapsed = time.time() - start_time
        logger.info("="*70)
        logger.info(f"COMPLETED: {edge_count:,} keyword edges in {elapsed/60:.1f} minutes")
        logger.info(f"Found {interdisciplinary_count:,} interdisciplinary connections")
        logger.info(f"Processing rate: {len(embeddings)/elapsed:.0f} papers/second")
        logger.info(f"Edge creation rate: {edge_count/elapsed:.0f} edges/second")
        
        # Estimate speedup vs CPU
        cpu_time_estimate = 32 * len(embeddings) / 1000  # ~32 seconds per 1000 papers on CPU
        logger.info(f"Estimated speedup vs CPU: {cpu_time_estimate/elapsed:.0f}x")
    
    def verify_gpu_usage(self):
        """Verify GPU is being used effectively."""
        logger.info("\nGPU Status:")
        logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")


@click.command()
@click.option('--threshold', default=0.65, help='Similarity threshold')
@click.option('--top-k', default=100, help='Number of nearest neighbors per paper')
def main(threshold, top_k):
    """Build keyword similarity edges using GPU acceleration."""
    
    builder = GPUGraphBuilder()
    builder.build_keyword_similarity_edges(threshold=threshold, top_k=top_k)
    builder.verify_gpu_usage()
    
    logger.info("\nGraph statistics:")
    try:
        client = ArangoClient(hosts='unix:///tmp/arangodb.sock')
    except:
        client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('academy_store', username='root', 
                   password=os.environ.get('ARANGO_PASSWORD'))
    
    for coll_name in ['same_field', 'temporal_proximity', 'keyword_similarity']:
        if coll_name in [c['name'] for c in db.collections()]:
            count = db.collection(coll_name).count()
            logger.info(f"  {coll_name}: {count:,} edges")


if __name__ == "__main__":
    main()