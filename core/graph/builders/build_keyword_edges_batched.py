#!/usr/bin/env python3
"""
Batched keyword edge builder to avoid GPU OOM with 2.8M papers.

Key strategy: Process similarity in chunks, only keeping top-k edges.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple
import click
from arango import ArangoClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchedKeywordEdgeBuilder:
    """Build keyword similarity edges with batched GPU processing."""
    
    def __init__(self, batch_size: int = 1000, db_batch_size: int = 10000):
        """
        Initialize builder.
        
        Args:
            batch_size: Number of papers to process at once on GPU
            db_batch_size: Number of edges to insert at once
        """
        self.batch_size = batch_size
        self.db_batch_size = db_batch_size
        
        # Connect to database
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Check GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.info("Using CPU (will be slower)")
    
    def load_embeddings(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Load keyword embeddings from database."""
        logger.info("Loading papers with keywords...")
        
        # Get papers with keywords
        query = "FOR p IN arxiv_papers FILTER p.keywords != null RETURN p"
        papers = list(self.db.aql.execute(query))
        logger.info(f"Found {len(papers):,} papers with keywords")
        
        # Load embeddings
        logger.info("Loading keyword embeddings...")
        embeddings = []
        paper_ids = []
        paper_categories = []
        
        embed_coll = self.db.collection('arxiv_embeddings')
        
        for paper in tqdm(papers[:100000], desc="Loading embeddings"):  # Limit for testing
            try:
                embed_doc = embed_coll.get(paper['_key'])
                if embed_doc and 'keyword_embedding' in embed_doc:
                    embeddings.append(embed_doc['keyword_embedding'])
                    paper_ids.append(paper['_key'])
                    cat = paper.get('categories', ['unknown'])[0]
                    paper_categories.append(cat)
            except:
                continue
        
        if not embeddings:
            raise ValueError("No keyword embeddings found!")
        
        embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"Loaded {len(embeddings):,} keyword embeddings")
        
        return embeddings, paper_ids, paper_categories
    
    def build_edges_batched(
        self,
        threshold: float = 0.65,
        top_k: int = 100
    ):
        """
        Build keyword similarity edges with batched processing.
        
        Strategy:
        1. Load all embeddings into memory (but not GPU)
        2. Process in batches on GPU
        3. For each batch, find top-k most similar papers
        4. Insert edges in database batches
        """
        # Load embeddings
        embeddings, paper_ids, categories = self.load_embeddings()
        n_papers = len(embeddings)
        
        # Normalize embeddings for cosine similarity
        logger.info("Normalizing embeddings...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        # Clear existing keyword edges
        logger.info("Clearing existing keyword similarity edges...")
        keyword_coll = self.db.collection('keyword_similarity')
        keyword_coll.truncate()
        
        # Process in batches
        logger.info(f"Building edges (batch_size={self.batch_size}, top_k={top_k})...")
        
        all_edges = []
        total_edges = 0
        
        # Move reference embeddings to GPU once
        if self.device.type == 'cuda':
            # Use smaller chunks to fit in GPU memory
            # With 48GB GPU, we can hold about 500k embeddings
            max_ref_size = min(500000, n_papers)
            embeddings_gpu = torch.from_numpy(embeddings[:max_ref_size]).to(self.device)
            logger.info(f"Loaded {max_ref_size:,} reference embeddings to GPU")
        else:
            embeddings_gpu = torch.from_numpy(embeddings)
        
        # Process queries in batches
        for batch_start in tqdm(range(0, n_papers, self.batch_size), desc="Processing batches"):
            batch_end = min(batch_start + self.batch_size, n_papers)
            batch_embeddings = embeddings[batch_start:batch_end]
            
            # Convert to torch and move to device
            batch_tensor = torch.from_numpy(batch_embeddings).to(self.device)
            
            # Compute similarities
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    # If we have more papers than fit on GPU, process in chunks
                    if n_papers > len(embeddings_gpu):
                        # Process against GPU chunk and CPU chunks
                        similarities = torch.mm(batch_tensor, embeddings_gpu.t())
                        
                        # Get top-k from GPU chunk
                        values, indices = torch.topk(similarities, min(top_k, similarities.shape[1]), dim=1)
                        
                        # TODO: Add processing for remaining embeddings on CPU
                        # For now, just use the GPU subset
                    else:
                        similarities = torch.mm(batch_tensor, embeddings_gpu.t())
                        values, indices = torch.topk(similarities, min(top_k, similarities.shape[1]), dim=1)
            else:
                similarities = torch.mm(batch_tensor, embeddings_gpu.t())
                values, indices = torch.topk(similarities, min(top_k, similarities.shape[1]), dim=1)
            
            # Convert to CPU for processing
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
                    
                    # Skip self-connections and below threshold
                    if target_idx == paper_idx or similarity < threshold:
                        continue
                    
                    target_id = paper_ids[target_idx]
                    target_cat = categories[target_idx]
                    
                    # Bonus weight for cross-category connections
                    weight = float(similarity)
                    if paper_cat != target_cat:
                        weight *= 1.2  # Boost interdisciplinary connections
                    
                    all_edges.append({
                        '_from': f'arxiv_papers/{paper_id}',
                        '_to': f'arxiv_papers/{target_id}',
                        'weight': weight,
                        'similarity': float(similarity),
                        'from_category': paper_cat,
                        'to_category': target_cat,
                        'cross_category': paper_cat != target_cat
                    })
                    
                    # Insert batch if full
                    if len(all_edges) >= self.db_batch_size:
                        keyword_coll.insert_many(all_edges)
                        total_edges += len(all_edges)
                        all_edges = []
            
            # Clear GPU cache periodically
            if self.device.type == 'cuda' and batch_start % (self.batch_size * 10) == 0:
                torch.cuda.empty_cache()
        
        # Insert remaining edges
        if all_edges:
            keyword_coll.insert_many(all_edges)
            total_edges += len(all_edges)
        
        logger.info(f"Created {total_edges:,} keyword similarity edges")
        
        # Report statistics
        self.report_stats()
    
    def report_stats(self):
        """Report final statistics."""
        logger.info("\n" + "="*70)
        logger.info("KEYWORD EDGE STATISTICS")
        logger.info("="*70)
        
        # Count edges
        keyword_coll = self.db.collection('keyword_similarity')
        total = keyword_coll.count()
        
        # Count cross-category edges
        query = """
        FOR e IN keyword_similarity
        FILTER e.cross_category == true
        COLLECT WITH COUNT INTO cross_count
        RETURN cross_count
        """
        cross_count = list(self.db.aql.execute(query))[0] if total > 0 else 0
        
        logger.info(f"Total keyword edges: {total:,}")
        logger.info(f"Cross-category edges: {cross_count:,} ({cross_count/total*100:.1f}%)" if total > 0 else "")
        
        # Sample interdisciplinary connections
        if cross_count > 0:
            logger.info("\nSample interdisciplinary connections:")
            query = """
            FOR e IN keyword_similarity
            FILTER e.cross_category == true
            SORT e.weight DESC
            LIMIT 10
            RETURN {
                from_cat: e.from_category,
                to_cat: e.to_category,
                weight: e.weight,
                similarity: e.similarity
            }
            """
            samples = list(self.db.aql.execute(query))
            for s in samples[:5]:
                logger.info(f"  {s['from_cat'][:20]:20s} -> {s['to_cat'][:20]:20s} (sim: {s['similarity']:.3f})")


@click.command()
@click.option('--batch-size', default=1000, help='GPU batch size')
@click.option('--threshold', default=0.65, help='Similarity threshold')
@click.option('--top-k', default=100, help='Top K similar papers per paper')
def main(batch_size: int, threshold: float, top_k: int):
    """Build keyword similarity edges with batched processing."""
    builder = BatchedKeywordEdgeBuilder(batch_size=batch_size)
    builder.build_edges_batched(threshold=threshold, top_k=top_k)


if __name__ == "__main__":
    main()