#!/usr/bin/env python3
"""
Complete graph builder - builds all edge types with optimizations.
Combines the fast methods from the optimized builder with GPU acceleration for keywords.
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
from collections import defaultdict
from multiprocessing import Pool
import click
import pickle
from unix_socket_client import get_unix_socket_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_category_batch(category_item):
    """Process a single category to create edges."""
    category, paper_ids = category_item
    
    if len(paper_ids) < 2:
        return []
    
    edges = []
    
    # For large categories, sample to avoid explosion
    if len(paper_ids) > 1000:
        # Create edges for first 500 papers to all others
        for i in range(min(500, len(paper_ids))):
            # Random sample of connections
            sample_size = min(50, len(paper_ids) - 1)
            indices = np.random.choice(
                [j for j in range(len(paper_ids)) if j != i],
                sample_size,
                replace=False
            )
            
            for j in indices:
                edges.append({
                    '_from': f"arxiv_papers/{paper_ids[i]}",
                    '_to': f"arxiv_papers/{paper_ids[j]}",
                    'weight': 1.0,
                    'category': category
                })
    else:
        # For small categories, connect all pairs
        for i in range(len(paper_ids)):
            for j in range(i + 1, min(i + 50, len(paper_ids))):
                edges.append({
                    '_from': f"arxiv_papers/{paper_ids[i]}",
                    '_to': f"arxiv_papers/{paper_ids[j]}",
                    'weight': 1.0,
                    'category': category
                })
    
    return edges


def process_temporal_batch(batch_data):
    """Process a single month's temporal edges in parallel."""
    month = batch_data['month']
    categories = batch_data['categories']
    next_month_data = batch_data['next_month_data']
    
    edges = []
    
    for category, papers in categories.items():
        if len(papers) < 2:
            continue
        
        # Within-month connections (same category, same month)
        if len(papers) <= 100:
            # Small groups: connect all
            for i in range(len(papers)):
                for j in range(i + 1, len(papers)):
                    edges.append({
                        '_from': f"arxiv_papers/{papers[i]}",
                        '_to': f"arxiv_papers/{papers[j]}",
                        'weight': 1.0,
                        'month': month,
                        'category': category
                    })
        else:
            # Large groups: sample connections
            for i in range(min(200, len(papers))):
                # Sample connections
                sample_size = min(10, len(papers) - 1)
                indices = np.random.choice(
                    [j for j in range(len(papers)) if j != i],
                    sample_size,
                    replace=False
                )
                for j in indices:
                    edges.append({
                        '_from': f"arxiv_papers/{papers[i]}",
                        '_to': f"arxiv_papers/{papers[j]}",
                        'weight': 0.9,
                        'month': month,
                        'category': category
                    })
        
        # Cross-month connections (to next month in same category)
        if next_month_data and category in next_month_data:
            next_papers = next_month_data[category]
            
            # Sample cross-month connections
            for i in range(min(50, len(papers))):
                for j in range(min(5, len(next_papers))):
                    edges.append({
                        '_from': f"arxiv_papers/{papers[i]}",
                        '_to': f"arxiv_papers/{next_papers[j]}",
                        'weight': 0.7,
                        'cross_month': True,
                        'category': category
                    })
    
    return edges


class CompleteGraphBuilder:
    """Build complete graph with all edge types."""
    
    def __init__(self, workers: int = 48):
        # Check if Unix socket is disabled
        if os.environ.get('DISABLE_UNIX_SOCKET'):
            logger.info("Unix socket disabled, using HTTP connection")
            from arango import ArangoClient
            self.client = ArangoClient(hosts='http://localhost:8529')
        else:
            # Use Unix socket for best performance (requires user in arangodb group)
            self.client = get_unix_socket_client(
                socket_path='/tmp/arangodb.sock',
                fallback_to_tcp=True  # Fall back to TCP if permissions issue
            )
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        self.workers = workers
        
        # Ensure edge collections exist
        edge_collections = ['same_field', 'temporal_proximity', 'keyword_similarity']
        for coll_name in edge_collections:
            if coll_name not in [c['name'] for c in self.db.collections()]:
                self.db.create_collection(coll_name, edge=True)
        
        self.edge_collections = {
            name: self.db.collection(name) for name in edge_collections
        }
        
        # Check GPU availability for keyword phase
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.device = torch.device('cuda:0')
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("GPU not available, will use CPU for keyword similarity")
    
    def build_category_edges(self):
        """Build same-field edges based on categories."""
        logger.info("="*70)
        logger.info("PHASE 1: CATEGORY EDGES")
        logger.info("="*70)
        
        self.edge_collections['same_field'].truncate()
        start_time = time.time()
        
        # Index papers by category
        papers_by_category = defaultdict(list)
        query = "FOR paper IN arxiv_papers RETURN paper"
        papers = self.db.aql.execute(query)
        
        for paper in tqdm(papers, desc="Indexing by category", total=2823744):
            if 'categories' in paper and paper['categories']:
                primary_cat = paper['categories'][0] if isinstance(paper['categories'], list) else paper['categories']
                papers_by_category[primary_cat].append(paper['_key'])
        
        logger.info(f"Found {len(papers_by_category)} categories")
        
        # Process categories in parallel
        edge_count = 0
        edges_buffer = []
        
        with Pool(self.workers) as pool:
            category_items = list(papers_by_category.items())
            
            for edges in tqdm(pool.imap_unordered(
                process_category_batch,
                category_items
            ), total=len(category_items), desc="Processing categories"):
                if edges:
                    edges_buffer.extend(edges)
                    edge_count += len(edges)
                    
                    # Batch insert with larger buffer
                    if len(edges_buffer) >= 100000:  # Increased from 10k
                        self.edge_collections['same_field'].insert_many(edges_buffer, overwrite=True, silent=True)
                        edges_buffer = []
        
        # Insert remaining edges
        if edges_buffer:
            self.edge_collections['same_field'].insert_many(edges_buffer, overwrite=True)
        
        elapsed = time.time() - start_time
        logger.info(f"Created {edge_count:,} category edges in {elapsed/60:.1f} minutes")
    
    def build_temporal_edges(self):
        """Build temporal proximity edges using ArXiv ID optimization."""
        logger.info("="*70)
        logger.info("PHASE 2: TEMPORAL EDGES (ArXiv ID Optimized)")
        logger.info("="*70)
        
        self.edge_collections['temporal_proximity'].truncate()
        start_time = time.time()
        
        # Load all papers with ArXiv IDs
        logger.info("Loading papers with ArXiv IDs...")
        query = "FOR paper IN arxiv_papers FILTER paper.arxiv_id != null RETURN paper"
        papers = list(self.db.aql.execute(query))
        logger.info(f"Found {len(papers):,} papers with ArXiv IDs")
        
        # Group by month (YYMM) and category
        papers_by_month_category = defaultdict(lambda: defaultdict(list))
        
        for paper in tqdm(papers, desc="Grouping by month and category"):
            if 'arxiv_id' in paper and 'categories' in paper and paper['categories']:
                arxiv_id = paper['arxiv_id']
                
                # Extract month key from ArXiv ID
                month_key = None
                if '/' in arxiv_id:
                    # Old format: category/YYMMNNN
                    parts = arxiv_id.split('/')
                    if len(parts) == 2 and len(parts[1]) >= 4:
                        month_key = parts[1][:4]
                elif '.' in arxiv_id and len(arxiv_id) >= 4:
                    # New format: YYMM.NNNNN
                    month_key = arxiv_id[:4]
                
                if month_key and month_key.isdigit():
                    primary_cat = paper['categories'][0]
                    papers_by_month_category[month_key][primary_cat].append(paper['_key'])
        
        logger.info(f"Found {len(papers_by_month_category)} unique months")
        
        # Process months in batches with parallel processing
        months = sorted(papers_by_month_category.keys())
        edge_count = 0
        
        logger.info(f"Processing {len(months)} months with {self.workers} workers...")
        
        # Process in chunks to avoid memory issues
        chunk_size = 100  # Increased from 50 - we have plenty of resources
        for chunk_start in range(0, len(months), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(months))
            chunk_months = months[chunk_start:chunk_end]
            
            logger.info(f"Processing months {chunk_start+1}-{chunk_end}...")
            
            # Prepare batch data
            batch_data_list = []
            for i, month in enumerate(chunk_months):
                next_month_data = None
                if chunk_start + i + 1 < len(months):
                    next_month = months[chunk_start + i + 1]
                    next_month_data = papers_by_month_category[next_month]
                
                batch_data_list.append({
                    'month': month,
                    'categories': papers_by_month_category[month],
                    'next_month_data': next_month_data
                })
            
            # Process in parallel
            edges_buffer = []
            with Pool(self.workers) as pool:
                for edges in tqdm(
                    pool.imap_unordered(process_temporal_batch, batch_data_list),
                    total=len(batch_data_list),
                    desc=f"Processing months {chunk_start+1}-{chunk_end}"
                ):
                    edges_buffer.extend(edges)
            
            # Batch insert with larger batches for better performance
            if edges_buffer:
                logger.info(f"Inserting edges for months {chunk_start+1}-{chunk_end}...")
                # Increase batch size to 500k for fewer round trips
                for i in range(0, len(edges_buffer), 500000):
                    batch = edges_buffer[i:i+500000]
                    self.edge_collections['temporal_proximity'].insert_many(
                        batch, overwrite=True, silent=True  # Silent mode for faster writes
                    )
                edge_count += len(edges_buffer)
                logger.info(f"Inserted {edge_count:,} edges so far...")
        
        elapsed = time.time() - start_time
        logger.info(f"Created {edge_count:,} temporal edges in {elapsed/60:.1f} minutes")
    
    def build_keyword_edges_gpu(self, threshold: float = 0.65, top_k: int = 100):
        """Build keyword similarity edges using GPU acceleration."""
        logger.info("="*70)
        logger.info("PHASE 3: KEYWORD EDGES (GPU Optimized)")
        logger.info("="*70)
        
        if not self.use_gpu:
            logger.warning("GPU not available, skipping keyword edges")
            return
        
        self.edge_collections['keyword_similarity'].truncate()
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
        
        # Process in large batches for better GPU utilization
        batch_size = 10000
        edge_count = 0
        interdisciplinary_count = 0
        edges_buffer = []
        
        logger.info(f"Finding top {top_k} similar papers for each paper...")
        
        for batch_start in tqdm(range(0, len(embeddings_gpu), batch_size), 
                                desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(embeddings_gpu))
            batch = embeddings_gpu[batch_start:batch_end]
            
            # Compute similarities
            with torch.cuda.amp.autocast():
                similarities = torch.mm(batch, embeddings_gpu.t())
            
            # Find top-k similar papers
            top_values, top_indices = torch.topk(similarities, 
                                                 min(top_k + 1, similarities.shape[1]), 
                                                 dim=1)
            
            # Move results to CPU
            top_values = top_values.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            # Process results
            for i in range(batch.shape[0]):
                paper_idx = batch_start + i
                paper_id = paper_ids[paper_idx]
                paper_cat = paper_categories[paper_idx]
                
                for j in range(1, len(top_indices[i])):
                    neighbor_idx = top_indices[i][j]
                    similarity = top_values[i][j]
                    
                    if similarity < threshold or neighbor_idx == paper_idx:
                        continue
                    
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
                    
                    if paper_cat != neighbor_cat:
                        interdisciplinary_count += 1
                    
                    edges_buffer.append(edge)
                    edge_count += 1
            
            # Batch insert with larger buffer for fewer DB round trips
            if len(edges_buffer) >= 200000:  # Increased from 50k
                self.edge_collections['keyword_similarity'].insert_many(edges_buffer, overwrite=True, silent=True)
                edges_buffer = []
            
            # Clear GPU cache
            if batch_start % 50000 == 0:
                torch.cuda.empty_cache()
        
        # Insert remaining edges
        if edges_buffer:
            self.edge_collections['keyword_similarity'].insert_many(edges_buffer, overwrite=True)
        
        elapsed = time.time() - start_time
        logger.info(f"Created {edge_count:,} keyword edges in {elapsed/60:.1f} minutes")
        logger.info(f"Found {interdisciplinary_count:,} interdisciplinary connections")
    
    def print_statistics(self):
        """Print final graph statistics."""
        logger.info("="*70)
        logger.info("FINAL GRAPH STATISTICS")
        logger.info("="*70)
        
        total_edges = 0
        for coll_name, coll in self.edge_collections.items():
            count = coll.count()
            total_edges += count
            logger.info(f"{coll_name:20s}: {count:>15,} edges")
        
        logger.info("-"*70)
        logger.info(f"{'TOTAL':20s}: {total_edges:>15,} edges")
        
        # Check papers
        paper_count = self.db.collection('arxiv_papers').count()
        logger.info(f"{'Papers':20s}: {paper_count:>15,}")


@click.command()
@click.option('--workers', default=48, help='Number of parallel workers')
@click.option('--skip-categories', is_flag=True, help='Skip category edges')
@click.option('--skip-temporal', is_flag=True, help='Skip temporal edges')
@click.option('--skip-keywords', is_flag=True, help='Skip keyword edges')
@click.option('--threshold', default=0.65, help='Keyword similarity threshold')
@click.option('--top-k', default=100, help='Number of nearest neighbors for keywords')
def main(workers, skip_categories, skip_temporal, skip_keywords, threshold, top_k):
    """Build complete graph with all edge types."""
    
    logger.info("="*70)
    logger.info("COMPLETE GRAPH BUILD")
    logger.info(f"Workers: {workers}")
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    builder = CompleteGraphBuilder(workers=workers)
    
    overall_start = time.time()
    
    if not skip_categories:
        builder.build_category_edges()
    
    if not skip_temporal:
        builder.build_temporal_edges()
    
    if not skip_keywords:
        builder.build_keyword_edges_gpu(threshold=threshold, top_k=top_k)
    
    overall_elapsed = time.time() - overall_start
    
    logger.info("="*70)
    logger.info(f"COMPLETE BUILD FINISHED")
    logger.info(f"Total time: {overall_elapsed/60:.1f} minutes")
    logger.info("="*70)
    
    builder.print_statistics()


if __name__ == "__main__":
    main()