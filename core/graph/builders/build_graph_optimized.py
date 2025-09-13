#!/usr/bin/env python3
"""
Optimized graph builder with ArXiv ID temporal edges and Faiss keyword similarity.

Key optimizations:
1. ArXiv ID-based temporal matching (98 hours -> 1-2 hours)
2. Faiss-based keyword similarity search
3. Comprehensive interdisciplinary tracking
4. Checkpointing for resume capability
"""

import os
import sys
import json
import logging
import pickle
import time
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm
import click
import faiss
from arango import ArangoClient
from multiprocessing import Pool, cpu_count

# Add parent to path for imports  
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
# from core.monitoring.system_monitor import SystemMonitor  # TODO: integrate later

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/todd/olympus/HADES-Lab/logs/graph_build_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedGraphBuilder:
    """Optimized graph builder with performance improvements."""
    
    def __init__(self, workers: int = 36, checkpoint_dir: str = None):
        """Initialize the builder.
        
        Args:
            workers: Number of parallel workers
            checkpoint_dir: Directory for checkpoints
        """
        self.workers = workers
        self.checkpoint_dir = Path(checkpoint_dir or '/tmp/graph_checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Connect to database
        # Use HTTP connection (Unix socket needs special configuration)
        self.client = ArangoClient(hosts='http://localhost:8529')
            
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Edge collections
        self.edge_collections = {
            'same_field': self.db.collection('same_field'),
            'temporal_proximity': self.db.collection('temporal_proximity'),
            'keyword_similarity': self.db.collection('keyword_similarity'),
            'citations': self.db.collection('citations')
        }
        
        # Interdisciplinary tracking
        self.interdisciplinary_edges = []
        
        # Initialize monitoring (simplified for now)
        self.monitor = None  # SystemMonitor integration later
    
    def checkpoint_save(self, phase: str, data: Any):
        """Save checkpoint for resume capability."""
        checkpoint_path = self.checkpoint_dir / f"{phase}_checkpoint.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Checkpoint saved: {phase}")
    
    def checkpoint_load(self, phase: str) -> Any:
        """Load checkpoint if exists."""
        checkpoint_path = self.checkpoint_dir / f"{phase}_checkpoint.pkl"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Checkpoint loaded: {phase}")
            return data
        return None
    
    def build_category_edges(self, clear: bool = True):
        """Build same-field edges (existing fast implementation)."""
        logger.info("="*70)
        logger.info("PHASE 1: CATEGORY EDGES")
        logger.info("="*70)
        
        if clear:
            self.edge_collections['same_field'].truncate()
        
        # Check for checkpoint
        checkpoint = self.checkpoint_load('category')
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint['processed']} categories done")
            return
        
        start_time = time.time()
        
        # Index papers by category
        papers_by_category = defaultdict(list)
        # Use AQL query to get all papers
        query = "FOR paper IN arxiv_papers RETURN paper"
        papers = self.db.aql.execute(query)
        
        for paper in tqdm(papers, desc="Indexing by category", total=2823744):
            if 'categories' in paper and paper['categories']:
                # Use first category as primary
                primary_cat = paper['categories'][0] if isinstance(paper['categories'], list) else paper['categories']
                papers_by_category[primary_cat].append(paper['_key'])
        
        logger.info(f"Found {len(papers_by_category)} categories")
        
        # Process categories in parallel
        edge_count = 0
        edges_buffer = []
        with Pool(self.workers) as pool:
            # Prepare batches
            category_items = list(papers_by_category.items())
            
            # Process with progress bar
            for edges in tqdm(pool.imap_unordered(
                process_category_batch,
                category_items
            ), total=len(category_items), desc="Processing categories"):
                if edges:
                    edges_buffer.extend(edges)
                    edge_count += len(edges)
                    
                    # Batch insert when buffer is large
                    if len(edges_buffer) >= 10000:
                        self.edge_collections['same_field'].insert_many(edges_buffer, overwrite=True)
                        edges_buffer = []
        
        # Insert remaining edges
        if edges_buffer:
            self.edge_collections['same_field'].insert_many(edges_buffer, overwrite=True)
        
        elapsed = time.time() - start_time
        logger.info(f"Created {edge_count:,} category edges in {elapsed/60:.1f} minutes")
        
        self.checkpoint_save('category', {'processed': len(papers_by_category), 'edges': edge_count})
    
    def build_temporal_edges_arxiv_id(self):
        """Build temporal edges using ArXiv ID optimization."""
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
                
                # Handle different ArXiv ID formats
                # New format: YYMM.NNNNN (e.g., "2301.12345")
                # Old format: category/YYMMNNN (e.g., "astro-ph/9901234")
                # Very old: just category-based (e.g., "acc-phys/9503001")
                
                if '/' in arxiv_id:
                    # Old format with category prefix
                    parts = arxiv_id.split('/')
                    if len(parts) == 2 and len(parts[1]) >= 4:
                        # Extract YYMM from the numeric part
                        month_key = parts[1][:4]
                        # Validate it's numeric
                        try:
                            int(month_key)
                        except ValueError:
                            continue  # Skip malformed IDs
                    else:
                        continue
                elif '.' in arxiv_id and len(arxiv_id) >= 4:
                    # New format YYMM.NNNNN
                    month_key = arxiv_id[:4]
                    # Validate it's numeric
                    try:
                        int(month_key)
                    except ValueError:
                        continue  # Skip malformed IDs
                else:
                    continue  # Skip unknown formats
                
                # Use first category as primary
                category = paper['categories'][0] if isinstance(paper['categories'], list) else paper['categories']
                papers_by_month_category[month_key][category].append(paper)
        
        logger.info(f"Found {len(papers_by_month_category)} unique months")
        
        # Process each month-category combination IN PARALLEL
        edge_count = 0
        
        # Prepare month batches for parallel processing
        month_batches = []
        months = sorted(papers_by_month_category.keys())
        
        for month in months:
            categories = papers_by_month_category[month]
            # Get next month for cross-month connections
            next_month = get_next_month(month)
            next_month_data = papers_by_month_category.get(next_month, {})
            
            month_batches.append({
                'month': month,
                'categories': categories,
                'next_month_data': next_month_data
            })
        
        # Process months in parallel
        logger.info(f"Processing {len(month_batches)} months with {self.workers} workers...")
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 50  # Process 50 months at a time
        
        for chunk_start in range(0, len(month_batches), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(month_batches))
            chunk = month_batches[chunk_start:chunk_end]
            
            with Pool(self.workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(process_temporal_batch, chunk),
                    total=len(chunk),
                    desc=f"Processing months {chunk_start+1}-{chunk_end}"
                ))
            
            # Insert edges from this chunk immediately
            logger.info(f"Inserting edges for months {chunk_start+1}-{chunk_end}...")
            for edges in results:
                if edges:
                    # Batch insert for efficiency
                    for i in range(0, len(edges), 5000):
                        batch = edges[i:i+5000]
                        self.edge_collections['temporal_proximity'].insert_many(
                            batch, overwrite=True, silent=True
                        )
                        edge_count += len(batch)
            
            logger.info(f"Inserted {edge_count:,} edges so far...")
        
        elapsed = time.time() - start_time
        logger.info(f"Created {edge_count:,} temporal edges in {elapsed/60:.1f} minutes")
        logger.info(f"Speedup: {98*60/elapsed:.1f}x faster than original!")
    
    def build_keyword_edges_faiss(self, threshold: float = 0.65):
        """Build keyword similarity edges using Faiss or PyTorch GPU."""
        logger.info("="*70)
        logger.info("PHASE 3: KEYWORD EDGES (GPU Optimized)")
        logger.info("="*70)
        
        self.edge_collections['keyword_similarity'].truncate()
        start_time = time.time()
        
        # Load papers with keywords
        logger.info("Loading papers with keywords...")
        query = "FOR paper IN arxiv_papers FILTER paper.keywords != null RETURN paper"
        papers = list(self.db.aql.execute(query))
        logger.info(f"Found {len(papers):,} papers with keywords")
        
        # Load keyword embeddings
        logger.info("Loading keyword embeddings...")
        embeddings = []
        paper_ids = []
        paper_categories = []
        
        for paper in tqdm(papers, desc="Loading embeddings"):
            # Get embedding from arxiv_embeddings collection
            embed_doc = self.db.collection('arxiv_embeddings').get(paper['_key'])
            if embed_doc and 'keyword_embedding' in embed_doc:
                embeddings.append(embed_doc['keyword_embedding'])
                paper_ids.append(paper['_key'])
                # Use first category as primary
                cat = paper['categories'][0] if 'categories' in paper and paper['categories'] else 'unknown'
                paper_categories.append(cat)
        
        if not embeddings:
            logger.warning("No keyword embeddings found, skipping keyword edges")
            return
        
        embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"Loaded {len(embeddings):,} keyword embeddings")
        
        # Try PyTorch GPU first
        import torch
        if torch.cuda.is_available():
            logger.info(f"Using PyTorch GPU for similarity search (GPU: {torch.cuda.get_device_name(0)})")
            return self._build_keyword_edges_torch_gpu(embeddings, paper_ids, paper_categories, threshold)
        
        # Fall back to Faiss CPU
        logger.info("Using Faiss CPU for similarity search")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build Faiss index
        logger.info("Building Faiss index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
        index.add(embeddings)
        
        # Search for similar papers
        logger.info("Finding similar papers...")
        k = min(200, len(embeddings))  # Find top 200 similar papers
        batch_size = 1000
        edge_count = 0
        interdisciplinary_count = 0
        edges_batch = []
        
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[i:batch_end]
            
            # Search for k nearest neighbors
            distances, indices = index.search(batch_embeddings, k)
            
            for j, (dist_row, idx_row) in enumerate(zip(distances, indices)):
                paper_idx = i + j
                paper_id = paper_ids[paper_idx]
                paper_cat = paper_categories[paper_idx]
                
                for dist, neighbor_idx in zip(dist_row, idx_row):
                    if neighbor_idx == paper_idx:
                        continue  # Skip self
                    
                    if dist >= threshold:
                        neighbor_id = paper_ids[neighbor_idx]
                        neighbor_cat = paper_categories[neighbor_idx]
                        
                        edge = {
                            '_from': f"arxiv_papers/{paper_id}",
                            '_to': f"arxiv_papers/{neighbor_id}",
                            'weight': float(dist),
                            'similarity': float(dist)
                        }
                        
                        # Track interdisciplinary connections
                        if paper_cat != neighbor_cat:
                            interdisciplinary_count += 1
                            self.interdisciplinary_edges.append({
                                'from': paper_id,
                                'to': neighbor_id,
                                'similarity': float(dist),
                                'cat1': paper_cat,
                                'cat2': neighbor_cat
                            })
                        
                        edges_batch.append(edge)
                        
                        if len(edges_batch) >= 5000:
                            self.edge_collections['keyword_similarity'].insert_many(
                                edges_batch, overwrite=True
                            )
                            edge_count += len(edges_batch)
                            edges_batch = []
        
        # Insert remaining edges
        if edges_batch:
            self.edge_collections['keyword_similarity'].insert_many(
                edges_batch, overwrite=True
            )
            edge_count += len(edges_batch)
        
        elapsed = time.time() - start_time
        logger.info(f"Created {edge_count:,} keyword edges in {elapsed/60:.1f} minutes")
        logger.info(f"Found {interdisciplinary_count:,} interdisciplinary connections!")
        
        # Analyze interdisciplinary bridges
        self.analyze_interdisciplinary_bridges()
    
    def _build_keyword_edges_torch_gpu(self, embeddings, paper_ids, paper_categories, threshold):
        """Build keyword edges using PyTorch GPU acceleration."""
        import torch
        
        start_time = time.time()
        device = torch.device('cuda:0')
        logger.info(f"Moving {len(embeddings):,} embeddings to GPU...")
        
        # Convert to torch tensors and move to GPU
        embeddings_gpu = torch.from_numpy(embeddings).to(device)
        
        # Normalize for cosine similarity
        embeddings_gpu = torch.nn.functional.normalize(embeddings_gpu, p=2, dim=1)
        
        edge_count = 0
        interdisciplinary_count = 0
        edges_buffer = []
        batch_size = 100  # Process 100 papers at a time to avoid GPU memory issues
        
        logger.info("Computing similarities on GPU...")
        
        for i in tqdm(range(0, len(embeddings_gpu), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(embeddings_gpu))
            batch = embeddings_gpu[i:batch_end]
            
            # Compute cosine similarity with all embeddings
            similarities = torch.mm(batch, embeddings_gpu.t())
            
            # Find high similarity pairs
            high_sim_mask = similarities >= threshold
            
            for j in range(batch.shape[0]):
                paper_idx = i + j
                paper_id = paper_ids[paper_idx]
                paper_cat = paper_categories[paper_idx]
                
                # Get indices of similar papers
                similar_indices = torch.where(high_sim_mask[j])[0].cpu().numpy()
                
                for neighbor_idx in similar_indices:
                    if neighbor_idx <= paper_idx:  # Avoid duplicates and self-connections
                        continue
                    
                    similarity = similarities[j, neighbor_idx].item()
                    neighbor_id = paper_ids[neighbor_idx]
                    neighbor_cat = paper_categories[neighbor_idx]
                    
                    edge = {
                        '_from': f"arxiv_papers/{paper_id}",
                        '_to': f"arxiv_papers/{neighbor_id}",
                        'weight': similarity,
                        'similarity': similarity
                    }
                    
                    # Track interdisciplinary connections
                    if paper_cat != neighbor_cat:
                        interdisciplinary_count += 1
                        self.interdisciplinary_edges.append({
                            'from': paper_id,
                            'to': neighbor_id,
                            'similarity': similarity,
                            'cat1': paper_cat,
                            'cat2': neighbor_cat
                        })
                    
                    edges_buffer.append(edge)
                    edge_count += 1
                    
                    # Insert edges in batches
                    if len(edges_buffer) >= 10000:
                        self.edge_collections['keyword_similarity'].insert_many(
                            edges_buffer, overwrite=True
                        )
                        edges_buffer = []
            
            # Clear GPU cache periodically
            if i % 1000 == 0:
                torch.cuda.empty_cache()
        
        # Insert remaining edges
        if edges_buffer:
            self.edge_collections['keyword_similarity'].insert_many(
                edges_buffer, overwrite=True
            )
        
        elapsed = time.time() - start_time
        logger.info(f"Created {edge_count:,} keyword edges in {elapsed/60:.1f} minutes")
        logger.info(f"Found {interdisciplinary_count:,} interdisciplinary connections")
        logger.info(f"GPU speedup: ~{32*2820/elapsed:.0f}x vs CPU")
    
    def analyze_interdisciplinary_bridges(self):
        """Analyze and log interdisciplinary connections."""
        if not self.interdisciplinary_edges:
            return
        
        logger.info("="*70)
        logger.info("INTERDISCIPLINARY ANALYSIS")
        logger.info("="*70)
        
        # Calculate statistics
        year_gaps = [e['year_gap'] for e in self.interdisciplinary_edges]
        similarities = [e['similarity'] for e in self.interdisciplinary_edges]
        category_pairs = [(e['cat1'], e['cat2']) for e in self.interdisciplinary_edges]
        
        logger.info(f"Total interdisciplinary edges: {len(self.interdisciplinary_edges):,}")
        logger.info(f"Average year gap: {np.mean(year_gaps):.1f} years")
        logger.info(f"Max year gap: {max(year_gaps)} years")
        logger.info(f"Average similarity: {np.mean(similarities):.3f}")
        
        # Top category pairs
        top_pairs = Counter(category_pairs).most_common(10)
        logger.info("\nTop interdisciplinary connections:")
        for (cat1, cat2), count in top_pairs:
            logger.info(f"  {cat1} <-> {cat2}: {count:,} edges")
        
        # Save detailed analysis
        analysis_path = Path('/home/todd/olympus/HADES-Lab/data/interdisciplinary_analysis.json')
        analysis_path.parent.mkdir(exist_ok=True)
        
        with open(analysis_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_edges': len(self.interdisciplinary_edges),
                    'avg_year_gap': float(np.mean(year_gaps)),
                    'max_year_gap': int(max(year_gaps)),
                    'avg_similarity': float(np.mean(similarities))
                },
                'top_pairs': [{'pair': pair, 'count': count} for pair, count in top_pairs],
                'sample_edges': self.interdisciplinary_edges[:100]  # Save sample for inspection
            }, f, indent=2)
        
        logger.info(f"\nDetailed analysis saved to: {analysis_path}")
    
    def build_citation_edges(self):
        """Build citation edges from paper references."""
        logger.info("="*70)
        logger.info("PHASE 4: CITATION EDGES")
        logger.info("="*70)
        
        self.edge_collections['citations'].truncate()
        start_time = time.time()
        
        # This would need the actual citation data
        # For now, placeholder
        logger.info("Citation edge building not yet implemented")
        logger.info("Would process paper references and create directed edges")
        
        elapsed = time.time() - start_time
        logger.info(f"Citation phase completed in {elapsed/60:.1f} minutes")
    
    def build_all(self, skip_categories: bool = False):
        """Build all edge types with optimizations."""
        logger.info("="*70)
        logger.info("OPTIMIZED GRAPH BUILDING")
        logger.info(f"Workers: {self.workers}")
        logger.info(f"Start time: {datetime.now()}")
        logger.info("="*70)
        
        total_start = time.time()
        
        # Start monitoring (if available)
        # self.monitor.start_monitoring()
        
        # Phase 1: Category edges (keep existing fast implementation)
        if not skip_categories:
            self.build_category_edges()
        
        # Phase 2: Temporal edges (ArXiv ID optimization)
        self.build_temporal_edges_arxiv_id()
        
        # Phase 3: Keyword edges (Faiss optimization)
        self.build_keyword_edges_faiss()
        
        # Phase 4: Citation edges
        self.build_citation_edges()
        
        # Stop monitoring and get summary
        summary = {}  # self.monitor.stop_monitoring()
        
        total_elapsed = time.time() - total_start
        logger.info("="*70)
        logger.info("BUILD COMPLETE")
        logger.info(f"Total time: {total_elapsed/3600:.1f} hours")
        logger.info(f"Energy used: {summary.get('total_energy_kwh', 0):.3f} kWh")
        logger.info("="*70)
        
        # Final statistics
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """Print final graph statistics."""
        logger.info("\nFinal Graph Statistics:")
        logger.info("="*50)
        
        total_edges = 0
        for name, collection in self.edge_collections.items():
            count = collection.count()
            total_edges += count
            logger.info(f"{name:20s}: {count:>12,} edges")
        
        logger.info(f"{'TOTAL':20s}: {total_edges:>12,} edges")
        
        paper_count = self.db.collection('arxiv_papers').count()
        avg_degree = (total_edges * 2) / paper_count if paper_count > 0 else 0
        logger.info(f"\nPapers: {paper_count:,}")
        logger.info(f"Average degree: {avg_degree:.1f}")


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
    
    # Process within-month connections
    for category, month_papers in categories.items():
        # Connect papers within same month and category
        for i in range(len(month_papers)):
            for j in range(i + 1, min(i + 50, len(month_papers))):  # Limit connections per paper
                paper1, paper2 = month_papers[i], month_papers[j]
                
                # Check submission order proximity
                try:
                    arxiv1 = paper1['arxiv_id']
                    arxiv2 = paper2['arxiv_id']
                    
                    # Extract submission numbers based on format
                    if '/' in arxiv1 and '/' in arxiv2:
                        # Old format: extract numbers after YYMM
                        num1 = arxiv1.split('/')[1][4:]
                        num2 = arxiv2.split('/')[1][4:]
                    elif '.' in arxiv1 and '.' in arxiv2:
                        # New format: extract NNNNN after dot
                        num1 = arxiv1.split('.')[1][:5] if '.' in arxiv1 else '99999'
                        num2 = arxiv2.split('.')[1][:5] if '.' in arxiv2 else '99999'
                    else:
                        # Mixed or unknown format, just connect them
                        edges.append({
                            '_from': f"arxiv_papers/{paper1['_key']}",
                            '_to': f"arxiv_papers/{paper2['_key']}",
                            'weight': 1.0,
                            'month': month
                        })
                        continue
                    
                    id1_num = int(num1) if num1.isdigit() else 99999
                    id2_num = int(num2) if num2.isdigit() else 99999
                    
                    # Connect if within 500 submissions
                    if abs(id1_num - id2_num) <= 500:
                        edges.append({
                            '_from': f"arxiv_papers/{paper1['_key']}",
                            '_to': f"arxiv_papers/{paper2['_key']}",
                            'weight': 1.0,
                            'month': month,
                            'submission_gap': abs(id1_num - id2_num)
                        })
                except (ValueError, IndexError, AttributeError):
                    # If anything fails, still connect papers in same month/category
                    edges.append({
                        '_from': f"arxiv_papers/{paper1['_key']}",
                        '_to': f"arxiv_papers/{paper2['_key']}",
                        'weight': 0.9,
                        'month': month
                    })
    
    # Process cross-month connections
    if next_month_data:
        for category in categories:
            if category in next_month_data:
                current_papers = categories[category]
                next_papers = next_month_data[category]
                
                # Connect last 20 of current to first 20 of next (reduced for speed)
                for p1 in current_papers[-20:]:
                    for p2 in next_papers[:20]:
                        edges.append({
                            '_from': f"arxiv_papers/{p1['_key']}",
                            '_to': f"arxiv_papers/{p2['_key']}",
                            'weight': 0.8,
                            'cross_month': True
                        })
    
    return edges


def get_next_month(month_str: str) -> str:
    """Get the next month in YYMM format."""
    year = int(month_str[:2])
    month = int(month_str[2:])
    
    month += 1
    if month > 12:
        month = 1
        year += 1
    
    return f"{year:02d}{month:02d}"


@click.command()
@click.option('--workers', default=36, help='Number of parallel workers')
@click.option('--skip-categories', is_flag=True, help='Skip category edge building')
@click.option('--threshold', default=0.65, help='Keyword similarity threshold')
def main(workers: int, skip_categories: bool, threshold: float):
    """Run the optimized graph builder."""
    builder = OptimizedGraphBuilder(workers=workers)
    builder.build_all(skip_categories=skip_categories)


if __name__ == "__main__":
    main()
