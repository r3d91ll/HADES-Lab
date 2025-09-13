#!/usr/bin/env python3
"""
Parallel graph builder - uses multiple workers to build edges faster.
CPU-bound process that benefits from multiprocessing.
"""

import os
import logging
from collections import defaultdict
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import numpy as np
from tqdm import tqdm
from arango import ArangoClient
import click
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_category_batch(args):
    """Process a batch of categories to create edges."""
    category_batch, max_edges_per_node, db_config = args
    
    # Each worker creates its own DB connection
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db(
        db_config['database'],
        username=db_config['username'],
        password=db_config['password']
    )
    
    edges = []
    
    for category, papers in category_batch:
        if len(papers) < 2:
            continue
            
        # Sample edges for large categories
        if len(papers) > 100:
            # Limit processing to avoid memory explosion
            sample_papers = papers[:2000]  # Process first 2000 papers
            
            for paper in sample_papers[:500]:  # Create edges for first 500
                neighbors = np.random.choice(
                    [p for p in sample_papers if p != paper],
                    min(max_edges_per_node, len(sample_papers)-1),
                    replace=False
                )
                for neighbor in neighbors:
                    edges.append({
                        '_from': f'arxiv_papers/{paper}',
                        '_to': f'arxiv_papers/{neighbor}',
                        'category': category,
                        'weight': 0.3
                    })
        else:
            # Small categories: connect more densely
            for i, paper1 in enumerate(papers):
                for paper2 in papers[i+1:min(i+max_edges_per_node, len(papers))]:
                    edges.append({
                        '_from': f'arxiv_papers/{paper1}',
                        '_to': f'arxiv_papers/{paper2}',
                        'category': category,
                        'weight': 0.3
                    })
    
    return edges


def process_coauthor_batch(args):
    """Process a batch of authors to create coauthorship edges."""
    author_batch, max_edges_per_node, db_config = args
    
    edges = []
    
    for author, papers in author_batch:
        if len(papers) < 2:
            continue
            
        # Limit edges per author to prevent explosion
        papers_subset = papers[:100]  # Max 100 papers per author
        
        for i, paper1 in enumerate(papers_subset):
            for paper2 in papers_subset[i+1:min(i+max_edges_per_node, len(papers_subset))]:
                edges.append({
                    '_from': f'arxiv_papers/{paper1}',
                    '_to': f'arxiv_papers/{paper2}',
                    'author': author,
                    'weight': 0.8
                })
    
    return edges


class ParallelGraphBuilder:
    """Build graph using parallel processing."""
    
    def __init__(self, num_workers=None):
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        self.db_config = {
            'database': 'academy_store',
            'username': 'root',
            'password': os.environ.get('ARANGO_PASSWORD')
        }
        
        self.num_workers = num_workers or max(1, cpu_count() - 2)  # Leave 2 cores free
        logger.info(f"Using {self.num_workers} worker processes")
        
    def build_category_edges_parallel(self, max_edges_per_node=30, batch_size=10):
        """Build category edges using parallel processing."""
        logger.info("Building category edges in parallel...")
        
        # Ensure collection exists
        if 'same_field' not in [c['name'] for c in self.db.collections()]:
            self.db.create_collection('same_field', edge=True)
        
        same_field = self.db.collection('same_field')
        
        # Load all paper-category mappings
        query = """
        FOR p IN arxiv_papers
            FILTER p.categories != null AND LENGTH(p.categories) > 0
            RETURN {
                id: p._key,
                categories: p.categories
            }
        """
        
        logger.info("Loading paper-category mappings...")
        category_papers = defaultdict(list)
        
        for paper in tqdm(self.db.aql.execute(query, batch_size=10000)):
            for category in paper['categories']:
                category_papers[category].append(paper['id'])
        
        logger.info(f"Found {len(category_papers)} categories")
        
        # Split categories into batches for parallel processing
        categories_list = list(category_papers.items())
        category_batches = [
            categories_list[i:i+batch_size] 
            for i in range(0, len(categories_list), batch_size)
        ]
        
        logger.info(f"Processing {len(category_batches)} batches with {self.num_workers} workers")
        
        # Process batches in parallel
        process_func = partial(
            process_category_batch,
            max_edges_per_node=max_edges_per_node,
            db_config=self.db_config
        )
        
        total_edges = 0
        edges_buffer = []
        
        with Pool(self.num_workers) as pool:
            # Map with progress bar
            args_list = [(batch, max_edges_per_node, self.db_config) for batch in category_batches]
            
            for batch_edges in tqdm(
                pool.imap_unordered(process_category_batch, args_list),
                total=len(category_batches),
                desc="Processing batches"
            ):
                edges_buffer.extend(batch_edges)
                
                # Batch insert when buffer is large enough
                if len(edges_buffer) >= 50000:
                    same_field.insert_many(edges_buffer, overwrite=True)
                    total_edges += len(edges_buffer)
                    logger.info(f"Inserted {total_edges:,} edges so far...")
                    edges_buffer = []
        
        # Insert remaining edges
        if edges_buffer:
            same_field.insert_many(edges_buffer, overwrite=True)
            total_edges += len(edges_buffer)
        
        logger.info(f"Created {total_edges:,} category edges")
        
    def build_coauthor_edges_parallel(self, max_edges_per_node=20):
        """Build coauthorship edges using parallel processing."""
        logger.info("Building coauthorship edges in parallel...")
        
        # Ensure collection exists
        if 'coauthorship' not in [c['name'] for c in self.db.collections()]:
            self.db.create_collection('coauthorship', edge=True)
        
        coauthorship = self.db.collection('coauthorship')
        
        # Load author-paper mappings
        query = """
        FOR p IN arxiv_papers
            FILTER p.authors != null AND LENGTH(p.authors) > 1
            RETURN {
                id: p._key,
                authors: p.authors
            }
        """
        
        logger.info("Loading author-paper mappings...")
        author_papers = defaultdict(list)
        
        for paper in tqdm(self.db.aql.execute(query, batch_size=10000)):
            for author in paper['authors']:
                author_papers[author].append(paper['id'])
        
        logger.info(f"Found {len(author_papers)} authors")
        
        # Split authors into batches
        authors_list = list(author_papers.items())
        author_batches = [
            authors_list[i:i+100]
            for i in range(0, len(authors_list), 100)
        ]
        
        total_edges = 0
        edges_buffer = []
        
        with Pool(self.num_workers) as pool:
            args_list = [(batch, max_edges_per_node, self.db_config) for batch in author_batches]
            
            for batch_edges in tqdm(
                pool.imap_unordered(process_coauthor_batch, args_list),
                total=len(author_batches),
                desc="Processing author batches"
            ):
                edges_buffer.extend(batch_edges)
                
                if len(edges_buffer) >= 50000:
                    coauthorship.insert_many(edges_buffer, overwrite=True)
                    total_edges += len(edges_buffer)
                    logger.info(f"Inserted {total_edges:,} edges so far...")
                    edges_buffer = []
        
        if edges_buffer:
            coauthorship.insert_many(edges_buffer, overwrite=True)
            total_edges += len(edges_buffer)
        
        logger.info(f"Created {total_edges:,} coauthorship edges")
        
    def analyze_graph(self):
        """Analyze the graph statistics."""
        logger.info("\n" + "="*60)
        logger.info("GRAPH ANALYSIS")
        logger.info("="*60)
        
        for coll_name in ['coauthorship', 'same_field', 'temporal_proximity', 'citations']:
            if coll_name in [c['name'] for c in self.db.collections()]:
                count = self.db.collection(coll_name).count()
                logger.info(f"{coll_name:20s}: {count:>12,} edges")


@click.command()
@click.option('--categories', is_flag=True, help='Build category edges')
@click.option('--coauthors', is_flag=True, help='Build coauthor edges')
@click.option('--workers', type=int, help='Number of worker processes')
@click.option('--analyze', is_flag=True, help='Analyze graph')
def main(categories, coauthors, workers, analyze):
    """Build graph using parallel processing."""
    
    builder = ParallelGraphBuilder(num_workers=workers)
    
    start_time = time.time()
    
    if categories:
        builder.build_category_edges_parallel()
        
    if coauthors:
        builder.build_coauthor_edges_parallel()
        
    if analyze or categories or coauthors:
        builder.analyze_graph()
    
    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()