#!/usr/bin/env python3
"""
Batched graph builder - processes papers in chunks to avoid memory issues.

Optimized for 2.8M papers with batched loading and processing.
"""

import os
import sys
import logging
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import click
from arango import ArangoClient
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchedGraphBuilder:
    """Build graph with batched processing for large datasets."""
    
    def __init__(self, batch_size: int = 100000):
        """Initialize with batch size."""
        self.batch_size = batch_size
        
        # Connect to ArangoDB
        logger.info("Connecting to ArangoDB...")
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Get collections
        self.papers = self.db.collection('arxiv_papers')
        self.temporal_edges = self.db.collection('temporal_proximity')
        
    def build_temporal_edges_batched(self):
        """Build temporal edges with batched processing."""
        logger.info("="*70)
        logger.info("TEMPORAL EDGES - BATCHED PROCESSING")
        logger.info(f"Batch size: {self.batch_size:,}")
        logger.info("="*70)
        
        # Clear existing edges
        logger.info("Clearing existing temporal edges...")
        self.temporal_edges.truncate()
        
        start_time = time.time()
        
        # Get total paper count
        total_papers = self.papers.count()
        logger.info(f"Total papers: {total_papers:,}")
        
        # Process in batches
        total_edges = 0
        papers_by_month = defaultdict(lambda: defaultdict(list))
        
        for offset in range(0, total_papers, self.batch_size):
            batch_start = time.time()
            
            # Load batch
            query = f"""
            FOR paper IN arxiv_papers
            FILTER paper.arxiv_id != null
            LIMIT {offset}, {self.batch_size}
            RETURN paper
            """
            
            batch = list(self.db.aql.execute(query))
            logger.info(f"Processing batch {offset//self.batch_size + 1}: {len(batch):,} papers")
            
            # Group by month
            for paper in batch:
                if 'arxiv_id' not in paper or 'categories' not in paper:
                    continue
                    
                arxiv_id = paper['arxiv_id']
                categories = paper.get('categories', [])
                if not categories:
                    continue
                
                # Extract month from ArXiv ID
                month_key = None
                
                if '/' in arxiv_id:
                    # Old format: category/YYMMNNN
                    parts = arxiv_id.split('/')
                    if len(parts) == 2 and len(parts[1]) >= 4:
                        try:
                            month_key = parts[1][:4]
                            int(month_key)  # Validate numeric
                        except ValueError:
                            continue
                elif '.' in arxiv_id and len(arxiv_id) >= 4:
                    # New format: YYMM.NNNNN
                    try:
                        month_key = arxiv_id[:4]
                        int(month_key)  # Validate numeric
                    except ValueError:
                        continue
                
                if month_key:
                    # Store minimal paper info
                    paper_info = {
                        '_key': paper['_key'],
                        'arxiv_id': arxiv_id,
                        'cat': categories[0]  # Primary category
                    }
                    papers_by_month[month_key][categories[0]].append(paper_info)
            
            batch_time = time.time() - batch_start
            logger.info(f"  Batch processed in {batch_time:.1f}s")
            
            # Periodically build edges to avoid too much memory
            if offset > 0 and offset % (self.batch_size * 5) == 0:
                logger.info("Building edges for accumulated papers...")
                edge_count = self._build_edges_for_months(papers_by_month)
                total_edges += edge_count
                papers_by_month.clear()  # Clear memory
        
        # Build edges for remaining papers
        logger.info("Building edges for final batch...")
        edge_count = self._build_edges_for_months(papers_by_month)
        total_edges += edge_count
        
        elapsed = time.time() - start_time
        logger.info("="*70)
        logger.info(f"Created {total_edges:,} temporal edges in {elapsed/60:.1f} minutes")
        logger.info(f"Rate: {total_edges/elapsed:.0f} edges/second")
        
    def _build_edges_for_months(self, papers_by_month: Dict) -> int:
        """Build edges for grouped papers."""
        edges = []
        
        # Process each month
        for month, categories in papers_by_month.items():
            for category, papers in categories.items():
                # Connect papers in same month/category
                for i in range(len(papers)):
                    for j in range(i + 1, min(i + 50, len(papers))):  # Limit connections
                        edges.append({
                            '_from': f"arxiv_papers/{papers[i]['_key']}",
                            '_to': f"arxiv_papers/{papers[j]['_key']}",
                            'weight': 0.95,
                            'month': month,
                            'category': category
                        })
                        
                        if len(edges) >= 10000:
                            # Insert batch
                            self.temporal_edges.insert_many(edges)
                            edge_count = len(edges)
                            edges = []
        
        # Insert remaining edges
        if edges:
            self.temporal_edges.insert_many(edges)
            
        return len(edges)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current graph statistics."""
        stats = {
            'papers': self.papers.count(),
            'temporal_edges': self.temporal_edges.count()
        }
        
        if stats['papers'] > 0:
            stats['avg_degree'] = (stats['temporal_edges'] * 2) / stats['papers']
        else:
            stats['avg_degree'] = 0
            
        return stats


@click.command()
@click.option('--batch-size', default=100000, help='Batch size for processing')
def main(batch_size: int):
    """Run batched graph builder."""
    builder = BatchedGraphBuilder(batch_size=batch_size)
    
    # Build temporal edges
    builder.build_temporal_edges_batched()
    
    # Show final stats
    stats = builder.get_stats()
    logger.info("\nFinal Statistics:")
    logger.info(f"  Papers: {stats['papers']:,}")
    logger.info(f"  Temporal edges: {stats['temporal_edges']:,}")
    logger.info(f"  Average degree: {stats['avg_degree']:.2f}")


if __name__ == "__main__":
    main()