#!/usr/bin/env python3
"""
ArXiv "Death of the Author" Graph v1.0

This orchestration creates a reproducible academic graph that treats papers as 
autonomous knowledge entities, explicitly avoiding author-based connections.

Philosophy: Following Barthes' "Death of the Author", we focus on the knowledge
itself rather than who created it. Papers connect through conceptual similarity,
temporal proximity, and field relationships - not authorship.

Graph Configuration:
- Papers: ~2.8M ArXiv papers with keywords
- Edge Types:
  1. keyword_similarity: Semantic connections (GPU-accelerated)
  2. temporal_proximity: Papers published within ±1 month (ArXiv-specific)
  3. same_field: Papers in the same ArXiv category
  4. NO author edges (philosophical choice)
  5. Minimal citations (de-emphasize traditional citation networks)

Requirements:
- 2x RTX A6000 GPUs with NVLink (96GB combined)
- 128GB+ system RAM
- ArangoDB with academy_store database
- ~2.8M papers with keyword embeddings
"""

import os
import sys
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import click

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from arango import ArangoClient
from core.graph.builders.build_graph_optimized import OptimizedGraphBuilder
from core.graph.builders.build_keyword_edges_gpu_nvlink import NVLinkKeywordEdgeBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeathOfAuthorGraphOrchestrator:
    """Orchestrates building of the Death of Author graph."""
    
    # Graph configuration (for reproducibility)
    CONFIG = {
        'version': '1.0',
        'philosophy': 'death_of_author',
        'source': 'arxiv',
        
        # Edge configurations
        'edges': {
            'same_field': {
                'enabled': True,
                'weight': 1.0,
                'description': 'Papers in same ArXiv category'
            },
            'temporal_proximity': {
                'enabled': True,
                'method': 'arxiv_id_optimized',  # ArXiv-specific
                'window_months': 1,
                'weight': 1.0,
                'cross_category_boost': 1.2,
                'description': 'Papers within ±1 month'
            },
            'keyword_similarity': {
                'enabled': True,
                'method': 'gpu_nvlink',
                'threshold': 0.7,
                'top_k': 30,
                'batch_size': 15000,
                'cross_category_boost': 1.2,
                'description': 'Semantic similarity via embeddings'
            },
            'citation': {
                'enabled': False,  # Minimal citations only
                'description': 'Traditional citations (disabled)'
            },
            'coauthor': {
                'enabled': False,  # Death of Author philosophy
                'description': 'Author connections (explicitly disabled)'
            }
        },
        
        # Dataset filters
        'filters': {
            'require_keywords': True,
            'require_embeddings': True,
            'min_year': None,  # Use all years
            'categories': None  # Use all categories
        }
    }
    
    def __init__(self, dry_run: bool = False):
        """Initialize orchestrator.
        
        Args:
            dry_run: If True, only show what would be built
        """
        self.dry_run = dry_run
        
        # Connect to database
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Create output directory for logs
        self.output_dir = Path('graph_builds') / f"death_of_author_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self):
        """Save configuration for reproducibility."""
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.CONFIG, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        
        # Also create a hash for verification
        config_str = json.dumps(self.CONFIG, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
        hash_path = self.output_dir / 'config_hash.txt'
        with open(hash_path, 'w') as f:
            f.write(f"Config hash: {config_hash}\n")
            f.write(f"This hash ensures reproducibility\n")
        
        logger.info(f"Config hash: {config_hash}")
    
    def validate_environment(self):
        """Validate that we have everything needed."""
        logger.info("Validating environment...")
        
        # Check database
        try:
            paper_count = self.db.collection('arxiv_papers').count()
            logger.info(f"✓ Database connected: {paper_count:,} papers")
        except Exception as e:
            logger.error(f"✗ Database error: {e}")
            return False
        
        # Check embeddings
        query = """
        FOR e IN arxiv_embeddings
        FILTER e.keyword_embedding != null
        COLLECT WITH COUNT INTO c
        RETURN c
        """
        embed_count = list(self.db.aql.execute(query))[0]
        logger.info(f"✓ Keyword embeddings: {embed_count:,} papers")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                for i in range(n_gpus):
                    props = torch.cuda.get_device_properties(i)
                    logger.info(f"✓ GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")
            else:
                logger.warning("⚠ No GPUs available - keyword similarity will be slow")
        except ImportError:
            logger.error("✗ PyTorch not installed")
            return False
        
        return True
    
    def build_same_field_edges(self):
        """Build edges between papers in same ArXiv category."""
        if not self.CONFIG['edges']['same_field']['enabled']:
            logger.info("Skipping same_field edges (disabled)")
            return
        
        logger.info("\n" + "="*70)
        logger.info("BUILDING SAME FIELD EDGES")
        logger.info("="*70)
        
        if self.dry_run:
            logger.info("DRY RUN: Would build same_field edges")
            return
        
        builder = OptimizedGraphBuilder(workers=48)
        builder.build_same_field_edges()
    
    def build_temporal_edges(self):
        """Build temporal proximity edges (ArXiv-specific)."""
        if not self.CONFIG['edges']['temporal_proximity']['enabled']:
            logger.info("Skipping temporal edges (disabled)")
            return
        
        logger.info("\n" + "="*70)
        logger.info("BUILDING TEMPORAL EDGES (ArXiv ID Optimized)")
        logger.info("="*70)
        
        if self.dry_run:
            logger.info("DRY RUN: Would build temporal_proximity edges")
            return
        
        builder = OptimizedGraphBuilder(workers=48)
        builder.build_temporal_edges_arxiv_id()
    
    def build_keyword_edges(self):
        """Build keyword similarity edges using GPU acceleration."""
        if not self.CONFIG['edges']['keyword_similarity']['enabled']:
            logger.info("Skipping keyword edges (disabled)")
            return
        
        logger.info("\n" + "="*70)
        logger.info("BUILDING KEYWORD SIMILARITY EDGES (GPU)")
        logger.info("="*70)
        
        if self.dry_run:
            logger.info("DRY RUN: Would build keyword_similarity edges")
            return
        
        config = self.CONFIG['edges']['keyword_similarity']
        builder = NVLinkKeywordEdgeBuilder(batch_size=config['batch_size'])
        builder.build_edges_gpu(
            threshold=config['threshold'],
            top_k=config['top_k']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get final graph statistics."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'config_version': self.CONFIG['version'],
            'nodes': {},
            'edges': {}
        }
        
        # Count nodes
        stats['nodes']['arxiv_papers'] = self.db.collection('arxiv_papers').count()
        
        # Count edges by type
        edge_collections = ['same_field', 'temporal_proximity', 'keyword_similarity']
        for coll_name in edge_collections:
            try:
                count = self.db.collection(coll_name).count()
                stats['edges'][coll_name] = count
                
                # Get cross-category stats for keyword edges
                if coll_name == 'keyword_similarity' and count > 0:
                    query = f"""
                    FOR e IN {coll_name}
                    FILTER e.cross_category == true
                    COLLECT WITH COUNT INTO c
                    RETURN c
                    """
                    cross_count = list(self.db.aql.execute(query))[0]
                    stats['edges'][f'{coll_name}_cross_category'] = cross_count
                    stats['edges'][f'{coll_name}_cross_category_pct'] = (cross_count/count)*100
            except:
                stats['edges'][coll_name] = 0
        
        return stats
    
    def run(self):
        """Run the complete orchestration."""
        start_time = time.time()
        
        logger.info("="*70)
        logger.info("DEATH OF THE AUTHOR GRAPH ORCHESTRATION v1.0")
        logger.info("="*70)
        
        # Validate environment
        if not self.validate_environment():
            logger.error("Environment validation failed")
            return False
        
        # Save configuration
        if not self.dry_run:
            self.save_config()
        
        # Build edges in order
        try:
            # 1. Same field edges (fastest)
            self.build_same_field_edges()
            
            # 2. Temporal edges (optimized with ArXiv IDs)
            self.build_temporal_edges()
            
            # 3. Keyword similarity (GPU-accelerated)
            self.build_keyword_edges()
            
        except Exception as e:
            logger.error(f"Build failed: {e}")
            return False
        
        # Get final statistics
        stats = self.get_statistics()
        
        # Save statistics
        if not self.dry_run:
            stats_path = self.output_dir / 'statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
        
        # Report results
        elapsed = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info("BUILD COMPLETE")
        logger.info("="*70)
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Nodes: {stats['nodes']['arxiv_papers']:,} papers")
        for edge_type, count in stats['edges'].items():
            if not edge_type.endswith('_pct'):
                logger.info(f"Edges ({edge_type}): {count:,}")
        
        if not self.dry_run:
            logger.info(f"\nResults saved to: {self.output_dir}")
        
        return True


@click.command()
@click.option('--dry-run', is_flag=True, help='Show what would be built without building')
@click.option('--skip-same-field', is_flag=True, help='Skip same_field edges')
@click.option('--skip-temporal', is_flag=True, help='Skip temporal edges')
@click.option('--skip-keyword', is_flag=True, help='Skip keyword edges')
def main(dry_run: bool, skip_same_field: bool, skip_temporal: bool, skip_keyword: bool):
    """Build the Death of Author ArXiv graph."""
    orchestrator = DeathOfAuthorGraphOrchestrator(dry_run=dry_run)
    
    # Override config if skipping
    if skip_same_field:
        orchestrator.CONFIG['edges']['same_field']['enabled'] = False
    if skip_temporal:
        orchestrator.CONFIG['edges']['temporal_proximity']['enabled'] = False
    if skip_keyword:
        orchestrator.CONFIG['edges']['keyword_similarity']['enabled'] = False
    
    success = orchestrator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()