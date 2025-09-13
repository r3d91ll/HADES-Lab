#!/usr/bin/env python3
"""
Complete Graph Build Orchestration for ArXiv Papers.

This script manages the entire graph construction process following
the "Death of the Author" philosophy - no author-based connections.
"""

import os
import sys
import time
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import click
from arango import ArangoClient
import torch
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GraphOrchestrator:
    """Orchestrate the complete graph building process."""
    
    def __init__(self, config_path: str = None):
        """Initialize orchestrator with configuration."""
        self.config = self._load_config(config_path)
        self.start_time = time.time()
        
        # Connect to database
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Track statistics
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'edges_built': {},
            'validation_results': {},
            'errors': []
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'edge_types': {
                'same_field': {
                    'enabled': True,
                    'method': 'category_match',
                    'priority': 1
                },
                'temporal_proximity': {
                    'enabled': True,
                    'method': 'time_window',
                    'window_days': 30,
                    'priority': 2
                },
                'abstract_similarity': {
                    'enabled': True,
                    'method': 'cosine_similarity',
                    'threshold': 0.75,
                    'batch_size': 15000,
                    'priority': 3
                },
                'keyword_similarity': {
                    'enabled': True,
                    'method': 'cosine_similarity',
                    'threshold': 0.65,
                    'batch_size': 15000,
                    'priority': 4
                },
                'paper_versions': {
                    'enabled': True,
                    'method': 'version_grouping',
                    'priority': 5
                }
            },
            'validation': {
                'check_symmetry': True,
                'check_orphans': True,
                'check_duplicates': True,
                'check_weights': True
            },
            'gpu': {
                'use_gpu': True,
                'devices': [0, 1],
                'batch_size': 15000,
                'use_fp16': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(loaded_config)
        
        return default_config
    
    def verify_data(self) -> bool:
        """Verify all required data is present."""
        logger.info("="*70)
        logger.info("VERIFYING DATA")
        logger.info("="*70)
        
        checks = []
        
        # Check papers
        paper_count = self.db.collection('arxiv_papers').count()
        logger.info(f"Papers: {paper_count:,}")
        checks.append(paper_count > 0)
        
        # Check embeddings
        for embedding_type in ['abstract', 'keyword']:
            query = f"""
            FOR e IN arxiv_embeddings
            FILTER e.{embedding_type}_embedding != null
            COLLECT WITH COUNT INTO c
            RETURN c
            """
            count = list(self.db.aql.execute(query))[0]
            logger.info(f"{embedding_type.capitalize()} embeddings: {count:,}")
            
            if embedding_type == 'abstract' and count < paper_count * 0.95:
                logger.warning(f"⚠ Abstract embeddings incomplete! Only {count/paper_count*100:.1f}% done")
                checks.append(False)
            elif embedding_type == 'keyword' and count < paper_count * 0.95:
                logger.warning(f"⚠ Keyword embeddings incomplete! Only {count/paper_count*100:.1f}% done")
                checks.append(False)
            else:
                checks.append(True)
        
        return all(checks)
    
    def build_same_field_edges(self) -> int:
        """Build edges between papers in the same field."""
        logger.info("Building same_field edges...")
        
        # Check if already exists
        if self.db.has_collection('same_field'):
            count = self.db.collection('same_field').count()
            if count > 4000000:  # Expected ~4.1M
                logger.info(f"✓ same_field edges already built: {count:,}")
                return count
        
        # Import the builder
        from builders.build_same_field import SameFieldBuilder
        builder = SameFieldBuilder()
        count = builder.build()
        
        logger.info(f"✓ Built {count:,} same_field edges")
        return count
    
    def build_temporal_edges(self) -> int:
        """Build temporal proximity edges."""
        logger.info("Building temporal_proximity edges...")
        
        # Check if already exists
        if self.db.has_collection('temporal_proximity'):
            count = self.db.collection('temporal_proximity').count()
            if count > 40000000:  # Expected ~40.6M
                logger.info(f"✓ temporal_proximity edges already built: {count:,}")
                return count
        
        # Import the builder
        from builders.build_temporal_edges_arxiv import TemporalEdgeBuilder
        builder = TemporalEdgeBuilder()
        count = builder.build(window_days=self.config['edge_types']['temporal_proximity']['window_days'])
        
        logger.info(f"✓ Built {count:,} temporal_proximity edges")
        return count
    
    def build_abstract_similarity_edges(self) -> int:
        """Build abstract similarity edges using GPU."""
        logger.info("Building abstract_similarity edges...")
        
        # Check embeddings are complete
        query = """
        FOR e IN arxiv_embeddings
        FILTER e.abstract_embedding != null
        COLLECT WITH COUNT INTO c
        RETURN c
        """
        embedding_count = list(self.db.aql.execute(query))[0]
        paper_count = self.db.collection('arxiv_papers').count()
        
        if embedding_count < paper_count * 0.95:
            logger.error(f"Abstract embeddings incomplete: {embedding_count:,}/{paper_count:,}")
            return 0
        
        # Import the GPU builder
        from builders.build_abstract_edges_gpu import AbstractSimilarityBuilder
        builder = AbstractSimilarityBuilder(
            batch_size=self.config['edge_types']['abstract_similarity']['batch_size'],
            threshold=self.config['edge_types']['abstract_similarity']['threshold']
        )
        count = builder.build()
        
        logger.info(f"✓ Built {count:,} abstract_similarity edges")
        return count
    
    def build_keyword_similarity_edges(self) -> int:
        """Build keyword similarity edges using GPU."""
        logger.info("Building keyword_similarity edges...")
        
        # Import the GPU builder with NVLink
        from builders.build_keyword_edges_gpu_nvlink import KeywordSimilarityGPUBuilder
        builder = KeywordSimilarityGPUBuilder(
            batch_size=self.config['edge_types']['keyword_similarity']['batch_size'],
            threshold=self.config['edge_types']['keyword_similarity']['threshold']
        )
        count = builder.build()
        
        logger.info(f"✓ Built {count:,} keyword_similarity edges")
        return count
    
    def build_version_edges(self) -> int:
        """Build edges between paper versions."""
        logger.info("Building paper_versions edges...")
        
        # Create collection if doesn't exist
        if not self.db.has_collection('paper_versions'):
            self.db.create_edge_collection('paper_versions')
        
        versions_coll = self.db.collection('paper_versions')
        
        # Find papers with multiple versions
        query = """
        FOR p IN arxiv_papers
        FILTER p.versions != null AND LENGTH(p.versions) > 1
        RETURN {
            _key: p._key,
            arxiv_id: p.arxiv_id,
            versions: p.versions
        }
        """
        
        papers_with_versions = list(self.db.aql.execute(query))
        logger.info(f"Found {len(papers_with_versions):,} papers with multiple versions")
        
        edges = []
        for paper in tqdm(papers_with_versions, desc="Building version edges"):
            # Extract base ID (remove version suffix)
            base_id = paper['arxiv_id'].split('v')[0] if 'v' in paper['arxiv_id'] else paper['arxiv_id']
            
            # Connect consecutive versions
            versions = sorted(paper['versions'], key=lambda x: x.get('created', ''))
            for i in range(len(versions) - 1):
                edge = {
                    '_from': f"arxiv_papers/{paper['_key']}_v{i+1}",
                    '_to': f"arxiv_papers/{paper['_key']}_v{i+2}",
                    'base_id': base_id,
                    'version_from': i + 1,
                    'version_to': i + 2,
                    'created_from': versions[i].get('created'),
                    'created_to': versions[i + 1].get('created')
                }
                edges.append(edge)
                
                if len(edges) >= 1000:
                    versions_coll.insert_many(edges)
                    edges = []
        
        # Insert remaining
        if edges:
            versions_coll.insert_many(edges)
        
        count = versions_coll.count()
        logger.info(f"✓ Built {count:,} paper_versions edges")
        return count
    
    def validate_graph(self) -> Dict[str, Any]:
        """Validate the constructed graph."""
        logger.info("="*70)
        logger.info("VALIDATING GRAPH")
        logger.info("="*70)
        
        validation_results = {}
        
        # Count edges in each collection
        edge_collections = [
            'same_field',
            'temporal_proximity',
            'keyword_similarity',
            'abstract_similarity',
            'paper_versions'
        ]
        
        total_edges = 0
        for coll_name in edge_collections:
            if self.db.has_collection(coll_name):
                count = self.db.collection(coll_name).count()
                validation_results[f'{coll_name}_count'] = count
                total_edges += count
                logger.info(f"{coll_name}: {count:,} edges")
        
        validation_results['total_edges'] = total_edges
        
        # Calculate average degree
        paper_count = self.db.collection('arxiv_papers').count()
        avg_degree = (total_edges * 2) / paper_count  # *2 for undirected
        validation_results['average_degree'] = avg_degree
        logger.info(f"Average degree: {avg_degree:.2f}")
        
        # Check for orphans (papers with no edges)
        if self.config['validation']['check_orphans']:
            # This is expensive, so we sample
            query = """
            FOR p IN arxiv_papers
            LIMIT 10000
            LET edges = (
                FOR e IN 1..1 ANY p 
                    same_field, temporal_proximity, keyword_similarity, 
                    abstract_similarity, paper_versions
                LIMIT 1
                RETURN 1
            )
            FILTER LENGTH(edges) == 0
            COLLECT WITH COUNT INTO orphans
            RETURN orphans
            """
            orphan_sample = list(self.db.aql.execute(query))[0]
            orphan_rate = orphan_sample / 10000
            validation_results['orphan_rate'] = orphan_rate
            logger.info(f"Orphan rate (sample): {orphan_rate*100:.2f}%")
        
        return validation_results
    
    def export_statistics(self, output_path: str = None):
        """Export graph statistics to file."""
        if not output_path:
            output_path = f"graph_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['total_time_minutes'] = (time.time() - self.start_time) / 60
        
        with open(output_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"✓ Statistics exported to {output_path}")
    
    def run(self, edge_types: List[str] = None):
        """Run the complete graph building pipeline."""
        logger.info("="*70)
        logger.info("ARXIV GRAPH BUILD ORCHESTRATION")
        logger.info("="*70)
        logger.info(f"Started at: {datetime.now()}")
        
        # Verify data
        if not self.verify_data():
            logger.error("Data verification failed! Please complete embeddings first.")
            return False
        
        # Determine which edges to build
        if not edge_types:
            edge_types = [k for k, v in self.config['edge_types'].items() if v['enabled']]
        
        # Sort by priority
        edge_types.sort(key=lambda x: self.config['edge_types'][x].get('priority', 999))
        
        logger.info(f"Building edge types: {edge_types}")
        
        # Build each edge type
        for edge_type in edge_types:
            logger.info(f"\n{'='*70}")
            logger.info(f"Building {edge_type}")
            logger.info("="*70)
            
            try:
                if edge_type == 'same_field':
                    count = self.build_same_field_edges()
                elif edge_type == 'temporal_proximity':
                    count = self.build_temporal_edges()
                elif edge_type == 'abstract_similarity':
                    count = self.build_abstract_similarity_edges()
                elif edge_type == 'keyword_similarity':
                    count = self.build_keyword_similarity_edges()
                elif edge_type == 'paper_versions':
                    count = self.build_version_edges()
                else:
                    logger.warning(f"Unknown edge type: {edge_type}")
                    continue
                
                self.stats['edges_built'][edge_type] = count
                
            except Exception as e:
                logger.error(f"Error building {edge_type}: {e}")
                self.stats['errors'].append({
                    'edge_type': edge_type,
                    'error': str(e),
                    'time': datetime.now().isoformat()
                })
        
        # Validate graph
        self.stats['validation_results'] = self.validate_graph()
        
        # Export statistics
        self.export_statistics()
        
        # Final summary
        elapsed = time.time() - self.start_time
        logger.info("\n" + "="*70)
        logger.info("GRAPH BUILD COMPLETE")
        logger.info("="*70)
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Total edges: {self.stats['validation_results']['total_edges']:,}")
        logger.info(f"Average degree: {self.stats['validation_results']['average_degree']:.2f}")
        
        return True


@click.command()
@click.option('--config', default=None, help='Path to configuration file')
@click.option('--verify-data', is_flag=True, help='Only verify data')
@click.option('--build-edges', default='all', help='Which edges to build (all/specific type)')
@click.option('--validate-graph', is_flag=True, help='Only validate existing graph')
@click.option('--export-stats', default=None, help='Export statistics to file')
def main(config, verify_data, build_edges, validate_graph, export_stats):
    """Orchestrate the complete ArXiv graph building process."""
    
    orchestrator = GraphOrchestrator(config_path=config)
    
    if verify_data:
        # Only verify data
        if orchestrator.verify_data():
            logger.info("✓ Data verification passed")
        else:
            logger.error("✗ Data verification failed")
        return
    
    if validate_graph:
        # Only validate
        results = orchestrator.validate_graph()
        logger.info(f"Validation results: {json.dumps(results, indent=2)}")
        if export_stats:
            orchestrator.export_statistics(export_stats)
        return
    
    # Run full pipeline
    edge_types = None if build_edges == 'all' else [build_edges]
    success = orchestrator.run(edge_types=edge_types)
    
    if success:
        logger.info("✓ Graph build completed successfully")
    else:
        logger.error("✗ Graph build failed")
        sys.exit(1)


if __name__ == "__main__":
    main()