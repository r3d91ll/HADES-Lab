#!/usr/bin/env python3
"""
GraphSAGE Pipeline - Main execution pipeline for GraphSAGE operations.

Orchestrates the complete workflow:
1. Load graph from ArangoDB into RAM
2. Train/load GraphSAGE model
3. Generate embeddings
4. Discover theory-practice bridges
5. Store results back to ArangoDB

From Information Reconstructionism: This pipeline operationalizes the
discovery of CONVEYANCE patterns across heterogeneous knowledge domains.
"""

import os
import sys
import argparse
import yaml
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.framework.memory_store import GraphMemoryStore
from core.framework.graph_embedders import GraphSAGEEmbedder, GraphSAGEConfig, AggregatorType
from tools.graphsage.utils.neighborhood_sampler import NeighborhoodSampler, SamplingConfig
from tools.graphsage.bridge_discovery.theory_practice_finder import (
    TheoryPracticeFinder, BridgeDiscoveryConfig
)
from arango import ArangoClient


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    # Database
    arango_host: str = "http://localhost:8529"
    arango_database: str = "academy_store"
    arango_username: str = "root"
    
    # GraphSAGE model
    model_config: GraphSAGEConfig = None
    
    # Sampling
    sampling_config: SamplingConfig = None
    
    # Bridge discovery
    bridge_config: BridgeDiscoveryConfig = None
    
    # Pipeline options
    load_from_cache: bool = False
    cache_path: str = "./cache/graph_cache.json"
    save_embeddings: bool = True
    save_bridges: bool = True
    
    # Memory settings
    max_memory_gb: float = 100.0
    use_shared_memory: bool = True
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = GraphSAGEConfig()
        if self.sampling_config is None:
            self.sampling_config = SamplingConfig()
        if self.bridge_config is None:
            self.bridge_config = BridgeDiscoveryConfig()


class GraphSAGEPipeline:
    """
    Main pipeline for GraphSAGE operations.
    
    Manages the complete workflow from data loading to bridge discovery.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.graph_store = None
        self.graphsage_model = None
        self.bridge_finder = None
        
        # Statistics
        self.stats = {
            'load_time': 0,
            'train_time': 0,
            'embedding_time': 0,
            'bridge_discovery_time': 0,
            'total_time': 0
        }
    
    def load_graph(self):
        """Load graph from ArangoDB or cache."""
        print("\n" + "="*60)
        print("LOADING GRAPH")
        print("="*60)
        
        start = time.time()
        
        # Initialize graph store
        self.graph_store = GraphMemoryStore(max_memory_gb=self.config.max_memory_gb)
        
        # Load from cache or database
        if self.config.load_from_cache and Path(self.config.cache_path).exists():
            print(f"Loading graph from cache: {self.config.cache_path}")
            self.graph_store.load_from_disk(self.config.cache_path)
        else:
            print("Loading graph from ArangoDB...")
            
            db_config = {
                'host': self.config.arango_host,
                'database': self.config.arango_database,
                'username': self.config.arango_username,
                'password': os.environ.get('ARANGO_PASSWORD')
            }
            
            stats = self.graph_store.load_from_arangodb(db_config)
            
            # Save to cache
            os.makedirs(os.path.dirname(self.config.cache_path), exist_ok=True)
            self.graph_store.save_to_disk(self.config.cache_path)
        
        # Create shared memory if requested
        if self.config.use_shared_memory:
            self.graph_store.create_shared_memory(embedding_dim=self.config.model_config.input_dim)
        
        self.stats['load_time'] = time.time() - start
        print(f"\nGraph loaded in {self.stats['load_time']:.2f} seconds")
    
    def initialize_model(self):
        """Initialize GraphSAGE model."""
        print("\n" + "="*60)
        print("INITIALIZING GRAPHSAGE MODEL")
        print("="*60)
        
        self.graphsage_model = GraphSAGEEmbedder(
            config=self.config.model_config,
            graph_store=self.graph_store
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.graphsage_model.parameters())
        trainable_params = sum(p.numel() for p in self.graphsage_model.parameters() if p.requires_grad)
        
        print(f"Model initialized:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Device: {self.config.model_config.device}")
    
    def train_model(self, num_epochs: int = 10):
        """
        Train GraphSAGE model (unsupervised).
        
        Args:
            num_epochs: Number of training epochs
        """
        print("\n" + "="*60)
        print("TRAINING GRAPHSAGE MODEL")
        print("="*60)
        
        start = time.time()
        
        # For now, skip training and use random initialization
        # In production, implement proper unsupervised training
        print("Using random initialization (training not implemented yet)")
        
        self.stats['train_time'] = time.time() - start
    
    def generate_embeddings(self):
        """Generate embeddings for all nodes."""
        print("\n" + "="*60)
        print("GENERATING EMBEDDINGS")
        print("="*60)
        
        start = time.time()
        
        embeddings = self.graphsage_model.generate_embeddings(
            batch_size=self.config.model_config.batch_size
        )
        
        # Store in graph store
        if self.graph_store.node_embeddings is not None:
            self.graph_store.node_embeddings[:] = embeddings
        
        self.stats['embedding_time'] = time.time() - start
        
        print(f"Generated embeddings: shape {embeddings.shape}")
        print(f"Time: {self.stats['embedding_time']:.2f} seconds")
        
        # Save embeddings if requested
        if self.config.save_embeddings:
            self._save_embeddings(embeddings)
    
    def discover_bridges(self):
        """Discover theory-practice bridges."""
        print("\n" + "="*60)
        print("DISCOVERING THEORY-PRACTICE BRIDGES")
        print("="*60)
        
        start = time.time()
        
        # Initialize bridge finder
        self.bridge_finder = TheoryPracticeFinder(
            graph_store=self.graph_store,
            graphsage_model=self.graphsage_model,
            config=self.config.bridge_config
        )
        
        # Discover bridges
        all_bridges = self.bridge_finder.discover_all_bridges()
        
        self.stats['bridge_discovery_time'] = time.time() - start
        
        # Print summary
        print("\nBridge Discovery Summary:")
        for bridge_type, bridges in all_bridges.items():
            print(f"  {bridge_type}: {len(bridges)} bridges")
        
        print(f"Time: {self.stats['bridge_discovery_time']:.2f} seconds")
        
        # Save bridges if requested
        if self.config.save_bridges:
            self._save_bridges(all_bridges)
        
        return all_bridges
    
    def _save_embeddings(self, embeddings):
        """Save embeddings to ArangoDB."""
        print("\nSaving embeddings to ArangoDB...")
        
        client = ArangoClient(hosts=self.config.arango_host)
        db = client.db(
            self.config.arango_database,
            username=self.config.arango_username,
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Create collection if it doesn't exist
        if not db.has_collection('graphsage_embeddings'):
            db.create_collection('graphsage_embeddings')
        
        collection = db.collection('graphsage_embeddings')
        
        # Save embeddings in batches
        batch_size = 1000
        num_saved = 0
        
        for start_idx in range(0, len(embeddings), batch_size):
            end_idx = min(start_idx + batch_size, len(embeddings))
            batch = []
            
            for idx in range(start_idx, end_idx):
                node_id = self.graph_store.index_to_node.get(idx, f"node_{idx}")
                
                doc = {
                    '_key': str(node_id).replace('/', '_'),
                    'node_id': node_id,
                    'node_index': idx,
                    'embedding': embeddings[idx].tolist(),
                    'embedding_dim': len(embeddings[idx]),
                    'model': 'graphsage',
                    'timestamp': time.time()
                }
                batch.append(doc)
            
            try:
                collection.insert_many(batch, overwrite=True)
                num_saved += len(batch)
            except Exception as e:
                print(f"Error saving batch: {e}")
        
        print(f"Saved {num_saved} embeddings to ArangoDB")
    
    def _save_bridges(self, all_bridges):
        """Save discovered bridges to ArangoDB."""
        print("\nSaving bridges to ArangoDB...")
        
        client = ArangoClient(hosts=self.config.arango_host)
        db = client.db(
            self.config.arango_database,
            username=self.config.arango_username,
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Create collection if it doesn't exist
        if not db.has_collection('bridge_predictions'):
            db.create_collection('bridge_predictions')
        
        collection = db.collection('bridge_predictions')
        
        # Save bridges
        num_saved = 0
        
        for bridge_type, bridges in all_bridges.items():
            batch = []
            
            for bridge in bridges:
                doc = {
                    '_key': f"{bridge.paper_id}_{bridge.code_id}_{bridge_type}".replace('/', '_'),
                    'paper_id': bridge.paper_id,
                    'code_id': bridge.code_id,
                    'confidence': bridge.confidence,
                    'bridge_type': bridge_type,
                    'evidence': bridge.evidence,
                    'discovered_by': 'graphsage',
                    'timestamp': time.time()
                }
                batch.append(doc)
            
            try:
                collection.insert_many(batch, overwrite=True)
                num_saved += len(batch)
            except Exception as e:
                print(f"Error saving {bridge_type} bridges: {e}")
        
        print(f"Saved {num_saved} bridges to ArangoDB")
    
    def run(self):
        """Run the complete pipeline."""
        print("\n" + "="*60)
        print("GRAPHSAGE PIPELINE")
        print("="*60)
        
        total_start = time.time()
        
        # Step 1: Load graph
        self.load_graph()
        
        # Step 2: Initialize model
        self.initialize_model()
        
        # Step 3: Train model (optional)
        # self.train_model()
        
        # Step 4: Generate embeddings
        self.generate_embeddings()
        
        # Step 5: Discover bridges
        bridges = self.discover_bridges()
        
        self.stats['total_time'] = time.time() - total_start
        
        # Print final statistics
        self._print_statistics()
        
        return bridges
    
    def _print_statistics(self):
        """Print pipeline statistics."""
        print("\n" + "="*60)
        print("PIPELINE STATISTICS")
        print("="*60)
        
        for key, value in self.stats.items():
            if 'time' in key:
                print(f"{key}: {value:.2f} seconds")
        
        # Memory usage
        if self.graph_store:
            print(f"\nMemory usage: {self.graph_store._calculate_memory_usage():.2f} GB")
        
        # Throughput
        if self.graph_store and self.stats['total_time'] > 0:
            nodes_per_sec = len(self.graph_store.node_ids) / self.stats['total_time']
            print(f"Throughput: {nodes_per_sec:.0f} nodes/second")


def load_config(config_path: str) -> PipelineConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config objects
    model_config = GraphSAGEConfig(**config_dict.get('model', {}))
    sampling_config = SamplingConfig(**config_dict.get('sampling', {}))
    bridge_config = BridgeDiscoveryConfig(**config_dict.get('bridge_discovery', {}))
    
    # Create pipeline config
    pipeline_config = PipelineConfig(
        arango_host=config_dict.get('arango', {}).get('host', 'http://localhost:8529'),
        arango_database=config_dict.get('arango', {}).get('database', 'academy_store'),
        arango_username=config_dict.get('arango', {}).get('username', 'root'),
        model_config=model_config,
        sampling_config=sampling_config,
        bridge_config=bridge_config,
        **config_dict.get('pipeline', {})
    )
    
    return pipeline_config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GraphSAGE Pipeline")
    parser.add_argument('--config', type=str, default='../configs/graphsage_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--arango-password', type=str, help='ArangoDB password')
    
    args = parser.parse_args()
    
    # Set password if provided
    if args.arango_password:
        os.environ['ARANGO_PASSWORD'] = args.arango_password
    
    # Check for password
    if not os.environ.get('ARANGO_PASSWORD'):
        print("Error: ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Run pipeline
    pipeline = GraphSAGEPipeline(config)
    bridges = pipeline.run()
    
    print("\n✅ GraphSAGE pipeline completed successfully!")
    
    return bridges


if __name__ == "__main__":
    main()