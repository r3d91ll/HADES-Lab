#!/usr/bin/env python3
"""
Graph Manager - Core infrastructure for graph creation, export, and loading.

Provides reusable graph functionality for any tool that needs graph data:
- GNN models (GraphSAGE, GAT, GCN)
- Graph analytics tools
- Visualization tools
- Retrieval systems (HiRAG)

From Information Reconstructionism: The graph topology represents the WHERE
dimension - the relational positioning of knowledge in semantic space.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from arango import ArangoClient
import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphManager:
    """Manages graph creation, storage, and loading for GNN training."""
    
    def __init__(self, export_dir: str = "/home/todd/olympus/HADES-Lab/data/graphs"):
        """Initialize graph manager.
        
        Args:
            export_dir: Directory for storing exported graphs
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to ArangoDB
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get current graph statistics from database."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'papers': self.db.collection('arxiv_papers').count(),
            'edges': {},
            'total_edges': 0
        }
        
        edge_collections = ['same_field', 'temporal_proximity', 'keyword_similarity', 'citations']
        for coll_name in edge_collections:
            if coll_name in [c['name'] for c in self.db.collections()]:
                count = self.db.collection(coll_name).count()
                stats['edges'][coll_name] = count
                stats['total_edges'] += count
        
        return stats
    
    def export_graph(self, name: str = "arxiv_graph", 
                     include_features: bool = True) -> Dict[str, str]:
        """Export graph from ArangoDB to PyTorch Geometric format.
        
        Args:
            name: Name for the exported graph
            include_features: Whether to include node features
            
        Returns:
            Dictionary with paths to exported files
        """
        logger.info(f"Exporting graph '{name}'...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{name}_{timestamp}"
        export_path = self.export_dir / export_name
        export_path.mkdir(exist_ok=True)
        
        # Get all papers and create node mapping
        logger.info("Creating node mapping...")
        papers = list(self.db.collection('arxiv_papers').find())
        node_mapping = {paper['_key']: idx for idx, paper in enumerate(papers)}
        
        # Export edges from all collections
        edge_index_list = []
        edge_attr_list = []
        edge_types = []
        
        edge_collections = {
            'same_field': 0,
            'temporal_proximity': 1,
            'keyword_similarity': 2,
            'citations': 3
        }
        
        for coll_name, edge_type in edge_collections.items():
            if coll_name not in [c['name'] for c in self.db.collections()]:
                continue
                
            logger.info(f"Processing {coll_name} edges...")
            edges = self.db.collection(coll_name).find()
            
            for edge in edges:
                # Extract paper IDs from _from and _to
                from_id = edge['_from'].split('/')[-1]
                to_id = edge['_to'].split('/')[-1]
                
                if from_id in node_mapping and to_id in node_mapping:
                    from_idx = node_mapping[from_id]
                    to_idx = node_mapping[to_id]
                    
                    # Add both directions for undirected graph
                    edge_index_list.append([from_idx, to_idx])
                    edge_index_list.append([to_idx, from_idx])
                    
                    # Edge type as attribute
                    edge_types.append(edge_type)
                    edge_types.append(edge_type)
                    
                    # Additional attributes if present
                    if 'weight' in edge:
                        edge_attr_list.append(edge['weight'])
                        edge_attr_list.append(edge['weight'])
        
        # Convert to tensors
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        # Node features
        x = None
        if include_features:
            logger.info("Loading node features...")
            # Load from embeddings collection
            features = []
            for paper in papers:
                embed_doc = self.db.collection('arxiv_embeddings').get(paper['_key'])
                if embed_doc and 'embedding' in embed_doc:
                    features.append(embed_doc['embedding'])
                else:
                    # Use zero vector if no embedding
                    features.append([0.0] * 2048)
            x = torch.tensor(features, dtype=torch.float)
        
        # Node labels (categories)
        y = torch.zeros(len(papers), dtype=torch.long)
        for idx, paper in enumerate(papers):
            if 'primary_category' in paper:
                # Map category to index (would need category mapping)
                y[idx] = 0  # Placeholder
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y,
            num_nodes=len(papers)
        )
        
        # Save graph data
        torch.save(data, export_path / "graph.pt")
        
        # Save metadata
        metadata = {
            'name': export_name,
            'timestamp': timestamp,
            'num_nodes': len(papers),
            'num_edges': edge_index.shape[1],
            'edge_types': edge_collections,
            'node_mapping': node_mapping,
            'stats': self.get_graph_stats()
        }
        
        with open(export_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save node mapping separately for efficiency
        with open(export_path / "node_mapping.pkl", 'wb') as f:
            pickle.dump(node_mapping, f)
        
        logger.info(f"Graph exported to {export_path}")
        
        return {
            'graph_path': str(export_path / "graph.pt"),
            'metadata_path': str(export_path / "metadata.json"),
            'mapping_path': str(export_path / "node_mapping.pkl")
        }
    
    def load_graph(self, graph_path: str = None, 
                   name: str = None) -> Tuple[Data, Dict[str, Any]]:
        """Load a pre-exported graph.
        
        Args:
            graph_path: Direct path to graph.pt file
            name: Name of graph to load (finds most recent)
            
        Returns:
            Tuple of (PyTorch Geometric Data, metadata dict)
        """
        if graph_path:
            graph_dir = Path(graph_path).parent
        elif name:
            # Find most recent export with this name
            pattern = f"{name}_*"
            dirs = sorted(self.export_dir.glob(pattern))
            if not dirs:
                raise ValueError(f"No graph found with name '{name}'")
            graph_dir = dirs[-1]
        else:
            # Load most recent graph
            dirs = sorted([d for d in self.export_dir.iterdir() if d.is_dir()])
            if not dirs:
                raise ValueError("No exported graphs found")
            graph_dir = dirs[-1]
        
        # Load graph data
        data = torch.load(graph_dir / "graph.pt")
        
        # Load metadata
        with open(graph_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded graph from {graph_dir}")
        logger.info(f"  Nodes: {metadata['num_nodes']:,}")
        logger.info(f"  Edges: {metadata['num_edges']:,}")
        
        return data, metadata
    
    def list_graphs(self) -> list:
        """List all available exported graphs."""
        graphs = []
        for graph_dir in sorted(self.export_dir.iterdir()):
            if graph_dir.is_dir() and (graph_dir / "metadata.json").exists():
                with open(graph_dir / "metadata.json", 'r') as f:
                    metadata = json.load(f)
                graphs.append({
                    'name': metadata['name'],
                    'path': str(graph_dir),
                    'timestamp': metadata['timestamp'],
                    'nodes': metadata['num_nodes'],
                    'edges': metadata['num_edges']
                })
        return graphs


if __name__ == "__main__":
    # Example usage
    manager = GraphManager()
    
    # Check current graph stats
    stats = manager.get_graph_stats()
    print(f"\nCurrent Graph Statistics:")
    print(f"  Papers: {stats['papers']:,}")
    print(f"  Total edges: {stats['total_edges']:,}")
    for edge_type, count in stats['edges'].items():
        print(f"    {edge_type}: {count:,}")
    
    # List available graphs
    print(f"\nAvailable Graphs:")
    for graph in manager.list_graphs():
        print(f"  {graph['name']}: {graph['nodes']:,} nodes, {graph['edges']:,} edges")
