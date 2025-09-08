#!/usr/bin/env python3
"""
Graph Memory Store - RAM-based graph storage for GraphSAGE.

This module implements a shared memory store for keeping the entire graph in RAM,
enabling <1ms access times through zero-copy operations.

From Information Reconstructionism: The WHERE dimension (graph topology) must be
instantly accessible for real-time conveyance calculations.
"""

import os
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, DefaultDict
from dataclasses import dataclass
from collections import defaultdict
from multiprocessing import shared_memory
import json
import time
from arango import ArangoClient


@dataclass
class GraphStats:
    """Statistics about the loaded graph."""
    num_nodes: int
    num_edges: int
    num_node_types: int
    num_edge_types: int
    memory_usage_gb: float
    load_time_seconds: float


class GraphMemoryStore:
    """
    RAM-based graph storage with shared memory support.
    
    Keeps entire graph in memory for ultra-fast access (<1ms).
    Supports heterogeneous graphs with multiple node and edge types.
    """
    
    def __init__(self, max_memory_gb: float = 100.0):
        """
        Initialize the graph memory store.
        
        Args:
            max_memory_gb: Maximum RAM to use for graph storage
        """
        self.max_memory_gb = max_memory_gb
        
        # Node data
        self.node_ids: Dict[str, int] = {}  # Map node_id -> index
        self.node_types: Dict[int, str] = {}  # Map index -> type
        self.node_embeddings: Optional[np.ndarray] = None  # Shape: (n_nodes, embedding_dim)
        
        # Edge data (adjacency lists for efficiency)
        self.adjacency: DefaultDict[int, Set[int]] = defaultdict(set)
        self.edge_types: Dict[Tuple[int, int], str] = {}
        
        # Reverse mappings
        self.index_to_node: Dict[int, str] = {}
        
        # Shared memory for multiprocessing
        self.shared_memory_name: Optional[str] = None
        self.shared_memory: Optional[shared_memory.SharedMemory] = None
        
        # Statistics
        self.stats: Optional[GraphStats] = None
        
    def load_from_arangodb(self, db_config: Dict[str, Any]) -> GraphStats:
        """
        Load graph from ArangoDB into memory.
        
        Args:
            db_config: Database configuration with host, database, username, password
            
        Returns:
            GraphStats object with loading statistics
        """
        start_time = time.time()
        
        # Validate required config
        if not db_config.get('database'):
            raise ValueError("Database name is required in db_config")
        if not db_config.get('username'):
            raise ValueError("Username is required in db_config")
        
        # Get password securely
        password = db_config.get('password') or os.environ.get('ARANGO_PASSWORD')
        if not password:
            raise ValueError("Password must be provided in config or ARANGO_PASSWORD environment variable")
        
        # Connect to ArangoDB
        try:
            client = ArangoClient(hosts=db_config.get('host', 'http://localhost:8529'))
            db = client.db(
                db_config['database'],
                username=db_config['username'],
                password=password
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ArangoDB: {e}")
        
        print("Loading graph from ArangoDB into memory...")
        
        # Load nodes
        self._load_nodes(db)
        
        # Load edges
        self._load_edges(db)
        
        # Calculate statistics
        load_time = time.time() - start_time
        memory_usage = self._calculate_memory_usage()
        
        # For undirected graphs, edges are counted twice in adjacency lists
        # Divide by 2 to get the actual edge count
        edge_count = sum(len(neighbors) for neighbors in self.adjacency.values()) // 2
        
        self.stats = GraphStats(
            num_nodes=len(self.node_ids),
            num_edges=edge_count,
            num_node_types=len(set(self.node_types.values())),
            num_edge_types=len(set(self.edge_types.values())),
            memory_usage_gb=memory_usage,
            load_time_seconds=load_time
        )
        
        print(f"Graph loaded: {self.stats.num_nodes:,} nodes, {self.stats.num_edges:,} edges")
        print(f"Memory usage: {self.stats.memory_usage_gb:.2f} GB")
        print(f"Load time: {self.stats.load_time_seconds:.2f} seconds")
        
        return self.stats
    
    def _load_nodes(self, db):
        """Load nodes from various collections."""
        node_collections = [
            ('arxiv_papers', 'paper'),
            ('arxiv_chunks', 'chunk'),
            ('github_repositories', 'repo'),
            ('github_papers', 'code'),
        ]
        
        index = 0
        for collection_name, node_type in node_collections:
            try:
                collection = db.collection(collection_name)
                print(f"Loading {collection_name}...")
                
                # Get all documents
                cursor = db.aql.execute(f"FOR doc IN {collection_name} RETURN doc")
                
                for doc in cursor:
                    node_id = doc.get('_key', doc.get('_id'))
                    self.node_ids[node_id] = index
                    self.node_types[index] = node_type
                    self.index_to_node[index] = node_id
                    index += 1
                    
                print(f"  Loaded {index} nodes from {collection_name}")
                
            except Exception as e:
                print(f"  Warning: Could not load {collection_name}: {e}")
    
    def _load_edges(self, db):
        """Load edges from edge collections."""
        edge_collections = [
            ('semantic_similarity', 'similar'),
            ('theory_practice_bridges', 'implements'),
            ('paper_implements_theory', 'implements'),
        ]
        
        for collection_name, edge_type in edge_collections:
            try:
                print(f"Loading edges from {collection_name}...")
                
                cursor = db.aql.execute(f"""
                    FOR edge IN {collection_name}
                    RETURN {{from: edge._from, to: edge._to}}
                """)
                
                edge_count = 0
                for edge in cursor:
                    # Extract node IDs from _from and _to
                    from_id = edge['from'].split('/')[-1] if '/' in edge['from'] else edge['from']
                    to_id = edge['to'].split('/')[-1] if '/' in edge['to'] else edge['to']
                    
                    if from_id in self.node_ids and to_id in self.node_ids:
                        from_idx = self.node_ids[from_id]
                        to_idx = self.node_ids[to_id]
                        
                        self.adjacency[from_idx].add(to_idx)
                        self.adjacency[to_idx].add(from_idx)  # Undirected
                        self.edge_types[(from_idx, to_idx)] = edge_type
                        edge_count += 1
                
                print(f"  Loaded {edge_count} edges from {collection_name}")
                
            except Exception as e:
                print(f"  Warning: Could not load {collection_name}: {e}")
    
    def create_shared_memory(self, embedding_dim: int = 2048):
        """
        Create shared memory for embeddings to enable zero-copy access.
        
        Args:
            embedding_dim: Dimension of node embeddings
        """
        num_nodes = len(self.node_ids)
        
        # Guard against zero-node graphs
        if num_nodes == 0:
            print("Warning: No nodes in graph, skipping shared memory creation")
            return
        
        # Clean up any existing shared memory
        if self.shared_memory is not None:
            self.cleanup()
        
        # Calculate size needed
        size = num_nodes * embedding_dim * np.float32().itemsize
        size_gb = size / (1024 ** 3)
        
        # Check memory limit
        if size_gb > self.max_memory_gb:
            raise MemoryError(
                f"Required memory ({size_gb:.2f} GB) exceeds limit ({self.max_memory_gb} GB). "
                f"Reduce embedding_dim or increase max_memory_gb."
            )
        
        try:
            # Create shared memory
            self.shared_memory = shared_memory.SharedMemory(create=True, size=size)
            self.shared_memory_name = self.shared_memory.name
            
            # Create numpy array backed by shared memory
            self.node_embeddings = np.ndarray(
                (num_nodes, embedding_dim),
                dtype=np.float32,
                buffer=self.shared_memory.buf
            )
            
            # Initialize with zeros (will be filled by GraphSAGE)
            self.node_embeddings[:] = 0
            
            print(f"Created shared memory '{self.shared_memory_name}' for embeddings ({size_gb:.2f} GB)")
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to create shared memory: {e}")
        
    def get_neighbors(self, node_index: int, max_neighbors: Optional[int] = None) -> List[int]:
        """
        Get neighbors of a node.
        
        Args:
            node_index: Index of the node
            max_neighbors: Maximum number of neighbors to return (random sampling)
            
        Returns:
            List of neighbor indices
        """
        # Use sorted() for deterministic ordering
        neighbors = sorted(list(self.adjacency.get(node_index, set())))
        
        if max_neighbors and len(neighbors) > max_neighbors:
            # Random sampling for scalability
            indices = np.random.choice(len(neighbors), max_neighbors, replace=False)
            neighbors = [neighbors[i] for i in indices]
        
        return neighbors
    
    def get_k_hop_neighbors(self, node_index: int, k: int = 2, 
                           max_neighbors_per_hop: int = 25) -> Dict[int, Set[int]]:
        """
        Get k-hop neighbors for GraphSAGE aggregation.
        
        Args:
            node_index: Starting node index
            k: Number of hops
            max_neighbors_per_hop: Max neighbors to sample per hop
            
        Returns:
            Dictionary mapping hop distance to set of node indices
        """
        hop_neighbors = {0: {node_index}}
        visited = {node_index}
        
        for hop in range(1, k + 1):
            current_neighbors = set()
            
            for node in hop_neighbors[hop - 1]:
                neighbors = self.get_neighbors(node, max_neighbors_per_hop)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        current_neighbors.add(neighbor)
                        visited.add(neighbor)
            
            hop_neighbors[hop] = current_neighbors
            
            if not current_neighbors:
                break  # No more neighbors to explore
        
        return hop_neighbors
    
    def _calculate_memory_usage(self) -> float:
        """Calculate current memory usage in GB."""
        memory_bytes = 0
        
        # Node data
        memory_bytes += len(self.node_ids) * 100  # Rough estimate for dict overhead
        
        # Edge data
        for neighbors in self.adjacency.values():
            memory_bytes += len(neighbors) * 8  # 8 bytes per edge
        
        # Embeddings if loaded
        if self.node_embeddings is not None:
            memory_bytes += self.node_embeddings.nbytes
        
        return memory_bytes / (1024 ** 3)
    
    def cleanup(self):
        """Clean up shared memory."""
        if self.shared_memory:
            try:
                self.shared_memory.close()
                self.shared_memory.unlink()
            except FileNotFoundError:
                # Already unlinked
                pass
            except Exception as e:
                print(f"Warning: Error cleaning up shared memory: {e}")
            finally:
                self.shared_memory = None
                self.shared_memory_name = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
    
    def save_to_disk(self, filepath: str):
        """
        Save graph structure to disk for faster loading.
        
        Args:
            filepath: Path to save the graph
        """
        data = {
            'node_ids': self.node_ids,
            'node_types': self.node_types,
            'adjacency': {k: list(v) for k, v in self.adjacency.items()},
            'edge_types': {f"{k[0]},{k[1]}": v for k, v in self.edge_types.items()},
            'index_to_node': self.index_to_node
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        print(f"Graph saved to {filepath}")
    
    def load_from_disk(self, filepath: str):
        """
        Load graph structure from disk.
        
        Args:
            filepath: Path to load the graph from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.node_ids = data['node_ids']
        self.node_types = {int(k): v for k, v in data['node_types'].items()}
        self.adjacency = {int(k): set(v) for k, v in data['adjacency'].items()}
        self.edge_types = {
            tuple(map(int, k.split(','))): v 
            for k, v in data['edge_types'].items()
        }
        self.index_to_node = {int(k): v for k, v in data['index_to_node'].items()}
        
        print(f"Graph loaded from {filepath}")