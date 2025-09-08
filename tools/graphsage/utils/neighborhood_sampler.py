#!/usr/bin/env python3
"""
Neighborhood Sampling for GraphSAGE.

Implements efficient sampling strategies for large-scale graphs,
including uniform sampling, importance sampling, and adaptive sampling
based on node properties.

From Information Reconstructionism: The WHERE dimension (graph topology)
must be sampled intelligently to maintain conveyance while being computationally
tractable.
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import random
from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """Configuration for neighborhood sampling."""
    strategy: str = "uniform"  # uniform, importance, adaptive
    num_neighbors: List[int] = None  # Neighbors per layer [25, 10]
    importance_weights: Optional[Dict[int, float]] = None  # Node importance scores
    seed: int = 42
    
    def __post_init__(self):
        if self.num_neighbors is None:
            self.num_neighbors = [25, 10]  # Default: 25 at layer 1, 10 at layer 2


class NeighborhoodSampler:
    """
    Efficient neighborhood sampling for GraphSAGE.
    
    Supports multiple sampling strategies to balance between
    computational efficiency and representation quality.
    """
    
    def __init__(self, graph_store, config: SamplingConfig):
        """
        Initialize the sampler.
        
        Args:
            graph_store: GraphMemoryStore instance
            config: Sampling configuration
        """
        self.graph_store = graph_store
        self.config = config
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        # Precompute sampling probabilities if needed
        self.sampling_probs = self._compute_sampling_probabilities()
    
    def _compute_sampling_probabilities(self) -> Optional[Dict[int, np.ndarray]]:
        """
        Compute sampling probabilities for importance sampling.
        
        Returns:
            Dictionary mapping node index to neighbor sampling probabilities
        """
        if self.config.strategy != "importance":
            return None
        
        probs = {}
        
        for node_idx in range(len(self.graph_store.node_ids)):
            neighbors = sorted(list(self.graph_store.adjacency.get(node_idx, set())))
            
            if not neighbors:
                continue
            
            # Compute importance scores for neighbors
            if self.config.importance_weights:
                scores = np.array([
                    self.config.importance_weights.get(n, 1.0) 
                    for n in neighbors
                ])
            else:
                # Default: use node degree as importance
                scores = np.array([
                    len(self.graph_store.adjacency.get(n, set()))
                    for n in neighbors
                ])
            
            # Normalize to probabilities
            if scores.sum() > 0:
                probs[node_idx] = scores / scores.sum()
            else:
                probs[node_idx] = np.ones(len(neighbors)) / len(neighbors)
        
        return probs
    
    def sample_neighbors(self, node: int, num_samples: int) -> List[int]:
        """
        Sample neighbors for a single node.
        
        Args:
            node: Node index
            num_samples: Number of neighbors to sample
            
        Returns:
            List of sampled neighbor indices
        """
        # Use sorted() for deterministic ordering
        neighbors = sorted(list(self.graph_store.adjacency.get(node, set())))
        
        if not neighbors:
            return [node]  # Self-loop if no neighbors
        
        if len(neighbors) <= num_samples:
            return neighbors
        
        # Apply sampling strategy
        if self.config.strategy == "uniform":
            return self._uniform_sample(neighbors, num_samples)
        elif self.config.strategy == "importance":
            return self._importance_sample(node, neighbors, num_samples)
        elif self.config.strategy == "adaptive":
            return self._adaptive_sample(node, neighbors, num_samples)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.strategy}")
    
    def _uniform_sample(self, neighbors: List[int], num_samples: int) -> List[int]:
        """Uniform random sampling."""
        return random.sample(neighbors, num_samples)
    
    def _importance_sample(self, node: int, neighbors: List[int], 
                          num_samples: int) -> List[int]:
        """Importance-based sampling."""
        if self.sampling_probs and node in self.sampling_probs:
            probs = self.sampling_probs[node]
            
            # Sample without replacement
            indices = np.random.choice(
                len(neighbors), 
                size=min(num_samples, len(neighbors)),
                replace=False,
                p=probs
            )
            return [neighbors[i] for i in indices]
        else:
            return self._uniform_sample(neighbors, num_samples)
    
    def _adaptive_sample(self, node: int, neighbors: List[int], 
                        num_samples: int) -> List[int]:
        """
        Adaptive sampling based on node properties.
        
        Prioritizes neighbors with different node types for heterogeneous graphs.
        """
        # Get node type of current node
        node_type = self.graph_store.node_types.get(node, "unknown")
        
        # Separate neighbors by type
        same_type = []
        diff_type = []
        
        for neighbor in neighbors:
            neighbor_type = self.graph_store.node_types.get(neighbor, "unknown")
            if neighbor_type == node_type:
                same_type.append(neighbor)
            else:
                diff_type.append(neighbor)
        
        # Sample proportionally (prefer cross-type connections)
        if diff_type:
            num_diff = min(len(diff_type), num_samples * 2 // 3)
            num_same = min(len(same_type), num_samples - num_diff)
            
            sampled = []
            if num_diff > 0:
                sampled.extend(random.sample(diff_type, num_diff))
            if num_same > 0:
                sampled.extend(random.sample(same_type, num_same))
            
            return sampled[:num_samples]
        else:
            return self._uniform_sample(neighbors, num_samples)
    
    def sample_k_hop(self, nodes: List[int], k: int = 2) -> Dict[int, Dict[int, Set[int]]]:
        """
        Sample k-hop neighborhoods for multiple nodes.
        
        Args:
            nodes: List of starting node indices
            k: Number of hops
            
        Returns:
            Nested dictionary: node -> hop -> set of neighbor indices
        """
        results = {}
        
        for node in nodes:
            hop_neighbors = {0: {node}}
            visited = {node}
            
            for hop in range(1, k + 1):
                current_neighbors = set()
                num_samples = self.config.num_neighbors[min(hop - 1, len(self.config.num_neighbors) - 1)]
                
                for prev_node in hop_neighbors[hop - 1]:
                    sampled = self.sample_neighbors(prev_node, num_samples)
                    
                    for neighbor in sampled:
                        if neighbor not in visited:
                            current_neighbors.add(neighbor)
                            visited.add(neighbor)
                
                hop_neighbors[hop] = current_neighbors
                
                if not current_neighbors:
                    break  # No more neighbors to explore
            
            results[node] = hop_neighbors
        
        return results
    
    def create_minibatch(self, node_indices: List[int]) -> Tuple[List[List[int]], List[int]]:
        """
        Create a minibatch with sampled neighborhoods.
        
        Args:
            node_indices: Indices of target nodes
            
        Returns:
            Tuple of (neighbor_indices per layer, unique node list)
        """
        k = len(self.config.num_neighbors)
        neighborhoods = self.sample_k_hop(node_indices, k)
        
        # Organize by layer
        layer_nodes = defaultdict(set)
        
        for node, hop_neighbors in neighborhoods.items():
            for hop, neighbors in hop_neighbors.items():
                layer_nodes[hop].update(neighbors)
        
        # Create ordered lists
        layer_lists = []
        unique_nodes = set()
        
        for layer in range(k + 1):
            nodes = list(layer_nodes[layer])
            layer_lists.append(nodes)
            unique_nodes.update(nodes)
        
        return layer_lists, list(unique_nodes)
    
    def compute_sampling_variance(self, node: int, num_trials: int = 100) -> float:
        """
        Compute variance of sampling for a node (for analysis).
        
        Args:
            node: Node index
            num_trials: Number of sampling trials
            
        Returns:
            Variance of sampled neighbor sets
        """
        if num_trials <= 0:
            raise ValueError("num_trials must be positive")
            
        neighbor_sets = []
        
        # Check if node has neighbors
        if node not in self.graph_store.adjacency or not self.graph_store.adjacency[node]:
            return 0.0  # No neighbors, no variance
            
        num_samples = self.config.num_neighbors[0] if self.config.num_neighbors else 10
        
        for _ in range(num_trials):
            sampled = set(self.sample_neighbors(node, num_samples))
            neighbor_sets.append(sampled)
        
        # Compute Jaccard similarity variance
        similarities = []
        for i in range(len(neighbor_sets)):
            for j in range(i + 1, len(neighbor_sets)):
                intersection = len(neighbor_sets[i] & neighbor_sets[j])
                union = len(neighbor_sets[i] | neighbor_sets[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        if similarities:
            return float(np.var(similarities))
        return 0.0
    
    def get_sampling_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the sampling process.
        
        Returns:
            Dictionary of sampling statistics
        """
        stats = {}
        
        # Average degree
        degrees = [len(neighbors) for neighbors in self.graph_store.adjacency.values()]
        stats['avg_degree'] = np.mean(degrees) if degrees else 0
        stats['max_degree'] = max(degrees) if degrees else 0
        stats['min_degree'] = min(degrees) if degrees else 0
        
        # Sampling coverage
        total_edges = sum(degrees)
        sampled_edges_per_node = sum(self.config.num_neighbors)
        stats['sampling_ratio'] = sampled_edges_per_node / stats['avg_degree'] if stats['avg_degree'] > 0 else 0
        
        # Node type distribution
        type_counts = defaultdict(int)
        for node_type in self.graph_store.node_types.values():
            type_counts[node_type] += 1
        
        for node_type, count in type_counts.items():
            stats[f'nodes_{node_type}'] = count
        
        return stats