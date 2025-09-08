#!/usr/bin/env python3
"""
GraphSAGE Embedder - Inductive graph embedding generation.

Implements GraphSAGE (SAmple and aggreGatE) for learning node embeddings
through neighborhood aggregation. Supports heterogeneous graphs and
incremental updates without retraining.

From Information Reconstructionism: GraphSAGE discovers the CONVEYANCE
patterns that emerge from graph structure, enabling theory-practice bridges.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from .memory_store import GraphMemoryStore


class AggregatorType(Enum):
    """Types of aggregation functions."""
    MEAN = "mean"
    POOL = "pool"
    LSTM = "lstm"
    GCN = "gcn"


@dataclass
class GraphSAGEConfig:
    """Configuration for GraphSAGE model."""
    input_dim: int = 2048  # Jina v4 embedding dimension
    hidden_dims: List[int] = None  # Hidden layer dimensions
    output_dim: int = 256  # Final embedding dimension
    num_layers: int = 2  # Number of GraphSAGE layers (K)
    aggregator_type: AggregatorType = AggregatorType.MEAN
    dropout: float = 0.5
    concat: bool = True  # Concatenate self and neighbor features
    batch_size: int = 512
    num_neighbors: List[int] = None  # Neighbors to sample per layer
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]
        if self.num_neighbors is None:
            self.num_neighbors = [25, 10]  # Sample 25 at layer 1, 10 at layer 2


class Aggregator(nn.Module):
    """Base class for aggregation functions."""
    
    def __init__(self, input_dim: int, output_dim: int, concat: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.concat = concat
        
    def forward(self, self_feats: torch.Tensor, neighbor_feats: torch.Tensor) -> torch.Tensor:
        """Aggregate neighbor features."""
        raise NotImplementedError


class MeanAggregator(Aggregator):
    """Mean aggregation of neighbor features."""
    
    def __init__(self, input_dim: int, output_dim: int, concat: bool = True):
        super().__init__(input_dim, output_dim, concat)
        
        if concat:
            self.fc = nn.Linear(input_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, self_feats: torch.Tensor, neighbor_feats: torch.Tensor) -> torch.Tensor:
        """
        Aggregate using mean of neighbors.
        
        Args:
            self_feats: Features of target nodes (batch_size, input_dim)
            neighbor_feats: Features of neighbors (batch_size, num_neighbors, input_dim)
            
        Returns:
            Aggregated features (batch_size, output_dim)
        """
        # Check for empty neighbors
        if neighbor_feats.shape[1] == 0:
            # No neighbors, just use self features
            if self.concat:
                # Concatenate with zeros
                zeros = torch.zeros_like(self_feats)
                combined = torch.cat([self_feats, zeros], dim=1)
            else:
                combined = self_feats
        else:
            # Mean pooling over neighbors
            neighbor_mean = neighbor_feats.mean(dim=1)
            
            if self.concat:
                combined = torch.cat([self_feats, neighbor_mean], dim=1)
            else:
                combined = neighbor_mean
        
        return F.relu(self.fc(combined))


class PoolingAggregator(Aggregator):
    """Max pooling aggregation of neighbor features."""
    
    def __init__(self, input_dim: int, output_dim: int, concat: bool = True):
        super().__init__(input_dim, output_dim, concat)
        
        # Transform neighbor features before pooling
        self.fc_pool = nn.Linear(input_dim, input_dim)
        
        if concat:
            self.fc = nn.Linear(input_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, self_feats: torch.Tensor, neighbor_feats: torch.Tensor) -> torch.Tensor:
        """
        Aggregate using max pooling over transformed neighbors.
        
        Args:
            self_feats: Features of target nodes (batch_size, input_dim)
            neighbor_feats: Features of neighbors (batch_size, num_neighbors, input_dim)
            
        Returns:
            Aggregated features (batch_size, output_dim)
        """
        # Transform neighbor features
        neighbor_transformed = F.relu(self.fc_pool(neighbor_feats))
        
        # Max pooling
        neighbor_pooled = neighbor_transformed.max(dim=1)[0]
        
        if self.concat:
            combined = torch.cat([self_feats, neighbor_pooled], dim=1)
        else:
            combined = neighbor_pooled
        
        return F.relu(self.fc(combined))


class LSTMAggregator(Aggregator):
    """LSTM aggregation of neighbor features."""
    
    def __init__(self, input_dim: int, output_dim: int, concat: bool = True):
        super().__init__(input_dim, output_dim, concat)
        
        self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True)
        
        if concat:
            self.fc = nn.Linear(input_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, self_feats: torch.Tensor, neighbor_feats: torch.Tensor) -> torch.Tensor:
        """
        Aggregate using LSTM over neighbor sequence.
        
        Args:
            self_feats: Features of target nodes (batch_size, input_dim)
            neighbor_feats: Features of neighbors (batch_size, num_neighbors, input_dim)
            
        Returns:
            Aggregated features (batch_size, output_dim)
        """
        # Random permutation of neighbors for LSTM
        batch_size, num_neighbors, _ = neighbor_feats.shape
        perm = torch.randperm(num_neighbors)
        neighbor_feats = neighbor_feats[:, perm, :]
        
        # LSTM aggregation
        lstm_out, _ = self.lstm(neighbor_feats)
        neighbor_lstm = lstm_out[:, -1, :]  # Take last output
        
        if self.concat:
            combined = torch.cat([self_feats, neighbor_lstm], dim=1)
        else:
            combined = neighbor_lstm
        
        return F.relu(self.fc(combined))


class GraphSAGELayer(nn.Module):
    """Single GraphSAGE layer."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 aggregator_type: AggregatorType, 
                 dropout: float = 0.5, concat: bool = True):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Select aggregator
        if aggregator_type == AggregatorType.MEAN:
            self.aggregator = MeanAggregator(input_dim, output_dim, concat)
        elif aggregator_type == AggregatorType.POOL:
            self.aggregator = PoolingAggregator(input_dim, output_dim, concat)
        elif aggregator_type == AggregatorType.LSTM:
            self.aggregator = LSTMAggregator(input_dim, output_dim, concat)
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator_type}")
    
    def forward(self, self_feats: torch.Tensor, neighbor_feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GraphSAGE layer.
        
        Args:
            self_feats: Features of target nodes
            neighbor_feats: Features of neighbors
            
        Returns:
            Updated node features
        """
        self_feats = self.dropout(self_feats)
        neighbor_feats = self.dropout(neighbor_feats)
        
        # Aggregate
        h = self.aggregator(self_feats, neighbor_feats)
        
        # L2 normalize
        h = F.normalize(h, p=2, dim=1)
        
        return h


class GraphSAGEEmbedder(nn.Module):
    """
    GraphSAGE model for generating node embeddings.
    
    This model learns embeddings through iterative neighborhood aggregation,
    enabling inductive learning on unseen nodes.
    """
    
    def __init__(self, config: GraphSAGEConfig, graph_store: GraphMemoryStore):
        super().__init__()
        
        self.config = config
        self.graph_store = graph_store
        
        # Build layers
        self.layers = nn.ModuleList()
        
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        
        for i in range(len(dims) - 1):
            layer = GraphSAGELayer(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                aggregator_type=config.aggregator_type,
                dropout=config.dropout,
                concat=config.concat
            )
            self.layers.append(layer)
        
        # Move to device
        self.to(config.device)
    
    def sample_neighbors(self, node_indices: List[int], 
                        num_neighbors: List[int]) -> Dict[int, List[List[int]]]:
        """
        Sample neighbors for batch of nodes.
        
        Args:
            node_indices: Indices of target nodes
            num_neighbors: Number of neighbors to sample per layer
            
        Returns:
            Dictionary mapping layer to list of neighbor lists
        """
        samples = {0: [[idx] for idx in node_indices]}
        
        for layer in range(len(num_neighbors)):
            layer_samples = []
            
            for node_list in samples[layer]:
                node_neighbors = []
                for node in node_list:
                    neighbors = self.graph_store.get_neighbors(node, num_neighbors[layer])
                    if not neighbors:
                        neighbors = [node]  # Self-loop if no neighbors
                    node_neighbors.extend(neighbors)
                layer_samples.append(node_neighbors)
            
            samples[layer + 1] = layer_samples
        
        return samples
    
    def forward(self, node_indices: List[int], 
                initial_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate embeddings for nodes.
        
        Args:
            node_indices: Indices of nodes to embed
            initial_features: Initial node features (if None, use from graph_store)
            
        Returns:
            Node embeddings (batch_size, output_dim)
        """
        # Sample neighbors
        samples = self.sample_neighbors(node_indices, self.config.num_neighbors)
        
        # Get initial features
        if initial_features is None:
            if self.graph_store.node_embeddings is not None:
                initial_features = torch.tensor(
                    self.graph_store.node_embeddings[node_indices],
                    device=self.config.device,
                    dtype=torch.float32
                )
            else:
                # Random initialization if no embeddings
                initial_features = torch.randn(
                    len(node_indices), 
                    self.config.input_dim,
                    device=self.config.device
                )
        
        # Forward through layers
        h = initial_features
        
        for layer_idx, layer in enumerate(self.layers):
            # Get neighbor features for this layer
            neighbor_indices = samples[layer_idx + 1]
            
            # Flatten neighbor indices
            all_neighbor_indices = []
            neighbor_counts = []
            for neighbors in neighbor_indices:
                all_neighbor_indices.extend(neighbors)
                neighbor_counts.append(len(neighbors))
            
            # Get neighbor features
            if self.graph_store.node_embeddings is not None:
                neighbor_feats = torch.tensor(
                    self.graph_store.node_embeddings[all_neighbor_indices],
                    device=self.config.device,
                    dtype=torch.float32
                )
            else:
                neighbor_feats = torch.randn(
                    len(all_neighbor_indices),
                    h.shape[1],
                    device=self.config.device
                )
            
            # Reshape to (batch_size, num_neighbors, feature_dim)
            max_neighbors = max(neighbor_counts)
            batch_neighbor_feats = torch.zeros(
                len(node_indices), 
                max_neighbors, 
                neighbor_feats.shape[1],
                device=self.config.device
            )
            
            start_idx = 0
            for i, count in enumerate(neighbor_counts):
                batch_neighbor_feats[i, :count] = neighbor_feats[start_idx:start_idx + count]
                start_idx += count
            
            # Apply layer
            h = layer(h, batch_neighbor_feats)
        
        return h
    
    def compute_loss(self, pos_pairs: torch.Tensor, neg_pairs: torch.Tensor,
                    embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute unsupervised loss using positive and negative pairs.
        
        Args:
            pos_pairs: Positive node pairs (batch_size, 2)
            neg_pairs: Negative node pairs (batch_size, 2)
            embeddings: Node embeddings
            
        Returns:
            Loss value
        """
        # Positive scores
        pos_u = embeddings[pos_pairs[:, 0]]
        pos_v = embeddings[pos_pairs[:, 1]]
        pos_scores = torch.sigmoid((pos_u * pos_v).sum(dim=1))
        
        # Negative scores
        neg_u = embeddings[neg_pairs[:, 0]]
        neg_v = embeddings[neg_pairs[:, 1]]
        neg_scores = torch.sigmoid((neg_u * neg_v).sum(dim=1))
        
        # Binary cross-entropy loss
        pos_loss = -torch.log(pos_scores + 1e-8).mean()
        neg_loss = -torch.log(1 - neg_scores + 1e-8).mean()
        
        return pos_loss + neg_loss
    
    def generate_embeddings(self, batch_size: int = 512) -> np.ndarray:
        """
        Generate embeddings for all nodes in the graph.
        
        Args:
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (num_nodes, output_dim)
        """
        self.eval()
        
        num_nodes = len(self.graph_store.node_ids)
        embeddings = np.zeros((num_nodes, self.config.output_dim), dtype=np.float32)
        
        with torch.no_grad():
            for start_idx in range(0, num_nodes, batch_size):
                end_idx = min(start_idx + batch_size, num_nodes)
                node_indices = list(range(start_idx, end_idx))
                
                batch_embeddings = self.forward(node_indices)
                embeddings[start_idx:end_idx] = batch_embeddings.cpu().numpy()
        
        return embeddings