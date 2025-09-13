#!/usr/bin/env python3
"""
Unified GNN Training Script - Train any GNN architecture on pre-built graphs.

Supports GraphSAGE, GAT, GCN, and custom architectures.
Loads pre-built graphs from graph_construction module.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, global_mean_pool
from torch_geometric.loader import NeighborLoader
import yaml
from datetime import datetime

# Import core graph functionality
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.graph.graph_manager import GraphManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphSAGE(torch.nn.Module):
    """GraphSAGE model."""
    def __init__(self, num_features, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, num_classes))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(torch.nn.Module):
    """Graph Attention Network."""
    def __init__(self, num_features, hidden_dim, num_classes, num_layers=2, heads=8, dropout=0.6):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim * heads, num_classes, heads=1, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GCN(torch.nn.Module):
    """Graph Convolutional Network."""
    def __init__(self, num_features, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def train_gnn(config_path: str, graph_name: str = None, model_type: str = "graphsage"):
    """Train a GNN model on a pre-built graph.
    
    Args:
        config_path: Path to training configuration
        graph_name: Name of graph to load (or uses most recent)
        model_type: Type of GNN model (graphsage, gat, gcn)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load graph
    logger.info(f"Loading graph...")
    manager = GraphManager()
    data, metadata = manager.load_graph(name=graph_name)
    
    logger.info(f"Graph loaded: {metadata['num_nodes']:,} nodes, {metadata['num_edges']:,} edges")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    num_features = data.x.shape[1] if data.x is not None else config['model']['num_features']
    num_classes = config['model']['num_classes']
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model'].get('num_layers', 2)
    dropout = config['model'].get('dropout', 0.5)
    
    if model_type == "graphsage":
        model = GraphSAGE(num_features, hidden_dim, num_classes, num_layers, dropout)
    elif model_type == "gat":
        heads = config['model'].get('heads', 8)
        model = GAT(num_features, hidden_dim, num_classes, num_layers, heads, dropout)
    elif model_type == "gcn":
        model = GCN(num_features, hidden_dim, num_classes, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Create data loader for mini-batch training
    batch_size = config['training']['batch_size']
    num_neighbors = config['training'].get('num_neighbors', [25, 10])
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Training loop
    logger.info(f"Starting training with {model_type.upper()}...")
    model.train()
    
    for epoch in range(config['training']['epochs']):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index)
            
            # Only compute loss on the batch nodes
            loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch.batch_size
            pred = out[:batch.batch_size].argmax(dim=1)
            total_correct += (pred == batch.y[:batch.batch_size]).sum().item()
            total_samples += batch.batch_size
        
        # Log epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(f"models/{model_type}_{timestamp}.pt")
    model_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metadata': metadata,
        'model_type': model_type
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN on pre-built graph")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--graph", help="Name of graph to load")
    parser.add_argument("--model", default="graphsage", 
                       choices=["graphsage", "gat", "gcn"],
                       help="Type of GNN model")
    
    args = parser.parse_args()
    
    train_gnn(args.config, args.graph, args.model)
