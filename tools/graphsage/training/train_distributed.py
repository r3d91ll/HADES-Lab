#!/usr/bin/env python3
"""
Distributed GraphSAGE training that actually uses multiple GPUs.
Uses pipeline parallelism to split layers across GPUs.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DistributedGraphSAGE(nn.Module):
    """GraphSAGE model split across multiple GPUs."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 4, dropout: float = 0.5):
        super(DistributedGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Determine GPU allocation
        self.num_gpus = min(torch.cuda.device_count(), 2)  # Use max 2 GPUs
        logger.info(f"Using {self.num_gpus} GPUs for model parallelism")
        
        # Create layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        # Distribute layers across GPUs
        if self.num_gpus > 1:
            # Split layers between GPU 0 and GPU 1
            # First half on GPU 0, second half on GPU 1
            mid_point = len(self.convs) // 2
            
            for i, (conv, bn) in enumerate(zip(self.convs[:mid_point], self.bns[:mid_point])):
                conv.cuda(0)
                bn.cuda(0)
            
            for i, conv in enumerate(self.convs[mid_point:]):
                conv.cuda(1)
                if i + mid_point < len(self.bns):
                    self.bns[i + mid_point].cuda(1)
            
            self.gpu_split = mid_point
            logger.info(f"Layers 0-{mid_point-1} on GPU 0, layers {mid_point}-{num_layers-1} on GPU 1")
        else:
            self.cuda(0)
            self.gpu_split = None
    
    def forward(self, x, edge_index):
        if self.num_gpus > 1:
            # Process first half on GPU 0
            x = x.cuda(0)
            edge_index = edge_index.cuda(0)
            
            for i in range(self.gpu_split):
                x = self.convs[i](x, edge_index)
                if i < len(self.bns):
                    x = self.bns[i](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Transfer to GPU 1
            x = x.cuda(1)
            edge_index = edge_index.cuda(1)
            
            # Process second half on GPU 1
            for i in range(self.gpu_split, len(self.convs)):
                x = self.convs[i](x, edge_index)
                if i < len(self.bns):
                    x = self.bns[i](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            return x
        else:
            # Standard single GPU processing
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.convs[-1](x, edge_index)
            return x


def train_distributed():
    """Main training function."""
    
    # Load features
    logger.info("Loading features...")
    data_np = np.load('features.npz', allow_pickle=True)
    
    features = torch.tensor(data_np['features'], dtype=torch.float)
    labels = torch.tensor(data_np['labels'], dtype=torch.long) if 'labels' in data_np else None
    
    # Convert adjacency to edge_index (simplified - assumes it's already done)
    logger.info("Loading edge_index...")
    if 'edge_index' in data_np:
        edge_index = torch.tensor(data_np['edge_index'], dtype=torch.long)
    else:
        # Convert from adjacency
        edges = []
        adjacency = data_np['adjacency'].item()
        logger.info("Converting adjacency to edge_index...")
        for node, neighbors in adjacency.items():
            for neighbor in neighbors:
                edges.append([int(node), int(neighbor)])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create data object
    data = Data(x=features, edge_index=edge_index, y=labels)
    
    # Create train/val/test splits
    num_nodes = features.shape[0]
    indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    logger.info(f"Data splits - Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
    
    # Adjusted parameters for memory efficiency
    batch_size = 2048  # Reduced from 4096
    num_neighbors = [15, 10, 5, 5]  # Reduced from [25, 15, 10, 10]
    hidden_dim = 1024
    num_layers = 4
    
    logger.info(f"Configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Neighbors: {num_neighbors}")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"  Layers: {num_layers}")
    
    # Create data loaders
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True,
        num_workers=8
    )
    
    # Create distributed model
    num_classes = len(torch.unique(labels)) if labels is not None else 172
    model = DistributedGraphSAGE(
        features.shape[1], 
        hidden_dim, 
        num_classes,
        num_layers=num_layers
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer - need to handle parameters on different devices
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    logger.info("Starting training...")
    epochs = 100
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            
            # Forward pass (model handles GPU transfer internally)
            out = model(batch.x, batch.edge_index)
            
            # Loss computation on the final GPU
            if model.num_gpus > 1:
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask].cuda(1))
                pred = out[batch.train_mask].argmax(dim=1)
                correct = (pred == batch.y[batch.train_mask].cuda(1)).sum()
            else:
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask].cuda(0))
                pred = out[batch.train_mask].argmax(dim=1)
                correct = (pred == batch.y[batch.train_mask].cuda(0)).sum()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_correct += correct.item()
            total_samples += batch.train_mask.sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct.item() / batch.train_mask.sum().item()})
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
        
        # Save periodically
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'acc': avg_acc
            }, f'checkpoint_epoch_{epoch+1}.pt')
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_neighbors': num_neighbors,
            'batch_size': batch_size
        }
    }, 'trained_graphsage_distributed.pt')
    
    logger.info("Training complete!")


if __name__ == "__main__":
    train_distributed()