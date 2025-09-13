#!/usr/bin/env python3
"""
Distributed GraphSAGE training for massive graphs using mini-batch sampling.
Designed for graphs with 100M+ edges.
"""

import os
import json
import logging
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import torch.multiprocessing as mp
from tqdm import tqdm
from arango import ArangoClient
import click

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class MassiveGraphDataLoader:
    """Load massive graphs efficiently using streaming and sampling."""
    
    def __init__(self, db_name: str = 'academy_store'):
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            db_name,
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
    def load_graph_metadata(self) -> Dict:
        """Load just graph statistics without loading full graph."""
        logger.info("Loading graph metadata...")
        
        stats = {}
        
        # Count nodes
        stats['num_nodes'] = self.db.collection('arxiv_papers').count()
        
        # Count edges
        edge_collections = ['coauthorship', 'same_field', 'temporal_proximity', 'citations',
                          'same_journal', 'same_submitter', 'same_conference']
        
        stats['num_edges'] = 0
        stats['edge_counts'] = {}
        
        for coll_name in edge_collections:
            if coll_name in [c['name'] for c in self.db.collections()]:
                count = self.db.collection(coll_name).count()
                stats['edge_counts'][coll_name] = count
                stats['num_edges'] += count
                logger.info(f"  {coll_name}: {count:,} edges")
        
        logger.info(f"Total: {stats['num_nodes']:,} nodes, {stats['num_edges']:,} edges")
        logger.info(f"Average degree: {2 * stats['num_edges'] / stats['num_nodes']:.1f}")
        
        return stats
    
    def create_edge_index_streaming(self, batch_size: int = 100000) -> torch.Tensor:
        """Create edge index by streaming edges in batches."""
        logger.info("Creating edge index with streaming...")
        
        edge_list = []
        
        # Create node ID mapping
        logger.info("Creating node ID mapping...")
        node_to_idx = {}
        
        query = "FOR p IN arxiv_papers RETURN p._key"
        for idx, node_id in enumerate(tqdm(self.db.aql.execute(query, batch_size=10000), 
                                          desc="Indexing nodes")):
            node_to_idx[node_id] = idx
        
        logger.info(f"Indexed {len(node_to_idx):,} nodes")
        
        # Stream edges from each collection
        edge_collections = ['coauthorship', 'same_field', 'temporal_proximity', 'citations']
        
        for coll_name in edge_collections:
            if coll_name not in [c['name'] for c in self.db.collections()]:
                continue
            
            logger.info(f"Processing {coll_name} edges...")
            
            query = f"""
            FOR e IN {coll_name}
                RETURN {{
                    from: PARSE_IDENTIFIER(e._from).key,
                    to: PARSE_IDENTIFIER(e._to).key
                }}
            """
            
            batch_edges = []
            for edge in tqdm(self.db.aql.execute(query, batch_size=batch_size), 
                           desc=f"Loading {coll_name}"):
                if edge['from'] in node_to_idx and edge['to'] in node_to_idx:
                    batch_edges.append([node_to_idx[edge['from']], node_to_idx[edge['to']]])
                    
                    if len(batch_edges) >= batch_size:
                        edge_list.extend(batch_edges)
                        batch_edges = []
                        
                        if len(edge_list) > 10000000:  # Sample for very large graphs
                            # Randomly sample edges to keep memory manageable
                            indices = np.random.choice(len(edge_list), 5000000, replace=False)
                            edge_list = [edge_list[i] for i in indices]
                            logger.info(f"Sampled down to {len(edge_list):,} edges")
            
            edge_list.extend(batch_edges)
        
        logger.info(f"Total edges loaded: {len(edge_list):,}")
        
        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return edge_index, node_to_idx
    
    def create_random_features(self, num_nodes: int, feature_dim: int = 128) -> torch.Tensor:
        """Create random initial features for nodes."""
        logger.info(f"Creating random features ({num_nodes} x {feature_dim})...")
        
        # Use smaller chunks to avoid memory issues
        chunk_size = 100000
        features = []
        
        for i in range(0, num_nodes, chunk_size):
            chunk_features = torch.randn(min(chunk_size, num_nodes - i), feature_dim) * 0.1
            features.append(chunk_features)
        
        return torch.cat(features, dim=0)


class GraphSAGEMassive(nn.Module):
    """GraphSAGE for massive graphs with memory-efficient operations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.5):
        super(GraphSAGEMassive, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Use checkpoint to save memory during backprop
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        return x


def train_minibatch(model, train_loader, optimizer, device):
    """Train using mini-batch sampling."""
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_examples = 0
    
    for batch in tqdm(train_loader, desc="Training batches"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index)
        
        # Only compute loss on the target nodes (center of subgraph)
        mask = batch.train_mask if hasattr(batch, 'train_mask') else batch.batch_size
        
        if isinstance(mask, int):
            # NeighborLoader returns batch_size for target nodes
            loss = F.cross_entropy(out[:mask], batch.y[:mask])
            pred = out[:mask].argmax(dim=1)
            correct = (pred == batch.y[:mask]).sum()
            examples = mask
        else:
            loss = F.cross_entropy(out[mask], batch.y[mask])
            pred = out[mask].argmax(dim=1)
            correct = (pred == batch.y[mask]).sum()
            examples = mask.sum()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * examples
        total_correct += correct
        total_examples += examples
    
    return total_loss / total_examples, total_correct / total_examples


@click.command()
@click.option('--feature-dim', default=128, help='Feature dimension (smaller for massive graphs)')
@click.option('--hidden-dim', default=64, help='Hidden dimension')
@click.option('--num-layers', default=2, help='Number of GraphSAGE layers')
@click.option('--batch-size', default=1024, help='Batch size for neighbor sampling')
@click.option('--num-neighbors', default='10,10', help='Number of neighbors per layer')
@click.option('--epochs', default=10, help='Number of epochs (fewer for massive graphs)')
@click.option('--lr', default=0.01, help='Learning rate')
@click.option('--num-workers', default=4, help='Number of data loading workers')
@click.option('--sample-edges', type=int, help='Sample this many edges (for testing)')
def main(feature_dim, hidden_dim, num_layers, batch_size, num_neighbors, 
         epochs, lr, num_workers, sample_edges):
    """Train GraphSAGE on massive academic graph using mini-batch sampling."""
    
    loader = MassiveGraphDataLoader()
    
    # Get graph statistics
    stats = loader.load_graph_metadata()
    
    if stats['num_edges'] > 100_000_000:
        logger.warning(f"Graph has {stats['num_edges']:,} edges - using sampling strategies")
    
    # Load edge index with streaming
    edge_index, node_to_idx = loader.create_edge_index_streaming()
    
    num_nodes = len(node_to_idx)
    
    # Create features (random initialization for massive graphs)
    x = loader.create_random_features(num_nodes, feature_dim)
    
    # Create random labels for unsupervised or self-supervised learning
    # In practice, you'd load actual labels or use link prediction
    y = torch.randint(0, 10, (num_nodes,))
    
    # Create masks
    n_train = int(0.6 * num_nodes)
    n_val = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    # Parse neighbor sampling
    num_neighbors_list = [int(x) for x in num_neighbors.split(',')]
    
    # Create neighbor loader for mini-batch training
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors_list,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors_list,
        batch_size=batch_size,
        input_nodes=data.val_mask,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info(f"Created data loaders with batch size {batch_size}")
    logger.info(f"Neighbor sampling: {num_neighbors_list}")
    
    # Initialize model
    model = GraphSAGEMassive(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=10,  # Adjust based on task
        num_layers=num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("Starting training on massive graph...")
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_minibatch(model, train_loader, optimizer, device)
        
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, "
                   f"Acc={train_acc:.4f}, Time={epoch_time:.1f}s")
        
        # Estimate time for full training
        if epoch == 0:
            estimated_total = epoch_time * epochs
            logger.info(f"Estimated total training time: {estimated_total/60:.1f} minutes")
    
    logger.info("Training complete!")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_nodes': num_nodes,
        'feature_dim': feature_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers
    }, 'graphsage_massive_model.pt')
    
    logger.info("Model saved to graphsage_massive_model.pt")


if __name__ == '__main__':
    main()