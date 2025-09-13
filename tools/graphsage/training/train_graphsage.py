#!/usr/bin/env python3
"""
Train GraphSAGE on the academic graph for node classification or link prediction.
Implements the Conveyance Framework: C = (W·R·H/T)·Ctx^α
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
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, NeighborSampler
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from arango import ArangoClient
import click

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class GraphDataLoader:
    """Load and prepare graph data from ArangoDB."""
    
    def __init__(self, db_name: str = 'academy_store'):
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            db_name,
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
    def load_graph_from_json(self, graph_path: str) -> Dict:
        """Load pre-exported graph from JSON."""
        logger.info(f"Loading graph from {graph_path}...")
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        logger.info(f"Loaded {graph_data['num_nodes']:,} nodes and {graph_data['num_edges']:,} edges")
        return graph_data
    
    def load_graph_from_db(self, sample_size: Optional[int] = None) -> Dict:
        """Load graph directly from ArangoDB."""
        logger.info("Loading graph from database...")
        
        # Get nodes
        node_query = """
        FOR p IN arxiv_papers
            %s
            RETURN {
                id: p._key,
                title: p.title,
                categories: p.categories,
                year: SUBSTRING(p.update_date, 0, 4)
            }
        """ % (f"LIMIT {sample_size}" if sample_size else "")
        
        nodes = list(self.db.aql.execute(node_query))
        node_to_idx = {node['id']: idx for idx, node in enumerate(nodes)}
        
        # Get edges from all collections
        edge_collections = ['coauthorship', 'same_field', 'temporal_proximity', 'citations']
        all_edges = []
        
        for coll_name in edge_collections:
            if coll_name not in [c['name'] for c in self.db.collections()]:
                continue
                
            edge_query = f"""
            FOR e IN {coll_name}
                LET from_key = PARSE_IDENTIFIER(e._from).key
                LET to_key = PARSE_IDENTIFIER(e._to).key
                FILTER from_key IN @node_ids AND to_key IN @node_ids
                RETURN {{
                    from: from_key,
                    to: to_key,
                    type: '{coll_name}',
                    weight: e.weight
                }}
            """
            
            node_ids = [n['id'] for n in nodes]
            edges = list(self.db.aql.execute(edge_query, bind_vars={'node_ids': node_ids}))
            all_edges.extend(edges)
            logger.info(f"  {coll_name}: {len(edges):,} edges")
        
        # Build adjacency
        adjacency = {i: [] for i in range(len(nodes))}
        for edge in all_edges:
            if edge['from'] in node_to_idx and edge['to'] in node_to_idx:
                from_idx = node_to_idx[edge['from']]
                to_idx = node_to_idx[edge['to']]
                adjacency[from_idx].append(to_idx)
                adjacency[to_idx].append(from_idx)  # Undirected
        
        graph_data = {
            'num_nodes': len(nodes),
            'num_edges': len(all_edges),
            'nodes': nodes,
            'node_to_idx': node_to_idx,
            'adjacency': adjacency,
            'edges': all_edges
        }
        
        logger.info(f"Loaded {len(nodes):,} nodes and {len(all_edges):,} edges")
        return graph_data
    
    def load_node_features(self, node_ids: List[str], feature_dim: int = 1024) -> np.ndarray:
        """Load node features from embeddings or create from metadata."""
        logger.info(f"Loading features for {len(node_ids):,} nodes...")
        
        # Try to load embeddings
        embedding_query = """
        FOR p IN @node_ids
            LET embedding = FIRST(
                FOR e IN arxiv_embeddings
                    FILTER e.paper_id == p
                    LIMIT 1
                    RETURN e.embedding
            )
            RETURN {
                id: p,
                embedding: embedding
            }
        """
        
        embeddings = {}
        for result in self.db.aql.execute(embedding_query, bind_vars={'node_ids': node_ids}):
            if result['embedding']:
                embeddings[result['id']] = result['embedding']
        
        logger.info(f"Found embeddings for {len(embeddings):,} nodes")
        
        # Create feature matrix
        features = np.zeros((len(node_ids), feature_dim))
        
        for idx, node_id in enumerate(node_ids):
            if node_id in embeddings:
                # Use actual embedding
                features[idx] = embeddings[node_id][:feature_dim]
            else:
                # Create random features (or could use metadata)
                features[idx] = np.random.randn(feature_dim) * 0.1
        
        return features
    
    def create_category_labels(self, nodes: List[Dict]) -> Tuple[np.ndarray, Dict[str, int]]:
        """Create labels from paper categories for classification."""
        logger.info("Creating category labels...")
        
        # Find most common categories
        category_counts = {}
        for node in nodes:
            if node.get('categories'):
                for cat in node['categories']:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Select top categories as labels
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        category_to_label = {cat: idx for idx, (cat, _) in enumerate(top_categories)}
        category_to_label['other'] = len(category_to_label)
        
        # Assign labels
        labels = np.zeros(len(nodes), dtype=np.long)
        for idx, node in enumerate(nodes):
            if node.get('categories'):
                # Use first category that's in our label set
                for cat in node['categories']:
                    if cat in category_to_label:
                        labels[idx] = category_to_label[cat]
                        break
                else:
                    labels[idx] = category_to_label['other']
            else:
                labels[idx] = category_to_label['other']
        
        logger.info(f"Created {len(category_to_label)} label classes")
        return labels, category_to_label


class GraphSAGE(nn.Module):
    """
    GraphSAGE model implementing the Conveyance Framework.
    
    WHERE (R): Graph topology through neighbor aggregation
    WHAT (W): Node features from embeddings
    CONVEYANCE (H): Model capacity through hidden dimensions
    TIME (T): Training efficiency
    Context^α: Attention mechanisms and dropout for context quality
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.5):
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        # Batch normalization for stability
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])
        
    def forward(self, x, edge_index):
        """
        Forward pass implementing conveyance calculation.
        
        Args:
            x: Node features (WHAT dimension)
            edge_index: Graph connectivity (WHERE dimension)
        
        Returns:
            Node embeddings incorporating W·R·H
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def compute_conveyance(self, output, labels, ctx_alpha: float = 1.5):
        """
        Compute conveyance score: C = (W·R·H/T)·Ctx^α
        
        This measures the model's ability to convey information
        from graph structure to predictions.
        """
        with torch.no_grad():
            # W: Feature quality (cosine similarity of embeddings)
            W = F.cosine_similarity(output, output.mean(dim=0, keepdim=True)).mean()
            
            # R: Relational quality (clustering coefficient proxy)
            pred = output.argmax(dim=1)
            R = (pred == labels).float().mean()
            
            # H: Model capacity (parameter count)
            H = sum(p.numel() for p in self.parameters()) / 1e6  # In millions
            
            # Context quality (classification confidence)
            probs = F.softmax(output, dim=1)
            Ctx = probs.max(dim=1)[0].mean()
            
            # Conveyance calculation
            C = (W * R * H) * (Ctx ** ctx_alpha)
            
        return C.item(), {'W': W.item(), 'R': R.item(), 'H': H, 'Ctx': Ctx.item()}


def prepare_pytorch_geometric_data(graph_data: Dict, features: np.ndarray, 
                                  labels: np.ndarray) -> Data:
    """Convert graph data to PyTorch Geometric format."""
    logger.info("Preparing PyTorch Geometric data...")
    
    # Create edge index
    edge_list = []
    adjacency = graph_data.get('adjacency', {})
    
    for node_idx, neighbors in adjacency.items():
        for neighbor_idx in neighbors:
            edge_list.append([node_idx, neighbor_idx])
    
    if not edge_list:
        # Create a minimal connected graph if no edges
        logger.warning("No edges found, creating minimal connections")
        num_nodes = graph_data['num_nodes']
        for i in range(num_nodes - 1):
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Convert to tensors
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    
    # Create masks for train/val/test
    num_nodes = x.size(0)
    indices = np.arange(num_nodes)
    
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    logger.info(f"Data: {data}")
    logger.info(f"  Train nodes: {train_mask.sum().item():,}")
    logger.info(f"  Val nodes: {val_mask.sum().item():,}")
    logger.info(f"  Test nodes: {test_mask.sum().item():,}")
    
    return data


def train_epoch(model, data, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    pred = out[data.train_mask].argmax(dim=1)
    correct = (pred == data.y[data.train_mask]).sum()
    acc = correct.float() / data.train_mask.sum()
    
    return loss.item(), acc.item()


def evaluate(model, data, mask):
    """Evaluate model on validation or test set."""
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        correct = (pred == data.y[mask]).sum()
        acc = correct.float() / mask.sum()
        
        # Compute conveyance
        conveyance, components = model.compute_conveyance(out[mask], data.y[mask])
    
    return acc.item(), conveyance, components


@click.command()
@click.option('--graph-path', type=str, help='Path to pre-exported graph JSON')
@click.option('--sample', type=int, help='Sample size for testing')
@click.option('--hidden-dim', default=128, help='Hidden dimension size')
@click.option('--num-layers', default=2, help='Number of GraphSAGE layers')
@click.option('--epochs', default=200, help='Number of training epochs')
@click.option('--lr', default=0.01, help='Learning rate')
@click.option('--dropout', default=0.5, help='Dropout rate')
@click.option('--feature-dim', default=1024, help='Feature dimension')
@click.option('--save-model', type=str, help='Path to save trained model')
def main(graph_path, sample, hidden_dim, num_layers, epochs, lr, dropout, feature_dim, save_model):
    """Train GraphSAGE on academic graph."""
    
    # Load data
    loader = GraphDataLoader()
    
    if graph_path and os.path.exists(graph_path):
        graph_data = loader.load_graph_from_json(graph_path)
        node_ids = [graph_data['node_ids'][i] for i in range(graph_data['num_nodes'])]
        nodes = [{'id': nid} for nid in node_ids]  # Minimal node info
    else:
        graph_data = loader.load_graph_from_db(sample_size=sample)
        nodes = graph_data['nodes']
        node_ids = [n['id'] for n in nodes]
    
    # Load features
    features = loader.load_node_features(node_ids, feature_dim=feature_dim)
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Create labels
    labels, label_map = loader.create_category_labels(nodes)
    
    # Prepare PyTorch Geometric data
    data = prepare_pytorch_geometric_data(graph_data, features, labels)
    data = data.to(device)
    
    # Initialize model
    model = GraphSAGE(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=len(label_map),
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer, criterion)
        
        # Validate
        val_acc, val_conveyance, val_components = evaluate(model, data, data.val_mask)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d}: "
                       f"Loss={train_loss:.4f}, "
                       f"Train Acc={train_acc:.4f}, "
                       f"Val Acc={val_acc:.4f}, "
                       f"Conveyance={val_conveyance:.4f}")
            logger.info(f"  Components: W={val_components['W']:.3f}, "
                       f"R={val_components['R']:.3f}, "
                       f"H={val_components['H']:.1f}M, "
                       f"Ctx={val_components['Ctx']:.3f}")
    
    # Load best model and test
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    test_acc, test_conveyance, test_components = evaluate(model, data, data.test_mask)
    
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Conveyance: {test_conveyance:.4f}")
    logger.info(f"Final Components:")
    logger.info(f"  W (Feature Quality): {test_components['W']:.3f}")
    logger.info(f"  R (Relational Quality): {test_components['R']:.3f}")
    logger.info(f"  H (Model Capacity): {test_components['H']:.1f}M params")
    logger.info(f"  Ctx (Context Quality): {test_components['Ctx']:.3f}")
    
    # Save model if requested
    if save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_dim': feature_dim,
            'hidden_dim': hidden_dim,
            'output_dim': len(label_map),
            'num_layers': num_layers,
            'label_map': label_map,
            'test_accuracy': test_acc,
            'test_conveyance': test_conveyance
        }, save_model)
        logger.info(f"Model saved to {save_model}")


if __name__ == '__main__':
    main()