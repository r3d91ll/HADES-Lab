#!/usr/bin/env python3
"""
Module 2: Node Feature Loader
Load and prepare node features for the exported graph.
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
from arango import ArangoClient
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import click

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NodeFeatureLoader:
    """Load and prepare node features for GNN training."""
    
    def __init__(self, graph_path: str, db_name: str = 'academy_store'):
        """
        Initialize with exported graph.
        
        Args:
            graph_path: Path to exported graph JSON
            db_name: ArangoDB database name
        """
        # Load graph structure
        logger.info(f"Loading graph from {graph_path}")
        with open(graph_path, 'r') as f:
            self.graph_data = json.load(f)
        
        self.node_ids = self.graph_data['node_ids']
        self.node_to_idx = self.graph_data['node_to_idx']
        self.num_nodes = self.graph_data['num_nodes']
        
        logger.info(f"Loaded graph with {self.num_nodes:,} nodes")
        
        # Connect to database for feature loading
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            db_name,
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
    
    def load_embeddings(self, embedding_dim: int = 1024) -> np.ndarray:
        """
        Load pre-computed embeddings from database.
        
        Returns:
            features: (num_nodes, embedding_dim) array
        """
        logger.info(f"Loading embeddings for {self.num_nodes:,} nodes...")
        
        features = np.zeros((self.num_nodes, embedding_dim), dtype=np.float32)
        found_count = 0
        
        # Load in batches
        batch_size = 1000
        for i in tqdm(range(0, len(self.node_ids), batch_size), desc="Loading embeddings"):
            batch_ids = self.node_ids[i:i+batch_size]
            
            query = """
            FOR paper_id IN @paper_ids
                LET embedding = FIRST(
                    FOR e IN arxiv_embeddings
                        FILTER e.paper_id == paper_id
                        LIMIT 1
                        RETURN e.embedding
                )
                RETURN {
                    id: paper_id,
                    embedding: embedding
                }
            """
            
            for result in self.db.aql.execute(query, bind_vars={'paper_ids': batch_ids}):
                if result['embedding']:
                    idx = self.node_to_idx[result['id']]
                    features[idx] = result['embedding'][:embedding_dim]
                    found_count += 1
        
        logger.info(f"Found embeddings for {found_count:,}/{self.num_nodes:,} nodes")
        
        # Fill missing with random initialization
        missing_mask = np.all(features == 0, axis=1)
        num_missing = missing_mask.sum()
        if num_missing > 0:
            logger.info(f"Initializing {num_missing:,} missing embeddings randomly")
            features[missing_mask] = np.random.randn(num_missing, embedding_dim) * 0.01
        
        return features
    
    def load_metadata_features(self, use_categories: bool = True, 
                              use_temporal: bool = True,
                              use_text: bool = False) -> np.ndarray:
        """
        Create features from paper metadata.
        
        Args:
            use_categories: Include one-hot encoded categories
            use_temporal: Include temporal features (year, month)
            use_text: Include title/abstract features (requires embeddings)
        
        Returns:
            features: (num_nodes, feature_dim) array
        """
        logger.info("Loading metadata features...")
        
        feature_components = []
        
        # Get metadata for all nodes
        query = """
        FOR p IN arxiv_papers
            FILTER p._key IN @node_ids
            RETURN {
                id: p._key,
                categories: p.categories,
                update_date: p.update_date,
                title_length: LENGTH(p.title),
                abstract_length: LENGTH(p.abstract),
                num_authors: LENGTH(p.authors)
            }
        """
        
        metadata = {doc['id']: doc for doc in self.db.aql.execute(
            query, bind_vars={'node_ids': self.node_ids}
        )}
        
        # Category features
        if use_categories:
            logger.info("Creating category features...")
            
            # Get all unique categories
            all_categories = set()
            for doc in metadata.values():
                if doc.get('categories'):
                    all_categories.update(doc['categories'])
            
            # Use top K categories
            category_counts = {}
            for doc in metadata.values():
                if doc.get('categories'):
                    for cat in doc['categories']:
                        category_counts[cat] = category_counts.get(cat, 0) + 1
            
            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:50]
            top_cat_list = [cat for cat, _ in top_categories]
            cat_to_idx = {cat: idx for idx, cat in enumerate(top_cat_list)}
            
            # Create one-hot encoding
            cat_features = np.zeros((self.num_nodes, len(top_cat_list)))
            
            for node_id in self.node_ids:
                idx = self.node_to_idx[node_id]
                if node_id in metadata and metadata[node_id].get('categories'):
                    for cat in metadata[node_id]['categories']:
                        if cat in cat_to_idx:
                            cat_features[idx, cat_to_idx[cat]] = 1
            
            feature_components.append(cat_features)
            logger.info(f"  Added {len(top_cat_list)} category features")
        
        # Temporal features
        if use_temporal:
            logger.info("Creating temporal features...")
            
            temporal_features = np.zeros((self.num_nodes, 13))  # year + 12 months
            
            for node_id in self.node_ids:
                idx = self.node_to_idx[node_id]
                if node_id in metadata and metadata[node_id].get('update_date'):
                    date = metadata[node_id]['update_date']
                    try:
                        year = int(date[:4])
                        month = int(date[5:7])
                        
                        # Normalize year (papers from 1990-2025)
                        temporal_features[idx, 0] = (year - 2000) / 25.0
                        # One-hot month
                        temporal_features[idx, month] = 1.0
                    except:
                        pass
            
            feature_components.append(temporal_features)
            logger.info("  Added 13 temporal features")
        
        # Statistical features
        logger.info("Creating statistical features...")
        stat_features = np.zeros((self.num_nodes, 3))
        
        for node_id in self.node_ids:
            idx = self.node_to_idx[node_id]
            if node_id in metadata:
                doc = metadata[node_id]
                stat_features[idx, 0] = np.log1p(doc.get('title_length', 0)) / 10.0
                stat_features[idx, 1] = np.log1p(doc.get('abstract_length', 0)) / 10.0
                stat_features[idx, 2] = np.log1p(doc.get('num_authors', 1)) / 3.0
        
        feature_components.append(stat_features)
        logger.info("  Added 3 statistical features")
        
        # Concatenate all features
        features = np.hstack(feature_components)
        logger.info(f"Created feature matrix: {features.shape}")
        
        return features
    
    def load_hybrid_features(self, embedding_dim: int = 256) -> np.ndarray:
        """
        Load hybrid features combining embeddings and metadata.
        
        Args:
            embedding_dim: Dimension to reduce embeddings to
        
        Returns:
            features: Combined feature matrix
        """
        logger.info("Loading hybrid features...")
        
        # Load embeddings
        embeddings = self.load_embeddings(1024)
        
        # Reduce dimensionality
        if embedding_dim < 1024:
            logger.info(f"Reducing embeddings from 1024 to {embedding_dim} dimensions...")
            pca = PCA(n_components=embedding_dim, random_state=42)
            embeddings = pca.fit_transform(embeddings)
            logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Load metadata
        metadata_features = self.load_metadata_features()
        
        # Combine
        features = np.hstack([embeddings, metadata_features])
        
        # Normalize
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        logger.info(f"Created hybrid features: {features.shape}")
        
        return features
    
    def load_labels(self, task: str = 'category') -> Tuple[np.ndarray, Dict]:
        """
        Load labels for supervised training.
        
        Args:
            task: Type of labels ('category', 'year', 'field')
        
        Returns:
            labels: (num_nodes,) array of labels
            label_map: Dict mapping label indices to names
        """
        logger.info(f"Loading labels for task: {task}")
        
        query = """
        FOR p IN arxiv_papers
            FILTER p._key IN @node_ids
            RETURN {
                id: p._key,
                categories: p.categories,
                year: SUBSTRING(p.update_date, 0, 4)
            }
        """
        
        metadata = {doc['id']: doc for doc in self.db.aql.execute(
            query, bind_vars={'node_ids': self.node_ids}
        )}
        
        labels = []
        
        if task == 'category':
            # Use primary category as label
            for node_id in self.node_ids:
                if node_id in metadata and metadata[node_id].get('categories'):
                    primary_cat = metadata[node_id]['categories'][0]
                    labels.append(primary_cat)
                else:
                    labels.append('unknown')
        
        elif task == 'year':
            # Use publication year as label
            for node_id in self.node_ids:
                if node_id in metadata and metadata[node_id].get('year'):
                    labels.append(metadata[node_id]['year'])
                else:
                    labels.append('unknown')
        
        elif task == 'field':
            # Use broad field (physics, cs, math, etc.)
            for node_id in self.node_ids:
                if node_id in metadata and metadata[node_id].get('categories'):
                    cat = metadata[node_id]['categories'][0]
                    field = cat.split('.')[0] if '.' in cat else cat.split('-')[0]
                    labels.append(field)
                else:
                    labels.append('unknown')
        
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        label_map = {idx: label for idx, label in enumerate(le.classes_)}
        
        logger.info(f"Created {len(label_map)} label classes")
        
        return encoded_labels, label_map
    
    def save_features(self, features: np.ndarray, labels: Optional[np.ndarray] = None,
                      output_path: str = 'features.npz'):
        """
        Save features and labels for training.
        
        Args:
            features: Node feature matrix
            labels: Optional label array
            output_path: Path to save features
        """
        logger.info(f"Saving features to {output_path}")
        
        save_dict = {
            'features': features,
            'node_ids': self.node_ids,
            'node_to_idx': self.node_to_idx
        }
        
        # Add edge_index if available
        if 'edge_index' in self.graph_data:
            save_dict['edge_index'] = self.graph_data['edge_index']
        elif 'adjacency' in self.graph_data:
            save_dict['adjacency'] = self.graph_data['adjacency']
        
        if labels is not None:
            save_dict['labels'] = labels
        
        np.savez_compressed(output_path, **save_dict)
        
        logger.info(f"Saved features: {features.shape}")
        if labels is not None:
            logger.info(f"Saved labels: {labels.shape}")


@click.command()
@click.option('--graph', required=True, help='Path to exported graph JSON')
@click.option('--feature-type', type=click.Choice(['embeddings', 'metadata', 'hybrid']),
              default='hybrid', help='Type of features to load')
@click.option('--output', default='features.npz', help='Output path for features')
@click.option('--task', type=click.Choice(['category', 'year', 'field']),
              help='Task for labels')
def main(graph, feature_type, output, task):
    """Load node features for GNN training."""
    
    loader = NodeFeatureLoader(graph)
    
    # Load features based on type
    if feature_type == 'embeddings':
        features = loader.load_embeddings()
    elif feature_type == 'metadata':
        features = loader.load_metadata_features()
    elif feature_type == 'hybrid':
        features = loader.load_hybrid_features()
    
    # Load labels if task specified
    labels = None
    if task:
        labels, label_map = loader.load_labels(task)
        logger.info(f"Label distribution: {np.bincount(labels)[:10]}")
    
    # Save
    loader.save_features(features, labels, output)
    
    logger.info("Feature loading complete!")


if __name__ == '__main__':
    main()