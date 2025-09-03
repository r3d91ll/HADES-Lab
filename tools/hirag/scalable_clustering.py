#!/usr/bin/env python3
"""
Scalable HiRAG Hierarchical Clustering
Memory-efficient clustering for large-scale entity datasets
Handles 243,405 entities without running out of memory
"""

import os
import sys
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timezone
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from arango import ArangoClient
from arango.database import StandardDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScalableHiRAGClustering:
    """
    Memory-efficient hierarchical clustering for large entity collections.
    
    Uses MiniBatch K-Means for scalability instead of HDBSCAN which requires
    O(n²) memory for distance matrices. This approach can handle hundreds of
    thousands of entities within reasonable memory constraints.
    """
    
    def __init__(self, host: str = "192.168.1.69", port: int = 8529, 
                 username: str = "root", password: str = None):
        """Initialize the scalable clustering system."""
        self.password = password or os.getenv('ARANGO_PASSWORD', 'root_password')
        self.client = ArangoClient(hosts=f"http://{host}:{port}")
        self.db: Optional[StandardDatabase] = None
        
        # Scalable clustering parameters
        self.batch_size = 10000  # Process entities in batches
        self.n_l1_clusters = 200  # Target Level 1 clusters (manageable size)
        self.embedding_dim = 500   # Reduced TF-IDF dimensions
        self.svd_components = 100  # SVD dimensionality reduction
        
    async def connect(self, database_name: str = "academy_store") -> bool:
        """Connect to ArangoDB."""
        try:
            self.db = self.client.db(database_name, username="root", password=self.password)
            logger.info(f"Connected to database: {database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def load_entities_for_scalable_clustering(self) -> List[Dict]:
        """Load entities with sampling for large datasets."""
        try:
            # First, get total count
            count_query = "RETURN LENGTH(FOR entity IN entities FILTER entity.layer == 0 RETURN 1)"
            total_entities = list(self.db.aql.execute(count_query))[0]
            logger.info(f"Total entities to cluster: {total_entities}")
            
            # Load all entities (we'll process in batches)
            query = """
            FOR entity IN entities
                FILTER entity.layer == 0
                RETURN {
                    _key: entity._key,
                    name: entity.name,
                    type: entity.type,
                    description: entity.description,
                    frequency: entity.frequency
                }
            """
            
            cursor = self.db.aql.execute(query)
            entities = list(cursor)
            logger.info(f"Loaded {len(entities)} entities for clustering")
            return entities
            
        except Exception as e:
            logger.error(f"Failed to load entities: {e}")
            return []
    
    def generate_scalable_embeddings(self, entities: List[Dict]) -> Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]:
        """Generate memory-efficient embeddings with dimensionality reduction."""
        # Prepare texts
        texts = []
        for entity in entities:
            text_parts = [
                entity['name'],
                entity['type'],
                entity.get('description', ''),
                f"freq_{entity.get('frequency', 1)}"
            ]
            texts.append(" ".join(text_parts))
        
        # Create TF-IDF with reduced dimensions
        logger.info("Creating TF-IDF embeddings...")
        vectorizer = TfidfVectorizer(
            max_features=self.embedding_dim,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF shape: {tfidf_matrix.shape}")
        
        # Apply SVD for further dimensionality reduction
        logger.info("Applying SVD dimensionality reduction...")
        svd = TruncatedSVD(n_components=self.svd_components, random_state=42)
        reduced_embeddings = svd.fit_transform(tfidf_matrix)
        
        logger.info(f"Final embeddings shape: {reduced_embeddings.shape}")
        logger.info(f"SVD explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
        
        return reduced_embeddings, vectorizer, svd
    
    def create_scalable_level1_clusters(self, entities: List[Dict], embeddings: np.ndarray) -> List[Dict]:
        """Create Level 1 clusters using MiniBatch K-Means for scalability."""
        logger.info(f"Creating {self.n_l1_clusters} Level 1 clusters using MiniBatch K-Means...")
        
        # Use MiniBatch K-Means for memory efficiency
        clusterer = MiniBatchKMeans(
            n_clusters=self.n_l1_clusters,
            batch_size=self.batch_size,
            random_state=42,
            n_init=3,
            max_iter=100
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        logger.info("Clustering completed")
        
        # Create cluster objects
        clusters = []
        for cluster_id in range(self.n_l1_clusters):
            member_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(member_indices) < 3:  # Skip very small clusters
                continue
                
            member_entities = [entities[i] for i in member_indices]
            
            # Generate cluster metadata
            cluster_name = self._generate_cluster_name(member_entities)
            cluster_summary = self._generate_cluster_summary(member_entities)
            key_concepts = self._extract_key_concepts(member_entities)
            
            # Calculate cohesion using cluster center distance
            cluster_center = clusterer.cluster_centers_[cluster_id]
            member_embeddings = embeddings[member_indices]
            distances = np.linalg.norm(member_embeddings - cluster_center, axis=1)
            cohesion = 1.0 / (1.0 + np.mean(distances))  # Convert distance to similarity
            
            cluster_dict = {
                "_key": self._generate_cluster_id(cluster_name, 1),
                "name": cluster_name,
                "layer": 1,
                "members": [entities[i]['_key'] for i in member_indices],
                "member_count": len(member_indices),
                "summary": cluster_summary,
                "key_concepts": key_concepts,
                "cohesion_score": float(cohesion),
                "parent_cluster": None,
                "created": datetime.now(timezone.utc).isoformat(),
                "clustering_algorithm": "minibatch_kmeans",
                "parameters": {
                    "n_clusters": self.n_l1_clusters,
                    "batch_size": self.batch_size
                }
            }
            clusters.append(cluster_dict)
        
        logger.info(f"Created {len(clusters)} valid Level 1 clusters")
        return clusters
    
    def create_scalable_level2_clusters(self, level1_clusters: List[Dict]) -> List[Dict]:
        """Create Level 2 super-clusters from Level 1 cluster summaries."""
        if len(level1_clusters) < 4:
            logger.warning("Insufficient Level 1 clusters for Level 2")
            return []
        
        logger.info("Creating Level 2 super-clusters...")
        
        # Generate embeddings from L1 summaries
        cluster_texts = []
        for cluster in level1_clusters:
            text = f"{cluster['name']} {cluster['summary']} {' '.join(cluster['key_concepts'])}"
            cluster_texts.append(text)
        
        # Simpler TF-IDF for cluster summaries
        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        summary_embeddings = vectorizer.fit_transform(cluster_texts).toarray()
        
        # Use smaller number of L2 clusters
        n_l2_clusters = max(3, min(15, int(np.sqrt(len(level1_clusters)))))
        
        l2_clusterer = MiniBatchKMeans(
            n_clusters=n_l2_clusters,
            batch_size=min(1000, len(level1_clusters)),
            random_state=42
        )
        
        l2_labels = l2_clusterer.fit_predict(summary_embeddings)
        
        # Create Level 2 clusters
        level2_clusters = []
        for l2_id in range(n_l2_clusters):
            l1_indices = np.where(l2_labels == l2_id)[0]
            
            if len(l1_indices) < 2:
                continue
                
            member_l1_clusters = [level1_clusters[i] for i in l1_indices]
            
            # Generate L2 metadata
            super_name = self._generate_super_cluster_name(member_l1_clusters)
            super_summary = self._generate_super_cluster_summary(member_l1_clusters)
            super_concepts = self._extract_super_concepts(member_l1_clusters)
            
            # Calculate L2 cohesion
            l2_center = l2_clusterer.cluster_centers_[l2_id]
            l1_embeddings = summary_embeddings[l1_indices]
            l2_distances = np.linalg.norm(l1_embeddings - l2_center, axis=1)
            l2_cohesion = 1.0 / (1.0 + np.mean(l2_distances))
            
            l2_cluster = {
                "_key": self._generate_cluster_id(super_name, 2),
                "name": super_name,
                "layer": 2,
                "members": [cluster['_key'] for cluster in member_l1_clusters],
                "member_count": len(member_l1_clusters),
                "summary": super_summary,
                "key_concepts": super_concepts,
                "cohesion_score": float(l2_cohesion),
                "parent_cluster": None,
                "created": datetime.now(timezone.utc).isoformat(),
                "clustering_algorithm": "minibatch_kmeans_l2",
                "parameters": {
                    "n_clusters": n_l2_clusters
                }
            }
            level2_clusters.append(l2_cluster)
            
            # Update L1 parent references
            for l1_cluster in member_l1_clusters:
                l1_cluster["parent_cluster"] = l2_cluster["_key"]
        
        logger.info(f"Created {len(level2_clusters)} Level 2 super-clusters")
        return level2_clusters
    
    def _generate_cluster_id(self, name: str, layer: int) -> str:
        """Generate cluster ID."""
        content = f"cluster_L{layer}_{name.lower().replace(' ', '_')}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_cluster_name(self, entities: List[Dict]) -> str:
        """Generate cluster name from most common entity types and names."""
        # Get entity types
        types = [e['type'] for e in entities]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        dominant_type = max(type_counts.keys(), key=type_counts.get)
        
        # Get most frequent entities
        entity_names = [e['name'].lower() for e in entities]
        name_counts = {}
        for name in entity_names:
            words = name.split()[:2]  # First 2 words
            for word in words:
                if len(word) > 2:
                    name_counts[word] = name_counts.get(word, 0) + 1
        
        top_words = sorted(name_counts.keys(), key=name_counts.get, reverse=True)[:2]
        
        if top_words:
            return f"{' '.join(top_words).title()} {dominant_type.title()}s"
        else:
            return f"{dominant_type.title()} Cluster"
    
    def _generate_cluster_summary(self, entities: List[Dict]) -> str:
        """Generate cluster summary."""
        total = len(entities)
        types = {}
        for e in entities:
            types[e['type']] = types.get(e['type'], 0) + 1
        
        type_parts = [f"{count} {t}{'s' if count > 1 else ''}" for t, count in types.items()]
        summary = f"Cluster of {total} entities: {', '.join(type_parts)}. "
        
        # Add top entities by frequency
        top_entities = sorted(entities, key=lambda x: x.get('frequency', 1), reverse=True)[:3]
        top_names = [e['name'] for e in top_entities]
        summary += f"Top entities: {', '.join(top_names)}."
        
        return summary
    
    def _extract_key_concepts(self, entities: List[Dict]) -> List[str]:
        """Extract key concepts from entity names."""
        words = []
        for entity in entities:
            words.extend(entity['name'].lower().split())
        
        # Filter and count
        filtered = [w for w in words if len(w) > 3 and w.isalpha()]
        word_counts = {}
        for word in filtered:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        return sorted(word_counts.keys(), key=word_counts.get, reverse=True)[:5]
    
    def _generate_super_cluster_name(self, l1_clusters: List[Dict]) -> str:
        """Generate super-cluster name."""
        all_concepts = []
        for cluster in l1_clusters:
            all_concepts.extend(cluster['key_concepts'])
        
        concept_counts = {}
        for concept in all_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        top_concepts = sorted(concept_counts.keys(), key=concept_counts.get, reverse=True)[:2]
        
        if top_concepts:
            return f"{' '.join(top_concepts).title()} Domain"
        else:
            return f"Domain of {len(l1_clusters)} Clusters"
    
    def _generate_super_cluster_summary(self, l1_clusters: List[Dict]) -> str:
        """Generate super-cluster summary."""
        total_clusters = len(l1_clusters)
        total_entities = sum(cluster['member_count'] for cluster in l1_clusters)
        cluster_names = [cluster['name'] for cluster in l1_clusters]
        
        summary = f"Super-cluster containing {total_clusters} topic clusters with {total_entities} total entities. "
        summary += f"Includes: {', '.join(cluster_names[:3])}{'...' if len(cluster_names) > 3 else ''}."
        
        return summary
    
    def _extract_super_concepts(self, l1_clusters: List[Dict]) -> List[str]:
        """Extract concepts for super-cluster."""
        all_concepts = []
        for cluster in l1_clusters:
            all_concepts.extend(cluster['key_concepts'])
        
        concept_counts = {}
        for concept in all_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        return sorted(concept_counts.keys(), key=concept_counts.get, reverse=True)[:7]
    
    async def store_scalable_clusters(self, all_clusters: List[Dict]) -> bool:
        """Store clusters in database."""
        try:
            clusters_collection = self.db.collection("clusters")
            
            # Clear existing clusters
            clusters_collection.truncate()
            logger.info("Cleared existing clusters")
            
            # Insert new clusters
            batch_size = 1000
            success_count = 0
            
            for i in range(0, len(all_clusters), batch_size):
                batch = all_clusters[i:i + batch_size]
                result = clusters_collection.insert_many(batch, overwrite=True)
                success_count += len([r for r in result if not r.get('error')])
            
            logger.info(f"✅ Stored {success_count} clusters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store clusters: {e}")
            return False
    
    async def create_scalable_cluster_edges(self, level1_clusters: List[Dict], level2_clusters: List[Dict]) -> bool:
        """Create cluster edges efficiently."""
        try:
            cluster_edges_collection = self.db.collection("cluster_edges")
            
            # Clear existing edges
            cluster_edges_collection.truncate()
            logger.info("Cleared existing cluster edges")
            
            edges = []
            
            # Entity → L1 cluster edges
            for cluster in level1_clusters:
                for entity_id in cluster['members']:
                    edges.append({
                        "_from": f"entities/{entity_id}",
                        "_to": f"clusters/{cluster['_key']}",
                        "type": "member_of",
                        "membership_score": 0.9,
                        "created": datetime.now(timezone.utc).isoformat(),
                        "clustering_round": 1
                    })
            
            # L1 → L2 cluster edges
            for l2_cluster in level2_clusters:
                for l1_cluster_id in l2_cluster['members']:
                    edges.append({
                        "_from": f"clusters/{l1_cluster_id}",
                        "_to": f"clusters/{l2_cluster['_key']}",
                        "type": "subcuster_of",
                        "aggregation_weight": 0.8,
                        "created": datetime.now(timezone.utc).isoformat(),
                        "clustering_round": 2
                    })
            
            # Insert edges in batches
            batch_size = 5000
            success_count = 0
            
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                result = cluster_edges_collection.insert_many(batch, overwrite=True)
                success_count += len([r for r in result if not r.get('error')])
                
                if i % 25000 == 0:  # Progress updates
                    logger.info(f"Inserted {i + len(batch)} / {len(edges)} edges")
            
            logger.info(f"✅ Created {success_count} cluster edges")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create cluster edges: {e}")
            return False
    
    async def build_scalable_hierarchical_clusters(self) -> bool:
        """Execute scalable clustering pipeline."""
        logger.info("Starting scalable hierarchical clustering...")
        
        # Load entities
        entities = await self.load_entities_for_scalable_clustering()
        if not entities:
            return False
        
        # Generate embeddings with dimensionality reduction
        embeddings, vectorizer, svd = self.generate_scalable_embeddings(entities)
        
        # Create clusters
        level1_clusters = self.create_scalable_level1_clusters(entities, embeddings)
        level2_clusters = self.create_scalable_level2_clusters(level1_clusters)
        
        # Store everything
        all_clusters = level1_clusters + level2_clusters
        if not await self.store_scalable_clusters(all_clusters):
            return False
        
        if not await self.create_scalable_cluster_edges(level1_clusters, level2_clusters):
            return False
        
        logger.info("✅ Scalable hierarchical clustering completed!")
        logger.info(f"Created {len(level1_clusters)} L1 clusters, {len(level2_clusters)} L2 clusters")
        
        return True


async def main():
    """Main execution function."""
    password = os.getenv('ARANGO_PASSWORD')
    if not password:
        print("Please set ARANGO_PASSWORD environment variable")
        return
    
    clustering = ScalableHiRAGClustering(password=password)
    
    if not await clustering.connect():
        return
    
    success = await clustering.build_scalable_hierarchical_clusters()
    
    if success:
        print("\n🎉 Scalable HiRAG clustering completed!")
        print("Ready for three-level retrieval implementation!")
    else:
        print("\n❌ Scalable clustering failed!")


if __name__ == "__main__":
    asyncio.run(main())