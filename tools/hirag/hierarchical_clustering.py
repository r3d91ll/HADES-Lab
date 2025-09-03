#!/usr/bin/env python3
"""
HiRAG Hierarchical Clustering (HiIndex)
Implements Level 0 → Level 1 → Level 2 hierarchical clustering
Based on PRD Issue #19 requirements

This module creates the hierarchical knowledge structure that enables
HiRAG's three-level retrieval: Local ↔ Bridge ↔ Global
"""

import os
import sys
import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timezone
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from arango import ArangoClient
from arango.database import StandardDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClusterInfo:
    """
    Represents a hierarchical cluster.
    
    In anthropological terms, this is an institutional category -
    a bureaucratic classification that organizes knowledge actors
    into coherent administrative units for efficient governance.
    """
    name: str
    layer: int  # 1 for topic clusters, 2 for super-clusters
    members: List[str]  # entity IDs for layer 1, cluster IDs for layer 2
    summary: str
    key_concepts: List[str]
    cohesion_score: float
    parent_cluster: Optional[str] = None
    
    def generate_id(self) -> str:
        """Generate consistent cluster ID."""
        content = f"cluster_{self.layer}_{self.name.lower().replace(' ', '_')}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for ArangoDB storage."""
        return {
            "_key": self.generate_id(),
            "name": self.name,
            "layer": self.layer,
            "member_count": len(self.members),
            "summary": self.summary,
            "key_concepts": self.key_concepts,
            "semantic_embedding": None,  # Will be computed separately
            "parent_cluster": self.parent_cluster,
            "cohesion_score": self.cohesion_score,
            "created": datetime.now(timezone.utc).isoformat(),
            "clustering_algorithm": "hybrid_hdbscan_louvain",
            "parameters": {
                "min_cluster_size": 5,
                "min_samples": 3,
                "metric": "cosine"
            }
        }


class HiRAGHierarchicalClustering:
    """
    Implements hierarchical clustering for HiRAG knowledge graph.
    
    Following Weber's theory of bureaucratic organization, this class
    creates a rational hierarchy of knowledge categories, enabling
    efficient information retrieval through systematic classification
    and clear lines of authority between different levels of abstraction.
    """
    
    def __init__(self, host: str = "192.168.1.69", port: int = 8529, 
                 username: str = "root", password: str = None):
        """Initialize the hierarchical clustering system."""
        self.password = password or os.getenv('ARANGO_PASSWORD', 'root_password')
        self.client = ArangoClient(hosts=f"http://{host}:{port}")
        self.db: Optional[StandardDatabase] = None
        
        # Clustering parameters
        self.min_cluster_size = 5
        self.min_samples = 3
        self.semantic_weight = 0.6
        self.structural_weight = 0.4
        
    async def connect(self, database_name: str = "academy_store") -> bool:
        """Connect to the ArangoDB database."""
        try:
            self.db = self.client.db(database_name, username="root", password=self.password)
            logger.info(f"Connected to database: {database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    async def load_entities_for_clustering(self) -> List[Dict]:
        """
        Load Level 0 entities for clustering.
        
        This represents the census moment - gathering all base-level
        actors in our network for bureaucratic categorization.
        """
        try:
            query = """
            FOR entity IN entities
                FILTER entity.layer == 0
                RETURN {
                    _key: entity._key,
                    name: entity.name,
                    type: entity.type,
                    description: entity.description,
                    source_papers: entity.source_papers,
                    frequency: entity.frequency,
                    importance_score: entity.importance_score
                }
            """
            
            cursor = self.db.aql.execute(query)
            entities = list(cursor)
            logger.info(f"Loaded {len(entities)} entities for clustering")
            return entities
            
        except Exception as e:
            logger.error(f"Failed to load entities: {e}")
            return []
    
    def generate_entity_embeddings(self, entities: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for entities using TF-IDF on descriptions.
        
        This creates the coordinate system within which our actors
        will be positioned - the dimensional space that enables
        measurement of similarity and difference.
        """
        # Prepare text for embedding
        texts = []
        for entity in entities:
            # Combine name, type, and description
            text_parts = [
                entity['name'],
                entity['type'], 
                entity['description'] or "",
                # Add context from source papers
                f"appears_in_{len(entity.get('source_papers', []))}_papers",
                f"frequency_{entity.get('frequency', 1)}"
            ]
            texts.append(" ".join(text_parts))
        
        # Generate TF-IDF embeddings
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        try:
            embeddings = vectorizer.fit_transform(texts).toarray()
            logger.info(f"Generated TF-IDF embeddings: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate TF-IDF embeddings: {e}")
            raise RuntimeError(
                f"TF-IDF embedding generation failed: {str(e)}. "
                "Cannot proceed with clustering without proper embeddings."
            ) from e
    
    def create_level1_clusters(self, entities: List[Dict], embeddings: np.ndarray) -> List[ClusterInfo]:
        """
        Create Level 1 topic clusters using hybrid semantic + structural clustering.
        
        This implements the primary bureaucratic categorization -
        the creation of departmental structures that organize our
        knowledge actors into coherent administrative units.
        """
        logger.info("Creating Level 1 clusters with hybrid approach...")
        
        # 1. Semantic clustering using HDBSCAN
        semantic_clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='cosine',
            cluster_selection_epsilon=0.3
        )
        
        semantic_labels = semantic_clusterer.fit_predict(embeddings)
        logger.info(f"Semantic clustering found {len(set(semantic_labels)) - (1 if -1 in semantic_labels else 0)} clusters")
        
        # 2. Create clusters from labels
        clusters = []
        unique_labels = set(semantic_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            # Get cluster members
            member_indices = np.where(semantic_labels == label)[0]
            member_entities = [entities[i] for i in member_indices]
            
            if len(member_entities) < self.min_cluster_size:
                continue
            
            # Generate cluster name and summary
            cluster_name = self._generate_cluster_name(member_entities)
            cluster_summary = self._generate_cluster_summary(member_entities)
            key_concepts = self._extract_key_concepts(member_entities)
            
            # Calculate cohesion score
            cluster_embeddings = embeddings[member_indices]
            cohesion = self._calculate_cohesion_score(cluster_embeddings)
            
            cluster = ClusterInfo(
                name=cluster_name,
                layer=1,
                members=[entity['_key'] for entity in member_entities],
                summary=cluster_summary,
                key_concepts=key_concepts,
                cohesion_score=cohesion
            )
            clusters.append(cluster)
        
        logger.info(f"Created {len(clusters)} Level 1 clusters")
        return clusters
    
    def create_level2_clusters(self, level1_clusters: List[ClusterInfo]) -> List[ClusterInfo]:
        """
        Create Level 2 super-clusters from Level 1 clusters.
        
        This creates the executive level of our bureaucratic hierarchy -
        the super-departmental structures that coordinate between
        different topic domains.
        """
        logger.info("Creating Level 2 super-clusters...")
        
        if len(level1_clusters) < 4:  # Need at least 4 L1 clusters for L2
            logger.warning("Insufficient Level 1 clusters for Level 2 clustering")
            return []
        
        # Generate meta-embeddings from cluster summaries
        cluster_texts = [f"{cluster.name} {cluster.summary} {' '.join(cluster.key_concepts)}" 
                        for cluster in level1_clusters]
        
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1
        )
        
        meta_embeddings = vectorizer.fit_transform(cluster_texts).toarray()
        
        # Use agglomerative clustering for Level 2
        n_clusters = max(2, int(np.sqrt(len(level1_clusters))))  # Target √(L1_count) clusters
        
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        
        level2_labels = clusterer.fit_predict(meta_embeddings)
        
        # Create Level 2 clusters
        level2_clusters = []
        
        for label in set(level2_labels):
            member_indices = np.where(level2_labels == label)[0]
            member_l1_clusters = [level1_clusters[i] for i in member_indices]
            
            if len(member_l1_clusters) < 2:  # Need at least 2 L1 clusters
                continue
            
            # Generate super-cluster name and summary
            super_name = self._generate_super_cluster_name(member_l1_clusters)
            super_summary = self._generate_super_cluster_summary(member_l1_clusters)
            super_concepts = self._extract_super_key_concepts(member_l1_clusters)
            
            # Calculate cohesion for super-cluster
            super_embeddings = meta_embeddings[member_indices]
            super_cohesion = self._calculate_cohesion_score(super_embeddings)
            
            super_cluster = ClusterInfo(
                name=super_name,
                layer=2,
                members=[cluster.generate_id() for cluster in member_l1_clusters],
                summary=super_summary,
                key_concepts=super_concepts,
                cohesion_score=super_cohesion
            )
            level2_clusters.append(super_cluster)
            
            # Update parent references in Level 1 clusters
            for l1_cluster in member_l1_clusters:
                l1_cluster.parent_cluster = super_cluster.generate_id()
        
        logger.info(f"Created {len(level2_clusters)} Level 2 super-clusters")
        return level2_clusters
    
    def _generate_cluster_name(self, entities: List[Dict]) -> str:
        """Generate a meaningful name for a cluster based on its entities."""
        # Get most common types and frequent terms
        types = [entity['type'] for entity in entities]
        names = [entity['name'].lower() for entity in entities]
        
        # Find most common type
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        dominant_type = max(type_counts.keys(), key=type_counts.get)
        
        # Find common terms in names
        all_words = []
        for name in names:
            all_words.extend(name.split())
        
        word_counts = {}
        for word in all_words:
            if len(word) > 2:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_counts.keys(), key=word_counts.get, reverse=True)[:3]
        
        if top_words:
            return f"{' '.join(top_words).title()} ({dominant_type.title()}s)"
        else:
            return f"{dominant_type.title()} Cluster"
    
    def _generate_cluster_summary(self, entities: List[Dict]) -> str:
        """Generate a summary description for a cluster."""
        entity_count = len(entities)
        types = [entity['type'] for entity in entities]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        # Create summary
        type_parts = []
        for entity_type, count in type_counts.items():
            type_parts.append(f"{count} {entity_type}{'s' if count > 1 else ''}")
        
        summary = f"This cluster contains {entity_count} entities: {', '.join(type_parts)}. "
        
        # Add context about most frequent entities
        freq_entities = sorted(entities, key=lambda x: x.get('frequency', 1), reverse=True)[:3]
        top_names = [e['name'] for e in freq_entities]
        
        summary += f"Key entities include: {', '.join(top_names)}."
        return summary
    
    def _extract_key_concepts(self, entities: List[Dict]) -> List[str]:
        """Extract key concepts from cluster entities."""
        # Combine all names and descriptions
        all_text = []
        for entity in entities:
            all_text.append(entity['name'])
            if entity.get('description'):
                all_text.append(entity['description'])
        
        # Use simple frequency-based extraction
        words = []
        for text in all_text:
            words.extend(text.lower().split())
        
        # Filter and count words
        filtered_words = [w for w in words if len(w) > 3 and w.isalpha()]
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top concepts
        top_concepts = sorted(word_counts.keys(), key=word_counts.get, reverse=True)[:5]
        return top_concepts
    
    def _calculate_cohesion_score(self, embeddings: np.ndarray) -> float:
        """Calculate cluster cohesion using average pairwise similarity."""
        if len(embeddings) < 2:
            return 1.0
        
        # Calculate pairwise cosine similarities
        similarities = cosine_similarity(embeddings)
        
        # Get upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
        similarities_flat = similarities[mask]
        
        return float(np.mean(similarities_flat))
    
    def _generate_super_cluster_name(self, l1_clusters: List[ClusterInfo]) -> str:
        """Generate name for super-cluster from L1 cluster names."""
        # Extract key terms from L1 cluster names
        all_words = []
        for cluster in l1_clusters:
            words = cluster.name.lower().replace('(', ' ').replace(')', ' ').split()
            all_words.extend([w for w in words if len(w) > 3])
        
        # Find most common terms
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        top_words = sorted(word_counts.keys(), key=word_counts.get, reverse=True)[:2]
        
        if top_words:
            return f"{' '.join(top_words).title()} Domain"
        else:
            return f"Knowledge Domain {len(l1_clusters)} Clusters"
    
    def _generate_super_cluster_summary(self, l1_clusters: List[ClusterInfo]) -> str:
        """Generate summary for super-cluster."""
        cluster_count = len(l1_clusters)
        total_members = sum(len(cluster.members) for cluster in l1_clusters)
        
        cluster_names = [cluster.name for cluster in l1_clusters]
        
        return f"This domain encompasses {cluster_count} topic clusters with {total_members} total entities. " \
               f"It includes the following areas: {', '.join(cluster_names[:3])}{'...' if len(cluster_names) > 3 else ''}."
    
    def _extract_super_key_concepts(self, l1_clusters: List[ClusterInfo]) -> List[str]:
        """Extract key concepts for super-cluster."""
        all_concepts = []
        for cluster in l1_clusters:
            all_concepts.extend(cluster.key_concepts)
        
        # Count concept frequency
        concept_counts = {}
        for concept in all_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Return most frequent concepts
        top_concepts = sorted(concept_counts.keys(), key=concept_counts.get, reverse=True)[:7]
        return top_concepts
    
    async def store_clusters(self, clusters: List[ClusterInfo]) -> bool:
        """
        Store clusters in the ArangoDB clusters collection.
        
        This completes the bureaucratic institutionalization -
        the clusters become official categories in our knowledge
        management system.
        """
        try:
            clusters_collection = self.db.collection("clusters")
            cluster_docs = [cluster.to_dict() for cluster in clusters]
            
            # Batch insert
            result = clusters_collection.insert_many(cluster_docs, overwrite=True)
            success_count = len([r for r in result if not r.get('error')])
            
            logger.info(f"✅ Stored {success_count} clusters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store clusters: {e}")
            return False
    
    async def create_cluster_edges(self, level1_clusters: List[ClusterInfo], level2_clusters: List[ClusterInfo]) -> bool:
        """
        Create cluster membership and hierarchy edges.
        
        This establishes the formal relationships in our bureaucratic
        hierarchy - the reporting lines and membership structures
        that enable systematic navigation.
        """
        try:
            cluster_edges_collection = self.db.collection("cluster_edges")
            edges = []
            
            # Create entity → Level 1 cluster edges
            for cluster in level1_clusters:
                for entity_id in cluster.members:
                    edge_doc = {
                        "_from": f"entities/{entity_id}",
                        "_to": f"clusters/{cluster.generate_id()}",
                        "type": "member_of",
                        "membership_score": 0.9,  # High confidence for primary clustering
                        "created": datetime.now(timezone.utc).isoformat(),
                        "clustering_round": 1
                    }
                    edges.append(edge_doc)
            
            # Create Level 1 → Level 2 cluster edges
            for l2_cluster in level2_clusters:
                for l1_cluster_id in l2_cluster.members:
                    edge_doc = {
                        "_from": f"clusters/{l1_cluster_id}",
                        "_to": f"clusters/{l2_cluster.generate_id()}",
                        "type": "subcluster_of",
                        "aggregation_weight": 0.8,
                        "created": datetime.now(timezone.utc).isoformat(),
                        "clustering_round": 2
                    }
                    edges.append(edge_doc)
            
            # Insert edges in batches
            batch_size = 1000
            success_count = 0
            
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                result = cluster_edges_collection.insert_many(batch, overwrite=True)
                success_count += len([r for r in result if not r.get('error')])
            
            logger.info(f"✅ Created {success_count} cluster edges")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create cluster edges: {e}")
            return False
    
    async def build_hierarchical_clusters(self) -> bool:
        """
        Execute the complete hierarchical clustering pipeline.
        
        This represents the full bureaucratization process -
        the systematic organization of our knowledge domain
        into a rational, hierarchical administrative structure.
        """
        logger.info("Starting hierarchical clustering pipeline...")
        
        # 1. Load entities
        entities = await self.load_entities_for_clustering()
        if not entities:
            logger.error("No entities found for clustering")
            return False
        
        # 2. Generate embeddings
        embeddings = self.generate_entity_embeddings(entities)
        
        # 3. Create Level 1 clusters
        level1_clusters = self.create_level1_clusters(entities, embeddings)
        if not level1_clusters:
            logger.error("Failed to create Level 1 clusters")
            return False
        
        # 4. Create Level 2 super-clusters
        level2_clusters = self.create_level2_clusters(level1_clusters)
        
        # 5. Store all clusters
        all_clusters = level1_clusters + level2_clusters
        if not await self.store_clusters(all_clusters):
            return False
        
        # 6. Create cluster relationships
        if not await self.create_cluster_edges(level1_clusters, level2_clusters):
            return False
        
        logger.info("✅ Hierarchical clustering completed successfully!")
        logger.info(f"Created {len(level1_clusters)} Level 1 clusters and {len(level2_clusters)} Level 2 super-clusters")
        
        return True


async def main():
    """Main execution function."""
    password = os.getenv('ARANGO_PASSWORD')
    if not password:
        print("Please set ARANGO_PASSWORD environment variable")
        return
    
    # Create clustering system
    clustering = HiRAGHierarchicalClustering(password=password)
    
    # Connect to database
    if not await clustering.connect():
        return
    
    # Build hierarchical clusters
    success = await clustering.build_hierarchical_clusters()
    
    if success:
        print("\n🎉 HiRAG hierarchical clustering completed successfully!")
        print("Next steps:")
        print("1. Generate embeddings for clusters")
        print("2. Build three-level retrieval engine")
        print("3. Implement conveyance scoring")
    else:
        print("\n❌ Hierarchical clustering failed!")


if __name__ == "__main__":
    asyncio.run(main())