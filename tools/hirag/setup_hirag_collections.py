#!/usr/bin/env python3
"""
HiRAG ArangoDB Collections Setup
Creates the required collections and indexes for HiRAG implementation
Based on PRD Issue #19 requirements
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import CollectionCreateError, IndexCreateError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HiRAGCollectionSetup:
    """
    Sets up ArangoDB collections and indexes for HiRAG implementation.
    
    From an anthropological perspective, this represents the construction of
    institutional infrastructure - the bureaucratic frameworks that enable
    hierarchical knowledge organization. Following Weber's ideal types, we're
    creating a rational-legal authority structure for information retrieval.
    """
    
    def __init__(self, host: str = "192.168.1.69", port: int = 8529, 
                 username: str = "root", password: str = None):
        """
        Initialize ArangoDB connection.
        
        In ANT terms, this establishes our obligatory passage point - 
        the database becomes the mediator through which all knowledge
        flows must pass.
        """
        self.password = password or os.getenv('ARANGO_PASSWORD', 'root_password')
        self.client = ArangoClient(hosts=f"http://{host}:{port}")
        self.db: Optional[StandardDatabase] = None
        
    def connect(self, database_name: str = "academy_store") -> bool:
        """Connect to the specified database."""
        try:
            self.db = self.client.db(database_name, username="root", password=self.password)
            logger.info(f"Connected to database: {database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def create_document_collections(self) -> bool:
        """
        Create document collections (vertices in the knowledge graph).
        
        These represent the ontological categories of our domain -
        the fundamental types of actors in our information network.
        """
        collections = {
            # Core entity collections
            "entities": {
                "description": "Extracted entities from papers (concepts, methods, algorithms, people)",
                "indexes": [
                    {"fields": ["layer"], "type": "persistent"},
                    {"fields": ["type"], "type": "persistent"},
                    {"fields": ["cluster_id"], "type": "persistent"},
                    {"fields": ["name"], "type": "persistent"},
                    {"fields": ["importance_score"], "type": "persistent"},
                    {"fields": ["semantic_embedding"], "type": "persistent"}  # Vector index
                ]
            },
            
            "clusters": {
                "description": "Hierarchical clusters (Level 1 and Level 2 summary nodes)",
                "indexes": [
                    {"fields": ["layer"], "type": "persistent"},
                    {"fields": ["name"], "type": "persistent"},
                    {"fields": ["cohesion_score"], "type": "persistent"},
                    {"fields": ["semantic_embedding"], "type": "persistent"}  # Vector index
                ]
            },
            
            # Performance and caching collections
            "query_logs": {
                "description": "Query performance tracking and user feedback",
                "indexes": [
                    {"fields": ["timestamp"], "type": "persistent"},
                    {"fields": ["latency_ms"], "type": "persistent"},
                    {"fields": ["retrieval_mode"], "type": "persistent"},
                    {"fields": ["conveyance_score"], "type": "persistent"}
                ]
            },
            
            "bridge_cache": {
                "description": "Precomputed hot bridge paths for performance",
                "indexes": [
                    {"fields": ["from_entity"], "type": "persistent"},
                    {"fields": ["to_entity"], "type": "persistent"},
                    {"fields": ["access_count"], "type": "persistent"},
                    {"fields": ["ttl"], "type": "persistent"}
                ]
            },
            
            # Weight configuration
            "weight_config": {
                "description": "Edge weight parameters and learning configuration",
                "indexes": [
                    {"fields": ["relation_type"], "type": "persistent"},
                    {"fields": ["updated"], "type": "persistent"}
                ]
            }
        }
        
        success = True
        for collection_name, config in collections.items():
            try:
                # Create collection
                collection = self.db.create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")
                
                # Create indexes
                for index_config in config["indexes"]:
                    try:
                        collection.add_persistent_index(
                            fields=index_config["fields"],
                            name=f"{collection_name}_{'_'.join(index_config['fields'])}_idx"
                        )
                        logger.info(f"Created index on {collection_name}: {index_config['fields']}")
                    except IndexCreateError as e:
                        logger.warning(f"Index creation warning for {collection_name}: {e}")
                        
            except CollectionCreateError as e:
                if "duplicate name" in str(e).lower():
                    logger.info(f"Collection {collection_name} already exists")
                else:
                    logger.error(f"Failed to create collection {collection_name}: {e}")
                    success = False
                    
        return success
    
    def create_edge_collections(self) -> bool:
        """
        Create edge collections (relationships in the knowledge graph).
        
        These represent the performative relationships between actors -
        the ways in which entities enroll and translate each other in
        our knowledge network.
        """
        edge_collections = {
            "relations": {
                "description": "Entity-to-entity relationships with weighted edges",
                "indexes": [
                    {"fields": ["type"], "type": "persistent"},
                    {"fields": ["conveyance_weight"], "type": "persistent"},
                    {"fields": ["layer_bridge"], "type": "persistent"},
                    {"fields": ["embedding_similarity"], "type": "persistent"},
                    {"fields": ["temporal_factors.age_days"], "type": "persistent"}
                ]
            },
            
            "cluster_edges": {
                "description": "Cluster membership and hierarchy relationships", 
                "indexes": [
                    {"fields": ["type"], "type": "persistent"},
                    {"fields": ["membership_score"], "type": "persistent"},
                    {"fields": ["clustering_round"], "type": "persistent"}
                ]
            },
            
            "paper_entities": {
                "description": "Paper-to-entity extraction links",
                "indexes": [
                    {"fields": ["type"], "type": "persistent"},
                    {"fields": ["mention_count"], "type": "persistent"},
                    {"fields": ["confidence"], "type": "persistent"},
                    {"fields": ["extraction_method"], "type": "persistent"}
                ]
            }
        }
        
        success = True
        for collection_name, config in edge_collections.items():
            try:
                # Create edge collection
                collection = self.db.create_collection(collection_name, edge=True)
                logger.info(f"Created edge collection: {collection_name}")
                
                # Create indexes
                for index_config in config["indexes"]:
                    try:
                        collection.add_persistent_index(
                            fields=index_config["fields"],
                            name=f"{collection_name}_{'_'.join(index_config['fields'][:2])}_idx"
                        )
                        logger.info(f"Created index on {collection_name}: {index_config['fields']}")
                    except IndexCreateError as e:
                        logger.warning(f"Index creation warning for {collection_name}: {e}")
                        
            except CollectionCreateError as e:
                if "duplicate name" in str(e).lower():
                    logger.info(f"Edge collection {collection_name} already exists")
                else:
                    logger.error(f"Failed to create edge collection {collection_name}: {e}")
                    success = False
                    
        return success
    
    def create_graph(self) -> bool:
        """
        Create the HiRAG graph definition for traversal queries.
        
        This establishes the topology of our knowledge space -
        the navigational infrastructure that enables boundary crossing
        between different levels of abstraction.
        """
        try:
            # Define edge collections
            edge_definitions = [
                {
                    "edge_collection": "relations",
                    "from_vertex_collections": ["entities"],
                    "to_vertex_collections": ["entities"]
                },
                {
                    "edge_collection": "cluster_edges", 
                    "from_vertex_collections": ["entities", "clusters"],
                    "to_vertex_collections": ["clusters"]
                },
                {
                    "edge_collection": "paper_entities",
                    "from_vertex_collections": ["arxiv_papers"],
                    "to_vertex_collections": ["entities"]
                },
                {
                    "edge_collection": "theory_practice_bridges",
                    "from_vertex_collections": ["arxiv_papers"],
                    "to_vertex_collections": ["github_papers"]
                }
            ]
            
            # Create graph
            graph = self.db.create_graph(
                name="hirag_graph",
                edge_definitions=edge_definitions
            )
            logger.info("Created HiRAG graph definition")
            return True
            
        except Exception as e:
            if "duplicate name" in str(e).lower():
                logger.info("HiRAG graph already exists")
                return True
            else:
                logger.error(f"Failed to create HiRAG graph: {e}")
                return False
    
    def insert_weight_config(self) -> bool:
        """
        Insert default weight configuration for edge relationships.
        
        This represents the bureaucratic rules that govern information flow -
        the procedural rationality that determines how knowledge moves through
        our institutional structure.
        """
        default_weights = [
            {
                "_key": "implements",
                "relation_type": "implements",
                "base_weight": 0.9,
                "description": "Direct implementation relationship",
                "temporal_decay_rate": 365,  # days
                "bridge_bonus": 1.5,
                "created": "2025-09-03T00:00:00Z",
                "updated": "2025-09-03T00:00:00Z"
            },
            {
                "_key": "cites", 
                "relation_type": "cites",
                "base_weight": 0.8,
                "description": "Citation relationship",
                "temporal_decay_rate": 365,
                "bridge_bonus": 1.2,
                "created": "2025-09-03T00:00:00Z",
                "updated": "2025-09-03T00:00:00Z"
            },
            {
                "_key": "extends",
                "relation_type": "extends", 
                "base_weight": 0.7,
                "description": "Extension/improvement relationship",
                "temporal_decay_rate": 365,
                "bridge_bonus": 1.3,
                "created": "2025-09-03T00:00:00Z",
                "updated": "2025-09-03T00:00:00Z"
            },
            {
                "_key": "summarizes",
                "relation_type": "summarizes",
                "base_weight": 0.6,
                "description": "Summary/aggregation relationship", 
                "temporal_decay_rate": 365,
                "bridge_bonus": 1.1,
                "created": "2025-09-03T00:00:00Z",
                "updated": "2025-09-03T00:00:00Z"
            }
        ]
        
        try:
            weight_collection = self.db.collection("weight_config")
            for weight in default_weights:
                weight_collection.insert(weight, overwrite=True)
            logger.info("Inserted default weight configuration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert weight configuration: {e}")
            return False
    
    def setup_all(self) -> bool:
        """
        Run complete HiRAG collection setup.
        
        This represents the institutional founding moment - the establishment
        of the organizational infrastructure that will govern all future
        knowledge interactions in our domain.
        """
        logger.info("Starting HiRAG ArangoDB collection setup...")
        
        success = True
        success &= self.create_document_collections()
        success &= self.create_edge_collections() 
        success &= self.create_graph()
        success &= self.insert_weight_config()
        
        if success:
            logger.info("✅ HiRAG collection setup completed successfully!")
        else:
            logger.error("❌ HiRAG collection setup failed!")
            
        return success


def main():
    """Main setup function."""
    # Get password from environment or prompt
    password = os.getenv('ARANGO_PASSWORD')
    if not password:
        print("Please set ARANGO_PASSWORD environment variable or provide password:")
        password = input("ArangoDB password: ").strip()
    
    # Setup collections
    setup = HiRAGCollectionSetup(password=password)
    if not setup.connect():
        sys.exit(1)
        
    if not setup.setup_all():
        sys.exit(1)
        
    print("\n🎉 HiRAG collections are ready for implementation!")
    print("Next steps:")
    print("1. Run entity extraction pipeline")
    print("2. Build hierarchical clusters")
    print("3. Implement three-level retrieval engine")


if __name__ == "__main__":
    main()