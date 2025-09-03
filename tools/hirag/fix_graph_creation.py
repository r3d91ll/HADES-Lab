#!/usr/bin/env python3
"""
Fix HiRAG Graph Creation
Creates the HiRAG graph without the problematic theory_practice_bridges collection
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from arango import ArangoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_hirag_graph():
    """Create HiRAG graph with available collections only."""
    # Load configuration from environment
    arango_host = os.getenv('ARANGO_HOST', 'localhost')
    arango_password = os.getenv('ARANGO_PASSWORD')
    
    if not arango_password:
        raise ValueError(
            "ARANGO_PASSWORD environment variable is required"
        )
    
    client = ArangoClient(hosts=f"http://{arango_host}:8529")
    db = client.db("academy_store", username="root", password=arango_password)
    
    try:
        # Delete existing graph if it exists
        try:
            db.delete_graph("hirag_graph")
            logger.info("Deleted existing HiRAG graph")
        except Exception as e:
            logger.info(f"No existing HiRAG graph to delete: {e}")
        
        # Create graph with available edge collections only
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
            }
            # Skip theory_practice_bridges for now - will add later as edge collection
        ]
        
        # Create graph
        graph = db.create_graph(
            name="hirag_graph",
            edge_definitions=edge_definitions
        )
        logger.info("✅ Created HiRAG graph successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create HiRAG graph: {e}")
        return False

if __name__ == "__main__":
    success = fix_hirag_graph()
    if success:
        print("🎉 HiRAG graph created successfully!")
    else:
        print("❌ Failed to create HiRAG graph")