#!/usr/bin/env python3
"""Debug HiRAG Retrieval Issues"""

import os
import sys
from pathlib import Path
import logging
from arango import ArangoClient
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_connectivity():
    """Test basic data connectivity"""
    # Load configuration from environment
    arango_host = os.getenv('ARANGO_HOST', '192.168.1.69')
    arango_password = os.getenv('ARANGO_PASSWORD')
    
    if not arango_password:
        raise ValueError(
            "ARANGO_PASSWORD environment variable is required for database connection"
        )
    
    # Connect to database
    client = ArangoClient(hosts=f'http://{arango_host}:8529')
    db = client.db('academy_store', username='root', password=arango_password)
    
    # Test entity retrieval for "computer vision"
    print("🔍 Testing Local Entity Retrieval")
    query = """
    FOR entity IN entities
        FILTER CONTAINS(LOWER(entity.name), "vision") OR CONTAINS(LOWER(entity.name), "computer")
        SORT entity.frequency DESC
        LIMIT 10
        RETURN {
            key: entity._key,
            name: entity.name, 
            type: entity.type,
            frequency: entity.frequency
        }
    """
    
    result = db.aql.execute(query)
    local_entities = [doc for doc in result]
    print(f"Found {len(local_entities)} local entities:")
    for entity in local_entities:
        print(f"  • {entity['name']} ({entity['type']}) - freq: {entity.get('frequency', 0)}")
    
    # Test global cluster retrieval
    print("\n🌐 Testing Global Cluster Retrieval")
    if local_entities:
        local_keys = [e['key'] for e in local_entities[:3]]  # Top 3
        cluster_query = """
        FOR entity_key IN @entity_keys
            FOR cluster IN clusters
                FILTER entity_key IN cluster.members
                RETURN DISTINCT {
                    name: cluster.name,
                    summary: cluster.summary,
                    member_count: cluster.member_count,
                    cluster_type: cluster.cluster_type
                }
        """
        
        result = db.aql.execute(cluster_query, bind_vars={'entity_keys': local_keys})
        clusters = [doc for doc in result]
        print(f"Found {len(clusters)} related clusters:")
        for cluster in clusters:
            print(f"  • {cluster['name']} ({cluster['cluster_type']}) - {cluster.get('member_count', 0)} members")
    
    # Test TF-IDF similarity
    print("\n📊 Testing TF-IDF Similarity")
    query_text = "computer vision applications"
    entity_names = [e['name'] for e in local_entities]
    
    if entity_names:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        all_texts = entity_names + [query_text]
        
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        query_vec = tfidf_matrix[-1]
        entity_vecs = tfidf_matrix[:-1]
        
        similarities = cosine_similarity(query_vec, entity_vecs).flatten()
        
        print(f"TF-IDF similarities (max: {similarities.max():.4f}, mean: {similarities.mean():.4f}):")
        for i, (entity, sim) in enumerate(zip(local_entities, similarities)):
            if sim > 0.01:
                print(f"  • {entity['name']}: {sim:.4f}")

if __name__ == "__main__":
    test_data_connectivity()