#!/usr/bin/env python3
"""Test specific HiRAG query patterns"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from hiretrieval_engine import HiRetrievalEngine, QueryContext

async def test_specific_queries():
    """Test HiRAG with specific successful patterns"""
    
    # Initialize engine
    engine = HiRetrievalEngine()
    await engine.connect()
    
    # Test queries that should succeed
    test_queries = [
        ("artificial intelligence research", "hi", "Should find AI entities and CS cluster"),
        ("computer vision", "hi_local", "Should find CV entities"),
        ("machine learning", "hi_global", "Should find ML entities and CS cluster"), 
        ("deep learning neural networks", "hi", "Should bridge DL and NN concepts")
    ]
    
    print("🤖 Testing Specific HiRAG Query Patterns")
    print("=" * 60)
    
    for query_text, mode, expectation in test_queries:
        print(f"\n🔍 Query: '{query_text}' (mode: {mode})")
        print(f"📋 Expected: {expectation}")
        print("-" * 60)
        
        # Create query context
        context = QueryContext(
            query_text=query_text,
            query_type=mode,
            top_k_entities=10,
            top_m_clusters=5
        )
        
        # Execute query
        result = await engine.hierarchical_query(query_text, mode)
        
        # Display results
        print(f"📊 Results: {len(result.local_entities)} entities, "
              f"{len(result.global_clusters)} clusters, "
              f"{len(result.bridge_paths)} bridges")
        print(f"⚡ Performance: {result.retrieval_time_ms}ms, "
              f"Conveyance: {result.conveyance_score:.3f}")
        
        # Show first few entities if found
        if result.local_entities:
            print("\n📝 Top Entities:")
            for entity in result.local_entities[:3]:
                print(f"  • {entity['name']} ({entity['type']}) - "
                      f"freq: {entity.get('frequency', 'N/A')}")
        
        # Show clusters if found
        if result.global_clusters:
            print(f"\n🌐 Clusters ({len(result.global_clusters)}):")
            for cluster in result.global_clusters[:2]:
                print(f"  • {cluster['name']} ({cluster['cluster_type']}) - "
                      f"{cluster.get('member_count', 'N/A')} members")
                      
        # Show answer synthesis
        if result.answer_synthesis:
            print(f"\n💬 Answer: {result.answer_synthesis[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_specific_queries())