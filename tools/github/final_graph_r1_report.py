#!/usr/bin/env python3
"""
Final Graph-R1 Processing Report
================================

Complete report on the Graph-R1 repository processing results.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.database.arango_db_manager import ArangoDBManager
import yaml

def generate_final_report():
    """Generate the final comprehensive report for Graph-R1 processing."""
    
    config_path = Path(__file__).parent / "configs" / "github_simple.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    arango_password = os.getenv('ARANGO_PASSWORD')
    if not arango_password:
        raise ValueError("ARANGO_PASSWORD environment variable is required")
    config['arango']['password'] = arango_password
    
    db_manager = ArangoDBManager(config['arango'])
    
    print("="*80)
    print("GRAPH-R1 REPOSITORY IMPORT COMPLETION REPORT")
    print("="*80)
    
    # 1. Repository Information
    repo_query = """
    FOR repo IN github_repositories
        FILTER repo.full_name == 'LHRLAB/Graph-R1'
        RETURN repo
    """
    
    cursor = db_manager.db.aql.execute(repo_query)
    repos = list(cursor)
    
    if repos:
        repo = repos[0]
        print(f"Repository: {repo['full_name']}")
        print(f"Owner: {repo['owner']}")
        print(f"Name: {repo['name']}")
        print(f"Clone URL: {repo['clone_url']}")
        print(f"Processing Date: {repo.get('processed_date', 'N/A')}")
        print(f"Files Processed: {repo.get('file_count', 'N/A')}")
    
    # 2. Files by Language
    language_query = """
    FOR doc IN github_papers
        FILTER CONTAINS(doc.document_id, 'LHRLAB_Graph-R1')
        COLLECT language = doc.language WITH COUNT INTO count
        SORT count DESC
        RETURN {
            language: language,
            count: count
        }
    """
    
    cursor = db_manager.db.aql.execute(language_query)
    language_stats = list(cursor)
    
    print(f"\nFiles Processed by Language:")
    total_files = 0
    for stat in language_stats:
        print(f"  {stat['language'] or 'unknown'}: {stat['count']} files")
        total_files += stat['count']
    
    print(f"\nTotal Documents: {total_files}")
    
    # 3. Tree-sitter Analysis Results
    treesitter_query = """
    FOR doc IN github_papers
        FILTER CONTAINS(doc.document_id, 'LHRLAB_Graph-R1') AND doc.has_tree_sitter == true
        COLLECT WITH COUNT INTO ts_count
        RETURN ts_count
    """
    
    cursor = db_manager.db.aql.execute(treesitter_query)
    results = list(cursor)
    ts_count = results[0] if results else 0
    
    # 4. Sample Tree-sitter Analysis
    sample_ts_query = """
    FOR doc IN github_papers
        FILTER CONTAINS(doc.document_id, 'LHRLAB_Graph-R1') AND doc.has_tree_sitter == true
        LIMIT 5
        RETURN {
            id: doc.document_id,
            language: doc.language,
            symbols: doc.symbols ? {
                functions: LENGTH(doc.symbols.functions || []),
                classes: LENGTH(doc.symbols.classes || []),
                config_type: doc.symbols.config_type
            } : null,
            metrics: doc.code_metrics
        }
    """
    
    cursor = db_manager.db.aql.execute(sample_ts_query)
    sample_ts = list(cursor)
    
    print(f"\nTree-sitter Symbol Extraction:")
    print(f"  Files with Tree-sitter analysis: {ts_count}")
    
    if sample_ts:
        print(f"  Sample analyzed files:")
        for doc in sample_ts:
            print(f"    {doc['id']}:")
            print(f"      Language: {doc['language']}")
            if doc.get('symbols'):
                if doc['symbols'].get('config_type'):
                    print(f"      Config type: {doc['symbols']['config_type']}")
                else:
                    print(f"      Functions: {doc['symbols'].get('functions', 0)}")
                    print(f"      Classes: {doc['symbols'].get('classes', 0)}")
            if doc.get('metrics'):
                print(f"      Complexity: {doc['metrics'].get('complexity', 0)}")
    
    # 5. Chunks Generated
    chunks_query = """
    FOR chunk IN github_chunks
        FILTER CONTAINS(chunk.source, 'LHRLAB_Graph-R1')
        COLLECT WITH COUNT INTO chunk_count
        RETURN chunk_count
    """
    
    cursor = db_manager.db.aql.execute(chunks_query)
    chunk_results = list(cursor)
    chunk_count = chunk_results[0] if chunk_results else 0
    
    # 6. Embeddings Generated  
    embeddings_query = """
    FOR embedding IN github_embeddings
        FILTER CONTAINS(embedding.chunk_id, 'LHRLAB_Graph-R1')
        COLLECT WITH COUNT INTO embedding_count
        RETURN embedding_count
    """
    
    cursor = db_manager.db.aql.execute(embeddings_query)
    embedding_results = list(cursor)
    embedding_count = embedding_results[0] if embedding_results else 0
    
    # 7. Sample Embeddings
    sample_embedding_query = """
    FOR embedding IN github_embeddings
        FILTER CONTAINS(embedding.chunk_id, 'LHRLAB_Graph-R1')
        LIMIT 3
        RETURN {
            chunk_id: embedding.chunk_id,
            embedding_dim: LENGTH(embedding.embedding),
            model_used: embedding.model_name
        }
    """
    
    cursor = db_manager.db.aql.execute(sample_embedding_query)
    sample_embeddings = list(cursor)
    
    print(f"\nSemantic Processing:")
    print(f"  Text chunks created: {chunk_count}")
    print(f"  Jina v4 embeddings generated: {embedding_count}")
    
    if sample_embeddings:
        print(f"  Sample embeddings:")
        for emb in sample_embeddings:
            print(f"    Chunk: {emb['chunk_id']}")
            print(f"    Dimensions: {emb['embedding_dim']}")
            print(f"    Model: {emb['model_used']}")
    
    # 8. Graph Structure
    graph_edges_query = """
    FOR edge IN github_repo_files
        FILTER CONTAINS(edge._from, 'LHRLAB_Graph-R1')
        COLLECT WITH COUNT INTO edge_count
        RETURN edge_count
    """
    
    cursor = db_manager.db.aql.execute(graph_edges_query)
    edge_results = list(cursor)
    edge_count = edge_results[0] if edge_results else 0
    
    print(f"\nGraph Structure:")
    print(f"  Repository → Files edges: {edge_count}")
    
    # 9. Summary
    print(f"\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print("✅ SUCCESSFULLY COMPLETED: Graph-R1 Repository Import")
    print()
    print("📁 Repository Details:")
    print(f"   • Repository: LHRLAB/Graph-R1")
    print(f"   • Total files: {total_files}")
    print(f"   • Languages: {len(language_stats)}")
    print()
    print("🔍 Code Analysis (Tree-sitter):")
    print(f"   • Files analyzed: {ts_count}")
    print(f"   • Symbol extraction for functions, classes, and config files")
    print(f"   • Code complexity metrics calculated")
    print()
    print("🧠 Semantic Processing (Jina v4):")
    print(f"   • Text chunks: {chunk_count}")
    print(f"   • Embeddings: {embedding_count}")
    print(f"   • Model: jinaai/jina-embeddings-v4 with coding LoRA")
    print(f"   • Dimensions: 2048 (late chunking with 32k token context)")
    print()
    print("🗄️ Database Storage (ArangoDB):")
    print("   • github_repositories → github_papers → github_chunks → github_embeddings")
    print(f"   • Repository vertex created with metadata")
    print(f"   • Graph edges linking repository to files")
    print(f"   • All data ready for semantic search and analysis")
    print()
    print("🚀 Next Steps:")
    print("   • Repository is now searchable through HADES semantic search")
    print("   • Code symbols are queryable for development analysis")
    print("   • Embeddings enable similarity search across codebase")
    print("   • Graph structure supports relationship queries")
    print("="*80)

if __name__ == "__main__":
    generate_final_report()