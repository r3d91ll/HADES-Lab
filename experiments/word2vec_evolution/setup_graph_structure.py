#!/usr/bin/env python3
"""
Setup graph structure for theory-practice bridge analysis.
Creates edge collections and graph to connect papers with their implementations.
"""

import logging
from arango import ArangoClient
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_graph():
    """Create graph structure for theory-practice analysis."""
    
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://192.168.1.69:8529')
    sys_db = client.db('_system', username='root', password='1luv93ngu1n$')
    
    # Ensure academy_store exists
    if not sys_db.has_database('academy_store'):
        logger.error("academy_store database not found!")
        return
    
    db = client.db('academy_store', username='root', password='1luv93ngu1n$')
    
    logger.info("=" * 80)
    logger.info("SETTING UP THEORY-PRACTICE BRIDGE GRAPH")
    logger.info("=" * 80)
    
    # 1. Create Edge Collections
    edge_collections = [
        {
            'name': 'paper_implements_theory',
            'description': 'Links papers to repositories that implement them'
        },
        {
            'name': 'semantic_similarity',
            'description': 'Links semantically similar content across papers and code'
        },
        {
            'name': 'temporal_evolution',
            'description': 'Links papers in chronological order (word2vec→doc2vec→code2vec)'
        },
        {
            'name': 'conveyance_bridge',
            'description': 'Special edges for pure conveyance (e.g., Gensim from paper alone)'
        }
    ]
    
    for edge_col in edge_collections:
        if not db.has_collection(edge_col['name']):
            db.create_collection(edge_col['name'], edge=True)
            logger.info(f"✓ Created edge collection: {edge_col['name']}")
            logger.info(f"  Purpose: {edge_col['description']}")
        else:
            logger.info(f"  Edge collection exists: {edge_col['name']}")
    
    # 2. Create Analysis Collections
    analysis_collections = [
        {
            'name': 'theory_practice_bridges',
            'description': 'Stores identified bridges between theory and practice'
        },
        {
            'name': 'entropy_maps',
            'description': 'Stores entropy calculations for paper-code relationships'
        },
        {
            'name': 'conveyance_scores',
            'description': 'Stores conveyance measurements (C = W×R×H/T × Ctx^α)'
        }
    ]
    
    for anal_col in analysis_collections:
        if not db.has_collection(anal_col['name']):
            db.create_collection(anal_col['name'])
            logger.info(f"✓ Created analysis collection: {anal_col['name']}")
            logger.info(f"  Purpose: {anal_col['description']}")
        else:
            logger.info(f"  Analysis collection exists: {anal_col['name']}")
    
    # 3. Create the Graph
    graph_name = 'theory_practice_graph'
    
    if not db.has_graph(graph_name):
        graph = db.create_graph(
            name=graph_name,
            edge_definitions=[
                {
                    'edge_collection': 'paper_implements_theory',
                    'from_vertex_collections': ['arxiv_papers'],
                    'to_vertex_collections': ['github_repositories']
                },
                {
                    'edge_collection': 'semantic_similarity',
                    'from_vertex_collections': ['arxiv_embeddings', 'github_embeddings'],
                    'to_vertex_collections': ['arxiv_embeddings', 'github_embeddings']
                },
                {
                    'edge_collection': 'temporal_evolution',
                    'from_vertex_collections': ['arxiv_papers'],
                    'to_vertex_collections': ['arxiv_papers']
                },
                {
                    'edge_collection': 'conveyance_bridge',
                    'from_vertex_collections': ['arxiv_papers'],
                    'to_vertex_collections': ['github_repositories']
                }
            ]
        )
        logger.info(f"✓ Created graph: {graph_name}")
    else:
        graph = db.graph(graph_name)
        logger.info(f"  Graph exists: {graph_name}")
    
    # 4. Create Initial Edges
    logger.info("\n" + "-" * 80)
    logger.info("Creating theory-practice relationships...")
    
    # Define our known relationships
    relationships = [
        {
            'paper_id': '1301.3781',
            'repo': 'dav/word2vec',
            'type': 'community_implementation',
            'year': 2013
        },
        {
            'paper_id': '1405.4053',
            'repo': 'piskvorky/gensim',
            'type': 'pure_conveyance',  # KEY INSIGHT!
            'year': 2014
        },
        {
            'paper_id': '1803.09473',
            'repo': 'tech-srl/code2vec',
            'type': 'official_implementation',
            'year': 2018
        }
    ]
    
    # Create paper → repo edges
    paper_impl_coll = db.collection('paper_implements_theory')
    
    for rel in relationships:
        edge_key = f"{rel['paper_id'].replace('.', '_')}_to_{rel['repo'].replace('/', '_')}"
        paper_key = rel['paper_id'].replace('.', '_')
        repo_key = rel['repo'].replace('/', '_')
        
        edge_doc = {
            '_key': edge_key,
            '_from': f"arxiv_papers/{paper_key}",
            '_to': f"github_repositories/{repo_key}",
            'implementation_type': rel['type'],
            'year': rel['year'],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            paper_impl_coll.insert(edge_doc)
            logger.info(f"✓ Connected {rel['paper_id']} → {rel['repo']} ({rel['type']})")
        except Exception as e:
            if 'duplicate' in str(e).lower():
                logger.info(f"  Edge exists: {rel['paper_id']} → {rel['repo']}")
            else:
                logger.error(f"  Failed: {e}")
    
    # Create temporal evolution edges
    temporal_coll = db.collection('temporal_evolution')
    
    temporal_edges = [
        {
            'from': '1301.3781',  # word2vec
            'to': '1405.4053',    # doc2vec
            'year_from': 2013,
            'year_to': 2014,
            'evolution': 'word_to_document'
        },
        {
            'from': '1405.4053',  # doc2vec
            'to': '1803.09473',   # code2vec
            'year_from': 2014,
            'year_to': 2018,
            'evolution': 'document_to_code'
        }
    ]
    
    for edge in temporal_edges:
        edge_key = f"{edge['from'].replace('.', '_')}_to_{edge['to'].replace('.', '_')}"
        from_key = edge['from'].replace('.', '_')
        to_key = edge['to'].replace('.', '_')
        
        edge_doc = {
            '_key': edge_key,
            '_from': f"arxiv_papers/{from_key}",
            '_to': f"arxiv_papers/{to_key}",
            'year_from': edge['year_from'],
            'year_to': edge['year_to'],
            'evolution_type': edge['evolution'],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            temporal_coll.insert(edge_doc)
            logger.info(f"✓ Evolution: {edge['from']} ({edge['year_from']}) → {edge['to']} ({edge['year_to']})")
        except Exception as e:
            if 'duplicate' in str(e).lower():
                logger.info(f"  Evolution edge exists: {edge['from']} → {edge['to']}")
            else:
                logger.error(f"  Failed: {e}")
    
    # Special edge for Gensim's pure conveyance
    conveyance_coll = db.collection('conveyance_bridge')
    
    gensim_edge = {
        '_key': 'doc2vec_pure_conveyance',
        '_from': 'arxiv_papers/1405_4053',
        '_to': 'github_repositories/piskvorky_gensim',
        'conveyance_type': 'pure',
        'significance': 'Implementation from paper alone, no original code',
        'became_standard': True,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        conveyance_coll.insert(gensim_edge)
        logger.info("✓ Special: Gensim pure conveyance edge created")
    except Exception as e:
        if 'duplicate' in str(e).lower():
            logger.info("  Gensim conveyance edge exists")
    
    # 5. Summary
    logger.info("\n" + "=" * 80)
    logger.info("GRAPH STRUCTURE COMPLETE")
    logger.info("=" * 80)
    
    # Count statistics
    stats = {
        'papers': db.collection('arxiv_papers').count(),
        'repos': db.collection('github_repositories').count(),
        'paper_impl_edges': paper_impl_coll.count(),
        'temporal_edges': temporal_coll.count(),
        'conveyance_edges': conveyance_coll.count()
    }
    
    logger.info("Graph Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nKey Insight:")
    logger.info("  Gensim's doc2vec represents PURE CONVEYANCE")
    logger.info("  - No access to original implementation")
    logger.info("  - Built from paper descriptions alone")
    logger.info("  - Became the de facto standard")
    logger.info("  - Empirical proof of high conveyance score")
    
    return graph

if __name__ == "__main__":
    setup_graph()