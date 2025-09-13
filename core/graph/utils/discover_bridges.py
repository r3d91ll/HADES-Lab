#!/usr/bin/env python3
"""
Discover theory-practice bridges in the academic graph.
Uses graph topology and semantic analysis to find papers that connect theoretical and practical domains.
"""

import os
import json
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
import numpy as np
from tqdm import tqdm
from arango import ArangoClient
import click
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BridgeDiscovery:
    """Discover papers that bridge theory and practice."""
    
    def __init__(self, db_name: str = 'academy_store'):
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            db_name,
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Theory and practice indicators
        self.theory_keywords = [
            'theory', 'theorem', 'proof', 'lemma', 'conjecture',
            'axiom', 'formal', 'mathematical', 'abstract', 'generalization'
        ]
        
        self.practice_keywords = [
            'implementation', 'application', 'system', 'practical',
            'experiment', 'empirical', 'deployment', 'real-world',
            'prototype', 'framework', 'toolkit', 'benchmark'
        ]
        
    def classify_papers(self, sample_size: int = None) -> Dict[str, List[str]]:
        """Classify papers as theory, practice, or bridge based on title/abstract."""
        logger.info("Classifying papers...")
        
        query = """
        FOR p IN arxiv_papers
            FILTER p.title != null
            %s
            RETURN {
                id: p._key,
                title: LOWER(p.title),
                abstract: LOWER(p.abstract),
                categories: p.categories
            }
        """ % (f"LIMIT {sample_size}" if sample_size else "")
        
        theory_papers = []
        practice_papers = []
        bridge_papers = []
        neutral_papers = []
        
        for paper in tqdm(self.db.aql.execute(query, batch_size=10000), desc="Classifying"):
            text = f"{paper['title']} {paper.get('abstract', '')}"
            
            theory_score = sum(1 for kw in self.theory_keywords if kw in text)
            practice_score = sum(1 for kw in self.practice_keywords if kw in text)
            
            if theory_score > 0 and practice_score > 0:
                bridge_papers.append(paper['id'])
            elif theory_score > practice_score:
                theory_papers.append(paper['id'])
            elif practice_score > theory_score:
                practice_papers.append(paper['id'])
            else:
                neutral_papers.append(paper['id'])
        
        logger.info(f"Classification results:")
        logger.info(f"  Theory papers: {len(theory_papers):,}")
        logger.info(f"  Practice papers: {len(practice_papers):,}")
        logger.info(f"  Bridge papers (both): {len(bridge_papers):,}")
        logger.info(f"  Neutral papers: {len(neutral_papers):,}")
        
        return {
            'theory': theory_papers,
            'practice': practice_papers,
            'bridge': bridge_papers,
            'neutral': neutral_papers
        }
    
    def find_topological_bridges(self, theory_papers: List[str], practice_papers: List[str], 
                                 max_papers: int = 1000) -> List[Dict]:
        """Find papers that connect theory and practice clusters topologically."""
        logger.info("Finding topological bridges...")
        
        # Sample if too many papers
        if len(theory_papers) > max_papers:
            theory_papers = np.random.choice(theory_papers, max_papers, replace=False)
        if len(practice_papers) > max_papers:
            practice_papers = np.random.choice(practice_papers, max_papers, replace=False)
        
        theory_set = set(theory_papers)
        practice_set = set(practice_papers)
        
        bridges = []
        
        # Find papers connected to both theory and practice papers
        query = """
        FOR theory_id IN @theory_papers
            LET theory = CONCAT('arxiv_papers/', theory_id)
            
            // Papers connected to this theory paper
            FOR v1 IN 1..2 ANY theory coauthorship, same_field
                FILTER v1._key NOT IN @theory_papers
                
                // Check if also connected to practice papers
                FOR v2 IN 1..2 ANY v1 coauthorship, same_field
                    FILTER v2._key IN @practice_papers
                    
                    COLLECT bridge = v1._key INTO connections = {
                        theory: theory_id,
                        practice: v2._key
                    }
                    
                    RETURN {
                        bridge: bridge,
                        theory_connections: LENGTH(connections),
                        practice_connections: LENGTH(
                            FOR c IN connections 
                                RETURN DISTINCT c.practice
                        )
                    }
        """
        
        params = {
            'theory_papers': list(theory_papers),
            'practice_papers': list(practice_papers)
        }
        
        bridge_scores = defaultdict(lambda: {'theory': 0, 'practice': 0})
        
        for result in self.db.aql.execute(query, bind_vars=params):
            bridge_id = result['bridge']
            bridge_scores[bridge_id]['theory'] += result['theory_connections']
            bridge_scores[bridge_id]['practice'] += result['practice_connections']
        
        # Score bridges by their connectivity to both sides
        for bridge_id, scores in bridge_scores.items():
            if scores['theory'] > 0 and scores['practice'] > 0:
                # Harmonic mean favors balanced connections
                bridge_score = 2 * scores['theory'] * scores['practice'] / (scores['theory'] + scores['practice'])
                bridges.append({
                    'id': bridge_id,
                    'theory_connections': scores['theory'],
                    'practice_connections': scores['practice'],
                    'bridge_score': bridge_score
                })
        
        bridges.sort(key=lambda x: x['bridge_score'], reverse=True)
        
        logger.info(f"Found {len(bridges)} topological bridge candidates")
        
        return bridges[:100]  # Top 100 bridges
    
    def analyze_bridge_patterns(self, bridges: List[Dict]) -> Dict:
        """Analyze patterns in bridge papers."""
        logger.info("Analyzing bridge patterns...")
        
        if not bridges:
            return {}
        
        bridge_ids = [b['id'] for b in bridges]
        
        # Get metadata for bridge papers
        query = """
        FOR p IN arxiv_papers
            FILTER p._key IN @bridge_ids
            RETURN {
                id: p._key,
                title: p.title,
                categories: p.categories,
                authors: p.authors,
                year: SUBSTRING(p.update_date, 0, 4)
            }
        """
        
        bridge_metadata = list(self.db.aql.execute(query, bind_vars={'bridge_ids': bridge_ids}))
        
        # Analyze categories
        category_counter = Counter()
        for paper in bridge_metadata:
            if paper.get('categories'):
                for cat in paper['categories']:
                    category_counter[cat] += 1
        
        # Analyze temporal patterns
        year_counter = Counter()
        for paper in bridge_metadata:
            if paper.get('year'):
                year_counter[paper['year']] += 1
        
        # Find prolific bridge authors
        author_counter = Counter()
        for paper in bridge_metadata:
            if paper.get('authors'):
                for author in paper['authors']:
                    author_counter[author] += 1
        
        patterns = {
            'top_categories': category_counter.most_common(10),
            'top_years': year_counter.most_common(10),
            'top_authors': author_counter.most_common(10),
            'total_bridges': len(bridges),
            'sample_titles': [p['title'] for p in bridge_metadata[:5]]
        }
        
        return patterns
    
    def find_semantic_bridges(self, sample_size: int = 10000) -> List[Dict]:
        """Find bridges using semantic similarity from embeddings."""
        logger.info("Finding semantic bridges using embeddings...")
        
        # Get papers with embeddings
        query = """
        FOR p IN arxiv_papers
            FILTER p.title != null
            LIMIT @sample_size
            
            LET embedding = FIRST(
                FOR e IN arxiv_embeddings
                    FILTER e.paper_id == p._key
                    RETURN e.embedding
            )
            
            FILTER embedding != null
            
            RETURN {
                id: p._key,
                title: LOWER(p.title),
                abstract: LOWER(p.abstract),
                embedding: embedding
            }
        """
        
        papers = list(self.db.aql.execute(query, bind_vars={'sample_size': sample_size}))
        
        if not papers:
            logger.warning("No papers with embeddings found")
            return []
        
        # Classify papers
        theory_papers = []
        practice_papers = []
        
        for paper in papers:
            text = f"{paper['title']} {paper.get('abstract', '')}"
            theory_score = sum(1 for kw in self.theory_keywords if kw in text)
            practice_score = sum(1 for kw in self.practice_keywords if kw in text)
            
            if theory_score > practice_score:
                theory_papers.append(paper)
            elif practice_score > theory_score:
                practice_papers.append(paper)
        
        if not theory_papers or not practice_papers:
            logger.warning("Not enough theory or practice papers for semantic analysis")
            return []
        
        # Compute embeddings matrices
        theory_embeddings = np.array([p['embedding'] for p in theory_papers])
        practice_embeddings = np.array([p['embedding'] for p in practice_papers])
        
        # Find papers similar to both theory and practice
        bridges = []
        
        for paper in papers:
            embedding = np.array(paper['embedding']).reshape(1, -1)
            
            # Compute similarities
            theory_sims = cosine_similarity(embedding, theory_embeddings)[0]
            practice_sims = cosine_similarity(embedding, practice_embeddings)[0]
            
            # High similarity to both domains indicates a bridge
            avg_theory_sim = np.mean(np.sort(theory_sims)[-10:])  # Top 10 similarities
            avg_practice_sim = np.mean(np.sort(practice_sims)[-10:])
            
            if avg_theory_sim > 0.7 and avg_practice_sim > 0.7:
                bridge_score = 2 * avg_theory_sim * avg_practice_sim / (avg_theory_sim + avg_practice_sim)
                bridges.append({
                    'id': paper['id'],
                    'theory_similarity': float(avg_theory_sim),
                    'practice_similarity': float(avg_practice_sim),
                    'bridge_score': float(bridge_score)
                })
        
        bridges.sort(key=lambda x: x['bridge_score'], reverse=True)
        
        logger.info(f"Found {len(bridges)} semantic bridge candidates")
        
        return bridges[:100]
    
    def export_bridges(self, bridges: List[Dict], output_path: str):
        """Export discovered bridges."""
        logger.info(f"Exporting bridges to {output_path}...")
        
        # Get full metadata for bridges
        bridge_ids = [b['id'] for b in bridges]
        
        query = """
        FOR p IN arxiv_papers
            FILTER p._key IN @bridge_ids
            RETURN {
                id: p._key,
                title: p.title,
                abstract: p.abstract,
                categories: p.categories,
                authors: p.authors,
                update_date: p.update_date,
                pdf_url: p.pdf_url
            }
        """
        
        bridge_metadata = {p['id']: p for p in self.db.aql.execute(
            query, bind_vars={'bridge_ids': bridge_ids}
        )}
        
        # Combine with bridge scores
        export_data = []
        for bridge in bridges:
            metadata = bridge_metadata.get(bridge['id'], {})
            export_data.append({
                **metadata,
                **bridge
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} bridges")
    
    def print_bridge_summary(self, bridges: List[Dict], patterns: Dict):
        """Print summary of discovered bridges."""
        print("\n" + "="*80)
        print("THEORY-PRACTICE BRIDGE DISCOVERY RESULTS")
        print("="*80)
        
        print(f"\nFound {len(bridges)} bridge papers")
        
        if patterns:
            print("\nTop Bridge Categories:")
            for cat, count in patterns.get('top_categories', [])[:5]:
                print(f"  {cat:20s}: {count:3d} papers")
            
            print("\nTop Bridge Authors:")
            for author, count in patterns.get('top_authors', [])[:5]:
                print(f"  {author[:30]:30s}: {count:3d} papers")
            
            print("\nSample Bridge Titles:")
            for title in patterns.get('sample_titles', []):
                print(f"  - {title[:70]}...")
        
        print("\nTop 10 Bridges by Score:")
        for i, bridge in enumerate(bridges[:10], 1):
            print(f"\n{i}. Paper ID: {bridge['id']}")
            print(f"   Bridge Score: {bridge.get('bridge_score', 0):.3f}")
            
            if 'theory_connections' in bridge:
                print(f"   Theory Connections: {bridge['theory_connections']}")
                print(f"   Practice Connections: {bridge['practice_connections']}")
            
            if 'theory_similarity' in bridge:
                print(f"   Theory Similarity: {bridge['theory_similarity']:.3f}")
                print(f"   Practice Similarity: {bridge['practice_similarity']:.3f}")


@click.command()
@click.option('--method', type=click.Choice(['topological', 'semantic', 'both']), 
              default='both', help='Bridge discovery method')
@click.option('--sample', type=int, help='Sample size for analysis')
@click.option('--export', type=str, help='Export path for results')
def main(method, sample, export):
    """Discover theory-practice bridges in the academic graph."""
    
    discovery = BridgeDiscovery()
    
    all_bridges = []
    
    if method in ['topological', 'both']:
        # Classify papers
        classification = discovery.classify_papers(sample_size=sample)
        
        # Find topological bridges
        topo_bridges = discovery.find_topological_bridges(
            classification['theory'][:1000],
            classification['practice'][:1000]
        )
        
        for bridge in topo_bridges:
            bridge['method'] = 'topological'
        
        all_bridges.extend(topo_bridges)
    
    if method in ['semantic', 'both']:
        # Find semantic bridges
        semantic_bridges = discovery.find_semantic_bridges(
            sample_size=sample or 10000
        )
        
        for bridge in semantic_bridges:
            bridge['method'] = 'semantic'
        
        all_bridges.extend(semantic_bridges)
    
    # Remove duplicates and sort
    seen = set()
    unique_bridges = []
    for bridge in all_bridges:
        if bridge['id'] not in seen:
            seen.add(bridge['id'])
            unique_bridges.append(bridge)
    
    unique_bridges.sort(key=lambda x: x.get('bridge_score', 0), reverse=True)
    
    # Analyze patterns
    patterns = discovery.analyze_bridge_patterns(unique_bridges)
    
    # Print summary
    discovery.print_bridge_summary(unique_bridges, patterns)
    
    # Export if requested
    if export:
        discovery.export_bridges(unique_bridges, export)


if __name__ == '__main__':
    main()