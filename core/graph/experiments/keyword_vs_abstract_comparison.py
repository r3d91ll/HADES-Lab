#!/usr/bin/env python3
"""
Experiment: Keywords vs Abstracts for Graph Construction

Tests whether TF-IDF extracted keywords can serve as a computationally 
efficient proxy for full abstracts in graph construction.

Hypothesis: Keywords capture 80% of the signal at 20% of the cost.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import click
import numpy as np
import torch
from tqdm import tqdm
from arango import ArangoClient
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphQualityExperiment:
    """Compare graph quality between keyword and abstract-based edges."""
    
    def __init__(self, sample_size: int = 10000):
        """
        Initialize experiment with a sample of papers.
        
        Args:
            sample_size: Number of papers to sample for experiment
        """
        self.sample_size = sample_size
        
        # Connect to database
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        self.results = {
            'sample_size': sample_size,
            'timestamp': datetime.now().isoformat(),
            'keyword_graph': {},
            'abstract_graph': {},
            'comparison': {}
        }
    
    def load_sample_data(self) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """Load a sample of papers with both keyword and abstract embeddings."""
        logger.info(f"Loading {self.sample_size} papers with both embeddings...")
        
        # Get papers that have BOTH embeddings
        query = f"""
        FOR p IN arxiv_papers
        FILTER p._key IN (
            FOR e IN arxiv_embeddings
            FILTER e.keyword_embedding != null AND e.abstract_embedding != null
            RETURN e.paper_id
        )
        SORT RAND()
        LIMIT {self.sample_size}
        RETURN {{
            _key: p._key,
            title: p.title,
            categories: p.categories,
            update_date: p.update_date
        }}
        """
        
        papers = list(self.db.aql.execute(query))
        paper_ids = [p['_key'] for p in papers]
        
        # Load embeddings
        keyword_embeddings = []
        abstract_embeddings = []
        
        for paper_id in tqdm(paper_ids, desc="Loading embeddings"):
            # Get keyword embedding
            key_doc = self.db.collection('arxiv_embeddings').get(f"{paper_id}_keyword")
            if key_doc and 'keyword_embedding' in key_doc:
                keyword_embeddings.append(key_doc['keyword_embedding'])
            else:
                keyword_embeddings.append(None)
            
            # Get abstract embedding
            abs_doc = self.db.collection('arxiv_embeddings').get(f"{paper_id}_abstract")
            if abs_doc and 'abstract_embedding' in abs_doc:
                abstract_embeddings.append(abs_doc['abstract_embedding'])
            else:
                abstract_embeddings.append(None)
        
        # Filter out None values
        valid_indices = [i for i in range(len(papers)) 
                        if keyword_embeddings[i] is not None 
                        and abstract_embeddings[i] is not None]
        
        papers = [papers[i] for i in valid_indices]
        keyword_embeddings = np.array([keyword_embeddings[i] for i in valid_indices])
        abstract_embeddings = np.array([abstract_embeddings[i] for i in valid_indices])
        
        logger.info(f"Loaded {len(papers)} papers with both embeddings")
        logger.info(f"Keyword embedding shape: {keyword_embeddings.shape}")
        logger.info(f"Abstract embedding shape: {abstract_embeddings.shape}")
        
        return papers, keyword_embeddings, abstract_embeddings
    
    def build_similarity_graph(self, embeddings: np.ndarray, threshold: float, 
                             name: str) -> Dict[str, Any]:
        """Build a similarity graph from embeddings."""
        logger.info(f"Building {name} similarity graph (threshold={threshold})...")
        
        start_time = time.time()
        
        # Compute similarity matrix (using cosine similarity)
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        
        # Compute similarities
        similarities = np.dot(normalized, normalized.T)
        
        # Apply threshold
        edges = []
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i, j] >= threshold:
                    edges.append((i, j, float(similarities[i, j])))
        
        build_time = time.time() - start_time
        
        # Create NetworkX graph for analysis
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        
        # Compute graph statistics
        stats = {
            'num_edges': len(edges),
            'build_time_seconds': build_time,
            'threshold': threshold,
            'avg_similarity': np.mean([e[2] for e in edges]) if edges else 0,
            'density': nx.density(G) if G.number_of_nodes() > 0 else 0,
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            'clustering_coefficient': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0,
            'num_components': nx.number_connected_components(G),
            'largest_component_size': len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0
        }
        
        logger.info(f"{name} graph: {stats['num_edges']} edges in {build_time:.2f}s")
        
        return {'graph': G, 'stats': stats, 'edges': edges}
    
    def compare_graphs(self, keyword_graph: Dict, abstract_graph: Dict, 
                       papers: List[Dict]) -> Dict[str, Any]:
        """Compare the two graphs in detail."""
        logger.info("Comparing keyword and abstract graphs...")
        
        kg = keyword_graph['graph']
        ag = abstract_graph['graph']
        
        # Edge overlap
        kg_edges = set([(min(u, v), max(u, v)) for u, v in kg.edges()])
        ag_edges = set([(min(u, v), max(u, v)) for u, v in ag.edges()])
        
        overlap = kg_edges.intersection(ag_edges)
        union = kg_edges.union(ag_edges)
        
        # Neighborhood preservation (sample nodes)
        sample_nodes = np.random.choice(list(kg.nodes()), 
                                      min(100, kg.number_of_nodes()), 
                                      replace=False)
        
        neighborhood_similarities = []
        for node in sample_nodes:
            kg_neighbors = set(kg.neighbors(node)) if kg.has_node(node) else set()
            ag_neighbors = set(ag.neighbors(node)) if ag.has_node(node) else set()
            
            if kg_neighbors or ag_neighbors:
                jaccard = len(kg_neighbors.intersection(ag_neighbors)) / len(kg_neighbors.union(ag_neighbors))
                neighborhood_similarities.append(jaccard)
        
        comparison = {
            'edge_overlap': {
                'intersection': len(overlap),
                'union': len(union),
                'jaccard_similarity': len(overlap) / len(union) if union else 0,
                'keyword_only': len(kg_edges - ag_edges),
                'abstract_only': len(ag_edges - kg_edges)
            },
            'neighborhood_preservation': {
                'mean_jaccard': np.mean(neighborhood_similarities) if neighborhood_similarities else 0,
                'std_jaccard': np.std(neighborhood_similarities) if neighborhood_similarities else 0,
                'min_jaccard': np.min(neighborhood_similarities) if neighborhood_similarities else 0,
                'max_jaccard': np.max(neighborhood_similarities) if neighborhood_similarities else 0
            },
            'computational_efficiency': {
                'keyword_time': keyword_graph['stats']['build_time_seconds'],
                'abstract_time': abstract_graph['stats']['build_time_seconds'],
                'speedup': abstract_graph['stats']['build_time_seconds'] / keyword_graph['stats']['build_time_seconds'],
                'keyword_edges_per_second': keyword_graph['stats']['num_edges'] / keyword_graph['stats']['build_time_seconds'],
                'abstract_edges_per_second': abstract_graph['stats']['num_edges'] / abstract_graph['stats']['build_time_seconds']
            },
            'quality_metrics': {
                'keyword_clustering': keyword_graph['stats']['clustering_coefficient'],
                'abstract_clustering': abstract_graph['stats']['clustering_coefficient'],
                'keyword_density': keyword_graph['stats']['density'],
                'abstract_density': abstract_graph['stats']['density'],
                'keyword_components': keyword_graph['stats']['num_components'],
                'abstract_components': abstract_graph['stats']['num_components']
            }
        }
        
        return comparison
    
    def run_experiment(self, keyword_threshold: float = 0.65, 
                       abstract_threshold: float = 0.75) -> Dict[str, Any]:
        """Run the complete experiment."""
        logger.info("="*70)
        logger.info("KEYWORD VS ABSTRACT GRAPH EXPERIMENT")
        logger.info("="*70)
        
        # Load sample data
        papers, keyword_embeddings, abstract_embeddings = self.load_sample_data()
        
        # Build keyword graph
        logger.info("\n" + "="*70)
        logger.info("BUILDING KEYWORD GRAPH")
        logger.info("="*70)
        keyword_graph = self.build_similarity_graph(
            keyword_embeddings, 
            keyword_threshold, 
            "Keyword"
        )
        self.results['keyword_graph'] = keyword_graph['stats']
        
        # Build abstract graph
        logger.info("\n" + "="*70)
        logger.info("BUILDING ABSTRACT GRAPH")
        logger.info("="*70)
        abstract_graph = self.build_similarity_graph(
            abstract_embeddings, 
            abstract_threshold, 
            "Abstract"
        )
        self.results['abstract_graph'] = abstract_graph['stats']
        
        # Compare graphs
        logger.info("\n" + "="*70)
        logger.info("COMPARING GRAPHS")
        logger.info("="*70)
        comparison = self.compare_graphs(keyword_graph, abstract_graph, papers)
        self.results['comparison'] = comparison
        
        # Print results
        self.print_results()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def print_results(self):
        """Print experiment results in a readable format."""
        print("\n" + "="*70)
        print("EXPERIMENT RESULTS")
        print("="*70)
        
        print("\n1. GRAPH STATISTICS")
        print("-" * 40)
        print(f"{'Metric':<30} {'Keywords':<15} {'Abstracts':<15}")
        print("-" * 40)
        
        kg = self.results['keyword_graph']
        ag = self.results['abstract_graph']
        
        metrics = [
            ('Number of edges', 'num_edges'),
            ('Build time (seconds)', 'build_time_seconds'),
            ('Average degree', 'avg_degree'),
            ('Clustering coefficient', 'clustering_coefficient'),
            ('Graph density', 'density'),
            ('Connected components', 'num_components'),
            ('Largest component size', 'largest_component_size')
        ]
        
        for label, key in metrics:
            kval = kg.get(key, 0)
            aval = ag.get(key, 0)
            if isinstance(kval, float):
                print(f"{label:<30} {kval:<15.4f} {aval:<15.4f}")
            else:
                print(f"{label:<30} {kval:<15} {aval:<15}")
        
        print("\n2. EDGE OVERLAP")
        print("-" * 40)
        overlap = self.results['comparison']['edge_overlap']
        print(f"Edges in both graphs:     {overlap['intersection']:,}")
        print(f"Edges only in keywords:   {overlap['keyword_only']:,}")
        print(f"Edges only in abstracts:  {overlap['abstract_only']:,}")
        print(f"Jaccard similarity:       {overlap['jaccard_similarity']:.3f}")
        
        print("\n3. COMPUTATIONAL EFFICIENCY")
        print("-" * 40)
        efficiency = self.results['comparison']['computational_efficiency']
        print(f"Keyword build time:       {efficiency['keyword_time']:.2f}s")
        print(f"Abstract build time:      {efficiency['abstract_time']:.2f}s")
        print(f"Speedup with keywords:    {efficiency['speedup']:.2f}x")
        
        print("\n4. NEIGHBORHOOD PRESERVATION")
        print("-" * 40)
        neighborhood = self.results['comparison']['neighborhood_preservation']
        print(f"Mean Jaccard similarity:  {neighborhood['mean_jaccard']:.3f}")
        print(f"Std deviation:            {neighborhood['std_jaccard']:.3f}")
        print(f"Min similarity:           {neighborhood['min_jaccard']:.3f}")
        print(f"Max similarity:           {neighborhood['max_jaccard']:.3f}")
        
        print("\n5. VERDICT")
        print("-" * 40)
        
        # Calculate quality score
        quality_score = neighborhood['mean_jaccard']
        speed_gain = efficiency['speedup']
        
        if quality_score > 0.7:
            verdict = "STRONG: Keywords preserve >70% of abstract graph structure"
        elif quality_score > 0.5:
            verdict = "MODERATE: Keywords preserve 50-70% of abstract graph structure"
        else:
            verdict = "WEAK: Keywords preserve <50% of abstract graph structure"
        
        print(f"Neighborhood preservation: {quality_score:.1%}")
        print(f"Computational speedup:     {speed_gain:.1f}x")
        print(f"Verdict: {verdict}")
        
        if quality_score > 0.6 and speed_gain > 2:
            print("\n✓ RECOMMENDATION: Keywords are a viable efficient proxy!")
        elif quality_score > 0.5:
            print("\n⚠ RECOMMENDATION: Keywords are acceptable with quality tradeoff")
        else:
            print("\n✗ RECOMMENDATION: Use abstracts for better graph quality")
    
    def save_results(self):
        """Save experiment results to file."""
        output_file = f"keyword_vs_abstract_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nResults saved to {output_file}")


@click.command()
@click.option('--sample-size', default=10000, help='Number of papers to sample')
@click.option('--keyword-threshold', default=0.65, help='Similarity threshold for keywords')
@click.option('--abstract-threshold', default=0.75, help='Similarity threshold for abstracts')
def main(sample_size: int, keyword_threshold: float, abstract_threshold: float):
    """Run keyword vs abstract graph quality experiment."""
    
    experiment = GraphQualityExperiment(sample_size=sample_size)
    results = experiment.run_experiment(
        keyword_threshold=keyword_threshold,
        abstract_threshold=abstract_threshold
    )


if __name__ == "__main__":
    main()