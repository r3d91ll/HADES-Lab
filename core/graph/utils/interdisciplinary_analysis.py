#!/usr/bin/env python3
"""
Interdisciplinary bridge analysis for the graph.

Identifies and analyzes cross-disciplinary connections created through
keyword similarity, revealing how concepts migrate across fields.
"""

import os
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from arango import ArangoClient
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterdisciplinaryAnalyzer:
    """Analyze interdisciplinary connections in the graph."""
    
    def __init__(self):
        """Initialize the analyzer."""
        # Connect to database
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
    
    def find_bridge_papers(self, min_connections: int = 10) -> List[Dict]:
        """Find papers that bridge multiple disciplines.
        
        Args:
            min_connections: Minimum interdisciplinary connections
            
        Returns:
            List of bridge papers with their statistics
        """
        logger.info("Finding interdisciplinary bridge papers...")
        
        # Query for papers with many cross-category keyword connections
        query = """
        FOR edge IN keyword_similarity
            LET from_paper = DOCUMENT(edge._from)
            LET to_paper = DOCUMENT(edge._to)
            FILTER from_paper.primary_category != to_paper.primary_category
            COLLECT paper_id = from_paper._key INTO connections
            LET connection_count = LENGTH(connections)
            FILTER connection_count >= @min_connections
            LET paper = DOCUMENT(CONCAT('arxiv_papers/', paper_id))
            RETURN {
                paper_id: paper_id,
                title: paper.title,
                category: paper.primary_category,
                year: SUBSTRING(paper.arxiv_id, 0, 2),
                interdisciplinary_connections: connection_count,
                connected_categories: (
                    FOR c IN connections
                        LET other = DOCUMENT(c.edge._to)
                        RETURN DISTINCT other.primary_category
                )
            }
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={'min_connections': min_connections}
        )
        
        bridge_papers = list(cursor)
        logger.info(f"Found {len(bridge_papers)} bridge papers")
        
        return bridge_papers
    
    def analyze_concept_migration(self, keyword: str) -> Dict:
        """Analyze how a concept/keyword migrates across disciplines over time.
        
        Args:
            keyword: The keyword to track
            
        Returns:
            Migration analysis dictionary
        """
        logger.info(f"Analyzing migration of concept: {keyword}")
        
        # Find papers containing this keyword
        papers_with_keyword = list(self.db.collection('arxiv_papers').find(
            {'keywords': {'like': f'%{keyword}%'}}
        ))
        
        # Group by year and category
        migration_map = defaultdict(lambda: defaultdict(int))
        
        for paper in papers_with_keyword:
            if 'arxiv_id' in paper and 'primary_category' in paper:
                year = 2000 + int(paper['arxiv_id'][:2])  # Convert YY to YYYY
                category = paper['primary_category'].split('.')[0]  # Main category
                migration_map[year][category] += 1
        
        # Find first appearance in each field
        first_appearance = {}
        for year in sorted(migration_map.keys()):
            for category in migration_map[year]:
                if category not in first_appearance:
                    first_appearance[category] = year
        
        # Calculate migration delays
        if first_appearance:
            origin_year = min(first_appearance.values())
            origin_fields = [cat for cat, year in first_appearance.items() if year == origin_year]
            
            migration_delays = {
                cat: year - origin_year 
                for cat, year in first_appearance.items() 
                if year > origin_year
            }
        else:
            origin_fields = []
            migration_delays = {}
        
        return {
            'keyword': keyword,
            'total_papers': len(papers_with_keyword),
            'origin_fields': origin_fields,
            'origin_year': origin_year if first_appearance else None,
            'migration_delays': migration_delays,
            'yearly_distribution': dict(migration_map)
        }
    
    def find_unexpected_connections(self, threshold: float = 0.8) -> List[Dict]:
        """Find surprising cross-disciplinary connections.
        
        Args:
            threshold: Minimum similarity for unexpected connections
            
        Returns:
            List of unexpected connections
        """
        logger.info("Finding unexpected interdisciplinary connections...")
        
        # Define "unexpected" category pairs (traditionally unrelated fields)
        unexpected_pairs = [
            ('cs.', 'q-bio.'),  # Computer Science <-> Quantitative Biology
            ('math.', 'q-fin.'),  # Mathematics <-> Quantitative Finance  
            ('physics.', 'econ.'),  # Physics <-> Economics
            ('cs.', 'astro-ph.'),  # Computer Science <-> Astrophysics
            ('math.', 'q-bio.'),  # Mathematics <-> Biology
            ('hep-', 'cs.'),  # High Energy Physics <-> Computer Science
        ]
        
        unexpected_connections = []
        
        for cat1_prefix, cat2_prefix in unexpected_pairs:
            # Query for high-similarity connections between these fields
            query = """
            FOR edge IN keyword_similarity
                FILTER edge.similarity >= @threshold
                LET from_paper = DOCUMENT(edge._from)
                LET to_paper = DOCUMENT(edge._to)
                FILTER CONTAINS(from_paper.primary_category, @cat1) 
                   AND CONTAINS(to_paper.primary_category, @cat2)
                LIMIT 100
                RETURN {
                    paper1: {
                        id: from_paper._key,
                        title: from_paper.title,
                        category: from_paper.primary_category,
                        year: SUBSTRING(from_paper.arxiv_id, 0, 2)
                    },
                    paper2: {
                        id: to_paper._key,
                        title: to_paper.title,
                        category: to_paper.primary_category,
                        year: SUBSTRING(to_paper.arxiv_id, 0, 2)
                    },
                    similarity: edge.similarity,
                    year_gap: ABS(
                        TO_NUMBER(SUBSTRING(from_paper.arxiv_id, 0, 2)) - 
                        TO_NUMBER(SUBSTRING(to_paper.arxiv_id, 0, 2))
                    )
                }
            """
            
            cursor = self.db.aql.execute(
                query,
                bind_vars={
                    'threshold': threshold,
                    'cat1': cat1_prefix,
                    'cat2': cat2_prefix
                }
            )
            
            connections = list(cursor)
            if connections:
                logger.info(f"Found {len(connections)} connections between {cat1_prefix}* and {cat2_prefix}*")
                unexpected_connections.extend(connections)
        
        return unexpected_connections
    
    def generate_report(self, output_path: str = None):
        """Generate comprehensive interdisciplinary analysis report.
        
        Args:
            output_path: Path to save the report
        """
        logger.info("Generating interdisciplinary analysis report...")
        
        report = {
            'timestamp': str(pd.Timestamp.now()),
            'bridge_papers': self.find_bridge_papers(min_connections=20),
            'unexpected_connections': self.find_unexpected_connections(threshold=0.75),
            'concept_migrations': []
        }
        
        # Analyze migration of key concepts
        key_concepts = [
            'transformer', 'attention', 'neural', 'quantum',
            'entropy', 'optimization', 'bayesian', 'graph'
        ]
        
        for concept in key_concepts:
            try:
                migration = self.analyze_concept_migration(concept)
                if migration['total_papers'] > 100:  # Only include well-represented concepts
                    report['concept_migrations'].append(migration)
                    logger.info(f"  {concept}: {migration['total_papers']} papers, "
                              f"originated in {migration['origin_fields']}")
            except Exception as e:
                logger.warning(f"Could not analyze concept '{concept}': {e}")
        
        # Calculate summary statistics
        report['summary'] = {
            'total_bridge_papers': len(report['bridge_papers']),
            'total_unexpected_connections': len(report['unexpected_connections']),
            'concepts_analyzed': len(report['concept_migrations']),
            'avg_migration_delay': np.mean([
                np.mean(list(m['migration_delays'].values())) 
                for m in report['concept_migrations'] 
                if m['migration_delays']
            ]) if report['concept_migrations'] else 0
        }
        
        # Save report
        if output_path is None:
            output_path = '/home/todd/olympus/HADES-Lab/data/interdisciplinary_report.json'
        
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {output_path}")
        logger.info(f"\nSummary:")
        logger.info(f"  Bridge papers found: {report['summary']['total_bridge_papers']}")
        logger.info(f"  Unexpected connections: {report['summary']['total_unexpected_connections']}")
        logger.info(f"  Average concept migration delay: {report['summary']['avg_migration_delay']:.1f} years")
        
        return report
    
    def visualize_interdisciplinary_network(self, output_dir: str = None):
        """Create visualizations of interdisciplinary connections.
        
        Args:
            output_dir: Directory to save visualizations
        """
        logger.info("Creating interdisciplinary visualizations...")
        
        if output_dir is None:
            output_dir = '/home/todd/olympus/HADES-Lab/data/visualizations'
        Path(output_dir).mkdir(exist_ok=True)
        
        # Get category pair counts
        query = """
        FOR edge IN keyword_similarity
            LET from_paper = DOCUMENT(edge._from)
            LET to_paper = DOCUMENT(edge._to)
            FILTER from_paper.primary_category != to_paper.primary_category
            COLLECT 
                cat1 = SPLIT(from_paper.primary_category, '.')[0],
                cat2 = SPLIT(to_paper.primary_category, '.')[0]
            WITH COUNT INTO count
            FILTER count >= 100
            RETURN {cat1: cat1, cat2: cat2, count: count}
        """
        
        cursor = self.db.aql.execute(query)
        connections = list(cursor)
        
        if connections:
            # Create heatmap of interdisciplinary connections
            categories = sorted(set(
                [c['cat1'] for c in connections] + 
                [c['cat2'] for c in connections]
            ))
            
            matrix = np.zeros((len(categories), len(categories)))
            cat_to_idx = {cat: i for i, cat in enumerate(categories)}
            
            for conn in connections:
                i = cat_to_idx[conn['cat1']]
                j = cat_to_idx[conn['cat2']]
                matrix[i, j] = conn['count']
                matrix[j, i] = conn['count']  # Symmetric
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                np.log1p(matrix),  # Log scale for better visualization
                xticklabels=categories,
                yticklabels=categories,
                cmap='YlOrRd',
                cbar_kws={'label': 'Log(Connection Count + 1)'}
            )
            plt.title('Interdisciplinary Keyword Connections Heatmap')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/interdisciplinary_heatmap.png", dpi=150)
            plt.close()
            
            logger.info(f"Heatmap saved to {output_dir}/interdisciplinary_heatmap.png")
        
        logger.info("Visualization complete")


if __name__ == "__main__":
    analyzer = InterdisciplinaryAnalyzer()
    
    # Generate full report
    report = analyzer.generate_report()
    
    # Create visualizations
    analyzer.visualize_interdisciplinary_network()
    
    # Find specific bridge papers
    bridges = analyzer.find_bridge_papers(min_connections=50)
    if bridges:
        logger.info("\nTop bridge papers:")
        for paper in bridges[:5]:
            logger.info(f"  {paper['title'][:80]}...")
            logger.info(f"    Category: {paper['category']}")
            logger.info(f"    Connections: {paper['interdisciplinary_connections']}")
