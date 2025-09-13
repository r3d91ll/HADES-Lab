#!/usr/bin/env python3
"""
Analyze interdisciplinary connections in the academic graph.
Export high-value cross-category edges for theory-practice bridges.
"""

import os
import json
import logging
from collections import defaultdict, Counter
from pathlib import Path
from arango import ArangoClient
import pandas as pd
import click

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InterdisciplinaryAnalyzer:
    """Analyze cross-category connections in the academic graph."""
    
    def __init__(self):
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
    
    def analyze_keyword_bridges(self, min_similarity: float = 0.75):
        """Analyze high-similarity keyword bridges between categories."""
        logger.info("Analyzing keyword similarity bridges...")
        
        query = """
        FOR e IN keyword_similarity
        FILTER e.cross_category == true AND e.similarity >= @min_sim
        COLLECT 
            from_cat = e.from_category, 
            to_cat = e.to_category 
        AGGREGATE 
            count = COUNT(1),
            avg_sim = AVG(e.similarity),
            max_sim = MAX(e.similarity)
        SORT count DESC
        LIMIT 100
        RETURN {
            from_category: from_cat,
            to_category: to_cat,
            edge_count: count,
            avg_similarity: avg_sim,
            max_similarity: max_sim
        }
        """
        
        results = list(self.db.aql.execute(query, bind_vars={'min_sim': min_similarity}))
        
        # Create DataFrame for better analysis
        df = pd.DataFrame(results)
        
        if not df.empty:
            logger.info(f"Found {len(df)} category pairs with strong keyword connections")
            logger.info("\nTop 10 interdisciplinary bridges by edge count:")
            for _, row in df.head(10).iterrows():
                logger.info(f"  {row['from_category'][:20]:20s} <-> {row['to_category'][:20]:20s}: "
                          f"{row['edge_count']:,} edges (avg sim: {row['avg_similarity']:.3f})")
        
        return df
    
    def find_theory_practice_bridges(self):
        """Find papers that bridge theoretical and applied domains."""
        logger.info("\nFinding theory-practice bridges...")
        
        # Categories considered theoretical vs applied
        theoretical = ['math.', 'physics.', 'stat.', 'q-bio.', 'quant-']
        applied = ['cs.', 'eess.', 'econ.']
        
        query = """
        FOR e IN keyword_similarity
        FILTER e.cross_category == true AND e.similarity >= 0.8
        LET from_paper = DOCUMENT(e._from)
        LET to_paper = DOCUMENT(e._to)
        FILTER from_paper != null AND to_paper != null
        LET from_cats = from_paper.categories[0]
        LET to_cats = to_paper.categories[0]
        // Check if one is theoretical and other is applied
        FILTER (
            (SUBSTRING(from_cats, 0, 5) IN @theoretical AND SUBSTRING(to_cats, 0, 3) IN @applied) OR
            (SUBSTRING(from_cats, 0, 3) IN @applied AND SUBSTRING(to_cats, 0, 5) IN @theoretical)
        )
        LIMIT 100
        RETURN {
            from_id: from_paper._key,
            from_title: from_paper.title,
            from_category: from_cats,
            to_id: to_paper._key,
            to_title: to_paper.title,
            to_category: to_cats,
            similarity: e.similarity
        }
        """
        
        bridges = list(self.db.aql.execute(
            query, 
            bind_vars={'theoretical': theoretical, 'applied': applied}
        ))
        
        if bridges:
            logger.info(f"Found {len(bridges)} theory-practice bridge papers")
            logger.info("\nSample bridges:")
            for b in bridges[:5]:
                logger.info(f"\n  Theory: {b['from_category'][:15]:15s} | {b['from_title'][:60]}")
                logger.info(f"  Practice: {b['to_category'][:15]:15s} | {b['to_title'][:60]}")
                logger.info(f"  Similarity: {b['similarity']:.3f}")
        
        return bridges
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in interdisciplinary connections."""
        logger.info("\nAnalyzing temporal patterns...")
        
        query = """
        FOR e IN temporal_proximity
        FILTER e.weight > 1.0  // Cross-category bonus applied
        COLLECT month = e.month WITH COUNT INTO count
        SORT month DESC
        LIMIT 24
        RETURN {month: month, cross_category_edges: count}
        """
        
        temporal = list(self.db.aql.execute(query))
        
        if temporal:
            logger.info("Recent months with most cross-category temporal edges:")
            for t in temporal[:12]:
                logger.info(f"  {t['month']}: {t['cross_category_edges']:,} edges")
        
        return temporal
    
    def export_high_value_edges(self, output_dir: str = "exports"):
        """Export high-value interdisciplinary edges for further analysis."""
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info(f"\nExporting high-value edges to {output_dir}/")
        
        # Export top keyword bridges
        keyword_bridges = self.analyze_keyword_bridges(min_similarity=0.75)
        if not keyword_bridges.empty:
            keyword_bridges.to_csv(f"{output_dir}/keyword_bridges.csv", index=False)
            logger.info(f"  Exported {len(keyword_bridges)} keyword bridges")
        
        # Export theory-practice bridges
        bridges = self.find_theory_practice_bridges()
        if bridges:
            with open(f"{output_dir}/theory_practice_bridges.json", 'w') as f:
                json.dump(bridges, f, indent=2)
            logger.info(f"  Exported {len(bridges)} theory-practice bridges")
        
        # Export temporal patterns
        temporal = self.analyze_temporal_patterns()
        if temporal:
            with open(f"{output_dir}/temporal_patterns.json", 'w') as f:
                json.dump(temporal, f, indent=2)
            logger.info(f"  Exported temporal patterns")
        
        # Export summary statistics
        stats = self.get_graph_statistics()
        with open(f"{output_dir}/graph_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"  Exported graph statistics")
    
    def get_graph_statistics(self):
        """Get overall graph statistics."""
        stats = {}
        
        # Count edges by type
        edge_collections = ['same_field', 'temporal_proximity', 'keyword_similarity']
        for coll_name in edge_collections:
            try:
                count = self.db.collection(coll_name).count()
                stats[f'{coll_name}_edges'] = count
                
                # Count cross-category edges
                if coll_name == 'keyword_similarity':
                    query = f"""
                    FOR e IN {coll_name}
                    FILTER e.cross_category == true
                    COLLECT WITH COUNT INTO c
                    RETURN c
                    """
                    cross = list(self.db.aql.execute(query))
                    stats[f'{coll_name}_cross_category'] = cross[0] if cross else 0
            except:
                pass
        
        # Count papers with embeddings
        query = """
        FOR p IN arxiv_papers
        FILTER p.keywords != null
        COLLECT WITH COUNT INTO c
        RETURN c
        """
        stats['papers_with_keywords'] = list(self.db.aql.execute(query))[0]
        
        return stats


@click.command()
@click.option('--output-dir', default='exports', help='Output directory for exports')
@click.option('--min-similarity', default=0.75, help='Minimum similarity for analysis')
def main(output_dir: str, min_similarity: float):
    """Analyze interdisciplinary connections in the academic graph."""
    analyzer = InterdisciplinaryAnalyzer()
    
    # Run all analyses
    analyzer.analyze_keyword_bridges(min_similarity)
    analyzer.find_theory_practice_bridges()
    analyzer.analyze_temporal_patterns()
    
    # Export results
    analyzer.export_high_value_edges(output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()