#!/usr/bin/env python3
"""
Explore the academic graph structure and find interesting patterns.
"""

import os
from arango import ArangoClient
import click

client = ArangoClient(hosts='http://localhost:8529')
db = client.db('academy_store', username='root', password=os.environ.get('ARANGO_PASSWORD'))


def analyze_graph():
    """Analyze graph structure."""
    print("="*80)
    print("GRAPH STRUCTURE ANALYSIS")
    print("="*80)
    
    # Edge statistics
    edge_stats = {}
    collections = ['coauthorship', 'same_field', 'temporal_proximity', 
                   'same_journal', 'same_submitter', 'same_conference']
    
    total = 0
    for coll_name in collections:
        if coll_name in [c['name'] for c in db.collections()]:
            count = db.collection(coll_name).count()
            edge_stats[coll_name] = count
            total += count
            print(f"{coll_name:20s}: {count:,} edges")
    
    print(f"{'TOTAL':20s}: {total:,} edges")
    
    # Find most connected papers
    print("\n" + "="*80)
    print("MOST CONNECTED PAPERS (by coauthorship)")
    print("="*80)
    
    query = """
    FOR paper IN arxiv_papers
        LET coauthor_count = LENGTH(
            FOR e IN coauthorship
                FILTER e._from == CONCAT('arxiv_papers/', paper._key) 
                   OR e._to == CONCAT('arxiv_papers/', paper._key)
                RETURN 1
        )
        FILTER coauthor_count > 0
        SORT coauthor_count DESC
        LIMIT 10
        RETURN {
            id: paper._key,
            title: paper.title,
            authors: paper.authors,
            connections: coauthor_count
        }
    """
    
    for paper in db.aql.execute(query):
        print(f"\n{paper['id']}: {paper['connections']} connections")
        print(f"  Title: {paper['title'][:70]}...")
        print(f"  Authors: {paper['authors'][:70]}...")
    
    # Find most prolific authors
    print("\n" + "="*80)
    print("MOST PROLIFIC AUTHORS")
    print("="*80)
    
    query = """
    FOR author IN authors
        SORT author.paper_count DESC
        LIMIT 20
        RETURN {
            name: author.name,
            papers: author.paper_count,
            categories: author.categories
        }
    """
    
    for author in db.aql.execute(query):
        cats = ', '.join(author['categories'][:3]) if author['categories'] else 'N/A'
        print(f"{author['name']:30s}: {author['papers']:4} papers | Fields: {cats}")
    
    # Category distribution
    print("\n" + "="*80)
    print("TOP RESEARCH CATEGORIES")
    print("="*80)
    
    query = """
    FOR p IN arxiv_papers
        FILTER p.categories != null
        FOR cat IN p.categories
            COLLECT category = cat WITH COUNT INTO count
            SORT count DESC
            LIMIT 20
            RETURN {category: category, papers: count}
    """
    
    for cat in db.aql.execute(query):
        print(f"{cat['category']:20s}: {cat['papers']:,} papers")
    
    # Find cross-disciplinary papers
    print("\n" + "="*80)
    print("MOST INTERDISCIPLINARY PAPERS")
    print("="*80)
    
    query = """
    FOR p IN arxiv_papers
        FILTER LENGTH(p.categories) > 3
        SORT LENGTH(p.categories) DESC
        LIMIT 10
        RETURN {
            id: p._key,
            title: p.title,
            categories: p.categories
        }
    """
    
    for paper in db.aql.execute(query):
        print(f"\n{paper['id']}: {len(paper['categories'])} categories")
        print(f"  Title: {paper['title'][:70]}...")
        print(f"  Categories: {', '.join(paper['categories'])}")
    
    # Temporal patterns
    print("\n" + "="*80)
    print("TEMPORAL PATTERNS")
    print("="*80)
    
    query = """
    FOR p IN arxiv_papers
        FILTER p.update_date != null
        COLLECT year = SUBSTRING(p.update_date, 0, 4) WITH COUNT INTO count
        SORT year
        RETURN {year: year, papers: count}
    """
    
    year_counts = list(db.aql.execute(query))
    
    # Show last 10 years
    print("\nPapers by year (last 10 years):")
    for item in year_counts[-10:]:
        bar = '█' * int(item['papers'] / 10000)
        print(f"  {item['year']}: {item['papers']:7,} {bar}")


def find_theory_practice_candidates():
    """Find potential theory-practice bridge papers."""
    print("\n" + "="*80)
    print("THEORY-PRACTICE BRIDGE CANDIDATES")
    print("="*80)
    
    # Papers with theory indicators
    theory_query = """
    FOR p IN arxiv_papers
        FILTER p.title != null
        FILTER CONTAINS(LOWER(p.title), 'theory') 
            OR CONTAINS(LOWER(p.title), 'theorem')
            OR CONTAINS(LOWER(p.title), 'proof')
        LIMIT 5
        RETURN {
            id: p._key,
            title: p.title,
            categories: p.categories
        }
    """
    
    # Papers with practice indicators  
    practice_query = """
    FOR p IN arxiv_papers
        FILTER p.title != null
        FILTER CONTAINS(LOWER(p.title), 'implementation')
            OR CONTAINS(LOWER(p.title), 'application') 
            OR CONTAINS(LOWER(p.title), 'system')
            OR CONTAINS(LOWER(p.title), 'practical')
        LIMIT 5
        RETURN {
            id: p._key,
            title: p.title,
            categories: p.categories
        }
    """
    
    print("\nTheoretical papers:")
    for paper in db.aql.execute(theory_query):
        print(f"  {paper['id']}: {paper['title'][:70]}...")
    
    print("\nPractical papers:")
    for paper in db.aql.execute(practice_query):
        print(f"  {paper['id']}: {paper['title'][:70]}...")
    
    # Papers that mention both
    bridge_query = """
    FOR p IN arxiv_papers
        FILTER p.title != null
        FILTER (CONTAINS(LOWER(p.title), 'theory') OR CONTAINS(LOWER(p.title), 'theorem'))
           AND (CONTAINS(LOWER(p.title), 'application') OR CONTAINS(LOWER(p.title), 'practical'))
        LIMIT 10
        RETURN {
            id: p._key,
            title: p.title,
            categories: p.categories
        }
    """
    
    print("\nPotential bridge papers (theory + application):")
    for paper in db.aql.execute(bridge_query):
        print(f"  {paper['id']}: {paper['title'][:70]}...")


@click.command()
@click.option('--bridges', is_flag=True, help='Find theory-practice bridges')
def main(bridges):
    """Explore the academic graph."""
    analyze_graph()
    
    if bridges:
        find_theory_practice_candidates()


if __name__ == '__main__':
    main()