#!/usr/bin/env python3
"""
Monitor the graph building progress and show statistics.
"""

import os
import time
import logging
from arango import ArangoClient
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def monitor_graph_build():
    """Monitor graph building progress."""
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db(
        'academy_store',
        username='root',
        password=os.environ.get('ARANGO_PASSWORD')
    )
    
    start_time = datetime.now()
    
    while True:
        print("\033[2J\033[H")  # Clear screen
        print("="*70)
        print("GRAPH BUILD MONITOR")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed: {str(datetime.now() - start_time).split('.')[0]}")
        print("="*70)
        
        # Check edge collections
        collections = {
            'same_field': 'Category edges',
            'temporal_proximity': 'Temporal edges', 
            'keyword_similarity': 'Keyword edges',
            'citations': 'Citation edges',
            'coauthorship': 'Coauthor edges'
        }
        
        total_edges = 0
        for coll_name, description in collections.items():
            try:
                if coll_name in [c['name'] for c in db.collections()]:
                    count = db.collection(coll_name).count()
                    total_edges += count
                    status = "✓" if count > 0 else "○"
                    print(f"{status} {description:20s}: {count:>15,} edges")
                else:
                    print(f"○ {description:20s}: Collection not found")
            except Exception as e:
                print(f"✗ {description:20s}: Error: {e}")
        
        print("-"*70)
        print(f"TOTAL EDGES: {total_edges:>15,}")
        
        # Check interdisciplinary connections
        if 'keyword_similarity' in [c['name'] for c in db.collections()]:
            try:
                query = """
                FOR edge IN keyword_similarity
                    LET from_paper = DOCUMENT(edge._from)
                    LET to_paper = DOCUMENT(edge._to)
                    FILTER from_paper.categories[0] != to_paper.categories[0]
                    COLLECT WITH COUNT INTO cross_field
                    RETURN cross_field
                """
                result = list(db.aql.execute(query))
                if result:
                    cross_count = result[0]
                    print(f"\nInterdisciplinary connections: {cross_count:,}")
            except:
                pass
        
        # Memory usage
        try:
            import psutil
            process = psutil.Process()
            mem = process.memory_info()
            print(f"\nMemory usage: {mem.rss / 1024**3:.1f} GB")
            print(f"CPU usage: {psutil.cpu_percent(interval=1):.1f}%")
        except:
            pass
        
        print("\n[Press Ctrl+C to exit]")
        time.sleep(10)  # Update every 10 seconds


if __name__ == "__main__":
    try:
        monitor_graph_build()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")