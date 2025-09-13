#!/usr/bin/env python3
"""
Check the progress of graph building without interrupting the process.
Can be run while build_graph_parallel.py is running.
"""

import os
import time
from datetime import datetime, timedelta
from arango import ArangoClient
import psutil

def check_graph_progress():
    """Check current graph building progress."""
    
    # Connect to database
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db(
        'academy_store',
        username='root',
        password=os.environ.get('ARANGO_PASSWORD')
    )
    
    print("\n" + "="*70)
    print(f"GRAPH BUILD PROGRESS CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check edge collections
    collections = {
        'same_field': 'Category edges',
        'temporal_proximity': 'Temporal edges',
        'keyword_similarity': 'Keyword edges',
        'citations': 'Citation edges'
    }
    
    total_edges = 0
    for coll_name, description in collections.items():
        if coll_name in [c['name'] for c in db.collections()]:
            count = db.collection(coll_name).count()
            total_edges += count
            print(f"{description:20s}: {count:>12,} edges")
        else:
            print(f"{description:20s}: Not created yet")
    
    print(f"{'TOTAL':20s}: {total_edges:>12,} edges")
    
    # Calculate average degree
    paper_count = db.collection('arxiv_papers').count()
    avg_degree = (total_edges * 2) / paper_count if paper_count > 0 else 0
    
    print(f"\nGraph Statistics:")
    print(f"  Papers: {paper_count:,}")
    print(f"  Total edges: {total_edges:,}")
    print(f"  Average degree: {avg_degree:.1f}")
    
    # Check for running processes
    print(f"\nProcess Status:")
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
        try:
            if 'python' in proc.info['name']:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'build_graph' in cmdline or 'graphsage' in cmdline:
                    python_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if python_processes:
        print(f"  Found {len(python_processes)} relevant Python processes:")
        for proc in python_processes[:5]:  # Show first 5
            try:
                cpu = proc.cpu_percent(interval=0.1)
                cmdline = ' '.join(proc.cmdline())
                if len(cmdline) > 80:
                    cmdline = cmdline[:77] + "..."
                print(f"    PID {proc.pid}: CPU {cpu:5.1f}% - {cmdline}")
            except:
                pass
    else:
        print("  No graph building processes detected")
    
    # Estimate completion for temporal edges
    print(f"\nTemporal Edge Estimation:")
    temporal_count = db.collection('temporal_proximity').count() if 'temporal_proximity' in [c['name'] for c in db.collections()] else 0
    
    # Based on observed patterns
    expected_temporal = 5_600_000  # ~2824 batches * ~2000 edges
    if temporal_count > 0:
        completion_pct = (temporal_count / expected_temporal) * 100
        print(f"  Current: {temporal_count:,} edges")
        print(f"  Expected: ~{expected_temporal:,} edges")
        print(f"  Estimated completion: {completion_pct:.1f}%")
        
        # Try to estimate time remaining (very rough)
        if completion_pct > 10:  # Need some data to estimate
            # Assume linear progress (not perfect but gives idea)
            remaining_pct = 100 - completion_pct
            # Can't know actual runtime without logs, but can guess
            print(f"  Remaining: ~{remaining_pct:.1f}% to process")
    else:
        print("  Temporal edges not started or collection doesn't exist")
    
    print("="*70)
    
    # Monitor edge growth
    print("\nMonitoring edge growth (checking every 30 seconds for 2 minutes)...")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        prev_counts = {}
        for coll_name in collections.keys():
            if coll_name in [c['name'] for c in db.collections()]:
                prev_counts[coll_name] = db.collection(coll_name).count()
        
        for i in range(4):  # Check 4 times (2 minutes total)
            time.sleep(30)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Edge count changes:")
            for coll_name, description in collections.items():
                if coll_name in [c['name'] for c in db.collections()]:
                    current = db.collection(coll_name).count()
                    if coll_name in prev_counts:
                        diff = current - prev_counts[coll_name]
                        if diff > 0:
                            rate = diff * 2  # per minute
                            print(f"  {description}: +{diff:,} edges (rate: {rate:,}/min)")
                    prev_counts[coll_name] = current
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

if __name__ == "__main__":
    check_graph_progress()