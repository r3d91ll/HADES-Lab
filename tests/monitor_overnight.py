#!/usr/bin/env python3
"""
Monitor the overnight test progress.
"""

import time
import json
from pathlib import Path
from datetime import datetime

def monitor_test():
    """Monitor the overnight test progress."""
    
    # Find the most recent log and results files
    log_dir = Path("tests/logs")
    
    # Get the most recent log file
    log_files = sorted(log_dir.glob("overnight_arxiv_test_*.log"), key=lambda x: x.stat().st_mtime)
    if not log_files:
        print("No test log files found")
        return
    
    latest_log = log_files[-1]
    print(f"Monitoring: {latest_log.name}")
    
    # Check if result file exists
    timestamp = latest_log.stem.split('_')[-1]
    result_file = log_dir / f"overnight_arxiv_results_{timestamp}.json"
    
    while True:
        # Check if test is complete
        if result_file.exists():
            with open(result_file) as f:
                results = json.load(f)
            
            if 'summary' in results:
                print("\n" + "="*50)
                print("TEST COMPLETE!")
                print("="*50)
                print(f"Best rate: {results['summary']['best_rate_papers_per_minute']:.1f} papers/minute")
                print(f"Best config: {results['summary']['best_configuration']}")
                break
        
        # Get latest progress from log
        with open(latest_log) as f:
            lines = f.readlines()
        
        # Find latest progress line
        for line in reversed(lines):
            if "Progress:" in line:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {line.strip()}")
                break
            elif "Results for" in line:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {line.strip()}")
                break
            elif "TEST COMPLETE" in line:
                print(f"\n{line.strip()}")
                return
        
        # Wait 60 seconds before next check
        time.sleep(60)

if __name__ == "__main__":
    monitor_test()