#!/usr/bin/env python3
"""
Monitor the phase-separated test progress.
"""

import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

def get_gpu_status():
    """Get GPU memory usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        gpus = []
        for line in lines:
            parts = line.split(', ')
            if len(parts) >= 4:
                gpus.append({
                    'id': int(parts[0]),
                    'util': int(parts[1]),
                    'mem_used': int(parts[2]) / 1024,  # Convert to GB
                    'mem_total': int(parts[3]) / 1024
                })
        return gpus
    except:
        return []

def monitor_test():
    """Monitor the phase-separated test."""
    
    log_file = Path("tests/logs/overnight_phased_dual_gpu.out")
    
    if not log_file.exists():
        print("Log file not found")
        return
    
    print(f"Monitoring: {log_file.name}")
    print("="*70)
    
    last_size = 0
    phase = "Unknown"
    
    while True:
        # Check file size to see if still running
        current_size = log_file.stat().st_size
        if current_size == last_size:
            inactive_count = getattr(monitor_test, 'inactive_count', 0) + 1
            monitor_test.inactive_count = inactive_count
            if inactive_count > 10:  # No activity for 10 minutes
                print("\nTest appears to have completed or stalled")
                break
        else:
            monitor_test.inactive_count = 0
            last_size = current_size
        
        # Get latest lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Find current phase and progress
        for line in reversed(lines[-100:]):
            if "PHASE 1: EXTRACTION" in line:
                phase = "Extraction"
            elif "PHASE 2: EMBEDDING" in line:
                phase = "Embedding"
            elif "Extraction progress:" in line or "Embedding progress:" in line:
                # Extract progress info
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase: {phase}")
                print(f"  {line.strip()}")
                
                # Show GPU status
                gpus = get_gpu_status()
                for gpu in gpus:
                    print(f"  GPU {gpu['id']}: {gpu['util']}% util, {gpu['mem_used']:.1f}/{gpu['mem_total']:.1f} GB")
                break
            elif "FINAL SUMMARY" in line:
                print("\n" + "="*70)
                print("TEST COMPLETE!")
                # Print summary lines
                for summary_line in lines[lines.index(line):]:
                    if "Overall Rate:" in summary_line or "Extraction:" in summary_line or "Embedding:" in summary_line:
                        print(summary_line.strip())
                return
        
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor_test()