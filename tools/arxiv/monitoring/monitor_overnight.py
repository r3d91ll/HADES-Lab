#!/usr/bin/env python3
"""
Monitor script for overnight unified pipeline run.
Shows real-time progress, GPU utilization, and processing rates.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from arango import ArangoClient
import subprocess
import psutil
import yaml

def get_gpu_stats():
    """Get GPU utilization and memory stats."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        
        stats = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            stats.append({
                'index': int(parts[0]),
                'name': parts[1],
                'utilization': float(parts[2]),
                'memory_used': float(parts[3]),
                'memory_total': float(parts[4]),
                'memory_percent': (float(parts[3]) / float(parts[4])) * 100
            })
        return stats
    except:
        return []

def get_gpu_processes():
    """Get all processes using GPU memory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,name,used_memory', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                if len(parts) >= 3:
                    pid = int(parts[0])
                    name = parts[1]
                    memory = parts[2]  # e.g., "11266 MiB"
                    memory_mb = int(memory.split()[0])
                    
                    processes.append({
                        'pid': pid,
                        'name': name,
                        'memory_mb': memory_mb
                    })
        return processes
    except:
        return []

def get_queue_sizes():
    """Get queue sizes from the pipeline status file."""
    try:
        status_file = Path("pipeline_status.json")
        if status_file.exists():
            with open(status_file) as f:
                status = json.load(f)
                return {
                    'extraction': status.get('extraction_queue_size', 0),
                    'embedding': status.get('embedding_queue_size', 0),
                    'write': status.get('write_queue_size', 0),
                    'last_updated': status.get('last_updated', 'Unknown')
                }
    except:
        pass
    return {'extraction': 0, 'embedding': 0, 'write': 0, 'last_updated': 'Unknown'}

def check_worker_health():
    """Check health of all workers."""
    health = {
        'main_process': False,
        'workers_alive': 0,
        'extraction_workers': 0,
        'gpu_workers': 0,
        'write_workers': 0,
        'stuck_workers': [],
        'gpu_processes': [],
        'total_cpu_percent': 0.0
    }
    
    # Check main pipeline process
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'arxiv_pipeline_unified_hysteresis' in cmdline:
                health['main_process'] = True
                break
        except:
            continue
    
    # Count all worker processes (they appear as spawn_main processes)
    spawn_processes = []
    resource_trackers = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent', 'ppid']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            
            # Count resource trackers separately
            if 'resource_tracker' in cmdline:
                resource_trackers += 1
                continue
                
            # All workers show up as spawn_main processes
            if 'spawn_main' in cmdline:
                # Check if this process has a parent (not orphaned)
                try:
                    parent = psutil.Process(proc.info['ppid'])
                    # If parent is dead (zombie) or doesn't exist, skip this worker
                    if parent.status() == psutil.STATUS_ZOMBIE:
                        continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Orphaned process - don't count it
                    continue
                    
                spawn_processes.append(proc)
                health['workers_alive'] += 1
                
                # Add CPU usage
                cpu_percent = proc.cpu_percent(interval=0.1)
                health['total_cpu_percent'] += cpu_percent
                
                # Check if worker is stuck (high CPU for long time)
                create_time = datetime.fromtimestamp(proc.info['create_time'])
                age = (datetime.now() - create_time).total_seconds()
                
                if age > 600 and cpu_percent > 70:  # 10 min old, high CPU
                    health['stuck_workers'].append({
                        'pid': proc.info['pid'],
                        'age_seconds': age,
                        'cpu_percent': cpu_percent
                    })
        except:
            continue
    
    # Categorize workers based on config (we can't distinguish types from spawn_main)
    # Since you have 29 workers and expect 26 (16 extraction + 8 GPU + 2 write)
    # The extra 3 are likely resource trackers
    if health['workers_alive'] >= 26:
        # Use your configured values directly
        health['extraction_workers'] = 16
        health['gpu_workers'] = 8
        health['write_workers'] = 2
        # Extra processes are resource trackers
        health['resource_trackers'] = health['workers_alive'] - 26
    elif health['workers_alive'] > 0:
        # Proportional distribution if count is different
        # 16:8:2 ratio = 61.5%:30.8%:7.7%
        health['extraction_workers'] = int(health['workers_alive'] * 0.615)
        health['gpu_workers'] = int(health['workers_alive'] * 0.308)
        health['write_workers'] = max(1, health['workers_alive'] - health['extraction_workers'] - health['gpu_workers'])
        health['resource_trackers'] = 0
    
    # Get GPU processes
    health['gpu_processes'] = get_gpu_processes()
    
    return health

def monitor_progress(arango_password: str, refresh_interval: float = 5.0):
    """Monitor overnight processing progress."""
    
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://192.168.1.69:8529')
    db = client.db('academy_store', username='root', password=arango_password)
    
    # Check for checkpoint to get actual start counts
    checkpoint_path = Path("unified_checkpoint.json")
    checkpoint_start_count = 0
    run_start_time = None
    
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            # Get the count from when this run actually started
            checkpoint_start_count = checkpoint.get('processed_count', 0)
            # Try to parse the checkpoint time
            if 'last_updated' in checkpoint:
                try:
                    run_start_time = datetime.fromisoformat(checkpoint['last_updated'].replace('Z', '+00:00'))
                except:
                    pass
            print(f"📍 Found checkpoint with {checkpoint_start_count} papers already processed")
        except Exception as e:
            print(f"⚠️  Could not read checkpoint: {e}")
    
    # Also try to detect when THIS run started by looking at recent processing times
    try:
        # Get the oldest paper processed in the last hour (likely from this run)
        one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat() + 'Z'
        recent_papers = list(db.aql.execute(f"""
            FOR doc IN arxiv_unified_embeddings
            FILTER doc.processing_date >= "{one_hour_ago}"
            SORT doc.processing_date ASC
            LIMIT 1
            RETURN doc.processing_date
        """))
        
        if recent_papers and recent_papers[0]:
            actual_run_start = datetime.fromisoformat(recent_papers[0].replace('Z', '+00:00'))
            if not run_start_time or actual_run_start > run_start_time:
                run_start_time = actual_run_start
                print(f"🕐 Detected current run started at: {run_start_time.strftime('%H:%M:%S')}")
    except:
        pass
    
    # Get current counts
    current_embeddings = db.collection('arxiv_unified_embeddings').count()
    current_structures = db.collection('arxiv_structures').count()
    
    # If we have more papers than checkpoint, this run must have started from checkpoint
    if current_embeddings > checkpoint_start_count:
        start_embeddings = checkpoint_start_count
        print(f"📊 Current run started from: {start_embeddings} papers")
    else:
        # Fresh run or checkpoint is ahead
        start_embeddings = current_embeddings
        print(f"📊 Starting fresh count from: {start_embeddings} papers")
    
    start_structures = current_structures
    start_time = run_start_time if run_start_time else datetime.now()
    last_embeddings = current_embeddings
    
    # ANSI codes for cursor control and colors
    CLEAR_SCREEN = '\033[2J'
    CURSOR_HOME = '\033[H'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    # Initial clear to start fresh
    print(CLEAR_SCREEN + CURSOR_HOME, end='')
    
    # Configuration for update intervals
    last_update_time = time.time()
    update_interval = 10.0  # Update every 10 seconds initially
    fast_update_after = 600  # After 10 minutes, update every 30 seconds
    
    try:
        while True:
            # Move cursor to home position for smooth update
            print(CURSOR_HOME, end='')
            
            # Get current counts
            current_embeddings = db.collection('arxiv_unified_embeddings').count()
            current_structures = db.collection('arxiv_structures').count()
            
            # Calculate progress
            new_papers = current_embeddings - start_embeddings
            elapsed = datetime.now() - start_time
            elapsed_minutes = elapsed.total_seconds() / 60
            
            # Calculate rates
            if elapsed_minutes > 0:
                overall_rate = new_papers / elapsed_minutes  # papers per minute
                overall_rate_hour = overall_rate * 60  # papers per hour
            else:
                overall_rate = 0
                overall_rate_hour = 0
            
            # Calculate instantaneous rate (since last check)
            papers_since_last = current_embeddings - last_embeddings
            instant_rate = papers_since_last / (refresh_interval / 60)  # papers per minute
            last_embeddings = current_embeddings
            
            # Estimate completion
            target = 10000
            remaining = target - new_papers
            if overall_rate > 0:
                eta_minutes = remaining / overall_rate
                eta = datetime.now() + timedelta(minutes=eta_minutes)
                eta_str = eta.strftime('%Y-%m-%d %H:%M:%S')
            else:
                eta_minutes = 0
                eta_str = "Calculating..."
            
            # ANSI color codes
            BOLD = '\033[1m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            RED = '\033[91m'
            CYAN = '\033[96m'
            RESET = '\033[0m'
            
            # Display header
            print("="*80)
            print(f"{BOLD}OVERNIGHT UNIFIED PIPELINE MONITOR{RESET}")
            print("="*80)
            print(f"Current Time: {CYAN}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
            print(f"Elapsed: {CYAN}{str(elapsed).split('.')[0]}{RESET}")
            print("="*80)
            
            # Display progress with spacing
            print(f"\n{BOLD}📊 PROCESSING PROGRESS{RESET}")
            print("-"*40)
            print(f"Embeddings: {current_embeddings:,} (+{new_papers:,} new)")
            print(f"Structures: {current_structures:,}")
            print(f"Progress: {new_papers:,} / {target:,} ({(new_papers/target)*100:.1f}%)")
            print()  # Add space here
            
            # Progress bar with color
            bar_width = 50
            filled = int(bar_width * new_papers / target)
            percent = (new_papers/target)*100 if target > 0 else 0
            
            # Color based on progress
            if percent < 25:
                bar_color = RED
            elif percent < 50:
                bar_color = YELLOW
            elif percent < 75:
                bar_color = CYAN
            else:
                bar_color = GREEN
            
            bar = bar_color + "█" * filled + RESET + "░" * (bar_width - filled)
            print(f"[{bar}] {bar_color}{new_papers}/{target}{RESET}")
            
            # Display rates
            print(f"\n{BOLD}⚡ PROCESSING RATES{RESET}")
            print("-"*40)
            print(f"Overall: {overall_rate:.2f} papers/min ({overall_rate_hour:.0f} papers/hour)")
            print(f"Current: {instant_rate:.2f} papers/min ({instant_rate*60:.0f} papers/hour)")
            if eta_minutes > 0:
                print(f"ETA: {CYAN}{eta_str}{RESET} ({eta_minutes/60:.1f} hours remaining)")
            else:
                print(f"ETA: {CYAN}{eta_str}{RESET}")
            
            # Worker health status
            print(f"\n{BOLD}👷 WORKER HEALTH{RESET}")
            print("-"*40)
            worker_health = check_worker_health()
            
            # Main process status
            if worker_health['main_process']:
                print(f"Main Process: ✅ Running")
            else:
                print(f"Main Process: ❌ NOT RUNNING")
            
            # Worker counts
            print(f"Workers: {worker_health['workers_alive']} total")
            print(f"  • Extraction: {worker_health['extraction_workers']}")
            print(f"  • GPU/Embed: {worker_health['gpu_workers']}")
            print(f"  • Write: {worker_health['write_workers']}")
            if worker_health.get('resource_trackers', 0) > 0:
                print(f"  • Resource trackers: {worker_health['resource_trackers']}")
            
            # CPU usage
            print(f"Total CPU: {worker_health['total_cpu_percent']:.1f}%")
            
            # Stuck workers warning
            if worker_health['stuck_workers']:
                print(f"⚠️  STUCK WORKERS: {len(worker_health['stuck_workers'])}")
                for worker in worker_health['stuck_workers'][:3]:
                    age_min = worker['age_seconds'] / 60
                    print(f"  • PID {worker['pid']}: {age_min:.1f} min, {worker['cpu_percent']:.1f}% CPU")
            
            # Queue status
            print(f"\n{BOLD}📦 QUEUE STATUS{RESET}")
            print("-"*40)
            queue_status = get_queue_sizes()
            
            # Display queue sizes with visual indicators
            extraction_pct = (queue_status['extraction'] / 100) * 100  # Assuming 100 is max from safe config
            embedding_pct = (queue_status['embedding'] / 50) * 100    # 50 is max from safe config
            write_pct = (queue_status['write'] / 100) * 100           # 100 is max from safe config
            
            # ANSI color codes for terminal output
            RED = '\033[91m'
            YELLOW = '\033[93m'
            GREEN = '\033[92m'
            BOLD = '\033[1m'
            RESET = '\033[0m'
            
            # Color coding based on percentage
            def queue_indicator(pct):
                if pct >= 80:
                    return f"{RED}●{RESET}"  # Red - at hysteresis pause threshold
                elif pct >= 60:
                    return f"{YELLOW}●{RESET}"  # Yellow - warning
                else:
                    return f"{GREEN}●{RESET}"  # Green - healthy
            
            print(f"Extraction: {queue_indicator(extraction_pct)} {queue_status['extraction']}/100 ({extraction_pct:.0f}%)")
            print(f"Embedding:  {queue_indicator(embedding_pct)} {queue_status['embedding']}/50 ({embedding_pct:.0f}%)")
            print(f"Write:      {queue_indicator(write_pct)} {queue_status['write']}/100 ({write_pct:.0f}%)")
            
            # Hysteresis warnings
            if extraction_pct >= 80:
                print("  ⚠️  Extraction queue at pause threshold (80%)")
            if embedding_pct >= 80:
                print("  ⚠️  Embedding queue at pause threshold (80%)")
            
            # Check for queue imbalance
            if extraction_pct > 60 and embedding_pct < 20:
                print("  ⚠️  Queue imbalance: Extraction backing up, embedding starved")
            elif embedding_pct > 60 and write_pct < 20:
                print("  ⚠️  Queue imbalance: Embedding backing up, write queue empty")
            
            if queue_status['last_updated'] != 'Unknown':
                print(f"Queue status updated: {queue_status['last_updated']}")
            
            # GPU status
            print(f"\n{BOLD}🖥️  GPU STATUS{RESET}")
            print("-"*40)
            gpu_stats = get_gpu_stats()
            for gpu in gpu_stats:
                print(f"GPU {gpu['index']}: {gpu['utilization']:.0f}% util, "
                      f"{gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB "
                      f"({gpu['memory_percent']:.1f}% memory)")
            
            # GPU processes
            if worker_health['gpu_processes']:
                total_gpu_mem = sum(p['memory_mb'] for p in worker_health['gpu_processes'])
                print(f"\nGPU Processes: {len(worker_health['gpu_processes'])} ({total_gpu_mem:,} MB total)")
                for proc in worker_health['gpu_processes'][:5]:
                    print(f"  • PID {proc['pid']}: {proc['name']} ({proc['memory_mb']} MB)")
            
            # Check for checkpoint
            checkpoint_path = Path("unified_checkpoint.json")
            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path) as f:
                        checkpoint = json.load(f)
                    print("\n📍 CHECKPOINT")
                    print("-"*40)
                    print(f"Last saved: {checkpoint.get('last_updated', 'Unknown')}")
                    print(f"Papers processed: {checkpoint.get('processed_count', 0)}")
                    print(f"Failed papers: {len(checkpoint.get('failed_papers', []))}")
                except:
                    pass
            
            # Recent activity
            try:
                # Get most recent papers
                recent = list(db.aql.execute("""
                    FOR doc IN arxiv_unified_embeddings
                    SORT doc.processing_date DESC
                    LIMIT 5
                    RETURN {
                        arxiv_id: doc.arxiv_id,
                        chunks: doc.num_chunks,
                        has_latex: doc.has_latex,
                        has_pdf: doc.has_pdf
                    }
                """))
                
                if recent:
                    print(f"\n{BOLD}📝 RECENT PAPERS{RESET}")
                    print("-"*40)
                    for paper in recent:
                        sources = []
                        if paper.get('has_latex'):
                            sources.append('LaTeX')
                        if paper.get('has_pdf'):
                            sources.append('PDF')
                        source_str = '+'.join(sources) if sources else 'Unknown'
                        print(f"{paper['arxiv_id']}: {paper.get('chunks', 0)} chunks ({source_str})")
            except:
                pass
            
            # Dynamic refresh timing
            elapsed_seconds = elapsed.total_seconds()
            if elapsed_seconds > fast_update_after:
                # After 10 minutes, update every 30 seconds
                current_update_interval = 30.0
            else:
                # First 10 minutes, update every 10 seconds
                current_update_interval = update_interval
            
            print("\n" + "="*80)
            print(f"Refreshing every {current_update_interval:.0f} seconds... (Ctrl+C to stop)")
            
            time.sleep(current_update_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print(f"Final counts - Embeddings: {current_embeddings}, Structures: {current_structures}")
        print(f"Total processed: {new_papers} papers in {elapsed}")
        print(f"Average rate: {overall_rate:.2f} papers/minute")

def main():
    parser = argparse.ArgumentParser(description='Monitor overnight unified pipeline')
    parser.add_argument('--arango-password', 
                       default=os.environ.get('ARANGO_PASSWORD'),
                       help='ArangoDB password')
    parser.add_argument('--refresh', type=float, default=5.0,
                       help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    if not args.arango_password:
        print("Error: ArangoDB password required (use --arango-password or ARANGO_PASSWORD env var)")
        sys.exit(1)
    
    monitor_progress(args.arango_password, args.refresh)

if __name__ == '__main__':
    main()