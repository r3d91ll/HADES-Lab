#!/usr/bin/env python3
"""
Weekend Test Monitor
====================

Real-time monitoring for the 15,000 paper weekend test run.
Tracks progress, performance, and system health.
"""

import os
import sys
import json
import time
import psutil
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeekendTestMonitor:
    """Monitor for long-running weekend test."""
    
    def __init__(self, config_file: str = "../configs/weekend_test_15k.yaml"):
        self.config_file = Path(config_file)
        self.start_time = datetime.now()
        self.checkpoint_file = Path("../pipelines/weekend_test_checkpoint.json")
        self.log_file = Path("../logs/weekend_test_15k.log")
        self.sqlite_db = Path("../pipelines/weekend_test_15k_index.db")
        self.staging_dir = Path("/dev/shm/weekend_staging")
        
        # Tracking
        self.last_checkpoint = {}
        self.performance_history = []
        self.error_counts = {}
        
    def get_gpu_status(self) -> Dict:
        """Get current GPU status."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 6:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_used': int(parts[2]),
                        'memory_total': int(parts[3]),
                        'utilization': int(parts[4]),
                        'temperature': int(parts[5])
                    })
            return {'gpus': gpus, 'available': True}
        except:
            return {'gpus': [], 'available': False}
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'used_gb': psutil.virtual_memory().used / (1024**3),
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'staging_used_gb': self.get_directory_size(self.staging_dir) / (1024**3) if self.staging_dir.exists() else 0,
                'staging_files': len(list(self.staging_dir.glob('*.json'))) if self.staging_dir.exists() else 0
            }
        }
    
    def get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        if path.exists():
            for p in path.rglob('*'):
                if p.is_file():
                    total += p.stat().st_size
        return total
    
    def load_checkpoint(self) -> Dict:
        """Load current checkpoint."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_database_stats(self) -> Dict:
        """Get statistics from SQLite index."""
        stats = {
            'total_processed': 0,
            'extraction_complete': 0,
            'embedding_complete': 0,
            'failed': 0,
            'current_phase': 'unknown'
        }
        
        if not self.sqlite_db.exists():
            return stats
        
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) FROM papers WHERE processing_status = 'PROCESSED'")
            stats['total_processed'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM papers WHERE phase_completed >= 1")
            stats['extraction_complete'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM papers WHERE phase_completed = 2")
            stats['embedding_complete'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM papers WHERE processing_status = 'FAILED'")
            stats['failed'] = cursor.fetchone()[0]
            
            # Determine current phase
            if stats['embedding_complete'] > 0 and stats['embedding_complete'] < 15000:
                stats['current_phase'] = 'embedding'
            elif stats['extraction_complete'] > 0 and stats['extraction_complete'] < 15000:
                stats['current_phase'] = 'extraction'
            elif stats['embedding_complete'] >= 15000:
                stats['current_phase'] = 'complete'
            
            conn.close()
        except Exception as e:
            logger.error(f"Error reading database: {e}")
        
        return stats
    
    def parse_recent_logs(self, lines: int = 100) -> Dict:
        """Parse recent log entries for errors and progress."""
        info = {
            'recent_errors': [],
            'last_paper': None,
            'current_rate': 0,
            'phase_info': {}
        }
        
        if not self.log_file.exists():
            return info
        
        try:
            # Get last N lines of log
            result = subprocess.run(
                ['tail', '-n', str(lines), str(self.log_file)],
                capture_output=True, text=True
            )
            
            for line in result.stdout.split('\n'):
                # Check for errors
                if 'ERROR' in line or 'CRITICAL' in line:
                    info['recent_errors'].append(line)
                
                # Check for paper processing
                if 'Processing paper' in line or 'Processed:' in line:
                    # Extract paper ID if possible
                    parts = line.split()
                    for part in parts:
                        if '.' in part and part[0].isdigit():
                            info['last_paper'] = part
                
                # Check for rate information
                if 'papers/minute' in line:
                    try:
                        rate_str = line.split('papers/minute')[0].split()[-1]
                        info['current_rate'] = float(rate_str)
                    except:
                        pass
                
                # Check for phase changes
                if 'PHASE' in line:
                    if 'Extraction' in line:
                        info['phase_info']['current'] = 'extraction'
                    elif 'Embedding' in line:
                        info['phase_info']['current'] = 'embedding'
        except Exception as e:
            logger.error(f"Error parsing logs: {e}")
        
        return info
    
    def calculate_eta(self, processed: int, total: int, rate: float) -> str:
        """Calculate estimated time of arrival."""
        if rate <= 0 or processed >= total:
            return "N/A"
        
        remaining = total - processed
        minutes_remaining = remaining / rate
        eta = datetime.now() + timedelta(minutes=minutes_remaining)
        
        # Format nicely
        if minutes_remaining < 60:
            return f"{int(minutes_remaining)} minutes"
        elif minutes_remaining < 1440:  # Less than a day
            hours = minutes_remaining / 60
            return f"{hours:.1f} hours ({eta.strftime('%H:%M')})"
        else:
            days = minutes_remaining / 1440
            return f"{days:.1f} days ({eta.strftime('%a %H:%M')})"
    
    def print_dashboard(self):
        """Print monitoring dashboard."""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Get all stats
        checkpoint = self.load_checkpoint()
        db_stats = self.get_database_stats()
        system_stats = self.get_system_status()
        gpu_stats = self.get_gpu_status()
        log_info = self.parse_recent_logs()
        
        # Calculate runtime
        runtime = datetime.now() - self.start_time
        runtime_str = str(runtime).split('.')[0]  # Remove microseconds
        
        # Print header
        print("=" * 80)
        print(f"WEEKEND TEST MONITOR - 15,000 Paper Processing Run".center(80))
        print("=" * 80)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runtime: {runtime_str}")
        print(f"Current: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Progress section
        print("PROGRESS")
        print("-" * 40)
        total_target = 15000
        processed = db_stats['embedding_complete']  # Use fully processed count
        progress_pct = (processed / total_target) * 100
        
        # Progress bar
        bar_width = 50
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"[{bar}] {progress_pct:.1f}%")
        print(f"Papers Processed: {processed:,} / {total_target:,}")
        print(f"Extraction Complete: {db_stats['extraction_complete']:,}")
        print(f"Embedding Complete: {db_stats['embedding_complete']:,}")
        print(f"Failed: {db_stats['failed']:,}")
        print()
        
        # Performance section
        print("PERFORMANCE")
        print("-" * 40)
        rate = log_info['current_rate'] if log_info['current_rate'] > 0 else 11.3  # Default rate
        print(f"Current Rate: {rate:.1f} papers/minute")
        print(f"Current Phase: {db_stats['current_phase'].upper()}")
        print(f"Last Paper: {log_info['last_paper'] or 'Unknown'}")
        print(f"ETA: {self.calculate_eta(processed, total_target, rate)}")
        
        # Calculate overall rate
        if runtime.total_seconds() > 0 and processed > 0:
            overall_rate = (processed / runtime.total_seconds()) * 60
            print(f"Overall Rate: {overall_rate:.1f} papers/minute")
        print()
        
        # System Resources
        print("SYSTEM RESOURCES")
        print("-" * 40)
        print(f"CPU Usage: {system_stats['cpu_percent']:.1f}%")
        print(f"Memory: {system_stats['memory']['used_gb']:.1f}GB / {system_stats['memory']['total_gb']:.1f}GB ({system_stats['memory']['percent']:.1f}%)")
        print(f"Staging: {system_stats['disk']['staging_used_gb']:.2f}GB ({system_stats['disk']['staging_files']} files)")
        print()
        
        # GPU Status
        if gpu_stats['available'] and gpu_stats['gpus']:
            print("GPU STATUS")
            print("-" * 40)
            for gpu in gpu_stats['gpus']:
                mem_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
                print(f"GPU {gpu['index']}: {gpu['name']}")
                print(f"  Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({mem_pct:.1f}%)")
                print(f"  Utilization: {gpu['utilization']}%")
                print(f"  Temperature: {gpu['temperature']}°C")
            print()
        
        # Recent Errors
        if log_info['recent_errors']:
            print("RECENT ERRORS (Last 5)")
            print("-" * 40)
            for error in log_info['recent_errors'][-5:]:
                # Truncate long errors
                if len(error) > 77:
                    print(f"  {error[:74]}...")
                else:
                    print(f"  {error}")
            print()
        
        # Milestones
        print("MILESTONES")
        print("-" * 40)
        milestones = [1000, 2500, 5000, 7500, 10000, 12500, 15000]
        for milestone in milestones:
            if processed >= milestone:
                print(f"  ✅ {milestone:,} papers")
            else:
                eta = self.calculate_eta(processed, milestone, rate)
                print(f"  ⏳ {milestone:,} papers (ETA: {eta})")
        print()
        
        # Footer
        print("=" * 80)
        print("Press Ctrl+C to exit monitor (pipeline will continue running)")
        print("Refreshing every 30 seconds...")
    
    def run(self, refresh_interval: int = 30):
        """Run the monitor with periodic refresh."""
        try:
            while True:
                self.print_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped. Pipeline continues running in background.")
            print(f"Log file: {self.log_file}")
            print(f"To resume monitoring: python {__file__}")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor weekend test run")
    parser.add_argument('--config', default='../configs/weekend_test_15k.yaml',
                       help='Configuration file path')
    parser.add_argument('--refresh', type=int, default=30,
                       help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    print("Starting Weekend Test Monitor...")
    print("This monitor displays real-time progress of the 15,000 paper test.")
    print()
    
    monitor = WeekendTestMonitor(args.config)
    monitor.run(args.refresh)


if __name__ == "__main__":
    main()