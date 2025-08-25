#!/usr/bin/env python3
"""
Monitor the hybrid pipeline processing in real-time.

This monitoring tool embodies the observer effect from Information Reconstructionism -
by observing the pipeline, we create a different reality of its performance metrics.
Each metric represents a dimension of conveyance optimization.
"""

import json
import time
import psutil
import psycopg2
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import argparse
import os

console = Console()

class PipelineMonitor:
    """
    Monitor for the hybrid pipeline, tracking C = (W·R·H)/T · Ctx^α.
    
    Where:
    - W: Embedding quality (successful embeddings)
    - R: Database connections (PostgreSQL + ArangoDB)
    - H: GPU utilization and capability
    - T: Processing time and throughput
    - Ctx: Context preservation (chunks, structures extracted)
    """
    
    def __init__(self, checkpoint_file: str, pg_password: str, database: str = "Avernus"):
        self.checkpoint_file = Path(checkpoint_file)
        self.pg_password = pg_password
        self.database = database
        self.start_time = datetime.now()
        self.last_checkpoint = None
        
    def read_checkpoint(self):
        """Read the current checkpoint file."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_database_stats(self):
        """Get statistics from PostgreSQL."""
        try:
            conn = psycopg2.connect(
                host="localhost",
                database=self.database,
                user="postgres",
                password=self.pg_password
            )
            cur = conn.cursor()
            
            # Get total papers
            cur.execute("SELECT COUNT(*) FROM arxiv_papers")
            total_papers = cur.fetchone()[0]
            
            # Get CS/AI/ML papers
            cur.execute("""
                SELECT COUNT(DISTINCT id) 
                FROM arxiv_papers 
                WHERE categories LIKE '%cs.%' 
                   OR categories LIKE '%stat.ML%'
                   OR categories LIKE '%math.OC%'
            """)
            cs_papers = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            return {
                'total_papers': total_papers,
                'cs_papers': cs_papers
            }
        except Exception as e:
            return {
                'total_papers': 0,
                'cs_papers': 0,
                'error': str(e)
            }
    
    def get_system_stats(self):
        """Get system resource usage."""
        stats = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_gb': psutil.virtual_memory().used / (1024**3),
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        # Try to get GPU stats
        try:
        # Try to get GPU stats
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                stats['gpus'] = []
                
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = {
                        'index': i,
                        'name': pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
                        'memory_used': pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3),
                        'memory_total': pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3),
                        'utilization': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                        'temperature': pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    }
                    stats['gpus'].append(info)
            finally:
                pynvml.nvmlShutdown()
        except (ImportError, pynvml.NVMLError) as e:
            logger.debug(f"GPU stats unavailable: {e}")
            stats['gpus'] = []
            stats['gpus'] = []
        
        return stats
    
    def create_display(self):
        """Create the monitoring display."""
        checkpoint = self.read_checkpoint()
        db_stats = self.get_database_stats()
        sys_stats = self.get_system_stats()
        
        # Main layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=10),
            Layout(name="body"),
            Layout(name="footer", size=5)
        )
        
        # Header
        layout["header"].update(Panel(
            f"[bold cyan]ArXiv Pipeline Monitor[/]\n"
            f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            style="bold white on blue"
        ))
        
        # Progress section
        if checkpoint:
            processed = checkpoint.get('processed_count', 0)
            total = checkpoint.get('total_papers', db_stats['cs_papers'])
            failed = checkpoint.get('failed_count', 0)
            
            # Calculate metrics
            elapsed = datetime.now() - self.start_time
            elapsed_seconds = elapsed.total_seconds()
            throughput = processed / elapsed_seconds if elapsed_seconds > 0 else 0
            eta_seconds = (total - processed) / throughput if throughput > 0 else 0
            eta = timedelta(seconds=int(eta_seconds))
            
            progress_table = Table(show_header=True, header_style="bold magenta")
            progress_table.add_column("Metric", style="cyan")
            progress_table.add_column("Value", style="green")
            
            progress_table.add_row("Papers Processed", f"{processed:,} / {total:,}")
            progress_table.add_row("Success Rate", f"{(processed-failed)/processed*100:.1f}%" if processed > 0 else "0%")
            progress_table.add_row("Failed Papers", f"{failed:,}")
            progress_table.add_row("Throughput", f"{throughput:.1f} papers/sec")
            progress_table.add_row("ETA", str(eta))
            progress_table.add_row("Progress", f"{processed/total*100:.2f}%" if total > 0 else "0%")
            
            if 'current_batch' in checkpoint:
                progress_table.add_row("Current Batch", checkpoint['current_batch'])
            
            layout["progress"].update(Panel(progress_table, title="Processing Progress"))
        else:
            layout["progress"].update(Panel("[yellow]No checkpoint file found[/]", title="Processing Progress"))
        
        # Body - split into two columns
        layout["body"].split_row(
            Layout(name="system", ratio=1),
            Layout(name="gpu", ratio=1)
        )
        
        # System stats
        sys_table = Table(show_header=True, header_style="bold cyan")
        sys_table.add_column("Resource", style="cyan")
        sys_table.add_column("Usage", style="green")
        
        sys_table.add_row("CPU", f"{sys_stats['cpu_percent']:.1f}%")
        sys_table.add_row("Memory", f"{sys_stats['memory_gb']:.1f} GB ({sys_stats['memory_percent']:.1f}%)")
        sys_table.add_row("Disk", f"{sys_stats['disk_usage']:.1f}%")
        
        layout["system"].update(Panel(sys_table, title="System Resources"))
        
        # GPU stats
        if sys_stats['gpus']:
            gpu_table = Table(show_header=True, header_style="bold cyan")
            gpu_table.add_column("GPU", style="cyan")
            gpu_table.add_column("Memory", style="green")
            gpu_table.add_column("Util", style="yellow")
            gpu_table.add_column("Temp", style="red")
            
            for gpu in sys_stats['gpus']:
                gpu_table.add_row(
                    f"GPU {gpu['index']}: {gpu['name'][:20]}",
                    f"{gpu['memory_used']:.1f}/{gpu['memory_total']:.1f} GB",
                    f"{gpu['utilization']}%",
                    f"{gpu['temperature']}°C"
                )
            
            layout["gpu"].update(Panel(gpu_table, title="GPU Status"))
        else:
            layout["gpu"].update(Panel("[yellow]No GPU information available[/]", title="GPU Status"))
        
        # Footer with tips
        tips = [
            "Press Ctrl+C to stop monitoring (pipeline continues)",
            f"Checkpoint: {self.checkpoint_file}",
            f"Database: {db_stats.get('total_papers', 0):,} total papers"
        ]
        layout["footer"].update(Panel("\n".join(tips), style="dim"))
        
        return layout
    
    def run(self, refresh_interval: int = 5):
        """Run the monitor with live updates."""
        console.print("[bold green]Starting pipeline monitor...[/]")
        console.print(f"Monitoring checkpoint: {self.checkpoint_file}")
        console.print(f"Refresh interval: {refresh_interval} seconds")
        console.print("[dim]Press Ctrl+C to stop monitoring[/]\n")
        
        try:
            with Live(self.create_display(), refresh_per_second=1/refresh_interval) as live:
                while True:
                    time.sleep(refresh_interval)
                    live.update(self.create_display())
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitor stopped. Pipeline continues running.[/]")

def main():
    parser = argparse.ArgumentParser(description="Monitor ArXiv pipeline processing")
    parser.add_argument('--checkpoint', default='hybrid_checkpoint.json',
                       help='Checkpoint file to monitor')
    parser.add_argument('--password', required=True, help='PostgreSQL password')
    parser.add_argument('--database', default='Avernus', help='PostgreSQL database name')
    parser.add_argument('--refresh', type=int, default=5,
                       help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    monitor = PipelineMonitor(args.checkpoint, args.password, args.database)
    monitor.run(args.refresh)

if __name__ == "__main__":
    main()