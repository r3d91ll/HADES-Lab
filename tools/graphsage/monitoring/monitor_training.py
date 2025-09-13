#!/usr/bin/env python3
"""
Real-time monitoring script for GraphSAGE training.
Captures comprehensive metrics for ANT documentation and paper writing.
"""

import os
import sys
import json
import time
import psutil
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import signal
import threading
from collections import deque
import numpy as np

class TrainingMonitor:
    """Monitor and log training metrics for documentation."""
    
    def __init__(self, process_name="train_distributed", log_dir="/home/todd/olympus/HADES-Lab/logs/graphsage"):
        self.process_name = process_name
        self.log_dir = Path(log_dir)
        self.metrics_dir = self.log_dir / "metrics"
        self.archives_dir = self.log_dir / "archives"
        
        # Ensure directories exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.archives_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.metrics_dir / f"training_metrics_{self.session_id}.jsonl"
        self.summary_file = self.metrics_dir / f"training_summary_{self.session_id}.json"
        
        # Metrics accumulators
        self.start_time = time.time()
        self.gpu_hours = 0.0
        self.total_power_kwh = 0.0
        self.peak_memory = {}
        self.samples_processed = 0
        self.last_epoch = -1
        self.last_accuracy = 0.0
        self.last_loss = float('inf')
        
        # Recent metrics for averaging
        self.recent_gpu_utils = deque(maxlen=12)  # 1 minute window at 5s intervals
        self.recent_memory = deque(maxlen=12)
        
        # Control flags
        self.running = True
        self.training_pid = None
        
        # Find training process
        self.find_training_process()
        
        # Capture system info once
        self.system_info = self.capture_system_info()
        
    def find_training_process(self):
        """Find the PID of the training process."""
        try:
            result = subprocess.run(
                f"ps aux | grep -E 'python.*{self.process_name}' | grep -v grep | head -1",
                shell=True, capture_output=True, text=True
            )
            if result.stdout:
                self.training_pid = int(result.stdout.split()[1])
                print(f"Found training process: PID {self.training_pid}")
            else:
                print(f"Warning: No training process found matching '{self.process_name}'")
        except Exception as e:
            print(f"Error finding process: {e}")
    
    def capture_system_info(self):
        """Capture system configuration for documentation."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "hardware": {
                "cpu": {
                    "model": subprocess.run("lscpu | grep 'Model name' | cut -d: -f2", 
                                          shell=True, capture_output=True, text=True).stdout.strip(),
                    "cores": psutil.cpu_count(logical=False),
                    "threads": psutil.cpu_count(logical=True)
                },
                "memory": {
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
                },
                "gpus": []
            },
            "software": {
                "python": sys.version.split()[0],
                "cuda": None,
                "pytorch": None
            }
        }
        
        # GPU information
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                shell=True, capture_output=True, text=True
            )
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) == 2:
                    info["hardware"]["gpus"].append({
                        "name": parts[0],
                        "memory_gb": int(parts[1].replace(' MiB', '')) / 1024
                    })
        except:
            pass
        
        # CUDA version
        try:
            result = subprocess.run("nvidia-smi | grep CUDA", shell=True, capture_output=True, text=True)
            if "CUDA" in result.stdout:
                info["software"]["cuda"] = result.stdout.split("CUDA Version:")[1].split()[0]
        except:
            pass
        
        # PyTorch version
        try:
            import torch
            info["software"]["pytorch"] = torch.__version__
        except:
            pass
        
        return info
    
    def capture_gpu_metrics(self):
        """Capture current GPU metrics."""
        metrics = []
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader",
                shell=True, capture_output=True, text=True
            )
            
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpu_idx = int(parts[0])
                    
                    # Parse values, handling units
                    util = float(parts[2].replace(' %', ''))
                    mem_used = float(parts[3].replace(' MiB', ''))
                    mem_total = float(parts[4].replace(' MiB', ''))
                    power = float(parts[5].replace(' W', '')) if ' W' in parts[5] else 0
                    temp = float(parts[6].replace(' C', '')) if parts[6] != '[N/A]' else 0
                    
                    metrics.append({
                        "gpu_id": gpu_idx,
                        "name": parts[1],
                        "utilization_percent": util,
                        "memory_used_mb": mem_used,
                        "memory_total_mb": mem_total,
                        "memory_percent": round(100 * mem_used / mem_total, 2),
                        "power_watts": power,
                        "temperature_c": temp
                    })
                    
                    # Track peak memory
                    if gpu_idx not in self.peak_memory:
                        self.peak_memory[gpu_idx] = 0
                    self.peak_memory[gpu_idx] = max(self.peak_memory[gpu_idx], mem_used)
                    
                    # Accumulate power usage (convert to kWh)
                    self.total_power_kwh += (power / 1000.0) * (5.0 / 3600.0)  # 5 second intervals
                    
        except Exception as e:
            print(f"Error capturing GPU metrics: {e}")
        
        return metrics
    
    def capture_process_metrics(self):
        """Capture training process metrics."""
        if not self.training_pid:
            return None
        
        try:
            process = psutil.Process(self.training_pid)
            
            # Get process info
            with process.oneshot():
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_info = process.memory_info()
                io_counters = process.io_counters() if hasattr(process, 'io_counters') else None
                
                metrics = {
                    "pid": self.training_pid,
                    "cpu_percent": cpu_percent,
                    "memory_rss_gb": round(memory_info.rss / (1024**3), 2),
                    "memory_percent": round(process.memory_percent(), 2),
                    "num_threads": process.num_threads(),
                    "status": process.status()
                }
                
                if io_counters:
                    metrics["io_read_mb"] = round(io_counters.read_bytes / (1024**2), 2)
                    metrics["io_write_mb"] = round(io_counters.write_bytes / (1024**2), 2)
                
                return metrics
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.training_pid = None
            return None
    
    def parse_training_output(self):
        """Parse training metrics from console output (if available)."""
        # This would ideally tail the training log file
        # For now, we'll just track what we can observe
        metrics = {
            "epoch": self.last_epoch,
            "loss": self.last_loss,
            "accuracy": self.last_accuracy,
            "samples_processed": self.samples_processed
        }
        return metrics
    
    def log_metrics(self):
        """Log current metrics to file."""
        timestamp = datetime.now().isoformat()
        elapsed_time = time.time() - self.start_time
        
        # Gather all metrics
        metrics = {
            "timestamp": timestamp,
            "elapsed_seconds": round(elapsed_time, 2),
            "elapsed_hours": round(elapsed_time / 3600, 3),
            "gpu_metrics": self.capture_gpu_metrics(),
            "process_metrics": self.capture_process_metrics(),
            "training_metrics": self.parse_training_output(),
            "accumulated": {
                "gpu_hours": round(len(self.system_info["hardware"]["gpus"]) * elapsed_time / 3600, 3),
                "total_power_kwh": round(self.total_power_kwh, 4),
                "peak_memory_mb": self.peak_memory
            }
        }
        
        # Write to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        # Update recent metrics for averaging
        if metrics["gpu_metrics"]:
            avg_util = np.mean([g["utilization_percent"] for g in metrics["gpu_metrics"]])
            self.recent_gpu_utils.append(avg_util)
            
            total_mem = sum(g["memory_used_mb"] for g in metrics["gpu_metrics"])
            self.recent_memory.append(total_mem)
        
        return metrics
    
    def generate_summary(self):
        """Generate summary statistics for the paper."""
        elapsed_time = time.time() - self.start_time
        num_gpus = len(self.system_info["hardware"]["gpus"])
        
        summary = {
            "session_id": self.session_id,
            "system_info": self.system_info,
            "training_summary": {
                "total_runtime_hours": round(elapsed_time / 3600, 2),
                "gpu_hours": round(num_gpus * elapsed_time / 3600, 2),
                "total_power_kwh": round(self.total_power_kwh, 3),
                "estimated_cost_usd": round(self.total_power_kwh * 0.30, 2),  # Assuming $0.30/kWh
                "peak_memory_per_gpu_mb": self.peak_memory,
                "average_gpu_utilization": round(np.mean(self.recent_gpu_utils), 2) if self.recent_gpu_utils else 0,
                "average_memory_usage_mb": round(np.mean(self.recent_memory), 2) if self.recent_memory else 0,
                "final_epoch": self.last_epoch,
                "final_accuracy": self.last_accuracy,
                "final_loss": self.last_loss
            },
            "ant_metrics": {
                "description": "GraphSAGE as librarian gatekeeper actant",
                "computational_investment": f"{round(num_gpus * elapsed_time / 3600, 2)} GPU-hours",
                "energy_consumption": f"{round(self.total_power_kwh, 3)} kWh",
                "infrastructure": f"{num_gpus}x {self.system_info['hardware']['gpus'][0]['name'] if self.system_info['hardware']['gpus'] else 'GPU'}",
                "graph_scale": "1.68M nodes, 74.2M edges",
                "purpose": "Transform intractable graph into navigable embedding space"
            }
        }
        
        # Write summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary for console
        print("\n" + "="*60)
        print("TRAINING METRICS SUMMARY")
        print("="*60)
        print(f"Runtime: {summary['training_summary']['total_runtime_hours']:.2f} hours")
        print(f"GPU-hours: {summary['training_summary']['gpu_hours']:.2f}")
        print(f"Power consumption: {summary['training_summary']['total_power_kwh']:.3f} kWh")
        print(f"Estimated cost: ${summary['training_summary']['estimated_cost_usd']:.2f}")
        print(f"Peak memory: {summary['training_summary']['peak_memory_per_gpu_mb']}")
        print(f"Average GPU utilization: {summary['training_summary']['average_gpu_utilization']:.1f}%")
        print("="*60)
        
        return summary
    
    def monitor_loop(self, interval=5):
        """Main monitoring loop."""
        print(f"Starting monitoring (interval: {interval}s)")
        print(f"Logging to: {self.metrics_file}")
        print("Press Ctrl+C to stop monitoring")
        
        while self.running:
            try:
                metrics = self.log_metrics()
                
                # Display current status
                if metrics["gpu_metrics"]:
                    gpu_status = []
                    for g in metrics["gpu_metrics"]:
                        gpu_status.append(f"GPU{g['gpu_id']}: {g['utilization_percent']:.0f}% {g['memory_used_mb']:.0f}MB")
                    
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] " + " | ".join(gpu_status), end='', flush=True)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError in monitoring loop: {e}")
                time.sleep(interval)
        
        # Generate final summary
        print("\n\nGenerating final summary...")
        self.generate_summary()
        print(f"Metrics saved to: {self.metrics_file}")
        print(f"Summary saved to: {self.summary_file}")
    
    def stop(self):
        """Stop monitoring gracefully."""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Monitor GraphSAGE training for documentation")
    parser.add_argument("--process", default="train_distributed", help="Process name to monitor")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--log-dir", default="/home/todd/olympus/HADES-Lab/logs/graphsage", 
                       help="Directory for logs")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = TrainingMonitor(process_name=args.process, log_dir=args.log_dir)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down monitor...")
        monitor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start monitoring
    monitor.monitor_loop(interval=args.interval)


if __name__ == "__main__":
    main()