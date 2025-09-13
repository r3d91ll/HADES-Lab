#!/usr/bin/env python3
"""
General-purpose system monitoring for computational experiments.
Captures metrics for ANT framework documentation and academic publications.

From the Conveyance Framework:
- H (Who): GPU/CPU capability and utilization
- T (Time): Computational time and latency  
- W (What): Processing throughput and quality
- R (Where): Resource allocation and topology
"""

import os
import sys
import json
import time
import psutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np
import threading
import signal


class SystemMonitor:
    """Monitor system resources for academic documentation of computational costs."""
    
    def __init__(self, 
                 experiment_name: str = "experiment",
                 log_dir: str = None,
                 capture_interval: int = 5):
        """
        Initialize system monitor.
        
        Args:
            experiment_name: Name for this monitoring session
            log_dir: Directory for metrics (defaults to logs/monitoring/)
            capture_interval: Seconds between metric captures
        """
        self.experiment_name = experiment_name
        self.capture_interval = capture_interval
        
        # Setup directories
        if log_dir is None:
            log_dir = Path.home() / "olympus/HADES-Lab/logs/monitoring"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"{experiment_name}_{self.session_id}.jsonl"
        self.summary_file = self.log_dir / f"{experiment_name}_{self.session_id}_summary.json"
        
        # Metrics accumulators
        self.start_time = time.time()
        self.metrics_buffer = []
        self.gpu_hours = 0.0
        self.total_power_kwh = 0.0
        self.peak_memory = {}
        self.peak_gpu_memory = {}
        
        # Recent metrics for averaging
        self.recent_gpu_utils = deque(maxlen=60)  # 5 min window at 5s intervals
        self.recent_cpu_utils = deque(maxlen=60)
        self.recent_memory = deque(maxlen=60)
        
        # Control flags
        self.running = False
        self.monitor_thread = None
        
        # Capture system info once
        self.system_info = self._capture_system_info()
        
        # Process tracking
        self.tracked_processes = {}
        
    def _capture_system_info(self) -> Dict[str, Any]:
        """Capture static system configuration."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "hostname": os.uname().nodename,
            "hardware": {
                "cpu": {
                    "model": self._get_cpu_model(),
                    "cores": psutil.cpu_count(logical=False),
                    "threads": psutil.cpu_count(logical=True)
                },
                "memory": {
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
                },
                "gpus": self._get_gpu_info()
            },
            "software": {
                "python": sys.version.split()[0],
                "cuda": self._get_cuda_version(),
                "platform": sys.platform
            }
        }
        return info
    
    def _get_cpu_model(self) -> str:
        """Get CPU model name."""
        try:
            if sys.platform == "linux":
                result = subprocess.run(
                    "lscpu | grep 'Model name' | cut -d: -f2",
                    shell=True, capture_output=True, text=True
                )
                return result.stdout.strip()
            return "Unknown"
        except:
            return "Unknown"
    
    def _get_gpu_info(self) -> List[Dict]:
        """Get GPU information."""
        gpus = []
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
                shell=True, capture_output=True, text=True
            )
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_gb": float(parts[2].replace(' MiB', '')) / 1024
                    })
        except:
            pass
        return gpus
    
    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version."""
        try:
            result = subprocess.run(
                "nvidia-smi | grep 'CUDA Version'",
                shell=True, capture_output=True, text=True
            )
            if "CUDA Version:" in result.stdout:
                return result.stdout.split("CUDA Version:")[1].split()[0]
        except:
            pass
        return None
    
    def track_process(self, pid: int, name: str = None):
        """
        Track a specific process.
        
        Args:
            pid: Process ID to track
            name: Optional name for the process
        """
        try:
            process = psutil.Process(pid)
            self.tracked_processes[pid] = {
                "name": name or process.name(),
                "process": process,
                "start_time": time.time()
            }
            print(f"Tracking process {pid}: {self.tracked_processes[pid]['name']}")
        except psutil.NoSuchProcess:
            print(f"Process {pid} not found")
    
    def capture_metrics(self) -> Dict[str, Any]:
        """Capture current system metrics."""
        timestamp = datetime.now().isoformat()
        elapsed_time = time.time() - self.start_time
        
        metrics = {
            "timestamp": timestamp,
            "elapsed_seconds": round(elapsed_time, 2),
            "elapsed_hours": round(elapsed_time / 3600, 4),
            "system": self._capture_system_metrics(),
            "gpu": self._capture_gpu_metrics(),
            "processes": self._capture_process_metrics()
        }
        
        # Update accumulators
        self._update_accumulators(metrics)
        
        return metrics
    
    def _capture_system_metrics(self) -> Dict[str, Any]:
        """Capture system-wide metrics."""
        mem = psutil.virtual_memory()
        
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_per_core": psutil.cpu_percent(interval=0.1, percpu=True),
            "memory": {
                "used_gb": round(mem.used / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent": mem.percent
            },
            "swap": {
                "used_gb": round(psutil.swap_memory().used / (1024**3), 2),
                "percent": psutil.swap_memory().percent
            }
        }
        
        # Network I/O
        try:
            net = psutil.net_io_counters()
            metrics["network"] = {
                "bytes_sent_mb": round(net.bytes_sent / (1024**2), 2),
                "bytes_recv_mb": round(net.bytes_recv / (1024**2), 2)
            }
        except:
            pass
        
        # Disk I/O
        try:
            disk = psutil.disk_io_counters()
            metrics["disk"] = {
                "read_mb": round(disk.read_bytes / (1024**2), 2),
                "write_mb": round(disk.write_bytes / (1024**2), 2)
            }
        except:
            pass
        
        return metrics
    
    def _capture_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Capture GPU metrics."""
        metrics = []
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                "power.draw,temperature.gpu,clocks.current.graphics,clocks.current.memory "
                "--format=csv,noheader",
                shell=True, capture_output=True, text=True
            )
            
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpu_idx = int(parts[0])
                    
                    # Parse values
                    util = float(parts[2].replace(' %', ''))
                    mem_used = float(parts[3].replace(' MiB', ''))
                    mem_total = float(parts[4].replace(' MiB', ''))
                    power = float(parts[5].replace(' W', '')) if ' W' in parts[5] else 0
                    temp = float(parts[6].replace(' C', '')) if parts[6] != '[N/A]' else 0
                    
                    gpu_metrics = {
                        "gpu_id": gpu_idx,
                        "name": parts[1],
                        "utilization_percent": util,
                        "memory_used_mb": mem_used,
                        "memory_total_mb": mem_total,
                        "memory_percent": round(100 * mem_used / mem_total, 2),
                        "power_watts": power,
                        "temperature_c": temp
                    }
                    
                    # Add clock speeds if available
                    if len(parts) >= 9:
                        try:
                            gpu_metrics["clock_graphics_mhz"] = int(parts[7].replace(' MHz', ''))
                            gpu_metrics["clock_memory_mhz"] = int(parts[8].replace(' MHz', ''))
                        except:
                            pass
                    
                    metrics.append(gpu_metrics)
                    
                    # Track peak memory
                    if gpu_idx not in self.peak_gpu_memory:
                        self.peak_gpu_memory[gpu_idx] = 0
                    self.peak_gpu_memory[gpu_idx] = max(self.peak_gpu_memory[gpu_idx], mem_used)
                    
                    # Accumulate power usage
                    self.total_power_kwh += (power / 1000.0) * (self.capture_interval / 3600.0)
                    
        except Exception as e:
            print(f"Error capturing GPU metrics: {e}")
        
        return metrics
    
    def _capture_process_metrics(self) -> Dict[str, Any]:
        """Capture metrics for tracked processes."""
        metrics = {}
        
        for pid, info in list(self.tracked_processes.items()):
            try:
                process = info["process"]
                with process.oneshot():
                    cpu_percent = process.cpu_percent(interval=0.1)
                    memory_info = process.memory_info()
                    
                    proc_metrics = {
                        "name": info["name"],
                        "pid": pid,
                        "status": process.status(),
                        "cpu_percent": cpu_percent,
                        "memory_rss_gb": round(memory_info.rss / (1024**3), 3),
                        "memory_percent": round(process.memory_percent(), 2),
                        "num_threads": process.num_threads(),
                        "runtime_seconds": round(time.time() - info["start_time"], 2)
                    }
                    
                    # I/O counters if available
                    try:
                        io = process.io_counters()
                        proc_metrics["io"] = {
                            "read_mb": round(io.read_bytes / (1024**2), 2),
                            "write_mb": round(io.write_bytes / (1024**2), 2)
                        }
                    except:
                        pass
                    
                    metrics[str(pid)] = proc_metrics
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process ended or access denied
                del self.tracked_processes[pid]
        
        return metrics
    
    def _update_accumulators(self, metrics: Dict[str, Any]):
        """Update running statistics."""
        # GPU utilization
        if metrics["gpu"]:
            avg_util = np.mean([g["utilization_percent"] for g in metrics["gpu"]])
            self.recent_gpu_utils.append(avg_util)
            
            total_mem = sum(g["memory_used_mb"] for g in metrics["gpu"])
            self.recent_memory.append(total_mem)
        
        # CPU utilization
        self.recent_cpu_utils.append(metrics["system"]["cpu_percent"])
        
        # Peak memory
        current_mem = metrics["system"]["memory"]["used_gb"]
        if "system" not in self.peak_memory:
            self.peak_memory["system"] = 0
        self.peak_memory["system"] = max(self.peak_memory["system"], current_mem)
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to file."""
        # Add to buffer
        self.metrics_buffer.append(metrics)
        
        # Write to file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        # Periodic buffer flush
        if len(self.metrics_buffer) >= 100:
            self.metrics_buffer = self.metrics_buffer[-50:]  # Keep recent 50
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                metrics = self.capture_metrics()
                self.log_metrics(metrics)
                
                # Display summary
                if metrics["gpu"]:
                    gpu_status = []
                    for g in metrics["gpu"]:
                        gpu_status.append(
                            f"GPU{g['gpu_id']}: {g['utilization_percent']:.0f}% "
                            f"{g['memory_used_mb']:.0f}MB {g['power_watts']:.0f}W"
                        )
                    
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] " + 
                          " | ".join(gpu_status), end='', flush=True)
                
                time.sleep(self.capture_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError in monitoring: {e}")
                time.sleep(self.capture_interval)
    
    def start(self, background: bool = True):
        """
        Start monitoring.
        
        Args:
            background: Run in background thread if True
        """
        if self.running:
            print("Monitor already running")
            return
        
        self.running = True
        print(f"Starting monitoring: {self.experiment_name}")
        print(f"Logging to: {self.metrics_file}")
        
        if background:
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
        else:
            self._monitor_loop()
    
    def stop(self):
        """Stop monitoring and generate summary."""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        elapsed_time = time.time() - self.start_time
        num_gpus = len(self.system_info["hardware"]["gpus"])
        
        summary = {
            "session_id": self.session_id,
            "experiment_name": self.experiment_name,
            "system_info": self.system_info,
            "runtime": {
                "total_hours": round(elapsed_time / 3600, 3),
                "gpu_hours": round(num_gpus * elapsed_time / 3600, 3),
                "total_power_kwh": round(self.total_power_kwh, 4),
                "estimated_cost_usd": round(self.total_power_kwh * 0.30, 2)
            },
            "peak_usage": {
                "memory_gb": self.peak_memory,
                "gpu_memory_mb": self.peak_gpu_memory
            },
            "average_usage": {
                "gpu_utilization": round(np.mean(self.recent_gpu_utils), 1) if self.recent_gpu_utils else 0,
                "cpu_utilization": round(np.mean(self.recent_cpu_utils), 1) if self.recent_cpu_utils else 0,
                "gpu_memory_mb": round(np.mean(self.recent_memory), 1) if self.recent_memory else 0
            },
            "ant_framework": {
                "description": f"{self.experiment_name} computational investment",
                "who_h": f"{num_gpus}x {self.system_info['hardware']['gpus'][0]['name'] if self.system_info['hardware']['gpus'] else 'GPU'}",
                "time_t": f"{round(elapsed_time / 3600, 2)} hours",
                "conveyance": f"{round(self.total_power_kwh, 3)} kWh energy invested"
            }
        }
        
        # Write summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print(f"MONITORING SUMMARY: {self.experiment_name}")
        print("="*60)
        print(f"Runtime: {summary['runtime']['total_hours']:.2f} hours")
        print(f"GPU-hours: {summary['runtime']['gpu_hours']:.2f}")
        print(f"Power consumption: {summary['runtime']['total_power_kwh']:.3f} kWh")
        print(f"Estimated cost: ${summary['runtime']['estimated_cost_usd']:.2f}")
        print(f"Peak GPU memory: {summary['peak_usage']['gpu_memory_mb']}")
        print(f"Average GPU utilization: {summary['average_usage']['gpu_utilization']:.1f}%")
        print("="*60)
        print(f"Metrics saved to: {self.metrics_file}")
        print(f"Summary saved to: {self.summary_file}")
        
        return summary


def main():
    """Example usage and CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="System monitoring for experiments")
    parser.add_argument("--experiment", default="experiment", help="Experiment name")
    parser.add_argument("--interval", type=int, default=5, help="Capture interval (seconds)")
    parser.add_argument("--track-pid", type=int, help="PID to track")
    parser.add_argument("--background", action="store_true", help="Run in background")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = SystemMonitor(
        experiment_name=args.experiment,
        capture_interval=args.interval
    )
    
    # Track specific process if requested
    if args.track_pid:
        monitor.track_process(args.track_pid)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down monitor...")
        monitor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start monitoring
    monitor.start(background=args.background)
    
    if args.background:
        print("Monitor running in background. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop()
    

if __name__ == "__main__":
    main()