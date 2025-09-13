#!/usr/bin/env python3
"""
Analyze monitoring metrics for academic publications and ANT documentation.
Generates publication-ready statistics and visualizations.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class MetricsAnalyzer:
    """Analyze training and system metrics for documentation."""
    
    def __init__(self, metrics_file: str):
        """
        Initialize analyzer with metrics file.
        
        Args:
            metrics_file: Path to JSONL metrics file
        """
        self.metrics_file = Path(metrics_file)
        self.metrics = []
        self.load_metrics()
        
    def load_metrics(self):
        """Load metrics from JSONL file."""
        with open(self.metrics_file, 'r') as f:
            for line in f:
                try:
                    self.metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(self.metrics)} metric entries")
        
    def generate_ant_summary(self) -> Dict:
        """
        Generate summary aligned with ANT framework.
        
        Returns:
            Dictionary with ANT-aligned metrics
        """
        if not self.metrics:
            return {}
        
        first = self.metrics[0]
        last = self.metrics[-1]
        
        # Extract GPU information
        gpu_info = []
        if "gpu" in first:
            for gpu in first["gpu"]:
                gpu_info.append(gpu["name"])
        elif "system_info" in first:
            gpu_info = [g["name"] for g in first["system_info"]["hardware"]["gpus"]]
        
        # Calculate statistics
        gpu_utils = []
        gpu_memory = []
        power_samples = []
        
        for m in self.metrics:
            if "gpu" in m:
                for gpu in m["gpu"]:
                    gpu_utils.append(gpu.get("utilization_percent", 0))
                    gpu_memory.append(gpu.get("memory_used_mb", 0))
                    power_samples.append(gpu.get("power_watts", 0))
        
        # Runtime calculation
        if "elapsed_hours" in last:
            runtime_hours = last["elapsed_hours"]
        else:
            runtime_hours = (last.get("elapsed_seconds", 0)) / 3600
        
        # Energy calculation
        total_energy = 0
        if "accumulated_energy_kwh" in last:
            total_energy = last["accumulated_energy_kwh"]
        elif power_samples:
            # Estimate from GPU power
            avg_power = np.mean(power_samples)
            total_energy = (avg_power / 1000.0) * runtime_hours * len(gpu_info)
        
        summary = {
            "ant_framework": {
                "actant": "GraphSAGE Neural Network",
                "role": "Librarian Gatekeeper - mediates 2.82M paper knowledge graph",
                "network_translation": {
                    "from": "Intractable graph (2.82M nodes, 74.2M edges)",
                    "to": "Navigable embedding space (1024 dimensions)",
                    "mechanism": "Graph convolution across 4 layers"
                },
                "creation_cost": {
                    "time_t": f"{runtime_hours:.2f} hours",
                    "who_h": f"{len(gpu_info)}x {gpu_info[0] if gpu_info else 'GPU'}",
                    "computational": f"{runtime_hours * len(gpu_info):.2f} GPU-hours",
                    "energy": f"{total_energy:.3f} kWh",
                    "monetary": f"${total_energy * 0.30:.2f}"
                },
                "capability_metrics": {
                    "papers_processed": "2,820,000",
                    "edges_created": "74,200,000",
                    "embedding_dimensions": "1024",
                    "layers": "4"
                }
            },
            "conveyance_framework": {
                "W_what": "Paper semantic content via abstracts/titles",
                "R_where": "Category/temporal/keyword graph topology",  
                "H_who": f"{len(gpu_info)} GPUs with {np.max(gpu_memory)/1024:.1f}GB peak memory",
                "T_time": f"{runtime_hours:.2f} hours",
                "context_alpha": "1.5-2.0 (super-linear with context quality)",
                "efficiency": f"{2820000 / (runtime_hours * 3600):.1f} papers/second"
            },
            "performance_metrics": {
                "gpu_utilization": {
                    "average": f"{np.mean(gpu_utils):.1f}%" if gpu_utils else "N/A",
                    "peak": f"{np.max(gpu_utils):.1f}%" if gpu_utils else "N/A",
                    "efficiency_rating": self._rate_efficiency(np.mean(gpu_utils) if gpu_utils else 0)
                },
                "memory_usage": {
                    "average_gb": f"{np.mean(gpu_memory)/1024:.1f}" if gpu_memory else "N/A",
                    "peak_gb": f"{np.max(gpu_memory)/1024:.1f}" if gpu_memory else "N/A"
                },
                "power": {
                    "average_watts": f"{np.mean(power_samples):.0f}" if power_samples else "N/A",
                    "peak_watts": f"{np.max(power_samples):.0f}" if power_samples else "N/A",
                    "total_kwh": f"{total_energy:.3f}"
                }
            },
            "paper_ready_statement": self._generate_paper_statement(
                runtime_hours, len(gpu_info), total_energy, gpu_info
            )
        }
        
        return summary
    
    def _rate_efficiency(self, utilization: float) -> str:
        """Rate GPU utilization efficiency."""
        if utilization > 80:
            return "Excellent (>80%)"
        elif utilization > 60:
            return "Good (60-80%)"
        elif utilization > 40:
            return "Moderate (40-60%)"
        else:
            return "Poor (<40%)"
    
    def _generate_paper_statement(self, runtime_hours: float, num_gpus: int, 
                                  energy_kwh: float, gpu_info: List[str]) -> str:
        """Generate statement ready for academic paper."""
        gpu_name = gpu_info[0] if gpu_info else "GPU"
        
        statement = (
            f"The GraphSAGE model, conceptualized as a 'librarian gatekeeper' actant "
            f"in our Actor-Network Theory framework, required {runtime_hours:.1f} hours "
            f"of training on {num_gpus}× {gpu_name} GPUs (totaling {runtime_hours * num_gpus:.1f} "
            f"GPU-hours). This computational investment, consuming {energy_kwh:.2f} kWh of energy "
            f"(approximately ${energy_kwh * 0.30:.2f} at typical datacenter rates), transformed "
            f"an intractable knowledge graph of 2.82 million academic papers and 74.2 million "
            f"edges into a navigable 1024-dimensional embedding space. The resulting model "
            f"mediates access to the arxiv corpus, acting as an algorithmic gatekeeper that "
            f"enables efficient semantic navigation through the academic knowledge landscape."
        )
        
        return statement
    
    def plot_metrics(self, output_dir: Optional[str] = None) -> Path:
        """
        Generate publication-ready visualizations.
        
        Args:
            output_dir: Directory for output files
            
        Returns:
            Path to saved figure
        """
        if not output_dir:
            output_dir = self.metrics_file.parent
        output_dir = Path(output_dir)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('GraphSAGE Training: Creating the Librarian Gatekeeper', 
                     fontsize=16, fontweight='bold')
        
        # Extract time series data
        timestamps = []
        gpu_utils = defaultdict(list)
        memory_usage = defaultdict(list)
        power_draw = defaultdict(list)
        cpu_usage = []
        
        for m in self.metrics:
            if "elapsed_hours" in m:
                timestamps.append(m["elapsed_hours"])
            elif "elapsed_seconds" in m:
                timestamps.append(m["elapsed_seconds"] / 3600)
            else:
                continue
                
            # GPU metrics
            if "gpu" in m:
                for gpu in m["gpu"]:
                    gpu_id = gpu.get("gpu_id", 0)
                    gpu_utils[gpu_id].append(gpu.get("utilization_percent", 0))
                    memory_usage[gpu_id].append(gpu.get("memory_used_mb", 0) / 1024)
                    power_draw[gpu_id].append(gpu.get("power_watts", 0))
            
            # CPU metrics
            if "system" in m:
                cpu_usage.append(m["system"].get("cpu_percent", 0))
        
        # Plot 1: GPU Utilization
        ax = axes[0, 0]
        for gpu_id in sorted(gpu_utils.keys())[:2]:  # Show first 2 GPUs
            ax.plot(timestamps, gpu_utils[gpu_id], 
                   label=f'GPU {gpu_id}', alpha=0.8, linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Utilization (%)')
        ax.set_title('GPU Utilization During Training')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Plot 2: Memory Usage
        ax = axes[0, 1]
        for gpu_id in sorted(memory_usage.keys())[:2]:
            ax.plot(timestamps, memory_usage[gpu_id],
                   label=f'GPU {gpu_id}', alpha=0.8, linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('GPU Memory Consumption')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Power Consumption
        ax = axes[0, 2]
        if power_draw:
            total_power = []
            for i in range(len(timestamps)):
                total = sum(power_draw[gpu_id][i] for gpu_id in power_draw 
                          if i < len(power_draw[gpu_id]))
                total_power.append(total)
            ax.plot(timestamps, total_power, color='orange', linewidth=2)
            ax.fill_between(timestamps, 0, total_power, alpha=0.3, color='orange')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Power (Watts)')
        ax.set_title('Total System Power Draw')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: CPU Usage
        ax = axes[1, 0]
        if cpu_usage:
            ax.plot(timestamps[:len(cpu_usage)], cpu_usage, 
                   color='green', linewidth=2)
            ax.fill_between(timestamps[:len(cpu_usage)], 0, cpu_usage, 
                          alpha=0.3, color='green')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('CPU Utilization')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Plot 5: Cumulative Energy
        ax = axes[1, 1]
        if power_draw and timestamps:
            cumulative_energy = []
            total_kwh = 0
            for i in range(len(timestamps)):
                if i > 0:
                    time_delta = timestamps[i] - timestamps[i-1]
                    power_w = sum(power_draw[gpu_id][i] for gpu_id in power_draw 
                                if i < len(power_draw[gpu_id]))
                    total_kwh += (power_w / 1000.0) * time_delta
                cumulative_energy.append(total_kwh)
            
            ax.plot(timestamps, cumulative_energy, color='red', linewidth=2)
            ax.fill_between(timestamps, 0, cumulative_energy, alpha=0.3, color='red')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Energy (kWh)')
        ax.set_title('Cumulative Energy Consumption')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Efficiency Timeline
        ax = axes[1, 2]
        if gpu_utils:
            # Calculate rolling efficiency
            window = 12  # 1 minute at 5s intervals
            efficiency = []
            for i in range(len(timestamps)):
                start_idx = max(0, i - window)
                window_utils = []
                for gpu_id in gpu_utils:
                    if i < len(gpu_utils[gpu_id]):
                        window_utils.extend(gpu_utils[gpu_id][start_idx:i+1])
                if window_utils:
                    efficiency.append(np.mean(window_utils))
                else:
                    efficiency.append(0)
            
            ax.plot(timestamps, efficiency, color='purple', linewidth=2)
            
            # Add efficiency zones
            ax.axhspan(80, 100, alpha=0.2, color='green', label='Excellent')
            ax.axhspan(60, 80, alpha=0.2, color='yellow', label='Good')
            ax.axhspan(40, 60, alpha=0.2, color='orange', label='Moderate')
            ax.axhspan(0, 40, alpha=0.2, color='red', label='Poor')
            
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Efficiency (%)')
        ax.set_title('Training Efficiency')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        # Save figure
        output_file = output_dir / f"metrics_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_file}")
        
        return output_file
    
    def save_summary(self, output_file: Optional[str] = None) -> Path:
        """
        Save ANT framework summary to JSON.
        
        Args:
            output_file: Path for output file
            
        Returns:
            Path to saved summary
        """
        if not output_file:
            output_file = self.metrics_file.parent / f"ant_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file = Path(output_file)
        summary = self.generate_ant_summary()
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved ANT summary to: {output_file}")
        
        # Print key points
        print("\n" + "="*70)
        print("KEY METRICS FOR 'DEATH OF THE AUTHOR' ARTICLE")
        print("="*70)
        print(summary["paper_ready_statement"])
        print("="*70)
        
        return output_file
    
    def export_for_paper(self, output_dir: Optional[str] = None) -> Dict[str, Path]:
        """
        Export all materials needed for academic paper.
        
        Args:
            output_dir: Directory for exports
            
        Returns:
            Dictionary of exported file paths
        """
        if not output_dir:
            output_dir = self.metrics_file.parent / "paper_exports"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        exports = {}
        
        # Generate and save all outputs
        exports["summary"] = self.save_summary(output_dir / "ant_summary.json")
        exports["visualization"] = self.plot_metrics(output_dir)
        
        # Create LaTeX-ready statistics file
        summary = self.generate_ant_summary()
        latex_stats = output_dir / "statistics.tex"
        
        with open(latex_stats, 'w') as f:
            f.write("% Statistics for GraphSAGE Training\n")
            f.write("% Generated: " + datetime.now().isoformat() + "\n\n")
            
            # ANT metrics
            ant = summary["ant_framework"]
            f.write("\\newcommand{\\RuntimeHours}{" + 
                   ant["creation_cost"]["time_t"].replace(" hours", "") + "}\n")
            f.write("\\newcommand{\\GPUHours}{" + 
                   ant["creation_cost"]["computational"].replace(" GPU-hours", "") + "}\n")
            f.write("\\newcommand{\\EnergyKWH}{" + 
                   ant["creation_cost"]["energy"].replace(" kWh", "") + "}\n")
            f.write("\\newcommand{\\EstimatedCost}{" + 
                   ant["creation_cost"]["monetary"] + "}\n")
            
            # Performance metrics
            perf = summary["performance_metrics"]
            if "average" in perf["gpu_utilization"]:
                f.write("\\newcommand{\\AvgGPUUtil}{" + 
                       perf["gpu_utilization"]["average"].replace("%", "") + "}\n")
            if "peak_gb" in perf["memory_usage"]:
                f.write("\\newcommand{\\PeakMemoryGB}{" + 
                       perf["memory_usage"]["peak_gb"] + "}\n")
        
        exports["latex_stats"] = latex_stats
        
        print(f"\nExported paper materials to: {output_dir}")
        return exports


def main():
    """CLI interface for metrics analysis."""
    parser = argparse.ArgumentParser(description="Analyze training metrics")
    parser.add_argument("metrics_file", help="Path to metrics JSONL file")
    parser.add_argument("--plot", action="store_true", help="Generate visualizations")
    parser.add_argument("--export", action="store_true", help="Export for paper")
    parser.add_argument("--output-dir", help="Output directory")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = MetricsAnalyzer(args.metrics_file)
    
    # Generate summary
    summary = analyzer.generate_ant_summary()
    
    # Save summary
    analyzer.save_summary()
    
    # Generate plots if requested
    if args.plot:
        analyzer.plot_metrics(args.output_dir)
    
    # Export for paper if requested
    if args.export:
        analyzer.export_for_paper(args.output_dir)
    
    # Print summary
    print("\nAnalysis Complete!")
    print(f"Infrastructure: {summary['ant_framework']['creation_cost']['who_h']}")
    print(f"Energy Cost: {summary['ant_framework']['creation_cost']['energy']}")
    print(f"Efficiency: {summary['performance_metrics']['gpu_utilization']['efficiency_rating']}")


if __name__ == "__main__":
    main()