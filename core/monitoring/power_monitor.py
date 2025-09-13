#!/usr/bin/env python3
"""
Power monitoring with UPS integration for precise energy measurement.
Essential for documenting computational costs in academic publications.
"""

import os
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import threading


class PowerMonitor:
    """Monitor power consumption through UPS and GPU sensors."""
    
    def __init__(self, log_dir: str = None):
        """
        Initialize power monitor.
        
        Args:
            log_dir: Directory for power logs
        """
        if log_dir is None:
            log_dir = Path.home() / "olympus/HADES-Lab/logs/power"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.power_log = self.log_dir / f"power_{self.session_id}.jsonl"
        
        # Power accumulators
        self.start_time = time.time()
        self.total_energy_kwh = 0.0
        self.peak_power_watts = 0.0
        self.samples = []
        
        # UPS detection
        self.ups_available = self._detect_ups()
        self.ups_command = None
        
        if self.ups_available:
            print("UPS detected - will capture accurate power measurements")
        else:
            print("No UPS detected - using GPU power estimates only")
    
    def _detect_ups(self) -> bool:
        """Detect if UPS monitoring is available."""
        ups_tools = [
            ("apcaccess", "apcupsd"),  # APC UPS
            ("upsc", "nut"),            # Network UPS Tools
            ("pwrstat", "cyberpower")   # CyberPower
        ]
        
        for cmd, ups_type in ups_tools:
            try:
                result = subprocess.run(
                    f"which {cmd}",
                    shell=True,
                    capture_output=True
                )
                if result.returncode == 0:
                    self.ups_command = (cmd, ups_type)
                    return True
            except:
                pass
        
        return False
    
    def capture_ups_power(self) -> Optional[Dict]:
        """Capture power data from UPS."""
        if not self.ups_available or not self.ups_command:
            return None
        
        cmd, ups_type = self.ups_command
        
        try:
            if ups_type == "apcupsd":
                return self._capture_apc_power()
            elif ups_type == "nut":
                return self._capture_nut_power()
            elif ups_type == "cyberpower":
                return self._capture_cyberpower_power()
        except Exception as e:
            print(f"Error reading UPS: {e}")
        
        return None
    
    def _capture_apc_power(self) -> Optional[Dict]:
        """Capture power from APC UPS via apcaccess."""
        try:
            result = subprocess.run(
                "apcaccess status",
                shell=True,
                capture_output=True,
                text=True
            )
            
            data = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip()] = value.strip()
            
            # Extract power metrics
            power_data = {
                "source": "APC UPS",
                "timestamp": datetime.now().isoformat()
            }
            
            # Load percentage and watts
            if "LOADPCT" in data:
                load_pct = float(data["LOADPCT"].replace(' Percent', ''))
                power_data["load_percent"] = load_pct
                
                # Calculate watts from load percentage and UPS capacity
                if "NOMPOWER" in data:
                    nom_power = float(data["NOMPOWER"].replace(' Watts', ''))
                    power_data["power_watts"] = (load_pct / 100.0) * nom_power
            
            # Battery and line voltage
            if "LINEV" in data:
                power_data["line_voltage"] = float(data["LINEV"].replace(' Volts', ''))
            if "BATTV" in data:
                power_data["battery_voltage"] = float(data["BATTV"].replace(' Volts', ''))
            if "BCHARGE" in data:
                power_data["battery_charge"] = float(data["BCHARGE"].replace(' Percent', ''))
            
            return power_data
            
        except Exception as e:
            print(f"Error reading APC UPS: {e}")
            return None
    
    def _capture_nut_power(self) -> Optional[Dict]:
        """Capture power from Network UPS Tools."""
        try:
            # Get UPS list
            result = subprocess.run(
                "upsc -l",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if not result.stdout.strip():
                return None
            
            ups_name = result.stdout.strip().split('\n')[0]
            
            # Get UPS data
            result = subprocess.run(
                f"upsc {ups_name}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            data = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip()] = value.strip()
            
            power_data = {
                "source": f"NUT UPS ({ups_name})",
                "timestamp": datetime.now().isoformat()
            }
            
            # Extract relevant metrics
            if "ups.load" in data:
                power_data["load_percent"] = float(data["ups.load"])
            if "ups.power" in data:
                power_data["power_watts"] = float(data["ups.power"])
            if "input.voltage" in data:
                power_data["line_voltage"] = float(data["input.voltage"])
            if "battery.charge" in data:
                power_data["battery_charge"] = float(data["battery.charge"])
            
            return power_data
            
        except Exception as e:
            print(f"Error reading NUT UPS: {e}")
            return None
    
    def _capture_cyberpower_power(self) -> Optional[Dict]:
        """Capture power from CyberPower UPS."""
        try:
            result = subprocess.run(
                "pwrstat -status",
                shell=True,
                capture_output=True,
                text=True
            )
            
            power_data = {
                "source": "CyberPower UPS",
                "timestamp": datetime.now().isoformat()
            }
            
            for line in result.stdout.split('\n'):
                if "Load" in line and "Watt" in line:
                    # Extract watts from line like "Load............. 120 Watt(20%)"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Watt" or "Watt" in part:
                            power_data["power_watts"] = float(parts[i-1])
                            # Extract percentage
                            if '(' in part:
                                pct = part.split('(')[1].replace('%)', '')
                                power_data["load_percent"] = float(pct)
                
                elif "Utility Voltage" in line:
                    parts = line.split()
                    for part in parts:
                        if part.replace('.', '').isdigit():
                            power_data["line_voltage"] = float(part)
                            break
                
                elif "Battery Capacity" in line:
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            power_data["battery_charge"] = float(part.replace('%', ''))
            
            return power_data
            
        except Exception as e:
            print(f"Error reading CyberPower UPS: {e}")
            return None
    
    def capture_gpu_power(self) -> List[Dict]:
        """Capture power data from GPUs."""
        gpu_power = []
        
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=index,name,power.draw,power.limit "
                "--format=csv,noheader",
                shell=True,
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    gpu_idx = int(parts[0])
                    power_draw = float(parts[2].replace(' W', '')) if ' W' in parts[2] else 0
                    power_limit = float(parts[3].replace(' W', '')) if ' W' in parts[3] else 0
                    
                    gpu_power.append({
                        "gpu_id": gpu_idx,
                        "name": parts[1],
                        "power_watts": power_draw,
                        "power_limit_watts": power_limit,
                        "power_percent": round(100 * power_draw / power_limit, 1) if power_limit > 0 else 0
                    })
        except:
            pass
        
        return gpu_power
    
    def capture_power_metrics(self) -> Dict:
        """Capture all power metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_hours": round((time.time() - self.start_time) / 3600, 4)
        }
        
        # UPS power (most accurate if available)
        ups_data = self.capture_ups_power()
        if ups_data:
            metrics["ups"] = ups_data
            system_power = ups_data.get("power_watts", 0)
        else:
            system_power = 0
        
        # GPU power
        gpu_data = self.capture_gpu_power()
        metrics["gpus"] = gpu_data
        
        # Calculate totals
        gpu_total = sum(g["power_watts"] for g in gpu_data)
        metrics["power_summary"] = {
            "gpu_total_watts": gpu_total,
            "system_total_watts": system_power if system_power > 0 else gpu_total * 1.3,  # Estimate 30% overhead
            "source": "UPS" if ups_data else "GPU estimate"
        }
        
        # Update accumulators
        current_power = metrics["power_summary"]["system_total_watts"]
        self.peak_power_watts = max(self.peak_power_watts, current_power)
        
        # Energy accumulation (kWh)
        if self.samples:
            time_delta_hours = metrics["elapsed_hours"] - self.samples[-1]["elapsed_hours"]
            energy_kwh = (current_power / 1000.0) * time_delta_hours
            self.total_energy_kwh += energy_kwh
            metrics["energy_delta_kwh"] = energy_kwh
        
        metrics["accumulated_energy_kwh"] = self.total_energy_kwh
        
        # Log metrics
        with open(self.power_log, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        self.samples.append(metrics)
        
        return metrics
    
    def get_summary(self) -> Dict:
        """Get power consumption summary."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        
        # Calculate average power
        if self.samples:
            avg_power = sum(s["power_summary"]["system_total_watts"] for s in self.samples) / len(self.samples)
        else:
            avg_power = 0
        
        summary = {
            "session_id": self.session_id,
            "runtime_hours": round(elapsed_hours, 3),
            "total_energy_kwh": round(self.total_energy_kwh, 4),
            "peak_power_watts": round(self.peak_power_watts, 1),
            "average_power_watts": round(avg_power, 1),
            "estimated_cost_usd": round(self.total_energy_kwh * 0.30, 2),
            "co2_kg": round(self.total_energy_kwh * 0.5, 2),  # ~0.5 kg CO2/kWh average
            "ups_available": self.ups_available,
            "measurement_source": self.ups_command[1] if self.ups_command else "GPU estimate"
        }
        
        return summary
    
    def monitor_continuous(self, interval: int = 10, duration: Optional[int] = None):
        """
        Monitor power continuously.
        
        Args:
            interval: Seconds between measurements
            duration: Total duration in seconds (None for indefinite)
        """
        print(f"Starting power monitoring (interval: {interval}s)")
        print(f"Logging to: {self.power_log}")
        
        start = time.time()
        
        try:
            while True:
                metrics = self.capture_power_metrics()
                
                # Display current power
                power = metrics["power_summary"]["system_total_watts"]
                energy = metrics["accumulated_energy_kwh"]
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Power: {power:.0f}W | Energy: {energy:.3f} kWh | "
                      f"Cost: ${energy * 0.30:.2f}", end='', flush=True)
                
                if duration and (time.time() - start) >= duration:
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            pass
        
        # Final summary
        summary = self.get_summary()
        summary_file = self.log_dir / f"power_summary_{self.session_id}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("POWER MONITORING SUMMARY")
        print("="*60)
        print(f"Total energy: {summary['total_energy_kwh']:.3f} kWh")
        print(f"Peak power: {summary['peak_power_watts']:.0f} W")
        print(f"Average power: {summary['average_power_watts']:.0f} W")
        print(f"Estimated cost: ${summary['estimated_cost_usd']:.2f}")
        print(f"CO2 emissions: {summary['co2_kg']:.1f} kg")
        print(f"Measurement source: {summary['measurement_source']}")
        print("="*60)
        print(f"Summary saved to: {summary_file}")


def main():
    """CLI interface for power monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Power monitoring with UPS integration")
    parser.add_argument("--interval", type=int, default=10, help="Measurement interval (seconds)")
    parser.add_argument("--duration", type=int, help="Monitoring duration (seconds)")
    parser.add_argument("--test", action="store_true", help="Test power capture once")
    
    args = parser.parse_args()
    
    monitor = PowerMonitor()
    
    if args.test:
        # Test capture
        metrics = monitor.capture_power_metrics()
        print(json.dumps(metrics, indent=2))
    else:
        # Continuous monitoring
        monitor.monitor_continuous(
            interval=args.interval,
            duration=args.duration
        )


if __name__ == "__main__":
    main()