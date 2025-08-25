#!/usr/bin/env python3
"""
Pipeline Status Reporter
========================

Shared status reporting system for the unified pipeline.
Creates a status file that the monitoring dashboard can read.
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class PipelineStatusReporter:
    """
    Shared status reporter for pipeline statistics.
    
    This class can be used by the pipeline to report real-time statistics
    that the monitoring dashboard can read.
    """
    
    def __init__(self, status_file: str = "pipeline_status.json"):
        self.status_file = Path(status_file).resolve()
        self.status_data = {
            "timestamp": None,
            "queues": {
                "extraction_queue": 0,
                "embedding_queue": 0, 
                "write_queue": 0
            },
            "workers": {
                "extraction_active": 0,
                "embedding_active": 0,
                "write_active": 0
            },
            "processing": {
                "papers_processed": 0,
                "current_batch_size": 0,
                "last_paper_id": None,
                "errors_count": 0
            },
            "rates": {
                "extraction_rate": 0.0,
                "embedding_rate": 0.0,
                "write_rate": 0.0
            }
        }
        
        self._lock = threading.Lock()
        self._update_thread = None
        self._stop_event = threading.Event()
    
    def start_reporting(self, update_interval: int = 5):
        """Start background status reporting thread."""
        if self._update_thread and self._update_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._update_thread = threading.Thread(
            target=self._update_loop, 
            args=(update_interval,),
            daemon=True
        )
        self._update_thread.start()
    
    def stop_reporting(self):
        """Stop background status reporting."""
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=10)
    
    def update_queue_sizes(self, extraction: int = None, embedding: int = None, write: int = None):
        """Update queue size statistics."""
        with self._lock:
            if extraction is not None:
                self.status_data["queues"]["extraction_queue"] = extraction
            if embedding is not None:
                self.status_data["queues"]["embedding_queue"] = embedding
            if write is not None:
                self.status_data["queues"]["write_queue"] = write
            self.status_data["timestamp"] = datetime.now().isoformat()
    
    def update_worker_counts(self, extraction: int = None, embedding: int = None, write: int = None):
        """Update active worker counts."""
        with self._lock:
            if extraction is not None:
                self.status_data["workers"]["extraction_active"] = extraction
            if embedding is not None:
                self.status_data["workers"]["embedding_active"] = embedding
            if write is not None:
                self.status_data["workers"]["write_active"] = write
            self.status_data["timestamp"] = datetime.now().isoformat()
    
    def update_processing_stats(self, papers_processed: int = None, current_batch: int = None, 
                              last_paper_id: str = None, errors: int = None):
        """Update processing statistics."""
        with self._lock:
            if papers_processed is not None:
                self.status_data["processing"]["papers_processed"] = papers_processed
            if current_batch is not None:
                self.status_data["processing"]["current_batch_size"] = current_batch
            if last_paper_id is not None:
                self.status_data["processing"]["last_paper_id"] = last_paper_id
            if errors is not None:
                self.status_data["processing"]["errors_count"] = errors
            self.status_data["timestamp"] = datetime.now().isoformat()
    
    def update_rates(self, extraction: float = None, embedding: float = None, write: float = None):
        """Update processing rates (items per minute)."""
        with self._lock:
            if extraction is not None:
                self.status_data["rates"]["extraction_rate"] = extraction
            if embedding is not None:
                self.status_data["rates"]["embedding_rate"] = embedding
            if write is not None:
                self.status_data["rates"]["write_rate"] = write
            self.status_data["timestamp"] = datetime.now().isoformat()
    
    def _update_loop(self, interval: int):
        """Background thread to write status to file."""
        while not self._stop_event.wait(interval):
            try:
                with self._lock:
                    status_copy = self.status_data.copy()
                
                # Write to file atomically
                temp_file = self.status_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(status_copy, f, indent=2)
                
                # Atomic move
                temp_file.replace(self.status_file)
                
            except Exception as e:
                # Log error but continue
                print(f"Status reporter error: {e}")
    
    def read_status(self) -> Optional[Dict[str, Any]]:
        """Read current status from file."""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error reading status file: {e}")
        return None
    
    def cleanup(self):
        """Clean up status file and stop reporting."""
        self.stop_reporting()
        try:
            if self.status_file.exists():
                self.status_file.unlink()
        except Exception:
            pass


# Global instance for easy use
_global_reporter = None

def get_reporter(status_file: str = None) -> PipelineStatusReporter:
    """Get or create the global status reporter instance."""
    global _global_reporter
    
    if _global_reporter is None:
        if status_file is None:
            # Default to the current working directory
            status_file = "pipeline_status.json"
        _global_reporter = PipelineStatusReporter(status_file)
    
    return _global_reporter


def start_status_reporting(status_file: str = None, update_interval: int = 5):
    """Start global status reporting."""
    reporter = get_reporter(status_file)
    reporter.start_reporting(update_interval)
    return reporter


def stop_status_reporting():
    """Stop global status reporting."""
    global _global_reporter
    if _global_reporter:
        _global_reporter.cleanup()
        _global_reporter = None


# Convenience functions for easy integration
def report_queue_sizes(**kwargs):
    """Report queue sizes to the global reporter."""
    reporter = get_reporter()
    reporter.update_queue_sizes(**kwargs)


def report_worker_counts(**kwargs):
    """Report worker counts to the global reporter."""
    reporter = get_reporter()
    reporter.update_worker_counts(**kwargs)


def report_processing_stats(**kwargs):
    """Report processing statistics to the global reporter."""
    reporter = get_reporter()
    reporter.update_processing_stats(**kwargs)


def report_rates(**kwargs):
    """Report processing rates to the global reporter."""
    reporter = get_reporter()
    reporter.update_rates(**kwargs)


if __name__ == "__main__":
    # Test the reporter
    print("Testing Pipeline Status Reporter...")
    
    reporter = start_status_reporting("test_status.json", update_interval=2)
    
    try:
        # Simulate some status updates
        for i in range(10):
            report_queue_sizes(extraction=i*10, embedding=i*5, write=i*2)
            report_worker_counts(extraction=8, embedding=4, write=2)
            report_processing_stats(papers_processed=i*50, current_batch=32)
            report_rates(extraction=120.5, embedding=45.2, write=78.3)
            
            time.sleep(3)
            
            # Read back the status
            status = reporter.read_status()
            if status:
                print(f"Iteration {i+1}: {status['queues']}")
    
    finally:
        stop_status_reporting()
        # Clean up test file
        import os
        if os.path.exists("test_status.json"):
            os.remove("test_status.json")
    
    print("Test completed!")