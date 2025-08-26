#!/usr/bin/env python3
"""
Pre-flight Validation Framework
================================

Generic pre-flight checks for any pipeline or expensive operation.
Ensures all prerequisites are met before starting.

Following Actor-Network Theory: Pre-flight checks establish the
"enrollment" of all necessary actants before beginning the translation
process, preventing costly failures from missing prerequisites.
"""

import os
import shutil
import logging
import psycopg2
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class PreflightChecker:
    """
    Runs comprehensive pre-flight checks before expensive operations.
    
    In Information Reconstructionism terms, this validates that all
    dimensional prerequisites are non-zero before attempting to
    create information.
    """
    
    def __init__(self, name: str = "Pipeline"):
        """
        Initialize preflight checker.
        
        Args:
            name: Name of the process being checked
        """
        self.name = name
        self.checks = []
        self.results = {}
        self.passed = True
    
    def add_check(self, name: str, check_func: Callable[[], bool], 
                  description: str = "", critical: bool = True):
        """
        Add a check to run.
        
        Args:
            name: Name of the check
            check_func: Function that returns True if check passes
            description: Description of what's being checked
            critical: If True, failure stops all checks
        """
        self.checks.append({
            'name': name,
            'func': check_func,
            'description': description,
            'critical': critical
        })
    
    def check_file_exists(self, path: Path, description: str = None) -> bool:
        """Check if a file exists."""
        if not path.exists():
            logger.error(f"❌ File not found: {path}")
            return False
        
        size_mb = path.stat().st_size / (1024**2)
        logger.info(f"✓ File found: {path.name} ({size_mb:.2f} MB)")
        return True
    
    def check_directory_exists(self, path: Path, description: str = None) -> bool:
        """Check if a directory exists."""
        if not path.exists():
            logger.error(f"❌ Directory not found: {path}")
            return False
        
        # Count files
        file_count = sum(1 for _ in path.iterdir())
        logger.info(f"✓ Directory found: {path} ({file_count} items)")
        return True
    
    def check_disk_space(self, path: Path, required_gb: float) -> bool:
        """Check if sufficient disk space is available."""
        try:
            stat = shutil.disk_usage(path)
            available_gb = stat.free / (1024**3)
            
            if available_gb < required_gb:
                logger.error(f"❌ Insufficient disk space: {available_gb:.2f} GB available, {required_gb:.2f} GB required")
                return False
            
            logger.info(f"✓ Disk space: {available_gb:.2f} GB available")
            return True
            
        except Exception as e:
            logger.error(f"❌ Could not check disk space: {e}")
            return False
    
    def check_database_connection(self, host: str, database: str, 
                                 user: str, password: str) -> bool:
        """Check PostgreSQL database connection."""
        try:
            conn = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )
            cur = conn.cursor()
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            cur.close()
            conn.close()
            
            logger.info(f"✓ Database connection successful: {database}")
            logger.debug(f"  PostgreSQL version: {version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def check_environment_variable(self, var_name: str) -> bool:
        """Check if an environment variable is set."""
        value = os.environ.get(var_name)
        if not value:
            logger.error(f"❌ Environment variable not set: {var_name}")
            return False
        
        # Mask sensitive values
        if 'PASSWORD' in var_name or 'KEY' in var_name or 'SECRET' in var_name:
            masked = value[:3] + '*' * (len(value) - 6) + value[-3:] if len(value) > 6 else '***'
            logger.info(f"✓ Environment variable set: {var_name} = {masked}")
        else:
            logger.info(f"✓ Environment variable set: {var_name}")
        
        return True
    
    def check_python_module(self, module_name: str) -> bool:
        """Check if a Python module is installed."""
        try:
            __import__(module_name)
            logger.info(f"✓ Python module available: {module_name}")
            return True
        except ImportError:
            logger.error(f"❌ Python module not found: {module_name}")
            return False
    
    def check_gpu_available(self) -> bool:
        """Check if GPU is available (for PyTorch)."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    logger.info(f"✓ GPU {i}: {props.name} ({memory_gb:.2f} GB)")
                return True
            else:
                logger.warning("⚠️  No GPU available, will use CPU")
                return True  # Not critical - can fall back to CPU
                
        except ImportError:
            logger.warning("⚠️  PyTorch not installed, cannot check GPU")
            return True  # Not critical if not using PyTorch
    
    def run_all_checks(self) -> bool:
        """
        Run all registered checks.
        
        Returns:
            True if all critical checks pass
        """
        logger.info("=" * 60)
        logger.info(f"RUNNING PRE-FLIGHT CHECKS FOR {self.name}")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        for i, check in enumerate(self.checks, 1):
            print(f"\n{i}. {check['description'] or check['name']}...")
            
            try:
                result = check['func']()
                self.results[check['name']] = result
                
                if not result and check['critical']:
                    logger.error(f"Critical check failed: {check['name']}")
                    self.passed = False
                    break
                    
            except Exception as e:
                logger.error(f"Check failed with exception: {e}")
                self.results[check['name']] = False
                
                if check['critical']:
                    self.passed = False
                    break
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "=" * 60)
        if self.passed:
            logger.info("✅ ALL PRE-FLIGHT CHECKS PASSED")
        else:
            logger.error("❌ PRE-FLIGHT CHECKS FAILED")
        logger.info(f"Time: {elapsed:.2f} seconds")
        logger.info("=" * 60)
        
        return self.passed
    
    def get_report(self) -> Dict[str, Any]:
        """Get detailed report of check results."""
        return {
            'process': self.name,
            'passed': self.passed,
            'timestamp': datetime.now().isoformat(),
            'checks': self.results,
            'summary': {
                'total': len(self.checks),
                'passed': sum(1 for v in self.results.values() if v),
                'failed': sum(1 for v in self.results.values() if not v)
            }
        }


def standard_pipeline_checks(
    metadata_file: Optional[Path] = None,
    required_disk_gb: float = 50.0,
    required_env_vars: Optional[List[str]] = None,
    required_modules: Optional[List[str]] = None
) -> PreflightChecker:
    """
    Create a standard set of pipeline checks.
    
    Args:
        metadata_file: Path to required metadata file
        required_disk_gb: Required disk space in GB
        required_env_vars: List of required environment variables
        required_modules: List of required Python modules
        
    Returns:
        Configured PreflightChecker
    """
    checker = PreflightChecker("Data Pipeline")
    
    # File checks
    if metadata_file:
        checker.add_check(
            "metadata_file",
            lambda: checker.check_file_exists(metadata_file),
            "Checking metadata file"
        )
    
    # Disk space
    checker.add_check(
        "disk_space",
        lambda: checker.check_disk_space(Path.home(), required_disk_gb),
        f"Checking disk space ({required_disk_gb} GB required)"
    )
    
    # Environment variables
    if required_env_vars:
        for var in required_env_vars:
            checker.add_check(
                f"env_{var}",
                lambda v=var: checker.check_environment_variable(v),
                f"Checking environment variable {var}"
            )
    
    # Python modules
    if required_modules:
        for module in required_modules:
            checker.add_check(
                f"module_{module}",
                lambda m=module: checker.check_python_module(m),
                f"Checking Python module {module}",
                critical=False  # Usually not critical
            )
    
    # GPU check (not critical)
    checker.add_check(
        "gpu",
        checker.check_gpu_available,
        "Checking GPU availability",
        critical=False
    )
    
    return checker