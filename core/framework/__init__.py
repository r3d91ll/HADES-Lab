"""
HADES Processor Framework
=========================

Core framework for all HADES processors providing:
- Configuration management
- Structured logging
- Metrics collection
- Database connections
- Shared utilities
"""

from .base_processor import BaseProcessor
from .config import ConfigManager, Config
from .logging import LogManager
from .metrics import MetricsCollector
from .storage import StorageManager

__all__ = [
    'BaseProcessor',
    'ConfigManager',
    'Config',
    'LogManager',
    'MetricsCollector',
    'StorageManager'
]

__version__ = '1.5.0'