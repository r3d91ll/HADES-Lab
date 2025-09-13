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

# Backward compatibility imports with deprecation warnings
try:
    from . import embedders_compat as embedders
    from . import extractors_compat as extractors
    from . import storage_compat as storage
except ImportError:
    pass

__all__ = [
    'BaseProcessor',
    'ConfigManager',
    'Config',
    'LogManager',
    'MetricsCollector',
    'embedders',
    'extractors',
    'storage'
]

__version__ = '1.5.0'