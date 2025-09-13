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

# Backward compatibility imports with deprecation warnings
import types
import warnings

from .base_processor import BaseProcessor
from .config import Config, ConfigManager
from .logging import LogManager
from .metrics import MetricsCollector

try:
    from . import embedders_compat as embedders
except ImportError:
    # Create safe fallback
    embedders = types.SimpleNamespace()
    warnings.warn(
        "embedders_compat module not found. Using empty namespace fallback.",
        ImportWarning,
        stacklevel=2
    )

try:
    from . import extractors_compat as extractors
except ImportError:
    # Create safe fallback
    extractors = types.SimpleNamespace()
    warnings.warn(
        "extractors_compat module not found. Using empty namespace fallback.",
        ImportWarning,
        stacklevel=2
    )

try:
    from . import storage_compat as storage
    # Add StorageManager alias for backward compatibility
    StorageManager = getattr(storage, 'StorageManager', None)
except ImportError:
    # Create safe fallback
    storage = types.SimpleNamespace()
    StorageManager = None
    warnings.warn(
        "storage_compat module not found. Using empty namespace fallback.",
        ImportWarning,
        stacklevel=2
    )

__all__ = [
    'BaseProcessor',
    'ConfigManager',
    'Config',
    'LogManager',
    'MetricsCollector',
    'embedders',
    'extractors',
    'storage',
    'StorageManager'
]

__version__ = '1.5.0'
