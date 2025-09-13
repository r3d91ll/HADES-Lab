"""
Backward Compatibility for Embedders

This file provides backward compatibility for code that imports from
core.framework.embedders. It redirects to the new core.embedders module.

DEPRECATED: This will be removed after the restructure is complete.
"""

import warnings

from core.embedders import *

warnings.warn(
    "Importing from core.framework.embedders is deprecated. "
    "Please use core.embedders instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backward compatibility

# Try to import specific embedders
try:
    from core.embedders.embedders_jina import JinaV4Embedder
except ImportError:
    pass
