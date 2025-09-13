"""
Backward Compatibility for Storage

This file provides backward compatibility for code that imports from
core.framework.storage. It redirects to the new core.workflows.storage module.

DEPRECATED: This will be removed after the restructure is complete.
"""

import warnings

warnings.warn(
    "Importing from core.framework.storage is deprecated. "
    "Please use core.workflows.storage instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backward compatibility
from core.workflows.storage import *

# Try to import specific storage components
try:
    from core.workflows.storage import (
        StorageBase,
        LocalStorage
    )
except ImportError:
    pass