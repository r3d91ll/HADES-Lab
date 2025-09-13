"""
Backward Compatibility for State Manager

This file provides backward compatibility for code that imports from
core.utils.state_manager. It redirects to the new
core.workflows.state module.

DEPRECATED: This will be removed after the restructure is complete.
"""

import warnings

warnings.warn(
    "Importing from core.utils.state_manager is deprecated. "
    "Please use 'from core.workflows.state import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backward compatibility
from core.workflows.state import *

# Make sure specific imports work
try:
    from core.workflows.state import (
        StateManager
    )
except ImportError:
    pass