"""
Backward Compatibility for RobustExtractor

This file provides backward compatibility for code that imports from
core.framework.extractors.robust_extractor. It redirects to the new
core.extractors module.

DEPRECATED: This will be removed after the restructure is complete.
"""

import warnings

warnings.warn(
    "Importing from core.framework.extractors.robust_extractor is deprecated. "
    "Please use 'from core.extractors import RobustExtractor' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export RobustExtractor for backward compatibility
from core.extractors import RobustExtractor