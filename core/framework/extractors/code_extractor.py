"""
Backward Compatibility for CodeExtractor

This file provides backward compatibility for code that imports from
core.framework.extractors.code_extractor. It redirects to the new
core.extractors module.

DEPRECATED: This will be removed after the restructure is complete.
"""

import warnings

warnings.warn(
    "Importing from core.framework.extractors.code_extractor is deprecated. "
    "Please use 'from core.extractors import CodeExtractor' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export CodeExtractor for backward compatibility
