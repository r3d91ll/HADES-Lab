"""
Backward Compatibility for DoclingExtractor

This file provides backward compatibility for code that imports from
core.framework.extractors.docling_extractor. It redirects to the new
core.extractors module.

DEPRECATED: This will be removed after the restructure is complete.
"""

import warnings

warnings.warn(
    "Importing from core.framework.extractors.docling_extractor is deprecated. "
    "Please use 'from core.extractors import DoclingExtractor' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export DoclingExtractor for backward compatibility
