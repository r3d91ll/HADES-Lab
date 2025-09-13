"""
Backward Compatibility for LaTeXExtractor

This file provides backward compatibility for code that imports from
core.framework.extractors.latex_extractor. It redirects to the new
core.extractors module.

DEPRECATED: This will be removed after the restructure is complete.
"""

import warnings

warnings.warn(
    "Importing from core.framework.extractors.latex_extractor is deprecated. "
    "Please use 'from core.extractors import LaTeXExtractor' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export LaTeXExtractor for backward compatibility
from core.extractors import LaTeXExtractor