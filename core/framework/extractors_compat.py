"""
Backward Compatibility for Extractors

This file provides backward compatibility for code that imports from
core.framework.extractors. It redirects to the new core.extractors module.

DEPRECATED: This will be removed after the restructure is complete.
"""

import warnings

warnings.warn(
    "Importing from core.framework.extractors is deprecated. "
    "Please use core.extractors instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backward compatibility
from core.extractors import *

# Try to import specific extractors that might be used
try:
    from core.extractors import (
        CodeExtractor,
        DoclingExtractor,
        ExtractorBase,
        ExtractorConfig,
        ExtractorFactory,
        LaTeXExtractor,
        RobustExtractor,
        TreeSitterExtractor,
    )
except ImportError:
    pass
