"""
Backward Compatibility for TreeSitterExtractor

This file provides backward compatibility for code that imports from
core.framework.extractors.tree_sitter_extractor. It redirects to the new
core.extractors module.

DEPRECATED: This will be removed after the restructure is complete.
"""

import warnings

warnings.warn(
    "Importing from core.framework.extractors.tree_sitter_extractor is deprecated. "
    "Please use 'from core.extractors import TreeSitterExtractor' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export TreeSitterExtractor for backward compatibility
