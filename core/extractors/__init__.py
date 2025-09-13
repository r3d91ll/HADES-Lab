"""
Extractors Module

Provides document extraction capabilities for various file formats.
All extractors follow the Conveyance Framework, maximizing the CONVEYANCE
dimension by transforming unstructured documents into actionable data.

This module replaces core.framework.extractors with a cleaner structure.
"""

from .extractors_base import ExtractorBase, ExtractorConfig, ExtractionResult
from .extractors_factory import ExtractorFactory

# Auto-register available extractors
try:
    from .extractors_docling import DoclingExtractor
    ExtractorFactory.register("docling", DoclingExtractor)
except ImportError:
    pass

try:
    from .extractors_latex import LatexExtractor
    ExtractorFactory.register("latex", LatexExtractor)
except ImportError:
    pass

try:
    from .extractors_code import CodeExtractor
    ExtractorFactory.register("code", CodeExtractor)
except ImportError:
    pass

try:
    from .extractors_treesitter import TreeSitterExtractor
    ExtractorFactory.register("treesitter", TreeSitterExtractor)
except ImportError:
    pass

try:
    from .extractors_robust import RobustExtractor
    ExtractorFactory.register("robust", RobustExtractor)
except ImportError:
    pass

# Backward compatibility exports
__all__ = [
    'ExtractorBase',
    'ExtractorConfig',
    'ExtractionResult',
    'ExtractorFactory',
    'DoclingExtractor',
    'LatexExtractor',
    'CodeExtractor',
    'TreeSitterExtractor',
    'RobustExtractor',
]

# Convenience function for backward compatibility
def create_extractor_for_file(file_path, **kwargs):
    """
    Create an extractor for a given file (backward compatibility).

    Args:
        file_path: Path to the file
        **kwargs: Additional configuration

    Returns:
        Extractor instance
    """
    return ExtractorFactory.create_for_file(file_path, **kwargs)