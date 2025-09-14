"""
Backward Compatibility for Document Processor

This file provides backward compatibility for code that imports from
core.processors.document_processor. It redirects to the new
core.workflows.workflow_pdf module.

DEPRECATED: This will be removed after the restructure is complete.
"""

import warnings

warnings.warn(
    "Importing from core.processors.document_processor is deprecated. "
    "Please use 'from core.workflows.workflow_pdf import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backward compatibility
from core.workflows.workflow_pdf import *

# Make sure specific imports work
from core.workflows.workflow_pdf import (
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult,
    ExtractionResult
)