"""
Workflows Module

Provides orchestration and pipeline management for document processing.
Workflows coordinate the flow of data through extraction, embedding,
and storage phases while managing state and error recovery.

This module replaces scattered orchestration code with a unified structure.
"""

from .workflow_base import WorkflowBase, WorkflowConfig, WorkflowResult

# Import specific workflows if they exist
PDFWorkflow = None
BatchPDFWorkflow = None
ArxivMetadataWorkflow = None

try:
    from .workflow_pdf import PDFWorkflow  # type: ignore[assignment]
except ImportError:
    PDFWorkflow = None

try:
    from .workflow_pdf_batch import BatchPDFWorkflow  # type: ignore[assignment]
except ImportError:
    BatchPDFWorkflow = None

try:
    from .workflow_arxiv_metadata import ArxivMetadataWorkflow  # type: ignore[assignment]
except ImportError:
    ArxivMetadataWorkflow = None

# State management
from .state import StateManager, CheckpointManager

# Backward compatibility exports
__all__ = [
    'WorkflowBase',
    'WorkflowConfig',
    'WorkflowResult',
    'StateManager',
    'CheckpointManager',
]

if PDFWorkflow is not None:
    __all__.append('PDFWorkflow')
if BatchPDFWorkflow is not None:
    __all__.append('BatchPDFWorkflow')
if ArxivMetadataWorkflow is not None:
    __all__.append('ArxivMetadataWorkflow')
