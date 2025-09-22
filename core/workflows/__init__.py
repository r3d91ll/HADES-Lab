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
ArxivInitialIngestWorkflow = None
ArxivSinglePDFWorkflow = None
ArxivGraphBuildWorkflow = None

try:
    from .workflow_pdf import PDFWorkflow  # type: ignore[assignment]
except ImportError:
    PDFWorkflow = None

try:
    from .workflow_pdf_batch import BatchPDFWorkflow  # type: ignore[assignment]
except ImportError:
    BatchPDFWorkflow = None

try:
    from .workflow_arxiv_initial_ingest import ArxivInitialIngestWorkflow  # type: ignore[assignment]
except ImportError:
    ArxivInitialIngestWorkflow = None

try:
    from .arxiv_repository.arxiv_graph_build.workflow_arxiv_graph_build import (  # type: ignore[assignment]
        ArxivGraphBuildWorkflow,
    )
except ImportError:
    ArxivGraphBuildWorkflow = None

try:
    from .workflow_arxiv_single_pdf import ArxivSinglePDFWorkflow  # type: ignore[assignment]
except ImportError:
    ArxivSinglePDFWorkflow = None

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
if ArxivInitialIngestWorkflow is not None:
    __all__.append('ArxivInitialIngestWorkflow')
if ArxivGraphBuildWorkflow is not None:
    __all__.append('ArxivGraphBuildWorkflow')
if ArxivSinglePDFWorkflow is not None:
    __all__.append('ArxivSinglePDFWorkflow')
