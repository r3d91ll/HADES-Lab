"""
Workflows Module

Provides orchestration and pipeline management for document processing.
Workflows coordinate the flow of data through extraction, embedding,
and storage phases while managing state and error recovery.

This module replaces scattered orchestration code with a unified structure.
"""

from .workflow_base import WorkflowBase, WorkflowConfig, WorkflowResult

# Import specific workflows if they exist
try:
    from .workflow_pdf import PDFWorkflow
except ImportError:
    pass

try:
    from .workflow_pdf_batch import BatchPDFWorkflow
except ImportError:
    pass

# State management
from .state import StateManager, CheckpointManager

# Backward compatibility exports
__all__ = [
    'WorkflowBase',
    'WorkflowConfig',
    'WorkflowResult',
    'StateManager',
    'CheckpointManager',
    'PDFWorkflow',
    'BatchPDFWorkflow',
]