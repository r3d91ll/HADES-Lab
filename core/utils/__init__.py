"""
Core Utilities for HADES
=========================

Generic utilities extracted from ArXiv tools for broader use across HADES.
These implement common patterns for state management, pre-flight checks,
and batch processing with proper error isolation.

Following Actor-Network Theory: These utilities act as "immutable mobiles" -
standardized tools that can be transported across different contexts while
maintaining their essential properties.
"""

# StateManager moved to core/workflows/workflow_state_manager.py
# Import from there if needed: from core.workflows.workflow_state_manager import StateManager, CheckpointManager
from .preflight import PreflightChecker, standard_pipeline_checks
from .batch_processor import (
    BatchProcessor,
    ParallelBatchProcessor,
    batch_insert_with_savepoints
)

__all__ = [
    'PreflightChecker',
    'standard_pipeline_checks',
    'BatchProcessor',
    'ParallelBatchProcessor',
    'batch_insert_with_savepoints'
]