"""
ArXiv Pipeline Orchestration Module

This module implements the search → preview → refine → process pattern
for the ArXiv metadata service, providing interface-agnostic backend
components that can be used by CLI, MCP servers, or GUI frontends.
"""

from .orchestrator import (
    ArxivPipelineOrchestrator,
    OrchestrationState,
    OrchestrationContext,
    SearchConfiguration,
    interactive_cli_session
)

__all__ = [
    'ArxivPipelineOrchestrator',
    'OrchestrationState', 
    'OrchestrationContext',
    'SearchConfiguration',
    'interactive_cli_session'
]