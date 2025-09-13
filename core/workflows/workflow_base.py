#!/usr/bin/env python3
"""
Base Workflow Class

Defines the contract for all workflow implementations in HADES.
Workflows orchestrate the processing of documents through extraction,
embedding, and storage phases.

Theory Connection:
Workflows represent the TIME dimension in our Conveyance Framework,
orchestrating the sequence of transformations that convert raw input
into actionable information. They manage the flow of data through
the system while preserving the multiplicative dependencies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for workflows."""
    name: str
    batch_size: int = 32
    num_workers: int = 4
    use_gpu: bool = True
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 100
    staging_path: Path = Path("/dev/shm/workflow_staging")
    timeout_seconds: int = 300


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_name: str
    success: bool
    items_processed: int
    items_failed: int
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Calculate workflow duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.items_processed + self.items_failed
        if total == 0:
            return 0.0
        return (self.items_processed / total) * 100


class WorkflowBase(ABC):
    """
    Abstract base class for all workflows.

    Provides common infrastructure for checkpointing, error handling,
    and progress tracking while enforcing a consistent interface.
    """

    def __init__(self, config: Optional[WorkflowConfig] = None):
        """
        Initialize workflow with configuration.

        Args:
            config: Workflow configuration
        """
        self.config = config or WorkflowConfig(name="unnamed_workflow")
        self.checkpoint_data = {}
        self._ensure_staging_directory()

    def _ensure_staging_directory(self):
        """Ensure staging directory exists."""
        if self.config.staging_path:
            self.config.staging_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate workflow inputs.

        Returns:
            True if inputs are valid, False otherwise
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the workflow.

        Returns:
            WorkflowResult with execution details
        """
        pass

    def save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """
        Save workflow checkpoint.

        Args:
            checkpoint_data: Data to checkpoint
        """
        if not self.config.checkpoint_enabled:
            return

        checkpoint_path = self.config.staging_path / f"{self.config.name}_checkpoint.json"

        try:
            import json
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, default=str)
            logger.debug(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load workflow checkpoint if exists.

        Returns:
            Checkpoint data or None
        """
        if not self.config.checkpoint_enabled:
            return None

        checkpoint_path = self.config.staging_path / f"{self.config.name}_checkpoint.json"

        if not checkpoint_path.exists():
            return None

        try:
            import json
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self):
        """Clear workflow checkpoint."""
        checkpoint_path = self.config.staging_path / f"{self.config.name}_checkpoint.json"

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.debug(f"Checkpoint cleared: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to clear checkpoint: {e}")

    @property
    def name(self) -> str:
        """Get workflow name."""
        return self.config.name

    @property
    def supports_batch(self) -> bool:
        """Whether this workflow supports batch processing."""
        return True

    @property
    def supports_streaming(self) -> bool:
        """Whether this workflow supports streaming processing."""
        return False

    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the workflow.

        Returns:
            Dictionary with workflow metadata
        """
        return {
            "name": self.config.name,
            "class": self.__class__.__name__,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "use_gpu": self.config.use_gpu,
            "checkpoint_enabled": self.config.checkpoint_enabled,
            "supports_batch": self.supports_batch,
            "supports_streaming": self.supports_streaming
        }