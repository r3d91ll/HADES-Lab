"""
Base Processor Class
====================

Abstract base class for all HADES processors.
Provides common functionality for configuration, logging, metrics, and storage.
"""

import logging
import signal
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Optional

from .config import ConfigManager
from .logging import LogManager
from .metrics import MetricsCollector

# StorageManager has been moved to core.workflows.storage
# from .storage import StorageManager

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    Abstract base class for all HADES processors.

    Provides:
    - Configuration loading with hierarchy
    - Structured logging per processor
    - Database connection management
    - Metrics collection
    - Standard lifecycle hooks
    """

    def __init__(self, processor_name: str, config_override: Optional[dict] = None):
        """
        Initialize base processor.

        Args:
            processor_name: Name of the processor (e.g., 'arxiv', 'github')
            config_override: Optional config overrides
        """
        # Generate run ID for this processing session
        self.run_id = f"{datetime.now(UTC).isoformat()}_{uuid.uuid4().hex[:8]}"
        self.processor_name = processor_name

        # Load configuration hierarchy
        self.config = ConfigManager.load(
            processor_name=processor_name,
            override=config_override
        )

        # Setup structured logging
        self.logger = LogManager.get_logger(
            processor_name=processor_name,
            run_id=self.run_id
        )

        # Initialize database connection with best-effort compatibility
        self.db = None
        try:
            # Try new storage location first
            from core.workflows.storage import StorageManager
            if hasattr(self.config, 'database'):
                self.db = StorageManager.get_connection(self.config.database)
                self.logger.info("Database connection established via new storage module")
        except ImportError:
            # Fall back to storage_compat if available
            try:
                from . import storage_compat
                if hasattr(storage_compat, 'StorageManager') and hasattr(self.config, 'database'):
                    self.db = storage_compat.StorageManager.get_connection(self.config.database)
                    self.logger.warning("Using deprecated storage_compat for database connection")
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"Could not establish database connection: {e}. Implement storage in derived classes.")

        # Initialize metrics collector
        self.metrics = MetricsCollector(processor_name)

        # Setup signal handlers for graceful shutdown
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Log initialization
        self.logger.info("processor_initialized",
            processor=processor_name,
            run_id=self.run_id,
            config=self.config.dict()
        )

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    @abstractmethod
    def process(self, **kwargs) -> dict[str, Any]:
        """
        Main processing logic - must be implemented by subclasses.

        Returns:
            Processing result dictionary
        """
        pass

    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input before processing.

        Returns:
            True if input is valid, False otherwise
        """
        pass

    def cleanup(self):
        """Cleanup resources after processing."""
        self.metrics.flush()
        self.logger.info("processor_cleanup", processor=self.processor_name)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.cleanup()
