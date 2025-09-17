"""
Structured Logging
==================

Centralized logging configuration using structlog for structured JSON logging.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import structlog

# Global flag to track if logging has been setup
_logging_initialized = False


class LogManager:
    """
    Centralized logging configuration using structlog.

    Features:
    - Structured JSON logging
    - Automatic log rotation
    - Context preservation (run_id, processor)
    - Multiple output targets
    - No database logging
    """

    @staticmethod
    def setup(config_path: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize structured JSON logging (idempotent).
        
        Sets up Python logging and structlog configuration, creates a "logs" directory two levels above this file if missing, and attaches two rotating file handlers: processors.log (10 MB, 5 backups) and errors.log (10 MB, 3 backups, level=ERROR). Configures structlog to produce JSON-formatted logs with contextual fields and enables caching of logger objects. Marks logging as initialized so subsequent calls are no-ops and emits a "logging_initialized" info event.
        
        Parameters:
            config_path (Optional[str]): Currently unused; reserved for future config file support.
            log_level (str): Logging level name (e.g., "INFO", "DEBUG") used for the main handlers and root logger.
        
        Side effects:
            - Creates filesystem entries under the repository logs directory.
            - Modifies global logging configuration and the module-level _logging_initialized flag.
        """
        global _logging_initialized

        if _logging_initialized:
            return

        # Setup log directory
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        # Configure Python logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(message)s'
        )

        # Setup handlers
        handlers = []

        # Main log file with rotation (10MB, keep 5 backups)
        main_handler = RotatingFileHandler(
            log_dir / "processors.log",
            maxBytes=10_485_760,
            backupCount=5
        )
        main_handler.setLevel(getattr(logging, log_level))
        handlers.append(main_handler)

        # Error log file (10MB, keep 3 backups)
        error_handler = RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=10_485_760,
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        handlers.append(error_handler)

        # Add handlers to root logger
        root_logger = logging.getLogger()
        for handler in handlers:
            root_logger.addHandler(handler)

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                    ]
                ),
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        _logging_initialized = True

        # Log initialization
        logger = structlog.get_logger()
        logger.info("logging_initialized", log_dir=str(log_dir), level=log_level)

    @staticmethod
    def get_logger(processor_name: str, run_id: str):
        """
        Return a structlog logger bound with processor and run_id context.
        
        If logging has not yet been initialized, this will call LogManager.setup() before returning the logger.
        
        Parameters:
            processor_name: Identifier for the processor to attach to every log record.
            run_id: Unique run identifier to attach to every log record.
        
        Returns:
            A structlog BoundLogger with `processor` and `run_id` already bound.
        """
        # Ensure logging is setup
        if not _logging_initialized:
            LogManager.setup()

        return structlog.get_logger().bind(
            processor=processor_name,
            run_id=run_id
        )
