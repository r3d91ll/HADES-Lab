"""
Database Ingestion Module for ArXiv Data

This module manages the PostgreSQL data lake and ingestion pipeline:
- ArXiv metadata import and management
- Tar file extraction and tracking
- File location updates (PDF and LaTeX)
- Coverage analysis and reporting
- Data preparation for ArangoDB processing
"""

# Version
__version__ = "1.0.0"

# Database name
DATABASE_NAME = "arxiv_datalake"