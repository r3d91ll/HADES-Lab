"""
Database Module for ArXiv Data

This module manages the SQLite local cache and ArangoDB integration:
- ArXiv metadata caching in SQLite
- PDF location tracking  
- File indexing and management
- Coverage analysis and reporting
- Data preparation for ArangoDB processing
"""

# Version
__version__ = "2.0.0"

# Database name
DATABASE_NAME = "arxiv_cache"