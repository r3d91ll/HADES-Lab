"""
Storage Backends for Workflows

Provides different storage backends for workflow outputs including
local filesystem, S3, and RamFS for high-speed staging.
"""

try:
    from .storage_local import LocalStorage
except ImportError:
    LocalStorage = None

try:
    from .storage_s3 import S3Storage
except ImportError:
    S3Storage = None

try:
    from .storage_ramfs import RamFSStorage
except ImportError:
    RamFSStorage = None

# Import base class
from .storage_base import StorageBase

__all__ = [
    'StorageBase',
    'LocalStorage',
    'S3Storage',
    'RamFSStorage',
]