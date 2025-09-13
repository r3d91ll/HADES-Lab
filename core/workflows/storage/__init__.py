"""
Storage Backends for Workflows

Provides different storage backends for workflow outputs including
local filesystem, S3, and RamFS for high-speed staging.
"""

try:
    from .storage_local import LocalStorage
except ImportError:
    pass

try:
    from .storage_s3 import S3Storage
except ImportError:
    pass

try:
    from .storage_ramfs import RamFSStorage
except ImportError:
    pass

__all__ = [
    'LocalStorage',
    'S3Storage',
    'RamFSStorage',
]