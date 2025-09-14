#!/usr/bin/env python3
"""
Base Storage Interface

Defines the contract for all storage backends in workflows.
Storage backends handle the persistence of processed data.

Theory Connection:
Storage represents the WHERE dimension after processing - the final
location where information resides. Different backends optimize for
different access patterns and persistence requirements.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StorageBase(ABC):
    """
    Abstract base class for all storage backends.

    Provides a consistent interface for storing and retrieving
    workflow outputs across different storage systems.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize storage backend with configuration.

        Args:
            config: Storage configuration
        """
        self.config = config or {}

    @abstractmethod
    def store(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store data with the given key.

        Args:
            key: Storage key/identifier
            data: Data to store
            metadata: Optional metadata

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve data for the given key.

        Args:
            key: Storage key/identifier

        Returns:
            Stored data or None if not found
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in storage.

        Args:
            key: Storage key/identifier

        Returns:
            True if exists, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete data for the given key.

        Args:
            key: Storage key/identifier

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """
        List all keys in storage.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            List of storage keys
        """
        pass

    def batch_store(self, items: Dict[str, Any]) -> Dict[str, bool]:
        """
        Store multiple items at once.

        Args:
            items: Dictionary of key-value pairs

        Returns:
            Dictionary of key-success pairs
        """
        results = {}
        for key, data in items.items():
            results[key] = self.store(key, data)
        return results

    def batch_retrieve(self, keys: List[str]) -> Dict[str, Any]:
        """
        Retrieve multiple items at once.

        Args:
            keys: List of keys to retrieve

        Returns:
            Dictionary of key-value pairs
        """
        results = {}
        for key in keys:
            data = self.retrieve(key)
            if data is not None:
                results[key] = data
        return results

    def clear(self, prefix: Optional[str] = None) -> int:
        """
        Clear all items or items with prefix.

        Args:
            prefix: Optional prefix to filter items

        Returns:
            Number of items deleted
        """
        keys = self.list_keys(prefix)
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count

    @property
    @abstractmethod
    def storage_type(self) -> str:
        """Get the type of storage backend."""
        pass

    @property
    def supports_metadata(self) -> bool:
        """Whether this backend supports metadata."""
        return False

    @property
    def supports_streaming(self) -> bool:
        """Whether this backend supports streaming."""
        return False

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the storage backend.

        Returns:
            Dictionary with storage metadata
        """
        return {
            "type": self.storage_type,
            "supports_metadata": self.supports_metadata,
            "supports_streaming": self.supports_streaming,
            "config": self.config
        }