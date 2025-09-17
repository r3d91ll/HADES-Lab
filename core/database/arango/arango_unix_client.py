#!/usr/bin/env python3
"""
ArangoDB Unix Socket Client
===========================

Provides connection to ArangoDB via Unix socket at /tmp/arangodb.sock
for improved performance and security.

Uses a custom HTTP client with requests-unixsocket to enable
HTTP over Unix domain sockets for minimal latency.
"""

import os
import logging
from typing import Optional, Any
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger(__name__)


class UnixSocketHTTPClient:
    """
    Custom HTTP client for python-arango that supports Unix sockets.

    This enables HTTP communication over Unix domain sockets for
    minimal latency between the application and ArangoDB.
    """

    def __init__(self):
        """Initialize the Unix socket HTTP client."""
        try:
            import requests_unixsocket
            self.requests_unixsocket = requests_unixsocket
        except ImportError:
            raise ImportError(
                "requests-unixsocket not installed. "
                "Run: pip install requests-unixsocket"
            )

        # Import the base client from python-arango
        try:
            from arango.http import DefaultHTTPClient
            self.base_client = DefaultHTTPClient()
        except ImportError:
            raise ImportError("python-arango not installed")

    def create_session(self, host: str):
        """
        Create a session with Unix socket support.

        Args:
            host: The host URL (may include http+unix:// scheme)

        Returns:
            Configured requests session
        """
        # Use the base client's session creation
        session = self.base_client.create_session(host)

        # Mount Unix socket adapter for http+unix:// URLs
        session.mount("http+unix://", self.requests_unixsocket.UnixAdapter())
        session.mount("https+unix://", self.requests_unixsocket.UnixAdapter())

        return session

    def send_request(self, *args, **kwargs):
        """Forward request to base client."""
        return self.base_client.send_request(*args, **kwargs)


class ArangoUnixClient:
    """
    ArangoDB client using Unix socket connection.

    Uses the Unix socket at /tmp/arangodb.sock for local connections,
    providing better performance than HTTP connections.

    Theory Connection (Conveyance Framework):
    Optimizes the TIME (T) dimension by minimizing network latency,
    directly improving Conveyance C = (W·R·H/T)·Ctx^α
    """

    SOCKET_PATH = "/tmp/arangodb.sock"

    def __init__(self,
                 database: str = "academy_store",
                 username: str = "root",
                 password: Optional[str] = None):
        """
        Initialize Unix socket connection to ArangoDB.

        Args:
            database: Database name
            username: Username for authentication
            password: Password (from env if not provided)
        """
        self.database = database
        self.username = username
        self.password = password or os.environ.get('ARANGO_PASSWORD')
        self.use_unix = False

        # Check if socket exists
        if not Path(self.SOCKET_PATH).exists():
            logger.warning(f"Unix socket not found at {self.SOCKET_PATH}")
            raise FileNotFoundError(f"Unix socket not found at {self.SOCKET_PATH}")

        # Import ArangoDB client
        try:
            from arango import ArangoClient
        except ImportError:
            raise ImportError("python-arango not installed")

        # Percent-encode the socket path for the URL
        # /tmp/arangodb.sock -> %2Ftmp%2Farangodb.sock
        encoded_path = quote(self.SOCKET_PATH, safe="")
        socket_url = f"http+unix://{encoded_path}"

        try:
            # Create custom HTTP client with Unix socket support
            http_client = UnixSocketHTTPClient()

            # Create ArangoDB client with custom HTTP client
            self.client = ArangoClient(
                hosts=socket_url,
                http_client=http_client
            )

            # Connect to database
            self.db = self.client.db(
                self.database,
                username=self.username,
                password=self.password,
                verify=True  # Verify the connection
            )

            # Test connection
            version = self.db.version()
            self.use_unix = True

            logger.info(
                f"✓ Connected to ArangoDB via Unix socket at {self.SOCKET_PATH} "
                f"(ArangoDB {version})"
            )

        except Exception as e:
            logger.error(f"Failed to connect via Unix socket: {e}")
            raise

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the database connection.

        This allows the Unix client to be used as a drop-in replacement
        for the standard database connection.
        """
        return getattr(self.db, name)