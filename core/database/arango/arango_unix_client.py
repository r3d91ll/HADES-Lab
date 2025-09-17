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
        """
        Create a Unix-socket-capable HTTP client by loading required dependencies and instantiating the base Arango HTTP client.
        
        This initializer dynamically imports `requests_unixsocket` and stores it on `self.requests_unixsocket` to enable mounting Unix socket adapters. It also imports and instantiates `arango.http.DefaultHTTPClient`, storing the instance on `self.base_client`.
        
        Raises:
            ImportError: If `requests-unixsocket` is not installed (suggests `pip install requests-unixsocket`) or if `python-arango` is not available.
        """
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
        Create and return a requests Session configured for HTTP-over-Unix-socket URLs.
        
        The session is created via the wrapped base HTTP client for the given host and has Unix socket adapters mounted for the `http+unix://` and `https+unix://` schemes so requests can be sent over a Unix domain socket.
        
        Parameters:
            host (str): Host URL passed to the base client's session factory (may be a `http+unix://` URL that encodes the socket path).
        
        Returns:
            requests.Session: A session with Unix socket adapters mounted.
        """
        # Use the base client's session creation
        session = self.base_client.create_session(host)

        # Mount Unix socket adapter for http+unix:// URLs
        session.mount("http+unix://", self.requests_unixsocket.UnixAdapter())
        session.mount("https+unix://", self.requests_unixsocket.UnixAdapter())

        return session

    def send_request(self, *args, **kwargs):
        """
        Forward the call to the underlying base client's `send_request` and return its result.
        
        This method proxies all positional and keyword arguments to `self.base_client.send_request`.
        """
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
                 Initialize an ArangoDB connection over a Unix domain socket.
                 
                 Creates a client that connects to SOCKET_PATH (default /tmp/arangodb.sock) using an HTTP-over-Unix-socket transport, opens the specified database, and verifies connectivity by calling the database's version() method.
                 
                 Parameters:
                     database (str): Name of the database to open.
                     username (str): Username for authentication.
                     password (Optional[str]): Password for authentication. If omitted, the value of the ARANGO_PASSWORD environment variable is used.
                 
                 Raises:
                     FileNotFoundError: If the Unix socket at SOCKET_PATH does not exist.
                     ImportError: If the python-arango package is not installed.
                     Exception: Re-raises any exception encountered while creating the client, opening the database, or testing the connection.
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
        Proxy attribute access to the underlying Arango database object.
        
        If an attribute is not found on this wrapper, the lookup is delegated to
        self.db so the ArangoUnixClient can act as a drop-in replacement for the
        regular database connection.
        
        Parameters:
            name (str): Attribute name being accessed.
        
        Returns:
            Any: The attribute value from the underlying database object.
        
        Raises:
            AttributeError: If the attribute does not exist on the underlying database.
        """
        return getattr(self.db, name)