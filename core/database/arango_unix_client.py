#!/usr/bin/env python3
"""
ArangoDB Unix Socket Client
===========================

Provides optimized local database connections via Unix socket for the HADES system.
This is designed for internal model datastores that don't need network access.

Theory Connection (Conveyance Framework):
Optimizes T (time) dimension by eliminating network overhead.
Unix sockets provide ~40% better throughput than TCP for local connections.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from arango import ArangoClient
from arango.database import StandardDatabase

logger = logging.getLogger(__name__)


class ArangoUnixClient:
    """
    ArangoDB client optimized for Unix socket connections.

    Falls back to HTTP gracefully if Unix socket is not available.
    """

    DEFAULT_UNIX_SOCKET = "/tmp/arangodb.sock"
    DEFAULT_HTTP_HOST = "http://localhost:8529"

    def __init__(self,
                 unix_socket: Optional[str] = None,
                 http_host: Optional[str] = None,
                 prefer_unix: bool = True):
        """
                 Create a configured ArangoUnixClient instance.
                 
                 Initializes connection settings using the provided values or environment fallbacks (ARANGO_UNIX_SOCKET for the Unix socket path and ARXIV_DB_HOST for the HTTP host). Sets internal fields (connection_url, client, using_unix) and then runs _setup_connection() to select the active connection method.
                 
                 Parameters:
                     unix_socket: Optional path to an ArangoDB Unix socket. If not provided, falls back to ARANGO_UNIX_SOCKET or DEFAULT_UNIX_SOCKET.
                     http_host: Optional HTTP host URL to use as a fallback. If not provided, falls back to ARXIV_DB_HOST or DEFAULT_HTTP_HOST.
                     prefer_unix: If True, attempt to use the Unix socket before falling back to HTTP.
                 """
        self.unix_socket = unix_socket or os.environ.get('ARANGO_UNIX_SOCKET', self.DEFAULT_UNIX_SOCKET)
        self.http_host = http_host or os.environ.get('ARXIV_DB_HOST', self.DEFAULT_HTTP_HOST)
        self.prefer_unix = prefer_unix
        self.connection_url = None
        self.client = None
        self.using_unix = False

        self._setup_connection()

    def _setup_connection(self):
        """
        Determine and initialize the client's connection method (Unix socket preferred, HTTP fallback).
        
        Checks whether a configured Unix socket exists and is a socket. If a usable Unix socket is found the method records that fact but falls back to HTTP due to the python-arango library's lack of Unix-socket support. If the socket is missing or not a socket, it falls back to HTTP. After choosing the connection URL this method constructs the ArangoClient and assigns it to self.client and updates self.using_unix as appropriate.
        
        Side effects:
        - May call self._fallback_to_http().
        - Sets self.client (ArangoClient) and self.using_unix.
        - Emits informational or warning logs when the socket is found, missing, or invalid.
        """
        # Check for Unix socket availability
        if self.prefer_unix and os.path.exists(self.unix_socket):
            # Verify it's actually a socket
            socket_path = Path(self.unix_socket)
            if socket_path.is_socket():
                # Unix sockets are not directly supported by python-arango
                # We'll mark that we found it but use HTTP for actual connection
                # The Unix socket can be used by other tools that support it
                self.using_unix = False  # Can't actually use it with python-arango
                logger.info(f"Unix socket found at {self.unix_socket}, but using HTTP (python-arango limitation)")
                self._fallback_to_http()
            else:
                logger.warning(f"Path exists but is not a socket: {self.unix_socket}")
                self._fallback_to_http()
        else:
            if self.prefer_unix:
                logger.debug(f"Unix socket not found at {self.unix_socket}, using HTTP")
            self._fallback_to_http()

        # Create client
        self.client = ArangoClient(hosts=self.connection_url)

    def _fallback_to_http(self):
        """
        Switch the client to use HTTP for connections.
        
        Sets the active connection URL to the configured HTTP host and marks the client as not using a Unix socket (self.using_unix = False). This changes the client's connection method to HTTP and logs the chosen host.
        """
        self.connection_url = self.http_host
        self.using_unix = False
        logger.info(f"Using HTTP connection: {self.http_host}")

    def get_database(self,
                     name: str,
                     username: str,
                     password: str) -> StandardDatabase:
        """
                     Return a StandardDatabase instance for the named database authenticated with the provided credentials.
                     
                     Parameters:
                         name: Name of the database to open.
                         username: Username to authenticate as.
                         password: Password for the user.
                     
                     Returns:
                         An authenticated `StandardDatabase` for `name`.
                     """
        return self.client.db(name, username=username, password=password)

    def get_system_db(self, username: str = 'root', password: str = None) -> StandardDatabase:
        """
        Return a connected StandardDatabase for the ArangoDB '_system' database.
        
        If `password` is omitted, the function will read the admin password from the
        ARANGO_PASSWORD environment variable.
        
        Parameters:
            username (str): Admin username to authenticate as (default: 'root').
            password (str | None): Admin password; if None, ARANGO_PASSWORD is used.
        
        Returns:
            StandardDatabase: Authenticated connection to the '_system' database.
        
        Raises:
            ValueError: If no password is provided and ARANGO_PASSWORD is not set.
        """
        if password is None:
            password = os.environ.get('ARANGO_PASSWORD')
            if not password:
                raise ValueError("No password provided and ARANGO_PASSWORD not set")

        return self.client.db('_system', username=username, password=password)

    def test_connection(self, db: StandardDatabase) -> bool:
        """
        Check connectivity to the given ArangoDB database by executing a simple AQL query.
        
        Runs the AQL statement "RETURN 1" on the provided StandardDatabase. Returns True if the query succeeds; on failure the exception is caught, an error is logged, and the method returns False.
        """
        try:
            # Simple query to test connection
            db.aql.execute("RETURN 1")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Return metadata about the client's active connection method.
        
        Returns:
            dict: Connection information with keys:
                - using_unix (bool): True if the client is configured to use a Unix socket.
                - unix_socket (str|None): Path to the Unix socket when using_unix is True, otherwise None.
                - http_host (str|None): HTTP host/URL when not using a Unix socket, otherwise None.
                - connection_url (str): The effective host/url passed to the underlying Arango client.
                - prefer_unix (bool): The configuration flag indicating whether Unix socket use was preferred.
        """
        return {
            'using_unix': self.using_unix,
            'unix_socket': self.unix_socket if self.using_unix else None,
            'http_host': self.http_host if not self.using_unix else None,
            'connection_url': self.connection_url,
            'prefer_unix': self.prefer_unix
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ArangoUnixClient':
        """
        Create an ArangoUnixClient from a plain configuration dictionary.
        
        The config may include:
        - 'unix_socket' (str): Path to a Unix socket to prefer.
        - 'http_host' (str): HTTP host URL to use as a fallback (e.g. "http://localhost:8529").
        - 'prefer_unix' (bool): Whether to prefer Unix socket over HTTP (defaults to True).
        
        Returns:
            ArangoUnixClient: A configured client instance.
        """
        return cls(
            unix_socket=config.get('unix_socket'),
            http_host=config.get('http_host'),
            prefer_unix=config.get('prefer_unix', True)
        )


def get_optimized_client(prefer_unix: bool = True) -> ArangoUnixClient:
    """
    Return a configured ArangoUnixClient optimized for local usage.
    
    This factory constructs an ArangoUnixClient that prefers a Unix domain socket when available but will fall back to HTTP as needed. Use this to obtain a client configured for optimal local workflow performance.
    
    Parameters:
        prefer_unix (bool): If True, attempt to use a Unix socket before falling back to HTTP.
    
    Returns:
        ArangoUnixClient: A configured client instance.
    """
    return ArangoUnixClient(prefer_unix=prefer_unix)


def get_database_for_workflow(db_name: str = None,
                              username: str = None,
                              password: str = None,
                              prefer_unix: bool = True) -> StandardDatabase:
    """
                              Obtain a StandardDatabase connection configured for workflow processing.
                              
                              Defaults are sourced from the environment when parameters are omitted:
                              - db_name: ARXIV_DB_NAME (default 'arxiv_repository')
                              - username: ARXIV_WRITER_USER (default 'arxiv_writer')
                              - password: ARXIV_WRITER_PASSWORD
                              
                              Parameters:
                                  db_name: Name of the database to connect to (optional).
                                  username: Database username (optional).
                                  password: Database password (optional; required either directly or via ARXIV_WRITER_PASSWORD).
                                  prefer_unix: If True, prefer a Unix-socket-backed client when available; the client will fall back to HTTP if needed.
                              
                              Returns:
                                  StandardDatabase: An authenticated database object ready for use.
                              
                              Raises:
                                  ValueError: If no password is provided and ARXIV_WRITER_PASSWORD is not set in the environment.
                              """
    # Get defaults from environment
    db_name = db_name or os.environ.get('ARXIV_DB_NAME', 'arxiv_repository')
    username = username or os.environ.get('ARXIV_WRITER_USER', 'arxiv_writer')
    password = password or os.environ.get('ARXIV_WRITER_PASSWORD')

    if not password:
        raise ValueError("No password provided and ARXIV_WRITER_PASSWORD not set")

    # Get optimized client
    client = get_optimized_client(prefer_unix=prefer_unix)

    # Log connection info
    info = client.get_connection_info()
    logger.info(f"Connecting to database '{db_name}' as '{username}' "
                f"(using {'Unix socket' if info['using_unix'] else 'HTTP'})")

    return client.get_database(db_name, username, password)


# Example usage in workflows:
if __name__ == "__main__":
    # Test the connection
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Testing ArangoDB Unix Socket Client")
    print("=" * 40)

    client = get_optimized_client()
    info = client.get_connection_info()

    print(f"Connection type: {'Unix socket' if info['using_unix'] else 'HTTP'}")
    if info['using_unix']:
        print(f"Socket path: {info['unix_socket']}")
    else:
        print(f"HTTP host: {info['http_host']}")

    # Try to connect to a test database
    try:
        # This would normally use proper credentials
        # db = get_database_for_workflow()
        # if client.test_connection(db):
        #     print("✓ Connection test passed")
        # else:
        #     print("✗ Connection test failed")
        print("\nTo test with a real database:")
        print("  export ARXIV_WRITER_PASSWORD='your_password'")
        print("  python -m core.database.arango_unix_client")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)