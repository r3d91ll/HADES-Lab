#!/usr/bin/env python3
"""
Unix socket HTTP client for ArangoDB using httpx.
Allows python-arango to use Unix sockets for better performance.
"""

import httpx
from arango import ArangoClient
from arango.http import HTTPClient
from arango.response import Response
from urllib.parse import urlparse, urlunparse
import logging

logger = logging.getLogger(__name__)


class UnixSocketHTTPClient(HTTPClient):
    """HTTP client that uses Unix domain sockets instead of TCP."""
    
    def __init__(self, unix_socket_path='/tmp/arangodb.sock'):
        """Initialize Unix socket HTTP client.
        
        Args:
            unix_socket_path: Path to the Unix socket file
        """
        super().__init__()
        self.unix_socket_path = unix_socket_path
        # Create httpx transport with Unix socket support
        self.transport = httpx.HTTPTransport(uds=unix_socket_path)
        self.client = httpx.Client(transport=self.transport, timeout=300.0)
        logger.info(f"Using Unix socket at {unix_socket_path}")
    
    def create_session(self, host):
        """Create a session (we use a single client)."""
        return self.client
    
    def send_request(self,
                    session,
                    method,
                    url, 
                    params=None,
                    data=None,
                    headers=None,
                    auth=None):
        """Send an HTTP request over Unix socket.
        
        Args:
            method: HTTP method
            url: URL (will be adapted for Unix socket)
            params: Query parameters
            data: Request body
            headers: HTTP headers
            auth: Authentication
            
        Returns:
            Response dictionary
        """
        # Parse the URL and replace with localhost for Unix socket
        parsed = urlparse(url)
        
        # Use localhost as dummy host for Unix socket connection
        unix_url = urlunparse(('http', 'localhost', parsed.path,
                              parsed.params, parsed.query, parsed.fragment))
        
        try:
            response = self.client.request(
                method=method,
                url=unix_url,
                params=params,
                content=data,
                headers=headers,
                auth=auth
            )
            
            # Create a proper Response object for ArangoDB
            resp = Response(
                url=url,
                method=method,
                headers=dict(response.headers),
                status_code=response.status_code,
                status_text=response.reason_phrase or '',
                raw_body=response.content
            )
            return resp
        except Exception as e:
            logger.error(f"Unix socket request failed: {e}")
            raise


def get_unix_socket_client(socket_path='/tmp/arangodb.sock', fallback_to_tcp=True):
    """Get ArangoDB client using Unix socket if available, otherwise TCP.
    
    Args:
        socket_path: Path to Unix socket
        fallback_to_tcp: Whether to fall back to TCP if socket unavailable
        
    Returns:
        ArangoClient instance
    """
    import os
    
    # Check if Unix socket exists
    if os.path.exists(socket_path):
        try:
            http_client = UnixSocketHTTPClient(socket_path)
            # Use a dummy host since we're using Unix socket
            client = ArangoClient(
                hosts='http://localhost:8529',
                http_client=http_client
            )
            logger.info("Successfully created Unix socket client")
            return client
        except Exception as e:
            logger.warning(f"Failed to create Unix socket client: {e}")
            if not fallback_to_tcp:
                raise
    
    if fallback_to_tcp:
        logger.info("Falling back to TCP connection")
        return ArangoClient(hosts='http://localhost:8529')
    
    raise RuntimeError(f"Unix socket not found at {socket_path} and fallback disabled")