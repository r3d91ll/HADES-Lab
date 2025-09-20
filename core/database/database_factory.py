#!/usr/bin/env python3
"""
Database Factory

Factory pattern for creating database connections.
Supports automatic connection type selection and configuration.

Theory Connection:
The factory enables flexible WHERE dimension management by abstracting
the specific database implementation from the workflow logic.
"""

import logging
import os
from typing import Any, Dict, Optional

from core.database.arango import ArangoMemoryClient, resolve_memory_config

logger = logging.getLogger(__name__)


class DatabaseFactory:
    """
    Factory for creating database connections.

    Manages the instantiation of different database types based on
    configuration, with support for connection pooling and optimization.
    """

    @classmethod
    def get_arango(cls,
                   database: str = "academy_store",
                   username: str = "root",
                   password: Optional[str] = None,
                   host: str = "localhost",
                   port: int = 8529,
                   use_unix: bool = True,
                   **kwargs) -> Any:
        """
        Get ArangoDB connection.

        Args:
            database: Database name
            username: Username
            password: Password (or from env)
            host: Database host
            port: Database port
            use_unix: Use Unix socket if available
            **kwargs: Additional connection options

        Returns:
            ArangoDB connection object
        """
        # Get password from environment if not provided
        if password is None:
            password = os.environ.get('ARANGO_PASSWORD')
            if not password:
                raise ValueError("ArangoDB password required (set ARANGO_PASSWORD env var)")

        # Try Unix socket first if requested
        if use_unix:
            try:
                from .arango_unix_client import get_database_for_workflow
                db = get_database_for_workflow(
                    db_name=database,
                    username=username,
                    password=password,
                    prefer_unix=True
                )
                logger.info("✓ Using Unix socket for ArangoDB connection")
                return db
            except ImportError as e:
                logger.error("Unix socket client not available - this is required for HADES")
                raise RuntimeError("Unix socket connection is mandatory for HADES. Network access not allowed.") from e
            except Exception as e:
                logger.error(f"Unix socket connection failed: {e}")
                raise RuntimeError(f"Unix socket connection failed. Network fallback not allowed for HADES: {e}") from e

        # Network connection - ONLY for human debugging, never for HADES
        if host not in {"localhost", "127.0.0.1", "::1"}:
            if os.environ.get("HADES_ALLOW_NETWORK_DEBUG") != "true":
                raise RuntimeError("Network connections are disabled (set HADES_ALLOW_NETWORK_DEBUG=true to override).")
            logger.warning("Network connection requested - debugging only")
        try:
            from arango import ArangoClient
            client = ArangoClient(hosts=f'http://{host}:{port}')
            db = client.db(database, username=username, password=password)
            logger.info(f"✓ Connected to ArangoDB via HTTP at {host}:{port} (debugging only)")
            return db
        except ImportError:
            raise ImportError("python-arango not installed. Run: pip install python-arango")
        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            raise

    @classmethod
    def get_postgres(cls,
                    database: str = "arxiv",
                    username: str = "postgres",
                    password: Optional[str] = None,
                    host: str = "localhost",
                    port: int = 5432,
                    **kwargs) -> Any:
        """
        Get PostgreSQL connection.

        Args:
            database: Database name
            username: Username
            password: Password (or from env)
            host: Database host
            port: Database port
            **kwargs: Additional connection options

        Returns:
            PostgreSQL connection object
        """
        # Get password from environment if not provided
        if password is None:
            password = os.environ.get('PGPASSWORD')
            if not password:
                raise ValueError("PostgreSQL password required (set PGPASSWORD env var)")

        try:
            import psycopg
            conn_string = f"host={host} port={port} dbname={database} user={username} password={password}"
            conn = psycopg.connect(conn_string, **kwargs)
            logger.info(f"✓ Connected to PostgreSQL at {host}:{port}/{database}")
            return conn
        except ImportError:
            # Try psycopg2 as fallback
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=username,
                    password=password,
                    **kwargs
                )
                logger.info(f"✓ Connected to PostgreSQL (psycopg2) at {host}:{port}/{database}")
                return conn
            except ImportError:
                raise ImportError("Neither psycopg nor psycopg2 installed. Run: pip install psycopg")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    @classmethod
    def get_arango_memory_service(
        cls,
        *,
        database: str = "arxiv_repository",
        username: str = "root",
        password: Optional[str] = None,
        socket_path: Optional[str] = None,
        read_socket: Optional[str] = None,
        write_socket: Optional[str] = None,
        use_proxies: Optional[bool] = None,
        base_url: Optional[str] = None,
        connect_timeout: Optional[float] = None,
        read_timeout: Optional[float] = None,
        write_timeout: Optional[float] = None,
    ) -> ArangoMemoryClient:
        """Return the optimized ArangoDB memory client.

        Args:
            database: Target database name.
            username: Authentication user (default "root").
            password: Password (falls back to ``ARANGO_PASSWORD``).
            socket_path: Explicit Unix socket used for both reads and writes.
            read_socket: Optional read-only proxy socket.
            write_socket: Optional read-write proxy socket.
            use_proxies: Force proxy usage (defaults to environment autodetect).
            base_url: Base URL for HTTP/2 client (defaults to http://localhost).
            connect_timeout: Override connection timeout.
            read_timeout: Override read timeout.
            write_timeout: Override write timeout.

        Returns:
            Configured :class:`ArangoMemoryClient` instance.
        """

        config = resolve_memory_config(
            database=database,
            username=username,
            password=password,
            socket_path=socket_path,
            read_socket=read_socket,
            write_socket=write_socket,
            use_proxies=use_proxies,
            base_url=base_url,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )

        logger.info(
            "✓ Using Arango memory client (read_socket=%s, write_socket=%s)",
            config.read_socket,
            config.write_socket,
        )
        return ArangoMemoryClient(config)

    @classmethod
    def get_redis(cls,
                  host: str = "localhost",
                  port: int = 6379,
                  db: int = 0,
                  password: Optional[str] = None,
                  **kwargs) -> Any:
        """
        Get Redis connection.

        Args:
            host: Redis host
            port: Redis port
            db: Database number
            password: Password if required
            **kwargs: Additional connection options

        Returns:
            Redis connection object
        """
        try:
            import redis
            conn = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                **kwargs
            )
            # Test connection
            conn.ping()
            logger.info(f"✓ Connected to Redis at {host}:{port}/{db}")
            return conn
        except ImportError:
            raise ImportError("redis not installed. Run: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    @classmethod
    def create_pool(cls, db_type: str, pool_size: int = 10, **kwargs) -> Any:
        """
        Create a connection pool for the specified database type.

        Args:
            db_type: Type of database (arango, postgres, redis)
            pool_size: Size of connection pool
            **kwargs: Database-specific connection options

        Returns:
            Connection pool object
        """
        if db_type == "postgres":
            try:
                import psycopg_pool
                conn_string = cls._build_postgres_conn_string(**kwargs)
                pool = psycopg_pool.ConnectionPool(
                    conn_string,
                    min_size=1,
                    max_size=pool_size
                )
                logger.info(f"✓ Created PostgreSQL connection pool (size={pool_size})")
                return pool
            except ImportError:
                logger.warning("psycopg_pool not available, returning single connection")
                return cls.get_postgres(**kwargs)

        elif db_type == "redis":
            try:
                import redis
                pool = redis.ConnectionPool(
                    max_connections=pool_size,
                    **kwargs
                )
                conn = redis.Redis(connection_pool=pool)
                logger.info(f"✓ Created Redis connection pool (size={pool_size})")
                return conn
            except ImportError:
                raise ImportError("redis not installed")

        else:
            # ArangoDB handles pooling internally
            return cls.get_arango(**kwargs)

    @staticmethod
    def _build_postgres_conn_string(**kwargs) -> str:
        """Build PostgreSQL connection string from kwargs."""
        from typing import Dict, Any

        # Normalize keys to libpq names
        key_map = {"database": "dbname", "username": "user"}
        mapped: Dict[str, Any] = {}

        for key, value in kwargs.items():
            if value is None:
                continue
            # Map to correct libpq key names
            mapped_key = key_map.get(key, key)
            mapped[mapped_key] = value

        # Password fallback from environment if not provided
        if "password" not in mapped:
            env_pw = os.environ.get("PGPASSWORD")
            if env_pw:
                mapped["password"] = env_pw

        return " ".join(f"{k}={v}" for k, v in mapped.items())
