#!/usr/bin/env python3
"""
Database Factory

Factory pattern for creating database connections.
Supports automatic connection type selection and configuration.

Theory Connection:
The factory enables flexible WHERE dimension management by abstracting
the specific database implementation from the workflow logic.
"""

from typing import Optional, Dict, Any
import logging
import os

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
                   Return an ArangoDB database connection, preferring a Unix-socket client when requested and available.
                   
                   If `password` is None, the function reads ARANGO_PASSWORD from the environment and raises ValueError if not set. When `use_unix` is True the function first attempts to obtain a connection via the local Unix-socket helper (`get_database_for_workflow`) and falls back to an HTTP connection using python-arango on ImportError or other failure.
                   
                   Parameters:
                       database (str): ArangoDB database name.
                       username (str): Username for authentication.
                       password (Optional[str]): Password for authentication; if None, ARANGO_PASSWORD env var is used.
                       host (str): Host for HTTP fallback.
                       port (int): Port for HTTP fallback.
                       use_unix (bool): If True, try a Unix-socket connection before HTTP.
                       **kwargs: Additional connection options (passed through where applicable).
                   
                   Returns:
                       Any: A python-arango database connection object (or whatever the Unix-socket helper returns).
                   
                   Raises:
                       ValueError: If no password is provided and ARANGO_PASSWORD is not set.
                       ImportError: If the HTTP path is chosen but python-arango is not installed.
                       Exception: Propagates other exceptions raised while attempting connections.
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
            except ImportError:
                logger.warning("Unix socket client not available, falling back to HTTP")
            except Exception as e:
                logger.warning(f"Unix socket connection failed: {e}, falling back to HTTP")

        # Fall back to standard HTTP connection
        try:
            from arango import ArangoClient
            client = ArangoClient(hosts=f'http://{host}:{port}')
            db = client.db(database, username=username, password=password)
            logger.info(f"✓ Connected to ArangoDB via HTTP at {host}:{port}")
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