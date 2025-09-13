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
                from .arango.arango_unix_client import ArangoUnixClient
                client = ArangoUnixClient(
                    database=database,
                    username=username,
                    password=password
                )
                if client.use_unix:
                    logger.info("✓ Using Unix socket for ArangoDB connection")
                    return client
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
        parts = []
        for key, value in kwargs.items():
            if value is not None:
                parts.append(f"{key}={value}")
        return " ".join(parts)