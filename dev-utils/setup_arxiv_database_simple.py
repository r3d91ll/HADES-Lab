#!/usr/bin/env python3
"""
Simplified database setup script that uses environment variables.
Called by setup_database.sh with all configuration passed via environment.
"""

import os
import sys
import secrets
import string
import json
from pathlib import Path
from arango import ArangoClient


def generate_password(length: int = 24) -> str:
    """Generate a secure random password."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def main():
    """
    Provision an ArangoDB database, users, collections, and a credentials file using environment variables and optional CLI flags.
    
    Reads configuration from environment variables (e.g. DB_NAME, ROOT_PASSWORD, DB_HOST/UNIX_SOCKET, user names/passwords, COLLECTIONS_JSON) and optional flags (--drop-existing, --quiet, --auto). Connects to the ArangoDB _system database as root to create or (optionally) recreate the target database, create or update three role users (admin, writer, reader) and their database permissions, create any specified collections, and write a restricted-permissions environment file with generated or provided credentials into CONFIG_DIR.
    
    Returns:
        int: 0 on successful completion; 1 if any step (connection, database creation, user management, collection creation, or writing the credentials file) fails.
    """

    # Get configuration from environment
    db_name = os.environ.get('DB_NAME', 'arxiv_repository')
    root_password = os.environ.get('ROOT_PASSWORD')

    # Check for Unix socket connection (for config file only)
    unix_socket = os.environ.get('UNIX_SOCKET', '/tmp/arangodb.sock')
    use_unix = os.environ.get('USE_UNIX_SOCKET', 'true').lower() == 'true'
    unix_socket_exists = use_unix and os.path.exists(unix_socket)

    # Always use HTTP for database setup operations
    # (python-arango doesn't support Unix sockets directly)
    db_host = os.environ.get('DB_HOST', 'http://localhost:8529')

    if unix_socket_exists:
        print(f"Unix socket found at {unix_socket} (will be saved in config)")
        print(f"Using HTTP for setup: {db_host}")
    else:
        if use_unix:
            print(f"Unix socket not found at {unix_socket}")
        print(f"Using HTTP connection: {db_host}")

    # User configuration
    users = {
        os.environ.get('DB_ADMIN_USER', 'arxiv_admin'): {
            'password': os.environ.get('DB_ADMIN_PASSWORD') or generate_password(),
            'permissions': os.environ.get('ADMIN_PERMISSIONS', 'rw'),
            'description': 'Admin user for database'
        },
        os.environ.get('DB_WRITER_USER', 'arxiv_writer'): {
            'password': os.environ.get('DB_WRITER_PASSWORD') or generate_password(),
            'permissions': os.environ.get('WRITER_PERMISSIONS', 'rw'),
            'description': 'Writer user for workflows'
        },
        os.environ.get('DB_READER_USER', 'arxiv_reader'): {
            'password': os.environ.get('DB_READER_PASSWORD') or generate_password(),
            'permissions': os.environ.get('READER_PERMISSIONS', 'ro'),
            'description': 'Read-only user for monitoring'
        }
    }

    # Collections from environment (passed as COLLECTIONS_JSON)
    collections_json = os.environ.get('COLLECTIONS_JSON', '[]')
    collections = json.loads(collections_json)

    # Parse command line arguments
    drop_existing = '--drop-existing' in sys.argv
    quiet_mode = '--quiet' in sys.argv
    auto_mode = '--auto' in sys.argv

    if not quiet_mode:
        print(f"Setting up database: {db_name}")
        print(f"Host: {db_host}")

    # Connect to ArangoDB
    try:
        client = ArangoClient(hosts=db_host)
        sys_db = client.db('_system', username='root', password=root_password)
    except Exception as e:
        print(f"❌ Failed to connect to ArangoDB: {e}")
        return 1

    # Create or drop database
    try:
        if sys_db.has_database(db_name):
            if drop_existing:
                if not quiet_mode:
                    print(f"Dropping existing database '{db_name}'...")
                sys_db.delete_database(db_name)
                sys_db.create_database(db_name)
                if not quiet_mode:
                    print(f"✅ Recreated database '{db_name}'")
            else:
                if not quiet_mode:
                    print(f"ℹ️  Database '{db_name}' already exists")
        else:
            sys_db.create_database(db_name)
            if not quiet_mode:
                print(f"✅ Created database '{db_name}'")
    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        return 1

    # Create users
    passwords = {}
    for username, config in users.items():
        try:
            password = config['password']
            passwords[username] = password

            # Check if user exists
            if sys_db.has_user(username):
                if not quiet_mode:
                    print(f"Updating user '{username}'...")
                sys_db.update_user(
                    username=username,
                    password=password,
                    active=True
                )
            else:
                sys_db.create_user(
                    username=username,
                    password=password,
                    active=True,
                    extra={'description': config['description']}
                )
                if not quiet_mode:
                    print(f"✅ Created user '{username}'")

            # Set permissions
            sys_db.update_permission(
                username=username,
                database=db_name,
                permission=config['permissions']
            )
            if not quiet_mode:
                print(f"✅ Set {config['permissions']} permissions for '{username}'")

        except Exception as e:
            print(f"❌ Failed to create user '{username}': {e}")
            return 1

    # Create collections
    try:
        db = client.db(db_name, username='root', password=root_password)

        for collection_name in collections:
            if not db.has_collection(collection_name):
                db.create_collection(collection_name)
                if not quiet_mode:
                    print(f"✅ Created collection '{collection_name}'")
            elif not quiet_mode:
                print(f"ℹ️  Collection '{collection_name}' already exists")

    except Exception as e:
        print(f"❌ Failed to create collections: {e}")
        return 1

    # Save credentials to config directory
    config_dir = Path(os.environ.get('CONFIG_DIR', 'config'))
    config_dir.mkdir(exist_ok=True)

    # Save environment file
    env_file = config_dir / f"{db_name}.env"
    with open(env_file, 'w') as f:
        f.write(f"# {db_name.upper()} Database Credentials\n")
        f.write(f"# Generated by setup_database.sh\n")
        f.write(f"# KEEP THIS FILE SECURE!\n\n")
        f.write(f"export ARXIV_DB_NAME={db_name}\n")

        # Save connection info
        if unix_socket_exists:
            f.write(f"export ARANGO_UNIX_SOCKET={unix_socket}\n")
            f.write(f"export USE_UNIX_SOCKET=true\n")
            f.write(f"# Fallback HTTP connection\n")
            f.write(f"export ARXIV_DB_HOST={db_host}\n\n")
        else:
            f.write(f"export ARXIV_DB_HOST={db_host}\n")
            f.write(f"export USE_UNIX_SOCKET=false\n\n")

        # Save usernames and passwords
        f.write(f"export ARXIV_ADMIN_USER={os.environ.get('DB_ADMIN_USER', 'arxiv_admin')}\n")
        f.write(f"export ARXIV_WRITER_USER={os.environ.get('DB_WRITER_USER', 'arxiv_writer')}\n")
        f.write(f"export ARXIV_READER_USER={os.environ.get('DB_READER_USER', 'arxiv_reader')}\n\n")

        # Use the original environment variable names
        for username, password in passwords.items():
            if 'admin' in username:
                f.write(f"export ARXIV_ADMIN_PASSWORD={password}\n")
            elif 'writer' in username:
                f.write(f"export ARXIV_WRITER_PASSWORD={password}\n")
            elif 'reader' in username:
                f.write(f"export ARXIV_READER_PASSWORD={password}\n")

    # Set restrictive permissions
    os.chmod(env_file, 0o600)

    if not quiet_mode:
        print(f"\n✅ Saved credentials to {env_file}")
        print("\nDatabase setup complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())