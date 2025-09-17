#!/usr/bin/env python3
"""
Production-grade database setup for ArXiv Repository.

This script implements database best practices:
1. Creates dedicated database with proper user management
2. Sets up role-based access control (RBAC)
3. Creates separate users for different access levels
4. Implements proper security practices

Theory Connection (Conveyance Framework):
Optimizes WHO dimension by establishing clear agent boundaries and access patterns.
"""

import os
import sys
import secrets
import string
from typing import Dict, Optional
from pathlib import Path
from arango import ArangoClient
import json


class ArxivDatabaseSetup:
    """Production database setup with proper security."""

    def __init__(self, root_password: str, host: str = 'http://localhost:8529'):
        """
        Initialize an ArxivDatabaseSetup instance and prepare ArangoDB configuration metadata.
        
        Creates an ArangoDB client connected to the _system database, sets the target database name (arxiv_repository), and populates in-memory configuration for RBAC users (admin, writer, reader) and the collections with their desired index definitions used later by the setup workflow.
        
        Parameters:
            root_password (str): Root password for the ArangoDB '_system' user used to perform administrative operations.
            host (str): ArangoDB HTTP endpoint (e.g., 'http://localhost:8529').
        """
        self.client = ArangoClient(hosts=host)
        self.sys_db = self.client.db('_system', username='root', password=root_password)
        self.db_name = 'arxiv_repository'

        # User configurations
        self.users = {
            'arxiv_admin': {
                'role': 'admin',
                'description': 'Admin user for arxiv_repository database',
                'permissions': 'rw'  # Read-write on database
            },
            'arxiv_writer': {
                'role': 'writer',
                'description': 'Writer user for workflow processing',
                'permissions': 'rw'  # Read-write for processing
            },
            'arxiv_reader': {
                'role': 'reader',
                'description': 'Read-only user for queries and monitoring',
                'permissions': 'ro'  # Read-only
            }
        }

        # Collections to create
        self.collections = {
            'arxiv_papers': {
                'type': 'document',
                'indexes': [
                    {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True},
                    {'type': 'skiplist', 'fields': ['size_order_position']},
                    {'type': 'skiplist', 'fields': ['token_count']},
                    {'type': 'skiplist', 'fields': ['processed_at']},
                    {'type': 'skiplist', 'fields': ['categories']},
                    {'type': 'fulltext', 'fields': ['title'], 'minLength': 3}
                ]
            },
            'arxiv_chunks': {
                'type': 'document',
                'indexes': [
                    {'type': 'hash', 'fields': ['arxiv_id', 'chunk_index']},
                    {'type': 'skiplist', 'fields': ['paper_key']}
                ]
            },
            'arxiv_abstract_embeddings': {
                'type': 'document',
                'indexes': [
                    {'type': 'hash', 'fields': ['arxiv_id', 'chunk_index']},
                    {'type': 'skiplist', 'fields': ['paper_key']}
                ]
            },
            'arxiv_processing_order': {
                'type': 'document',
                'indexes': [
                    {'type': 'skiplist', 'fields': ['position']},
                    {'type': 'hash', 'fields': ['arxiv_id'], 'unique': True}
                ]
            },
            'arxiv_processing_stats': {
                'type': 'document',
                'indexes': [
                    {'type': 'skiplist', 'fields': ['timestamp']},
                    {'type': 'hash', 'fields': ['workflow_run_id']}
                ]
            }
        }

    def generate_password(self, length: int = 24) -> str:
        """
        Generate a cryptographically secure random password.
        
        Parameters:
            length (int): Desired length of the password (default 24). Must be a positive integer.
        
        Returns:
            str: A password of the requested length containing ASCII letters, digits, and the symbols !@#$%^&*.
        """
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def create_database(self, drop_existing: bool = False) -> bool:
        """
        Create the target ArangoDB database for the repository, optionally dropping an existing one.
        
        If the database named by self.db_name already exists, this will either return successfully (no-op)
        or drop it first when drop_existing is True, then create a fresh database. Any exceptions are
        caught and cause the method to return False.
        
        Parameters:
            drop_existing (bool): If True, delete an existing database with the same name before creating it.
        
        Returns:
            bool: True when the database exists or was created successfully; False on error.
        """
        try:
            if self.sys_db.has_database(self.db_name):
                if drop_existing:
                    print(f"⚠️  Dropping existing database '{self.db_name}'...")
                    self.sys_db.delete_database(self.db_name)
                    print(f"✅ Dropped database '{self.db_name}'")
                else:
                    print(f"ℹ️  Database '{self.db_name}' already exists")
                    return True

            # Create database
            self.sys_db.create_database(self.db_name)
            print(f"✅ Created database '{self.db_name}'")
            return True

        except Exception as e:
            print(f"❌ Failed to create database: {e}")
            return False

    def create_users(self) -> Dict[str, str]:
        """
        Create or update the configured ArangoDB users and assign database permissions.
        
        For each user defined in self.users this will:
        - Generate a secure random password and include it in the returned mapping.
        - Create the user if missing, or update the existing user's password and activate the account.
        - Set the user's permission for the target database (self.db_name) according to the user's configuration.
        
        Errors for individual users are caught and logged (function continues processing other users); exceptions are not propagated.
        
        Returns:
            Dict[str, str]: Mapping of username -> generated password for each processed user.
        """
        passwords = {}

        for username, config in self.users.items():
            try:
                # Generate secure password
                password = self.generate_password()
                passwords[username] = password

                # Check if user exists
                if self.sys_db.has_user(username):
                    print(f"ℹ️  User '{username}' already exists, updating...")
                    self.sys_db.update_user(
                        username=username,
                        password=password,
                        active=True
                    )
                else:
                    # Create user
                    self.sys_db.create_user(
                        username=username,
                        password=password,
                        active=True,
                        extra={'description': config['description']}
                    )
                    print(f"✅ Created user '{username}'")

                # Set database permissions
                self.sys_db.update_permission(
                    username=username,
                    database=self.db_name,
                    permission=config['permissions']
                )
                print(f"✅ Set {config['permissions']} permissions for '{username}' on '{self.db_name}'")

            except Exception as e:
                print(f"❌ Failed to create user '{username}': {e}")

        return passwords

    def create_collections(self) -> bool:
        """
        Create the configured collections in the target database and ensure their indexes exist.
        
        This connects to the target ArangoDB database using administrative credentials (currently `root`) and, for each collection declared in self.collections:
        - creates the collection if it does not exist;
        - creates configured indexes of types `hash`, `skiplist`, and `fulltext` with the provided fields and options (`unique` for hash/skiplist, `minLength` for fulltext).
        
        Index creation tolerates already-existing indexes (duplicate-index errors are ignored); other index creation errors are reported and cause the method to continue or abort depending on the collection-level error.
        
        Returns:
            bool: True when all collections and their indexes were ensured successfully; False if any collection or global operation failed.
        """
        try:
            # Connect to the arxiv database with admin user
            # For now use root, but in production use arxiv_admin
            db = self.client.db(self.db_name, username='root', password=self.sys_db.password)

            for collection_name, config in self.collections.items():
                try:
                    # Create collection if it doesn't exist
                    if not db.has_collection(collection_name):
                        collection = db.create_collection(collection_name)
                        print(f"✅ Created collection '{collection_name}'")
                    else:
                        collection = db.collection(collection_name)
                        print(f"ℹ️  Collection '{collection_name}' already exists")

                    # Create indexes
                    for index_config in config.get('indexes', []):
                        try:
                            if index_config['type'] == 'hash':
                                collection.add_hash_index(
                                    fields=index_config['fields'],
                                    unique=index_config.get('unique', False)
                                )
                            elif index_config['type'] == 'skiplist':
                                collection.add_skiplist_index(
                                    fields=index_config['fields'],
                                    unique=index_config.get('unique', False)
                                )
                            elif index_config['type'] == 'fulltext':
                                collection.add_fulltext_index(
                                    fields=index_config['fields'],
                                    min_length=index_config.get('minLength', 3)
                                )

                            fields_str = ', '.join(index_config['fields'])
                            print(f"  ✅ Created {index_config['type']} index on {fields_str}")

                        except Exception as e:
                            # Index might already exist, that's okay
                            if "duplicate" not in str(e).lower():
                                print(f"  ⚠️  Index creation warning: {e}")

                except Exception as e:
                    print(f"❌ Failed to create collection '{collection_name}': {e}")
                    return False

            return True

        except Exception as e:
            print(f"❌ Failed to create collections: {e}")
            return False

    def save_credentials(self, passwords: Dict[str, str], output_dir: str = "config") -> bool:
        """
        Write generated user passwords and a machine-readable config to disk.
        
        Creates `output_dir` (if missing) and writes two files for the configured database:
        - an environment file named `<db_name>.env` containing ARXIV_DB_NAME, ARXIV_DB_HOST, and per-user password variables. Username keys are uppercased and hyphens replaced with underscores (e.g. `arxiv-writer` -> `ARXIV_WRITER_PASSWORD`). The env file is written with restrictive permissions (0600).
        - a JSON config named `<db_name>_config.json` containing the database name, host, user metadata (description, role, permissions) and the list of collection names. The JSON file is written with permissions 0644.
        
        Parameters:
            passwords (Dict[str, str]): Mapping of username -> plaintext password to persist.
            output_dir (str): Directory path where the files will be created (default: "config").
        
        Returns:
            bool: True if both files were written and permission bits applied successfully; False if any error occurred.
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # Save credentials to environment file
            env_file = output_path / f"{self.db_name}.env"
            with open(env_file, 'w') as f:
                f.write(f"# ArXiv Repository Database Credentials\n")
                f.write(f"# Generated by setup_arxiv_database.py\n")
                f.write(f"# KEEP THIS FILE SECURE!\n\n")

                f.write(f"ARXIV_DB_NAME={self.db_name}\n")
                f.write(f"ARXIV_DB_HOST=http://localhost:8529\n\n")

                for username, password in passwords.items():
                    user_upper = username.upper().replace('-', '_')
                    f.write(f"{user_upper}_PASSWORD={password}\n")

            # Set restrictive permissions (owner read/write only)
            os.chmod(env_file, 0o600)
            print(f"✅ Saved credentials to {env_file}")

            # Save JSON config for programmatic access
            config_file = output_path / f"{self.db_name}_config.json"
            config_data = {
                'database': self.db_name,
                'host': 'http://localhost:8529',
                'users': {
                    username: {
                        'description': self.users[username]['description'],
                        'role': self.users[username]['role'],
                        'permissions': self.users[username]['permissions']
                    }
                    for username in passwords.keys()
                },
                'collections': list(self.collections.keys())
            }

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            os.chmod(config_file, 0o644)  # Readable by all, writable by owner
            print(f"✅ Saved configuration to {config_file}")

            return True

        except Exception as e:
            print(f"❌ Failed to save credentials: {e}")
            return False

    def print_connection_info(self, passwords: Dict[str, str]):
        """
        Print a concise connection summary and usage examples for the created database users.
        
        This writes a human-readable summary to standard output including:
        - the database name, host, and collection count;
        - a list of created users with their role and permissions;
        - example shell commands showing how to export the generated passwords for the `arxiv_writer` and `arxiv_reader` users and example commands to run common workflows/monitoring tools;
        - brief security reminders about storing and protecting the generated credentials.
        
        Parameters:
            passwords (Dict[str, str]): Mapping of username -> generated password used to populate the example export commands. If a password is missing for a user, the example output points callers to the config file path.
        """
        print("\n" + "=" * 60)
        print("DATABASE SETUP COMPLETE")
        print("=" * 60)

        print(f"\n📊 Database: {self.db_name}")
        print(f"🌐 Host: http://localhost:8529")
        print(f"📁 Collections: {len(self.collections)}")

        print("\n👥 Users Created:")
        for username, config in self.users.items():
            print(f"  • {username}: {config['description']}")
            print(f"    Role: {config['role']}, Permissions: {config['permissions']}")

        print("\n🔐 Connection Examples:")
        print("\nFor workflow processing (use arxiv_writer):")
        print(f"  export ARXIV_WRITER_PASSWORD='{passwords.get('arxiv_writer', 'check core/config/arxiv_repository.env')}'")
        print(f"  python -m core.workflows.workflow_arxiv_sorted \\")
        print(f"    --database {self.db_name} \\")
        print(f"    --username arxiv_writer")

        print("\nFor monitoring (use arxiv_reader):")
        print(f"  export ARXIV_READER_PASSWORD='{passwords.get('arxiv_reader', 'check core/config/arxiv_repository.env')}'")
        print(f"  python dev-utils/simple_monitor.py \\")
        print(f"    --database {self.db_name} \\")
        print(f"    --username arxiv_reader")

        print("\n⚠️  SECURITY NOTES:")
        print("  1. Credentials saved to core/config/arxiv_repository.env")
        print("  2. Keep this file secure and never commit to git")
        print("  3. Add 'config/*.env' to .gitignore")
        print("  4. Use environment variables in production")
        print("=" * 60)

    def setup(self, drop_existing: bool = False) -> bool:
        """
        Orchestrate the end-to-end setup of the `arxiv_repository` ArangoDB instance.
        
        Performs the full workflow in sequence: create (or optionally drop and recreate) the database, provision RBAC users and generated passwords, create collections and indexes, persist credentials/config files, and print connection/usage information. The process stops and returns False immediately if any step fails.
        Parameters:
            drop_existing (bool): If True, drop an existing `arxiv_repository` database before creating it.
        
        Returns:
            bool: True when all setup steps complete successfully; False if any step fails.
        """
        print("🚀 Starting ArXiv Repository Database Setup")
        print("=" * 60)

        # Create database
        if not self.create_database(drop_existing):
            return False

        # Create users
        passwords = self.create_users()
        if not passwords:
            print("❌ No users were created")
            return False

        # Create collections
        if not self.create_collections():
            return False

        # Save credentials
        if not self.save_credentials(passwords):
            return False

        # Print connection info
        self.print_connection_info(passwords)

        return True


def main():
    """
    CLI entry point to create and configure the arxiv_repository ArangoDB database.
    
    Parses command-line flags:
    - --drop-existing: if present, an existing database with the same name will be dropped before creation.
    - --host: ArangoDB host URL (default: "http://localhost:8529").
    
    Requires the ARANGO_PASSWORD environment variable to contain the ArangoDB root password; if it is not set the function prints an error and exits with code 1.
    
    Orchestrates the setup by instantiating ArxivDatabaseSetup(root_password, host) and calling its setup(drop_existing) method. Exits the process with status code 0 on successful setup or 1 on failure.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up ArXiv Repository database with proper security"
    )
    parser.add_argument(
        '--drop-existing',
        action='store_true',
        help='Drop existing database if it exists'
    )
    parser.add_argument(
        '--host',
        default='http://localhost:8529',
        help='ArangoDB host URL'
    )

    args = parser.parse_args()

    # Get root password
    root_password = os.environ.get('ARANGO_PASSWORD')
    if not root_password:
        print("❌ ERROR: ARANGO_PASSWORD environment variable not set")
        print("Please set: export ARANGO_PASSWORD='your-root-password'")
        sys.exit(1)

    # Run setup
    setup = ArxivDatabaseSetup(root_password, args.host)
    success = setup.setup(args.drop_existing)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()