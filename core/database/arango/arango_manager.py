#!/usr/bin/env python3
"""
ArangoDB Manager - Production utility for database operations.

Provides safe, reusable database management operations including:
- Collection management (create, drop, truncate, backup)
- Data verification and statistics
- Safe rebuild operations with confirmation
- Backup and restore capabilities

This is a PRODUCTION tool - all destructive operations require confirmation.
"""

import os
import sys
import json
import time
import click
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from arango import ArangoClient
from tabulate import tabulate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArangoManager:
    """Production-ready ArangoDB management utility."""
    
    def __init__(self, database: str = 'academy_store', 
                 host: str = 'http://localhost:8529',
                 username: str = 'root',
                 password: Optional[str] = None):
        """
        Initialize ArangoDB manager.
        
        Args:
            database: Database name
            host: ArangoDB host URL
            username: Database username
            password: Database password (defaults to ARANGO_PASSWORD env var)
        """
        self.database_name = database
        self.host = host
        self.username = username
        self.password = password or os.environ.get('ARANGO_PASSWORD')
        
        if not self.password:
            raise ValueError("No password provided. Set ARANGO_PASSWORD environment variable.")
        
        # Connect to database
        self.client = ArangoClient(hosts=host)
        self.db = self.client.db(database, username=username, password=self.password)
        
    def get_collections_info(self) -> List[Dict[str, Any]]:
        """Get information about all collections."""
        collections = []
        for collection in self.db.collections():
            if not collection['system']:  # Skip system collections
                info = {
                    'name': collection['name'],
                    'type': 'edge' if collection['type'] == 3 else 'document',
                    'count': self.db.collection(collection['name']).count(),
                    'size': collection.get('size', 0)
                }
                collections.append(info)
        return sorted(collections, key=lambda x: x['name'])
    
    def show_statistics(self):
        """Display database statistics in a formatted table."""
        print("\n" + "="*70)
        print(f"DATABASE: {self.database_name} @ {self.host}")
        print("="*70)
        
        collections = self.get_collections_info()
        
        if not collections:
            print("No collections found.")
            return
        
        # Prepare table data
        table_data = []
        total_docs = 0
        
        for coll in collections:
            total_docs += coll['count']
            table_data.append([
                coll['name'],
                coll['type'],
                f"{coll['count']:,}",
                f"{coll['size'] / (1024*1024):.2f} MB" if coll['size'] else "N/A"
            ])
        
        # Display table
        headers = ['Collection', 'Type', 'Documents', 'Size']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print(f"\nTotal documents: {total_docs:,}")
        
    def backup_collection(self, collection_name: str, output_dir: str = "./backups"):
        """
        Backup a collection to JSON file.
        
        Args:
            collection_name: Name of collection to backup
            output_dir: Directory to save backup files
        """
        if not self.db.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist")
            return False
        
        # Create backup directory
        backup_path = Path(output_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Generate backup filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = backup_path / f"{collection_name}_{timestamp}.json"
        
        # Export collection
        logger.info(f"Backing up {collection_name} to {filename}")
        collection = self.db.collection(collection_name)
        
        documents = []
        for doc in collection:
            documents.append(doc)
        
        with open(filename, 'w') as f:
            json.dump(documents, f, indent=2)
        
        logger.info(f"✓ Backed up {len(documents):,} documents to {filename}")
        return True
    
    def truncate_collection(self, collection_name: str, confirm: bool = False):
        """
        Truncate (empty) a collection while preserving its structure.
        
        Args:
            collection_name: Name of collection to truncate
            confirm: Skip confirmation prompt if True
        """
        if not self.db.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist")
            return False
        
        collection = self.db.collection(collection_name)
        count = collection.count()
        
        if not confirm:
            response = input(f"⚠️  Truncate '{collection_name}' ({count:,} documents)? [y/N]: ")
            if response.lower() != 'y':
                logger.info("Truncate cancelled")
                return False
        
        collection.truncate()
        logger.info(f"✓ Truncated {collection_name} ({count:,} documents removed)")
        return True
    
    def drop_collection(self, collection_name: str, confirm: bool = False):
        """
        Drop (delete) a collection completely.
        
        Args:
            collection_name: Name of collection to drop
            confirm: Skip confirmation prompt if True
        """
        if not self.db.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist")
            return False
        
        collection = self.db.collection(collection_name)
        count = collection.count()
        
        if not confirm:
            response = input(f"⚠️  DROP '{collection_name}' ({count:,} documents)? This cannot be undone! [y/N]: ")
            if response.lower() != 'y':
                logger.info("Drop cancelled")
                return False
        
        self.db.delete_collection(collection_name)
        logger.info(f"✓ Dropped {collection_name}")
        return True
    
    def drop_all_collections(self, pattern: Optional[str] = None, 
                            exclude: Optional[List[str]] = None,
                            backup_first: bool = True):
        """
        Drop all collections matching pattern.
        
        Args:
            pattern: Only drop collections containing this string
            exclude: List of collection names to exclude
            backup_first: Create backups before dropping
        """
        collections = self.get_collections_info()
        exclude = exclude or []
        
        # Filter collections
        to_drop = []
        for coll in collections:
            if coll['name'] in exclude:
                continue
            if pattern and pattern not in coll['name']:
                continue
            to_drop.append(coll)
        
        if not to_drop:
            logger.info("No collections to drop")
            return
        
        # Show what will be dropped
        print("\n" + "="*70)
        print("Collections to be DROPPED:")
        print("="*70)
        for coll in to_drop:
            print(f"  - {coll['name']}: {coll['count']:,} documents")
        
        total_docs = sum(c['count'] for c in to_drop)
        print(f"\nTotal: {len(to_drop)} collections, {total_docs:,} documents")
        
        # Confirm
        response = input("\n⚠️  This action CANNOT be undone! Type 'DELETE ALL' to confirm: ")
        if response != 'DELETE ALL':
            logger.info("Operation cancelled")
            return
        
        # Backup if requested
        if backup_first:
            logger.info("\nCreating backups...")
            for coll in to_drop:
                self.backup_collection(coll['name'])
        
        # Drop collections
        logger.info("\nDropping collections...")
        for coll in to_drop:
            self.db.delete_collection(coll['name'])
            logger.info(f"  ✓ Dropped {coll['name']}")
        
        logger.info(f"\n✓ Dropped {len(to_drop)} collections")
    
    def create_collections(self, collections: Dict[str, str]):
        """
        Create multiple collections.
        
        Args:
            collections: Dict of {name: type} where type is 'document' or 'edge'
        """
        for name, coll_type in collections.items():
            if self.db.has_collection(name):
                logger.info(f"Collection '{name}' already exists")
                continue
            
            if coll_type == 'edge':
                self.db.create_edge_collection(name)
            else:
                self.db.create_collection(name)
            
            logger.info(f"✓ Created {coll_type} collection: {name}")
    
    def verify_schema(self, expected_collections: Dict[str, Dict[str, Any]]) -> bool:
        """
        Verify database schema matches expectations.
        
        Args:
            expected_collections: Dict of collection definitions
            
        Returns:
            True if schema matches, False otherwise
        """
        all_valid = True
        
        for name, config in expected_collections.items():
            if not self.db.has_collection(name):
                logger.error(f"✗ Missing collection: {name}")
                all_valid = False
                continue
            
            collection = self.db.collection(name)
            
            # Check type
            is_edge = collection.properties()['type'] == 3
            expected_edge = config.get('type') == 'edge'
            
            if is_edge != expected_edge:
                logger.error(f"✗ {name}: Wrong type (expected {'edge' if expected_edge else 'document'})")
                all_valid = False
            
            # Check minimum document count if specified
            min_count = config.get('min_count')
            if min_count and collection.count() < min_count:
                logger.warning(f"⚠ {name}: Has {collection.count():,} docs (expected ≥{min_count:,})")
            
            logger.info(f"✓ {name}: Valid")
        
        return all_valid


@click.group()
def cli():
    """ArangoDB management utility."""
    pass


@cli.command()
@click.option('--database', default='academy_store', help='Database name')
def stats(database):
    """Show database statistics."""
    manager = ArangoManager(database=database)
    manager.show_statistics()


@cli.command()
@click.option('--database', default='academy_store', help='Database name')
@click.option('--pattern', default=None, help='Only drop collections matching pattern')
@click.option('--exclude', multiple=True, help='Collections to exclude')
@click.option('--no-backup', is_flag=True, help='Skip backup before dropping')
def drop_all(database, pattern, exclude, no_backup):
    """Drop all collections (with safety checks)."""
    manager = ArangoManager(database=database)
    manager.drop_all_collections(
        pattern=pattern,
        exclude=list(exclude),
        backup_first=not no_backup
    )


@cli.command()
@click.option('--database', default='academy_store', help='Database name')
@click.argument('collection')
def backup(database, collection):
    """Backup a collection to JSON."""
    manager = ArangoManager(database=database)
    manager.backup_collection(collection)


@cli.command()
@click.option('--database', default='academy_store', help='Database name')
@click.argument('collection')
@click.option('--confirm', is_flag=True, help='Skip confirmation')
def truncate(database, collection, confirm):
    """Truncate (empty) a collection."""
    manager = ArangoManager(database=database)
    manager.truncate_collection(collection, confirm=confirm)


@cli.command()
@click.option('--database', default='academy_store', help='Database name')
def arxiv_rebuild(database):
    """Prepare database for ArXiv rebuild."""
    manager = ArangoManager(database=database)
    
    print("\n" + "="*70)
    print("ARXIV DATABASE REBUILD PREPARATION")
    print("="*70)
    
    # Show current state
    manager.show_statistics()
    
    # Define ArXiv collections
    arxiv_collections = [
        'arxiv_papers',
        'arxiv_chunks',
        'arxiv_embeddings',
        'arxiv_equations',
        'arxiv_tables',
        'arxiv_images',
        'arxiv_structures',
        'same_field',
        'temporal_proximity',
        'keyword_similarity',
        'abstract_similarity',
        'paper_versions'
    ]
    
    # Drop ArXiv collections
    print("\nPreparing to drop ArXiv collections...")
    manager.drop_all_collections(
        pattern='arxiv',
        backup_first=True
    )
    manager.drop_all_collections(
        pattern='similarity',
        backup_first=False
    )
    manager.drop_all_collections(
        pattern='same_field',
        backup_first=False
    )
    manager.drop_all_collections(
        pattern='temporal_proximity',
        backup_first=False
    )
    manager.drop_all_collections(
        pattern='paper_versions',
        backup_first=False
    )
    
    # Create fresh collections
    print("\nCreating fresh collections...")
    collections_to_create = {
        'arxiv_papers': 'document',
        'arxiv_chunks': 'document',
        'arxiv_embeddings': 'document',
        'arxiv_equations': 'document',
        'arxiv_tables': 'document',
        'arxiv_images': 'document',
        'arxiv_structures': 'document',
        'same_field': 'edge',
        'temporal_proximity': 'edge',
        'keyword_similarity': 'edge',
        'abstract_similarity': 'edge',
        'paper_versions': 'edge'
    }
    manager.create_collections(collections_to_create)
    
    print("\n✓ Database ready for ArXiv rebuild")
    print("\nNext steps:")
    print("1. Run: cd tools/arxiv/utils/")
    print("2. Run: python complete_arxiv_ingestion.py")


if __name__ == '__main__':
    cli()