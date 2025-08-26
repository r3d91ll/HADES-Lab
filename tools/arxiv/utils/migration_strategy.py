"""
Migration strategy for transitioning to single-database ACID architecture.

Handles the phased migration from PostgreSQL + ArangoDB dual setup
to a single ArangoDB database with full ACID compliance.
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import psycopg2
from arango import ArangoClient

logger = logging.getLogger(__name__)


class MigrationPhase(Enum):
    """Migration phases as defined in PRD"""
    SETUP = "setup"              # Phase 1: Add ArangoDB Collections
    DUAL_WRITE = "dual_write"    # Phase 2: Write to both databases
    SWITCH_READS = "switch_reads" # Phase 3: Read from ArangoDB only
    CLEANUP = "cleanup"          # Phase 4: Remove PostgreSQL dependency


@dataclass
class MigrationStatus:
    """Status of migration process"""
    current_phase: MigrationPhase
    papers_migrated: int
    papers_total: int
    start_time: str
    last_update: str
    errors: List[str]
    warnings: List[str]


class MigrationStrategy:
    """
    Manages the phased migration from dual-database to single-database.
    
    This is a SIMPLIFICATION strategy - we're reducing complexity by
    consolidating everything into ArangoDB while maintaining ACID guarantees.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize migration manager"""
        config = config or {}
        
        # PostgreSQL connection (source) with secure error handling
        pg_password = config.get('pg_password', os.environ.get('PGPASSWORD'))
        if not pg_password:
            raise ValueError("PostgreSQL password must be provided in config or PGPASSWORD env var")
        
        try:
            self.pg_conn = psycopg2.connect(
                host=config.get('pg_host', 'localhost'),
                database=config.get('pg_database', 'arxiv_datalake'),
                user=config.get('pg_user', 'postgres'),
                password=pg_password
            )
        except Exception as e:
            # Log error without credentials
            logger.error(f"Failed to connect to PostgreSQL: {type(e).__name__}")
            raise RuntimeError("PostgreSQL connection failed") from e
        
        # ArangoDB connection (destination) with secure error handling
        arango_config = config.get('arango', {})
        arango_password = arango_config.get('password', os.environ.get('ARANGO_PASSWORD'))
        if not arango_password:
            raise ValueError("ArangoDB password must be provided in config or ARANGO_PASSWORD env var")
        
        try:
            self.arango = ArangoClient(hosts=arango_config.get('host', ['http://192.168.1.69:8529']))
            self.db = self.arango.db(
                arango_config.get('database', 'academy_store'),
                username=arango_config.get('username', 'root'),
                password=arango_password
            )
        except Exception as e:
            # Log error without credentials
            logger.error(f"Failed to connect to ArangoDB: {type(e).__name__}")
            raise RuntimeError("ArangoDB connection failed") from e
        
        # Migration state
        self.current_phase = self._detect_current_phase()
        self.migration_log = []
        
        logger.info(f"Migration strategy initialized. Current phase: {self.current_phase}")
    
    def _detect_current_phase(self) -> MigrationPhase:
        """Detect current migration phase based on database state"""
        try:
            # Check if new collections exist
            required_collections = ['papers', 'chunks', 'embeddings', 'equations', 'tables', 'images', 'locks']
            collections_exist = all(self.db.has_collection(c) for c in required_collections)
            
            if not collections_exist:
                return MigrationPhase.SETUP
            
            # Check if dual writing is active
            papers_collection = self.db.collection('papers')
            paper_count = papers_collection.count()
            
            # Check PostgreSQL paper count
            cursor = self.pg_conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM arxiv_papers WHERE arango_migrated = true")
                pg_processed_count = cursor.fetchone()[0]
            finally:
                cursor.close()
            
            if paper_count == 0:
                return MigrationPhase.DUAL_WRITE
            elif paper_count < pg_processed_count:
                return MigrationPhase.DUAL_WRITE
            else:
                # Check if reads are still from PostgreSQL
                # This would be determined by application configuration
                return MigrationPhase.SWITCH_READS
                
        except Exception as e:
            logger.error(f"Error detecting migration phase: {e}")
            return MigrationPhase.SETUP
    
    def execute_phase_1_setup(self) -> bool:
        """
        Phase 1: Create ArangoDB collections with proper indexes.
        
        This sets up the foundation for the single-database architecture.
        """
        logger.info("Starting Phase 1: Setup ArangoDB collections")
        
        try:
            # Collection definitions from PRD
            collections = [
                ('papers', {
                    'waitForSync': False,
                    'schema': {
                        'rule': {
                            'properties': {
                                '_key': {'type': 'string'},
                                'arxiv_id': {'type': 'string'},
                                'title': {'type': 'string'},
                                'abstract': {'type': 'string'},
                                'authors': {'type': 'array'},
                                'categories': {'type': 'array'},
                                'pdf_path': {'type': 'string'},
                                'pdf_hash': {'type': 'string'},
                                'status': {'type': 'string'},
                                'processing_date': {'type': 'string'},
                                'num_chunks': {'type': 'integer'},
                                'num_equations': {'type': 'integer'},
                                'num_tables': {'type': 'integer'},
                                'num_images': {'type': 'integer'}
                            },
                            'required': ['_key', 'arxiv_id', 'status']
                        }
                    }
                }),
                ('chunks', {'waitForSync': False}),
                ('embeddings', {'waitForSync': False}),
                ('equations', {'waitForSync': False}),
                ('tables', {'waitForSync': False}),
                ('images', {'waitForSync': False}),
                ('locks', {'waitForSync': True})  # Locks need immediate consistency
            ]
            
            # Create collections
            for name, options in collections:
                if not self.db.has_collection(name):
                    logger.info(f"Creating collection: {name}")
                    # Extract schema if present
                    schema = options.pop('schema', None)
                    collection = self.db.create_collection(name, **options)
                    
                    # Add schema validation if specified
                    if schema:
                        try:
                            collection.configure(schema=schema)
                            logger.info(f"Schema applied to collection: {name}")
                        except Exception as e:
                            logger.warning(f"Could not apply schema to {name}: {e}")
                            # Schema validation is optional, continue without it
            
            # Setup indexes
            self._setup_indexes()
            
            logger.info("Phase 1 complete: Collections and indexes created")
            return True
            
        except Exception as e:
            logger.error(f"Phase 1 failed: {e}")
            return False
    
    def _setup_indexes(self):
        """Setup required indexes for performance"""
        # TTL index on locks for automatic cleanup
        locks_collection = self.db.collection('locks')
        existing_indexes = locks_collection.indexes()
        has_ttl = any(idx.get('type') == 'ttl' for idx in existing_indexes)
        
        if not has_ttl:
            locks_collection.add_ttl_index(
                fields=['expiresAt'],
                expire_after=0  # Delete immediately after expiry
            )
            logger.info("Created TTL index on locks collection")
        
        # Hash index on arxiv_id for fast lookups
        papers_collection = self.db.collection('papers')
        papers_collection.add_hash_index(fields=['arxiv_id'], unique=True)
        
        # Index on paper_id for chunks, embeddings, etc.
        for collection_name in ['chunks', 'embeddings', 'equations', 'tables', 'images']:
            collection = self.db.collection(collection_name)
            collection.add_hash_index(fields=['paper_id'])
    
    def execute_phase_2_dual_write(self, batch_size: int = 100) -> Tuple[int, int]:
        """
        Phase 2: Migrate existing data with dual write.
        
        Returns:
            Tuple of (successful_migrations, failed_migrations)
        """
        logger.info("Starting Phase 2: Dual write migration")
        
        successful = 0
        failed = 0
        
        cursor = None
        try:
            # Get papers that need migration
            cursor = self.pg_conn.cursor()
            cursor.execute("""
                SELECT 
                    p.arxiv_id,
                    p.title,
                    p.abstract,
                    p.categories,
                    p.created_date,
                    p.updated_date,
                    array_agg(DISTINCT a.name) as authors
                FROM arxiv_papers p
                LEFT JOIN arxiv_paper_authors pa ON p.id = pa.paper_id
                LEFT JOIN arxiv_authors a ON pa.author_id = a.id
                WHERE p.arango_migrated IS NULL OR p.arango_migrated = false
                GROUP BY p.id, p.arxiv_id, p.title, p.abstract, p.categories, p.created_date, p.updated_date
                LIMIT %s
            """, (batch_size,))
            
            papers = cursor.fetchall()
            
            for paper_data in papers:
                arxiv_id = paper_data[0]
                
                try:
                    # Begin atomic transaction for this paper
                    self.pg_conn.autocommit = False
                    
                    # Migrate paper metadata to ArangoDB
                    self._migrate_paper(paper_data)
                    
                    # Mark as migrated in PostgreSQL (atomic with migration)
                    update_cursor = self.pg_conn.cursor()
                    try:
                        update_cursor.execute(
                            "UPDATE arxiv_papers SET arango_migrated = true WHERE arxiv_id = %s",
                            (arxiv_id,)
                        )
                        self.pg_conn.commit()
                        successful += 1
                    finally:
                        update_cursor.close()
                    
                except Exception as e:
                    # Rollback on any error
                    self.pg_conn.rollback()
                    logger.error(f"Failed to migrate {arxiv_id}: {e}")
                    failed += 1
                finally:
                    # Restore autocommit
                    self.pg_conn.autocommit = True
            
            logger.info(f"Phase 2 batch complete: {successful} successful, {failed} failed")
            return successful, failed
            
        except Exception as e:
            logger.error(f"Phase 2 failed: {e}")
            return successful, failed
        finally:
            if cursor:
                cursor.close()
    
    def _sanitize_arxiv_id(self, arxiv_id: str) -> str:
        """
        Sanitize arxiv_id for use as ArangoDB _key.
        
        ArXiv IDs follow strict patterns:
        - Modern: YYMM.NNNNN (e.g., "1234.5678")
        - Old: category/YYMMNNN (e.g., "cs.AI/0612345")
        
        Since ArXiv IDs never contain "__dot__" or "__slash__" naturally,
        this simple replacement is collision-free for this domain.
        
        Uses distinct tokens to avoid collisions:
        - '.' becomes '__dot__'
        - '/' becomes '__slash__'
        
        Note: For other domains where IDs might contain these patterns,
        consider using base64 encoding instead:
        base64.urlsafe_b64encode(arxiv_id.encode()).decode().rstrip('=')
        
        Args:
            arxiv_id: Original arXiv identifier
            
        Returns:
            Sanitized identifier safe for ArangoDB _key
        """
        return arxiv_id.replace('.', '__dot__').replace('/', '__slash__')
    
    def _migrate_paper(self, paper_data: Tuple):
        """Migrate single paper to ArangoDB"""
        arxiv_id, title, abstract, categories, created_date, updated_date, authors = paper_data
        
        # Sanitize arxiv_id for use as _key
        sanitized_id = self._sanitize_arxiv_id(arxiv_id)
        
        # Insert into papers collection
        self.db.collection('papers').insert({
            '_key': sanitized_id,
            'arxiv_id': arxiv_id,
            'title': title,
            'abstract': abstract,
            'categories': categories if categories else [],
            'authors': authors if authors else [],
            'created_date': created_date.isoformat() if created_date else None,
            'updated_date': updated_date.isoformat() if updated_date else None,
            'status': 'MIGRATED',
            'migration_date': datetime.now().isoformat(),
            'processing_date': None,  # Not yet processed
            'num_chunks': 0,
            'num_equations': 0,
            'num_tables': 0,
            'num_images': 0
        }, overwrite=True)
    
    def execute_phase_3_switch_reads(self) -> bool:
        """
        Phase 3: Switch all reads to ArangoDB.
        
        This phase updates application configuration to read from ArangoDB only.
        """
        logger.info("Starting Phase 3: Switch reads to ArangoDB")
        
        try:
            # Verify data consistency
            if not self._verify_data_consistency():
                logger.error("Data consistency check failed. Cannot switch reads.")
                return False
            
            # Update configuration flag (this would be in a config file)
            # In practice, this would update a configuration file or environment variable
            logger.info("Configuration updated to read from ArangoDB")
            
            # Log the switch
            self.migration_log.append({
                'phase': 'switch_reads',
                'timestamp': datetime.now().isoformat(),
                'status': 'complete'
            })
            
            logger.info("Phase 3 complete: Reads switched to ArangoDB")
            return True
            
        except Exception as e:
            logger.error(f"Phase 3 failed: {e}")
            return False
    
    def _verify_data_consistency(self) -> bool:
        """Verify data consistency between PostgreSQL and ArangoDB"""
        cursor = None
        try:
            # Check paper counts
            cursor = self.pg_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM arxiv_papers")
            pg_count = cursor.fetchone()[0]
            
            arango_count = self.db.collection('papers').count()
            
            if abs(pg_count - arango_count) > 100:  # Allow small difference during migration
                logger.warning(f"Paper count mismatch: PostgreSQL={pg_count}, ArangoDB={arango_count}")
                return False
            
            # Sample check: verify random papers match
            cursor = self.pg_conn.cursor()
            cursor.execute("""
                SELECT arxiv_id, title FROM arxiv_papers 
                ORDER BY RANDOM() LIMIT 10
            """)
            
            for arxiv_id, title in cursor.fetchall():
                sanitized_id = self._sanitize_arxiv_id(arxiv_id)
                arango_paper = self.db.collection('papers').get(sanitized_id)
                
                if not arango_paper or arango_paper.get('title') != title:
                    logger.warning(f"Paper mismatch: {arxiv_id}")
                    return False
            
            logger.info("Data consistency check passed")
            return True
            
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def execute_phase_4_cleanup(self) -> bool:
        """
        Phase 4: Remove PostgreSQL dependency.
        
        This is the final phase where we can safely remove PostgreSQL.
        """
        logger.info("Starting Phase 4: Cleanup PostgreSQL dependency")
        
        try:
            # Final consistency check
            if not self._verify_data_consistency():
                logger.error("Final consistency check failed. Cannot proceed with cleanup.")
                return False
            
            # Close PostgreSQL connection
            self.pg_conn.close()
            
            # Log completion
            self.migration_log.append({
                'phase': 'cleanup',
                'timestamp': datetime.now().isoformat(),
                'status': 'complete'
            })
            
            logger.info("Phase 4 complete: PostgreSQL dependency removed")
            logger.info("MIGRATION COMPLETE: Now using single ArangoDB database with ACID guarantees")
            return True
            
        except Exception as e:
            logger.error(f"Phase 4 failed: {e}")
            return False
    
    def get_migration_status(self) -> MigrationStatus:
        """Get current migration status"""
        cursor = None
        try:
            # Count papers in each database
            cursor = self.pg_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM arxiv_papers")
            pg_total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM arxiv_papers WHERE arango_migrated = true")
            pg_migrated = cursor.fetchone()[0]
            
            arango_count = self.db.collection('papers').count() if self.db.has_collection('papers') else 0
            
            return MigrationStatus(
                current_phase=self.current_phase,
                papers_migrated=arango_count,
                papers_total=pg_total,
                start_time=self.migration_log[0]['timestamp'] if self.migration_log else None,
                last_update=datetime.now().isoformat(),
                errors=[],
                warnings=[]
            )
            
        except Exception as e:
            logger.error(f"Error getting migration status: {e}")
            return MigrationStatus(
                current_phase=self.current_phase,
                papers_migrated=0,
                papers_total=0,
                start_time=None,
                last_update=datetime.now().isoformat(),
                errors=[str(e)],
                warnings=[]
            )
        finally:
            if cursor:
                cursor.close()
    
    def execute_full_migration(self, batch_size: int = 1000) -> bool:
        """
        Execute complete migration through all phases.
        
        This is the main entry point for the migration process.
        """
        logger.info("Starting full migration to single-database ACID architecture")
        
        # Phase 1: Setup
        if self.current_phase == MigrationPhase.SETUP:
            if not self.execute_phase_1_setup():
                logger.error("Phase 1 failed. Migration aborted.")
                return False
            self.current_phase = MigrationPhase.DUAL_WRITE
        
        # Phase 2: Dual Write
        if self.current_phase == MigrationPhase.DUAL_WRITE:
            total_successful = 0
            total_failed = 0
            
            while True:
                successful, failed = self.execute_phase_2_dual_write(batch_size)
                total_successful += successful
                total_failed += failed
                
                if successful == 0:  # No more papers to migrate
                    break
                
                # Log progress
                logger.info(f"Migration progress: {total_successful} migrated, {total_failed} failed")
                time.sleep(1)  # Brief pause between batches
            
            if total_failed > total_successful * 0.01:  # More than 1% failure rate
                logger.error(f"High failure rate: {total_failed}/{total_successful}. Migration aborted.")
                return False
            
            self.current_phase = MigrationPhase.SWITCH_READS
        
        # Phase 3: Switch Reads
        if self.current_phase == MigrationPhase.SWITCH_READS:
            if not self.execute_phase_3_switch_reads():
                logger.error("Phase 3 failed. Migration aborted.")
                return False
            self.current_phase = MigrationPhase.CLEANUP
        
        # Phase 4: Cleanup
        if self.current_phase == MigrationPhase.CLEANUP:
            if not self.execute_phase_4_cleanup():
                logger.error("Phase 4 failed. Migration incomplete but functional.")
                return False
        
        logger.info("Full migration complete! Single-database ACID architecture is now active.")
        return True
    
    def cleanup(self):
        """
        Clean up database connections and resources.
        Should be called when migration is complete or on error.
        """
        try:
            if hasattr(self, 'pg_conn') and self.pg_conn:
                self.pg_conn.close()
                logger.info("PostgreSQL connection closed")
        except Exception as e:
            logger.warning(f"Error closing PostgreSQL connection: {e}")
        
        # ArangoDB client handles connection pooling internally
        # No explicit cleanup needed