"""
Storage Management
==================

Database connection management and utilities.
"""

from typing import Optional, Dict, Any, List
from arango import ArangoClient
from arango.database import StandardDatabase
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manage database connections and provide storage utilities.
    
    Features:
    - Connection pooling
    - Automatic retry logic
    - Collection management
    - Index management
    """
    
    # Cache connections to avoid recreating
    _connections: Dict[str, StandardDatabase] = {}
    
    @classmethod
    def get_connection(cls, config: Any) -> StandardDatabase:
        """
        Get or create a database connection.
        
        Args:
            config: Database configuration object
            
        Returns:
            ArangoDB database connection
        """
        # Create connection key from config
        connection_key = f"{config.host}:{config.port}/{config.database}"
        
        # Return cached connection if available
        if connection_key in cls._connections:
            return cls._connections[connection_key]
        
        # Create new connection
        client = ArangoClient(hosts=f"http://{config.host}:{config.port}")
        
        # Connect to system database first
        sys_db = client.db(
            '_system',
            username=config.username,
            password=config.password
        )
        
        # Create database if it doesn't exist
        if not sys_db.has_database(config.database):
            sys_db.create_database(config.database)
            logger.info(f"Created database: {config.database}")
        
        # Connect to target database
        db = client.db(
            config.database,
            username=config.username,
            password=config.password
        )
        
        # Cache the connection
        cls._connections[connection_key] = db
        
        logger.info(f"Connected to database: {connection_key}")
        return db
    
    @staticmethod
    def ensure_collection(db: StandardDatabase, name: str, 
                         edge: bool = False, **kwargs) -> None:
        """
        Ensure a collection exists.
        
        Args:
            db: Database connection
            name: Collection name
            edge: Whether this is an edge collection
            **kwargs: Additional collection parameters
        """
        if not db.has_collection(name):
            if edge:
                db.create_collection(name, edge=True, **kwargs)
            else:
                db.create_collection(name, **kwargs)
            logger.info(f"Created {'edge' if edge else 'document'} collection: {name}")
    
    @staticmethod
    def create_index(db: StandardDatabase, collection: str, 
                    index_type: str, fields: list, **kwargs) -> None:
        """
        Create an index on a collection.
        
        Args:
            db: Database connection
            collection: Collection name
            index_type: Type of index (persistent, geo, fulltext, etc.)
            fields: Fields to index
            **kwargs: Additional index parameters
        """
        coll = db.collection(collection)
        
        try:
            if index_type == "persistent":
                coll.add_persistent_index(fields=fields, **kwargs)
            elif index_type == "geo":
                coll.add_geo_index(fields=fields, **kwargs)
            elif index_type == "fulltext":
                coll.add_fulltext_index(fields=fields, **kwargs)
            elif index_type == "hash":
                coll.add_hash_index(fields=fields, **kwargs)
            elif index_type == "skiplist":
                coll.add_skiplist_index(fields=fields, **kwargs)
            else:
                logger.warning(f"Unknown index type: {index_type}")
                return
                
            logger.info(f"Created {index_type} index on {collection}.{fields}")
        except Exception as e:
            # Index might already exist
            if "duplicate" not in str(e).lower():
                logger.error(f"Failed to create index: {e}")
    
    @staticmethod
    def verify_vector_support(db: StandardDatabase) -> bool:
        """
        Check if the database supports vector indexes.
        
        Args:
            db: Database connection
            
        Returns:
            True if vector indexes are supported
        """
        try:
            version_info = db.version()
            if isinstance(version_info, dict):
                version_str = version_info.get('server_version', '0.0.0')
            else:
                version_str = version_info
            
            major, minor = map(int, version_str.split('.')[:2])
            
            # Vector indexes require ArangoDB 3.11+
            if (major, minor) >= (3, 11):
                logger.info(f"ArangoDB {version_str} supports vector indexes")
                return True
            else:
                logger.warning(f"ArangoDB {version_str} does not support vector indexes")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify vector support: {e}")
            return False


class ArangoStorage:
    """
    Generic ArangoDB storage backend for document processing.
    
    Implements the WHERE dimension of the Conveyance Framework:
    - Stores documents in graph structure
    - Maintains relationships between documents, chunks, and embeddings
    - Provides atomic transaction support
    
    This class is source-agnostic and can be used by any tool
    that needs to store processed documents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ArangoDB storage.
        
        Args:
            config: Database configuration including:
                - host: ArangoDB host
                - port: ArangoDB port (default 8529)
                - database: Database name
                - username: Database username
                - password: Database password
                - collection_prefix: Prefix for collection names (default 'documents')
        """
        self.config = config
        self.collection_prefix = config.get('collection_prefix', 'documents')
        
        # Connect to database
        client = ArangoClient(hosts=f"http://{config['host']}:{config.get('port', 8529)}")
        self.db = client.db(
            config['database'],
            username=config.get('username', 'root'),
            password=config['password']
        )
        
        # Initialize collections
        self._init_collections()
        
        logger.info(f"ArangoStorage initialized with prefix '{self.collection_prefix}'")
    
    def _init_collections(self):
        """Initialize required collections."""
        # Document collection names
        self.papers_collection = f"{self.collection_prefix}_papers"
        self.chunks_collection = f"{self.collection_prefix}_chunks"
        self.embeddings_collection = f"{self.collection_prefix}_embeddings"
        self.structures_collection = f"{self.collection_prefix}_structures"
        
        # Ensure collections exist
        collections = [
            (self.papers_collection, False),
            (self.chunks_collection, False),
            (self.embeddings_collection, False),
            (self.structures_collection, False)
        ]
        
        for collection_name, is_edge in collections:
            if not self.db.has_collection(collection_name):
                self.db.create_collection(collection_name, edge=is_edge)
                logger.info(f"Created collection: {collection_name}")
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes for efficient querying."""
        # Papers collection indexes
        papers_coll = self.db.collection(self.papers_collection)
        try:
            papers_coll.add_persistent_index(fields=['document_id'], unique=True)
            papers_coll.add_persistent_index(fields=['status'])
            papers_coll.add_persistent_index(fields=['processing_timestamp'])
        except Exception as e:
            if "duplicate" not in str(e).lower():
                logger.error(f"Failed to create papers index: {e}")
        
        # Chunks collection indexes
        chunks_coll = self.db.collection(self.chunks_collection)
        try:
            chunks_coll.add_persistent_index(fields=['paper_id'])
            chunks_coll.add_persistent_index(fields=['chunk_index'])
            chunks_coll.add_fulltext_index(fields=['text'])
        except Exception as e:
            if "duplicate" not in str(e).lower():
                logger.error(f"Failed to create chunks index: {e}")
        
        # Embeddings collection indexes
        embeddings_coll = self.db.collection(self.embeddings_collection)
        try:
            embeddings_coll.add_persistent_index(fields=['paper_id'])
            embeddings_coll.add_persistent_index(fields=['chunk_id'])
        except Exception as e:
            if "duplicate" not in str(e).lower():
                logger.error(f"Failed to create embeddings index: {e}")
        
        # Structures collection indexes
        structures_coll = self.db.collection(self.structures_collection)
        try:
            structures_coll.add_persistent_index(fields=['paper_id'])
            structures_coll.add_persistent_index(fields=['type'])
        except Exception as e:
            if "duplicate" not in str(e).lower():
                logger.error(f"Failed to create structures index: {e}")
    
    def store_document(self, document_id: str, chunks: List[Dict[str, Any]], 
                      metadata: Dict[str, Any], extracted_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a processed document with its chunks and embeddings.
        
        Uses atomic transactions to ensure all-or-nothing storage.
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of document chunks with embeddings
            metadata: Document metadata
            extracted_content: Extracted content (equations, tables, images)
            
        Returns:
            Storage result with counts and status
        """
        try:
            # Start transaction
            transaction_db = self.db.begin_transaction(
                write=[
                    self.papers_collection,
                    self.chunks_collection,
                    self.embeddings_collection,
                    self.structures_collection
                ]
            )
            
            # Sanitize document ID for use as ArangoDB key
            safe_id = document_id.replace('/', '_').replace('.', '_')
            
            # Store paper document
            paper_doc = {
                '_key': safe_id,
                'document_id': document_id,
                'metadata': metadata,
                'status': 'PROCESSED',
                'processing_timestamp': datetime.now().isoformat(),
                'num_chunks': len(chunks),
                'num_equations': len(extracted_content.get('equations', [])),
                'num_tables': len(extracted_content.get('tables', [])),
                'num_images': len(extracted_content.get('images', []))
            }
            
            transaction_db.collection(self.papers_collection).insert(paper_doc)
            
            # Store chunks and embeddings
            for idx, chunk in enumerate(chunks):
                # Store chunk
                chunk_doc = {
                    'paper_id': safe_id,
                    'chunk_index': idx,
                    'text': chunk['text'],
                    'context_window_used': chunk.get('context_window_used', 0),
                    'metadata': chunk.get('metadata', {})
                }
                
                chunk_result = transaction_db.collection(self.chunks_collection).insert(chunk_doc)
                chunk_id = chunk_result['_key']
                
                # Store embedding
                embedding_doc = {
                    'paper_id': safe_id,
                    'chunk_id': chunk_id,
                    'chunk_index': idx,
                    'vector': chunk['embedding'],
                    'model': 'jina-v4',
                    'dimensions': len(chunk['embedding'])
                }
                
                transaction_db.collection(self.embeddings_collection).insert(embedding_doc)
            
            # Store structures (equations, tables, images)
            for eq_idx, equation in enumerate(extracted_content.get('equations', [])):
                eq_doc = {
                    'paper_id': safe_id,
                    'type': 'equation',
                    'index': eq_idx,
                    'content': equation.get('latex', ''),
                    'metadata': equation.get('metadata', {})
                }
                transaction_db.collection(self.structures_collection).insert(eq_doc)
            
            for table_idx, table in enumerate(extracted_content.get('tables', [])):
                table_doc = {
                    'paper_id': safe_id,
                    'type': 'table',
                    'index': table_idx,
                    'content': table.get('content', ''),
                    'metadata': table.get('metadata', {})
                }
                transaction_db.collection(self.structures_collection).insert(table_doc)
            
            for img_idx, image in enumerate(extracted_content.get('images', [])):
                img_doc = {
                    'paper_id': safe_id,
                    'type': 'image',
                    'index': img_idx,
                    'content': image.get('caption', ''),
                    'metadata': image.get('metadata', {})
                }
                transaction_db.collection(self.structures_collection).insert(img_doc)
            
            # Commit transaction
            transaction_db.commit_transaction()
            
            logger.info(f"Successfully stored document {document_id}: "
                       f"{len(chunks)} chunks, "
                       f"{paper_doc['num_equations']} equations, "
                       f"{paper_doc['num_tables']} tables, "
                       f"{paper_doc['num_images']} images")
            
            return {
                'success': True,
                'document_id': document_id,
                'num_chunks': len(chunks),
                'num_equations': paper_doc['num_equations'],
                'num_tables': paper_doc['num_tables'],
                'num_images': paper_doc['num_images']
            }
            
        except Exception as e:
            logger.error(f"Failed to store document {document_id}: {e}")
            try:
                transaction_db.abort_transaction()
            except:
                pass
            
            return {
                'success': False,
                'document_id': document_id,
                'error': str(e)
            }
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document and its associated data.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document data or None if not found
        """
        try:
            safe_id = document_id.replace('/', '_').replace('.', '_')
            
            # Get paper document
            papers_coll = self.db.collection(self.papers_collection)
            paper = papers_coll.get(safe_id)
            
            if not paper:
                return None
            
            # Get chunks
            chunks_coll = self.db.collection(self.chunks_collection)
            chunks_cursor = chunks_coll.find({'paper_id': safe_id})
            chunks = list(chunks_cursor)
            
            # Get embeddings
            embeddings_coll = self.db.collection(self.embeddings_collection)
            embeddings_cursor = embeddings_coll.find({'paper_id': safe_id})
            embeddings = list(embeddings_cursor)
            
            # Get structures
            structures_coll = self.db.collection(self.structures_collection)
            structures_cursor = structures_coll.find({'paper_id': safe_id})
            structures = list(structures_cursor)
            
            return {
                'paper': paper,
                'chunks': chunks,
                'embeddings': embeddings,
                'structures': structures
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve document {document_id}: {e}")
            return None
    
    def document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists in storage.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if document exists
        """
        try:
            safe_id = document_id.replace('/', '_').replace('.', '_')
            papers_coll = self.db.collection(self.papers_collection)
            return papers_coll.has(safe_id)
        except Exception as e:
            logger.error(f"Failed to check document existence: {e}")
            return False