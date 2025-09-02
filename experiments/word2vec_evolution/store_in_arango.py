#!/usr/bin/env python3
"""
Store the extracted papers and embeddings in ArangoDB.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging
from arango import ArangoClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_arango():
    """Connect to ArangoDB with correct credentials."""
    client = ArangoClient(hosts='http://192.168.1.69:8529')
    
    # Connect to system database
    sys_db = client.db('_system', username='root', password='1luv93ngu1n$')
    
    # Check if academy_store exists, create if not
    if not sys_db.has_database('academy_store'):
        logger.info("Creating academy_store database...")
        sys_db.create_database('academy_store')
    
    # Connect to academy_store
    db = client.db('academy_store', username='root', password='1luv93ngu1n$')
    
    # Ensure collections exist
    collections = ['arxiv_papers', 'arxiv_chunks', 'arxiv_embeddings', 'arxiv_structures']
    for coll_name in collections:
        if not db.has_collection(coll_name):
            logger.info(f"Creating collection: {coll_name}")
            db.create_collection(coll_name)
    
    return db

def store_paper(db, extracted_path: Path, embeddings_path: Path):
    """Store a single paper with its embeddings in ArangoDB."""
    
    # Load extracted data
    with open(extracted_path, 'r') as f:
        extracted = json.load(f)
    
    # Load embeddings
    with open(embeddings_path, 'r') as f:
        embeddings = json.load(f)
    
    arxiv_id = extracted['arxiv_id']
    logger.info(f"Storing {arxiv_id}...")
    
    # Prepare paper document
    paper_doc = {
        '_key': arxiv_id.replace('.', '_'),  # ArangoDB key format
        'arxiv_id': arxiv_id,
        'title': extracted['metadata'].get('title', 'Unknown'),
        'authors': extracted['metadata'].get('authors', []),
        'pdf_path': extracted['pdf_path'],
        'full_text_length': len(extracted['full_text']),
        'num_chunks': embeddings['num_chunks'],
        'processing_date': datetime.now().isoformat(),
        'status': 'PROCESSED',
        'embedding_dim': len(embeddings['paper_embedding']),
        'paper_embedding': embeddings['paper_embedding']  # Store paper-level embedding
    }
    
    # Store paper
    papers_coll = db.collection('arxiv_papers')
    try:
        papers_coll.insert(paper_doc)
        logger.info(f"  ✓ Paper document stored")
    except Exception as e:
        if 'duplicate' in str(e).lower():
            logger.info(f"  Paper already exists, updating...")
            papers_coll.update(paper_doc)
        else:
            raise
    
    # Store chunks and embeddings
    chunks_coll = db.collection('arxiv_chunks')
    embeddings_coll = db.collection('arxiv_embeddings')
    
    chunks = embeddings['chunks']
    chunk_embeddings = embeddings['chunk_embeddings']
    
    for i, (chunk_text, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings)):
        # Store chunk
        chunk_doc = {
            '_key': f"{arxiv_id.replace('.', '_')}_chunk_{i}",
            'paper_id': arxiv_id,
            'chunk_index': i,
            'text': chunk_text,
            'text_length': len(chunk_text),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            chunks_coll.insert(chunk_doc)
        except Exception as e:
            if 'duplicate' in str(e).lower():
                chunks_coll.update(chunk_doc)
        
        # Store embedding
        embedding_doc = {
            '_key': f"{arxiv_id.replace('.', '_')}_emb_{i}",
            'paper_id': arxiv_id,
            'chunk_id': f"{arxiv_id.replace('.', '_')}_chunk_{i}",
            'chunk_index': i,
            'vector': chunk_embedding,
            'model': 'jina-v4',
            'dimensions': len(chunk_embedding),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            embeddings_coll.insert(embedding_doc)
        except Exception as e:
            if 'duplicate' in str(e).lower():
                embeddings_coll.update(embedding_doc)
    
    logger.info(f"  ✓ Stored {len(chunks)} chunks with embeddings")
    
    # Store structures if present
    if 'structures' in extracted and extracted['structures']:
        structures_coll = db.collection('arxiv_structures')
        structures = extracted['structures']
        
        # Store tables
        for j, table in enumerate(structures.get('tables', [])):
            table_doc = {
                '_key': f"{arxiv_id.replace('.', '_')}_table_{j}",
                'paper_id': arxiv_id,
                'type': 'table',
                'index': j,
                'content': table,
                'timestamp': datetime.now().isoformat()
            }
            try:
                structures_coll.insert(table_doc)
            except Exception as e:
                if 'duplicate' in str(e).lower():
                    structures_coll.update(table_doc)
        
        if structures.get('tables'):
            logger.info(f"  ✓ Stored {len(structures['tables'])} tables")

def main():
    logger.info("=" * 80)
    logger.info("STORING PAPERS IN ARANGODB")
    logger.info("=" * 80)
    
    # Connect to ArangoDB
    logger.info("Connecting to ArangoDB...")
    db = connect_arango()
    logger.info("✓ Connected to academy_store database")
    
    # Process each paper
    extracted_dir = Path(__file__).parent / 'extracted_papers'
    embeddings_dir = Path(__file__).parent / 'embeddings'
    
    papers = [
        ('1301.3781', 'word2vec'),
        ('1405.4053', 'doc2vec'),
        ('1803.09473', 'code2vec')
    ]
    
    for arxiv_id, name in papers:
        logger.info("-" * 80)
        logger.info(f"Processing {name} ({arxiv_id})")
        
        extracted_path = extracted_dir / f"{arxiv_id}.json"
        embeddings_path = embeddings_dir / f"{arxiv_id}_embeddings.json"
        
        if not extracted_path.exists() or not embeddings_path.exists():
            logger.error(f"Missing files for {arxiv_id}")
            continue
        
        store_paper(db, extracted_path, embeddings_path)
    
    # Verify storage
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION")
    logger.info("=" * 80)
    
    papers_coll = db.collection('arxiv_papers')
    chunks_coll = db.collection('arxiv_chunks')
    embeddings_coll = db.collection('arxiv_embeddings')
    structures_coll = db.collection('arxiv_structures')
    
    logger.info(f"Papers stored: {papers_coll.count()}")
    logger.info(f"Chunks stored: {chunks_coll.count()}")
    logger.info(f"Embeddings stored: {embeddings_coll.count()}")
    logger.info(f"Structures stored: {structures_coll.count()}")
    
    # Show our specific papers
    logger.info("\nOur experiment papers:")
    for arxiv_id, name in papers:
        try:
            doc = papers_coll.get({'_key': arxiv_id.replace('.', '_')})
            if doc:
                logger.info(f"  • {doc['arxiv_id']}: {doc['title'][:60]}...")
                logger.info(f"    Chunks: {doc.get('num_chunks', 'N/A')}, Status: {doc.get('status', 'N/A')}")
        except:
            pass
    
    logger.info("\n✓ All papers successfully stored in ArangoDB!")
    return 0

if __name__ == "__main__":
    exit(main())