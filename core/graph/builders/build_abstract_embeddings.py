#!/usr/bin/env python3
"""
Generate abstract embeddings for all ArXiv papers.
Critical for meaningful semantic similarity in the graph.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import click
from arango import ArangoClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AbstractEmbeddingBuilder:
    """Generate embeddings for paper abstracts."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', batch_size: int = 32):
        """
        Initialize embedding builder.
        
        Args:
            model_name: Sentence transformer model to use
            batch_size: Batch size for GPU processing
        """
        self.batch_size = batch_size
        
        # Connect to database
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Load embedding model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU (will be slower)")
    
    def generate_abstract_embeddings(self, limit: int = None, resume_from: str = None):
        """
        Generate embeddings for all paper abstracts.
        
        Args:
            limit: Maximum number of papers to process (for testing)
            resume_from: Paper ID to resume from (for interrupted runs)
        """
        logger.info("="*70)
        logger.info("ABSTRACT EMBEDDING GENERATION")
        logger.info("="*70)
        
        # Get papers with abstracts
        logger.info("Loading papers with abstracts...")
        
        if resume_from:
            query = f"""
            FOR p IN arxiv_papers
            FILTER p.abstract != null AND p.abstract != '' AND p._key > '{resume_from}'
            SORT p._key
            RETURN {{_key: p._key, abstract: p.abstract}}
            """
        else:
            query = """
            FOR p IN arxiv_papers
            FILTER p.abstract != null AND p.abstract != ''
            SORT p._key
            RETURN {_key: p._key, abstract: p.abstract}
            """
        
        if limit:
            query = query.replace("RETURN", f"LIMIT {limit} RETURN")
        
        papers = list(self.db.aql.execute(query))
        logger.info(f"Found {len(papers):,} papers to process")
        
        if len(papers) == 0:
            logger.info("No papers to process")
            return
        
        # Process in batches
        embeddings_coll = self.db.collection('arxiv_embeddings')
        total_processed = 0
        start_time = time.time()
        
        for batch_start in tqdm(range(0, len(papers), self.batch_size), desc="Processing batches"):
            batch_end = min(batch_start + self.batch_size, len(papers))
            batch = papers[batch_start:batch_end]
            
            # Extract abstracts and IDs
            abstracts = [p['abstract'] for p in batch]
            paper_ids = [p['_key'] for p in batch]
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model.encode(
                    abstracts,
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
            
            # Store embeddings
            docs_to_insert = []
            docs_to_update = []
            
            for paper_id, embedding in zip(paper_ids, embeddings):
                doc_key = f"{paper_id}_abstract"
                
                # Check if document exists
                existing = embeddings_coll.get(doc_key)
                
                if existing:
                    # Update existing document
                    docs_to_update.append({
                        '_key': doc_key,
                        'abstract_embedding': embedding.tolist(),
                        'embedding_model': self.model.get_sentence_embedding_dimension(),
                        'embedding_date': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                else:
                    # Create new document
                    docs_to_insert.append({
                        '_key': doc_key,
                        'paper_id': paper_id,
                        'abstract_embedding': embedding.tolist(),
                        'embedding_model': self.model.get_sentence_embedding_dimension(),
                        'embedding_date': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            # Batch insert/update
            if docs_to_insert:
                embeddings_coll.insert_many(docs_to_insert)
            if docs_to_update:
                for doc in docs_to_update:
                    embeddings_coll.update(doc)
            
            total_processed += len(batch)
            
            # Report progress
            if batch_start % (self.batch_size * 100) == 0 and batch_start > 0:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed
                remaining = (len(papers) - total_processed) / rate
                logger.info(f"Processed {total_processed:,} papers ({rate:.1f} papers/sec, {remaining/60:.1f} min remaining)")
            
            # Clear memory periodically
            if batch_start % (self.batch_size * 1000) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final report
        elapsed = time.time() - start_time
        logger.info("="*70)
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total papers processed: {total_processed:,}")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Average rate: {total_processed/elapsed:.1f} papers/sec")
    
    def verify_embeddings(self):
        """Verify that embeddings were created correctly."""
        # Count abstract embeddings
        query = """
        FOR e IN arxiv_embeddings
        FILTER e.abstract_embedding != null
        COLLECT WITH COUNT INTO c
        RETURN c
        """
        count = list(self.db.aql.execute(query))[0]
        logger.info(f"Abstract embeddings created: {count:,}")
        
        # Sample check
        query = """
        FOR e IN arxiv_embeddings
        FILTER e.abstract_embedding != null
        LIMIT 5
        RETURN {
            paper_id: e.paper_id,
            embedding_dims: LENGTH(e.abstract_embedding)
        }
        """
        samples = list(self.db.aql.execute(query))
        logger.info("Sample embeddings:")
        for s in samples:
            logger.info(f"  Paper {s['paper_id']}: {s['embedding_dims']} dimensions")


@click.command()
@click.option('--model', default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model to use')
@click.option('--batch-size', default=32, help='Batch size for GPU processing')
@click.option('--limit', default=None, type=int, help='Limit number of papers (for testing)')
@click.option('--resume-from', default=None, help='Resume from paper ID')
@click.option('--verify-only', is_flag=True, help='Only verify existing embeddings')
def main(model: str, batch_size: int, limit: int, resume_from: str, verify_only: bool):
    """Generate abstract embeddings for ArXiv papers."""
    builder = AbstractEmbeddingBuilder(model_name=model, batch_size=batch_size)
    
    if verify_only:
        builder.verify_embeddings()
    else:
        builder.generate_abstract_embeddings(limit=limit, resume_from=resume_from)
        builder.verify_embeddings()


if __name__ == "__main__":
    main()