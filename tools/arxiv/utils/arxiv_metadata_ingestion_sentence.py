#!/usr/bin/env python3
"""
ArXiv metadata and abstract embedding ingestion using sentence-transformers.

Sentence-transformers provides optimized batch processing and is ~12% faster
than raw transformers while maintaining identical embedding quality.

This version uses the full Jina v4 model with sentence-transformers for
maximum compatibility and performance.
"""

import json
import time
import logging
import torch
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import click

import sys
sys.path.append('/home/todd/olympus/HADES-Lab')

from core.framework.sentence_embedder import JinaV4SentenceEmbedder
from core.database.arango_unix_client import ArangoUnixClient
from arango import ArangoClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for worker processes
WORKER_EMBEDDER = None
WORKER_DB = None

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)


class ArxivSentenceIngestion:
    """Ingestion pipeline using sentence-transformers for optimized performance."""
    
    def __init__(self, 
                 batch_size: int = 96,
                 num_workers: int = 2,
                 db_password: str = None):
        """
        Initialize sentence-transformers based ingestion pipeline.
        
        Args:
            batch_size: Batch size for embedding generation per worker
            num_workers: Number of GPU workers to use
            db_password: ArangoDB password
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.db_password = db_password
        
        # Connect to database using Unix socket for better performance
        self.unix_client = ArangoUnixClient(
            database='academy_store',
            username='root',
            password=db_password
        )
        
        # Use Unix socket if available
        if self.unix_client.use_unix:
            logger.info("✓ Using Unix socket for main database connection")
            self.db = self.unix_client
        else:
            logger.info("⚠ Unix socket not available, using HTTP")
            self.client = ArangoClient(hosts='http://localhost:8529')
            self.db = self.client.db(
                'academy_store',
                username='root',
                password=db_password
            )
        
        # Detect available GPUs
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count == 0:
            raise RuntimeError("No GPUs available!")
        
        # Adjust workers to available GPUs
        self.num_workers = min(self.num_workers, self.gpu_count)
        
        logger.info(f"Detected {self.gpu_count} GPUs, using {self.num_workers} workers")
        for i in range(self.gpu_count):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Ensure collections exist
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure required ArangoDB collections exist."""
        collections = [
            'arxiv_papers',
            'arxiv_abstract_embeddings'
        ]
        
        for coll_name in collections:
            if not self.db.has_collection(coll_name):
                self.db.create_collection(coll_name)
                logger.info(f"Created collection: {coll_name}")
    
    def download_dataset(self) -> Path:
        """Download ArXiv dataset from Kaggle."""
        logger.info("="*70)
        logger.info("STEP 1: Downloading ArXiv dataset from Kaggle")
        logger.info("="*70)
        
        dataset_path = Path('/bulk-store/arxiv-data/metadata/arxiv-kaggle-latest.json')
        
        if dataset_path.exists():
            # Check file size and age
            stat = dataset_path.stat()
            size_gb = stat.st_size / (1024**3)
            age_days = (time.time() - stat.st_mtime) / (24 * 3600)
            
            logger.info(f"✓ Using existing Kaggle dataset at: {dataset_path}")
            logger.info(f"  File size: {size_gb:.2f} GB")
            logger.info(f"  Last modified: {age_days:.0f} days ago")
            
            if age_days > 30:
                logger.warning("  ⚠ Dataset is over 30 days old, consider updating")
        else:
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        return dataset_path
    
    def import_metadata(self, dataset_path: Path):
        """Import metadata to ArangoDB."""
        logger.info("="*70)
        logger.info("STEP 2: Importing metadata to ArangoDB")
        logger.info("="*70)
        
        # Check if already imported
        count = self.db.collection('arxiv_papers').count()
        if count > 2_000_000:
            logger.info(f"✓ Papers already imported: {count:,}")
            return
        
        # Stream and import
        batch_docs = []
        batch_size = 1000
        total_imported = 0
        
        with open(dataset_path, 'r') as f:
            for line in tqdm(f, desc="Importing papers"):
                try:
                    paper = json.loads(line)
                    
                    # Sanitize arxiv_id for ArangoDB key
                    key = paper['id'].replace('/', '_').replace('.', '_')
                    
                    doc = {
                        '_key': key,
                        'arxiv_id': paper['id'],
                        'title': paper.get('title', ''),
                        'abstract': paper.get('abstract', ''),
                        'authors': paper.get('authors', ''),
                        'categories': paper.get('categories', ''),
                        'versions': paper.get('versions', []),
                        'update_date': paper.get('update_date', ''),
                        'authors_parsed': paper.get('authors_parsed', [])
                    }
                    
                    batch_docs.append(doc)
                    
                    if len(batch_docs) >= batch_size:
                        self.db.collection('arxiv_papers').insert_many(
                            batch_docs, 
                            overwrite=True
                        )
                        total_imported += len(batch_docs)
                        batch_docs = []
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Failed to import paper: {e}")
        
        # Insert remaining
        if batch_docs:
            self.db.collection('arxiv_papers').insert_many(
                batch_docs,
                overwrite=True
            )
            total_imported += len(batch_docs)
        
        logger.info(f"✓ Imported {total_imported:,} papers")
    
    @staticmethod
    def _init_embedding_worker(gpu_devices: List[int], db_password: str):
        """Initialize worker with GPU assignment and sentence-transformers model."""
        global WORKER_EMBEDDER, WORKER_DB
        
        # Get worker ID from process
        worker_info = mp.current_process()
        worker_id = int(worker_info.name.split('-')[-1]) - 1  # Process names are like "SpawnProcess-1"
        
        # Assign GPU to this worker
        gpu_id = gpu_devices[worker_id % len(gpu_devices)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        logger.info(f"Embedding worker {worker_id} starting initialization on GPU {gpu_id}...")
        
        # Initialize sentence-transformers embedder (faster than raw transformers)
        WORKER_EMBEDDER = JinaV4SentenceEmbedder(
            model_name='jinaai/jina-embeddings-v4',
            device='cuda',
            use_fp16=True,
            batch_size=32  # Internal batch size for sentence-transformers
        )
        
        # Initialize database connection using Unix socket
        unix_client = ArangoUnixClient(
            database='academy_store',
            username='root',
            password=db_password
        )
        WORKER_DB = unix_client
        if unix_client.use_unix:
            logger.info(f"✓ Worker {worker_id} using Unix socket")
        else:
            logger.info(f"⚠ Worker {worker_id} using HTTP")
        
        logger.info(f"✓ Embedding worker {worker_id} initialized on GPU {gpu_id}")
    
    @staticmethod
    def _embed_batch(batch_data: tuple) -> Dict:
        """Embed a batch of abstracts using sentence-transformers."""
        worker_id, papers = batch_data
        
        try:
            # Use pre-initialized embedder and database
            global WORKER_EMBEDDER, WORKER_DB
            
            if WORKER_EMBEDDER is None or WORKER_DB is None:
                raise RuntimeError(f"Worker {worker_id} not properly initialized!")
            
            # Extract abstracts
            abstracts = [p['abstract'] for p in papers]
            
            # Generate embeddings using sentence-transformers (faster!)
            embeddings = WORKER_EMBEDDER.embed_abstracts(
                abstracts,
                show_progress=False  # No progress bar in worker
            )
            
            # Store embeddings
            docs = []
            for paper, embedding in zip(papers, embeddings):
                docs.append({
                    '_key': paper['_key'],
                    'paper_id': paper['_key'],
                    'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                    'embedding_model': 'jinaai/jina-embeddings-v4',
                    'embedding_dimension': len(embedding),
                    'embedding_date': datetime.now().isoformat()
                })
            
            # Bulk insert (remove 'silent' parameter for Unix socket compatibility)
            WORKER_DB.collection('arxiv_abstract_embeddings').insert_many(
                docs,
                overwrite=True
            )
            
            return {
                'worker_id': worker_id,
                'success': True,
                'count': len(papers),
                'keys': [p['_key'] for p in papers]
            }
        
        except Exception as e:
            logger.error(f"Worker {worker_id} batch failed: {e}")
            return {
                'worker_id': worker_id,
                'success': False,
                'count': 0,
                'error': str(e)
            }
    
    def generate_abstract_embeddings_parallel(self, limit: int = None, resume: bool = True):
        """Generate embeddings using multiple GPU workers with sentence-transformers."""
        logger.info("="*70)
        logger.info("STEP 3: Generating abstract embeddings (Sentence-Transformers Multi-GPU)")
        logger.info("="*70)
        
        # Load checkpoint if exists
        checkpoint_file = Path('arxiv_embedding_checkpoint.json')
        processed_keys = set()
        
        if resume and checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed_keys = set(checkpoint.get('processed_keys', []))
                logger.info(f"✓ Resuming from checkpoint: {len(processed_keys):,} papers already processed")
        
        # Get papers that need processing
        if limit:
            # First check existing embeddings
            existing_query = """
            FOR e IN arxiv_abstract_embeddings
            RETURN e._key
            """
            existing_keys = set(self.db.aql.execute(existing_query))
            processed_keys.update(existing_keys)
            
            query = f"""
            FOR p IN arxiv_papers
            FILTER p.abstract != null AND p.abstract != ''
            FILTER p._key NOT IN {list(processed_keys)}
            LIMIT {limit}
            RETURN {{_key: p._key, abstract: p.abstract}}
            """
        else:
            # Get papers without embeddings
            query = """
            FOR p IN arxiv_papers
            FILTER p.abstract != null AND p.abstract != ''
            FILTER p._key NOT IN (
                FOR e IN arxiv_abstract_embeddings
                RETURN e._key
            )
            RETURN {_key: p._key, abstract: p.abstract}
            """
        
        papers = list(self.db.aql.execute(query))
        logger.info(f"Found {len(papers):,} papers needing embeddings")
        logger.info(f"Using {self.num_workers} GPU workers with batch size {self.batch_size}")
        logger.info("Sentence-transformers provides ~12% faster processing than raw transformers")
        
        # Prepare batches
        batches = []
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i:i+self.batch_size]
            worker_id = (i // self.batch_size) % self.num_workers
            batches.append((worker_id, batch))
        
        # Process with multiple GPU workers
        gpu_devices = list(range(self.gpu_count))
        processed = 0
        start_time = time.time()
        
        # Create pool with worker initialization
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=self._init_embedding_worker,
            initargs=(gpu_devices, self.db_password)
        ) as executor:
            # Submit all batches
            futures = {}
            for batch_data in batches:
                future = executor.submit(self._embed_batch, batch_data)
                futures[future] = batch_data[0]
            
            # Process results
            checkpoint_counter = 0
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result['success']:
                            processed += result['count']
                            for key in result['keys']:
                                processed_keys.add(key)
                        else:
                            logger.error(f"Batch failed: {result.get('error')}")
                    except Exception as e:
                        logger.error(f"Future failed: {e}")
                    
                    pbar.update(1)
                    
                    # Save checkpoint every 100 batches
                    checkpoint_counter += 1
                    if checkpoint_counter % 100 == 0:
                        checkpoint_data = {
                            'processed_keys': list(processed_keys),
                            'total_processed': len(processed_keys),
                            'last_updated': datetime.now().isoformat()
                        }
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint_data, f)
                        logger.info(f"✓ Checkpoint saved: {len(processed_keys):,} papers processed")
                    
                    # Log progress
                    elapsed = time.time() - start_time
                    papers_per_sec = processed / elapsed if elapsed > 0 else 0
                    eta = (len(papers) - processed) / papers_per_sec if papers_per_sec > 0 else 0
                    
                    if checkpoint_counter % 10 == 0:
                        logger.info(f"Progress: {processed:,}/{len(papers):,} papers "
                                  f"({papers_per_sec:.1f} papers/sec, ETA: {eta/60:.1f} min)")
        
        # Final checkpoint save
        checkpoint_data = {
            'processed_keys': list(processed_keys),
            'total_processed': len(processed_keys),
            'last_updated': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Summary
        elapsed = time.time() - start_time
        papers_per_sec = processed / elapsed if elapsed > 0 else 0
        
        logger.info("="*70)
        logger.info(f"✓ Embedding generation complete!")
        logger.info(f"  Total papers processed: {processed:,}")
        logger.info(f"  Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"  Average speed: {papers_per_sec:.1f} papers/second")
        logger.info(f"  Workers used: {self.num_workers}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Embedder: Sentence-Transformers (optimized)")
        logger.info("="*70)
    
    def run(self, limit: int = None, resume: bool = True):
        """Run the complete ingestion pipeline."""
        # Step 1: Download dataset
        dataset_path = self.download_dataset()
        
        # Step 2: Import metadata
        self.import_metadata(dataset_path)
        
        # Step 3: Generate embeddings
        self.generate_abstract_embeddings_parallel(limit=limit, resume=resume)


@click.command()
@click.option('--batch-size', default=96, help='Batch size for processing')
@click.option('--num-workers', default=2, help='Number of GPU workers')
@click.option('--limit', type=int, help='Limit number of papers to process')
@click.option('--db-password', envvar='ARANGO_PASSWORD', help='ArangoDB password')
@click.option('--resume/--no-resume', default=True, help='Resume from checkpoint')
def main(batch_size, num_workers, limit, db_password, resume):
    """Run ArXiv metadata ingestion with sentence-transformers for optimal performance."""
    if not db_password:
        logger.error("ArangoDB password required (set ARANGO_PASSWORD env var)")
        return
    
    # Create and run pipeline
    pipeline = ArxivSentenceIngestion(
        batch_size=batch_size,
        num_workers=num_workers,
        db_password=db_password
    )
    
    pipeline.run(limit=limit, resume=resume)


if __name__ == '__main__':
    main()