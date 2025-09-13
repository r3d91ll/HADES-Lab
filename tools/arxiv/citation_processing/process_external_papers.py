#!/usr/bin/env python3
"""
Process non-ArXiv papers from /bulk-store/random_pdfs with their metadata.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from arango import ArangoClient
from core.framework.extractors import DoclingExtractor
from core.framework.embedders import JinaV4Embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExternalPaperProcessor:
    """Process external (non-ArXiv) papers with metadata."""

    def __init__(self, arango_password: str):
        """Initialize processor with database connection."""
        # Initialize ArangoDB
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db('academy_store', username='root', password=arango_password)

        # Ensure collections exist
        self._ensure_collections()

        # Initialize processors
        logger.info("Initializing DoclingExtractor...")
        self.extractor = DoclingExtractor(
            use_ocr=False,
            extract_tables=True,
            use_fallback=True
        )

        logger.info("Initializing JinaV4Embedder...")
        self.embedder = JinaV4Embedder(
            device='cuda',
            use_fp16=True,
            chunk_size_tokens=1000,
            chunk_overlap_tokens=200
        )

    def _ensure_collections(self):
        """Ensure required collections exist."""
        collections = ['external_papers', 'external_chunks', 'external_embeddings']
        for coll_name in collections:
            if not self.db.has_collection(coll_name):
                self.db.create_collection(coll_name)
                logger.info(f"Created collection: {coll_name}")

    def load_metadata(self, metadata_path: Path) -> Dict:
        """Load metadata from JSON file."""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            return None

    def process_paper(self, pdf_path: Path, metadata_path: Optional[Path] = None) -> bool:
        """Process a single paper with its metadata."""
        paper_id = pdf_path.stem
        logger.info(f"Processing: {paper_id}")

        # Load metadata if available
        metadata = {}
        if metadata_path and metadata_path.exists():
            metadata = self.load_metadata(metadata_path)
            if metadata:
                paper_id = metadata.get('paper_id', paper_id)

        # Check if already processed
        try:
            existing = self.db.collection('external_papers').get(paper_id.replace('.', '_').replace('-', '_'))
            if existing:
                logger.info(f"Paper {paper_id} already processed, skipping")
                return True
        except:
            pass  # Paper doesn't exist, continue processing

        try:
            # Extract text and structure
            logger.info(f"  Extracting content from {pdf_path.name}...")
            chunks = self.extractor.extract(str(pdf_path))

            if not chunks:
                logger.warning(f"  No content extracted from {paper_id}")
                return False

            logger.info(f"  Extracted {len(chunks)} chunks")

            # Generate embeddings
            logger.info(f"  Generating embeddings...")
            # Handle both string and dict chunks
            texts = []
            for chunk in chunks:
                if isinstance(chunk, str):
                    texts.append(chunk)
                elif isinstance(chunk, dict):
                    texts.append(chunk.get('text', ''))
                else:
                    texts.append(str(chunk))
            embeddings = self.embedder.embed_texts(texts)

            # Store paper metadata
            paper_doc = {
                '_key': paper_id.replace('.', '_').replace('-', '_'),
                'paper_id': paper_id,
                'source': metadata.get('source', 'external'),
                'pdf_path': str(pdf_path),
                'metadata': metadata.get('metadata', {}),
                'title': metadata.get('metadata', {}).get('title', paper_id),
                'authors': metadata.get('metadata', {}).get('authors', []),
                'year': metadata.get('metadata', {}).get('year'),
                'abstract': metadata.get('metadata', {}).get('abstract', ''),
                'num_chunks': len(chunks),
                'processing_date': datetime.now().isoformat(),
                'has_embeddings': True,
                'bibliography': metadata.get('bibliography', [])
            }

            self.db.collection('external_papers').insert(paper_doc, overwrite=True)

            # Store chunks and embeddings
            chunks_collection = self.db.collection('external_chunks')
            embeddings_collection = self.db.collection('external_embeddings')

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Store chunk
                if isinstance(chunk, str):
                    chunk_text = chunk
                    chunk_metadata = {}
                    chunk_page = None
                    chunk_type = 'text'
                elif isinstance(chunk, dict):
                    chunk_text = chunk.get('text', '')
                    chunk_metadata = chunk.get('metadata', {})
                    chunk_page = chunk.get('page')
                    chunk_type = chunk.get('type', 'text')
                else:
                    chunk_text = str(chunk)
                    chunk_metadata = {}
                    chunk_page = None
                    chunk_type = 'text'

                chunk_doc = {
                    '_key': f"{paper_id}_{i}".replace('.', '_').replace('-', '_'),
                    'paper_id': paper_id,
                    'chunk_index': i,
                    'text': chunk_text,
                    'metadata': chunk_metadata,
                    'page': chunk_page,
                    'type': chunk_type
                }
                chunks_collection.insert(chunk_doc, overwrite=True)

                # Store embedding
                embedding_doc = {
                    '_key': f"{paper_id}_{i}".replace('.', '_').replace('-', '_'),
                    'paper_id': paper_id,
                    'chunk_index': i,
                    'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    'model': 'jina-embeddings-v4',
                    'dimension': len(embedding)
                }
                embeddings_collection.insert(embedding_doc, overwrite=True)

            logger.info(f"  ✓ Successfully processed {paper_id}")
            return True

        except Exception as e:
            logger.error(f"  ✗ Failed to process {paper_id}: {e}")
            return False

    def process_directory(self, directory: Path):
        """Process all papers in a directory."""
        # Find all PDFs
        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        success_count = 0
        failed_papers = []

        for pdf_path in sorted(pdf_files):
            # Skip if it's a symlink or has .pdf.metadata.json pattern
            if pdf_path.is_symlink():
                continue

            # Look for corresponding metadata file
            metadata_path = pdf_path.with_suffix('.metadata.json')
            if not metadata_path.exists():
                # Try without .pdf extension
                metadata_path = directory / f"{pdf_path.stem}.metadata.json"

            if self.process_paper(pdf_path, metadata_path if metadata_path.exists() else None):
                success_count += 1
            else:
                failed_papers.append(pdf_path.name)

        # Report results
        logger.info("="*60)
        logger.info(f"Processing complete!")
        logger.info(f"  Success: {success_count}/{len(pdf_files)}")
        logger.info(f"  Failed: {len(failed_papers)}")

        if failed_papers:
            logger.info("Failed papers:")
            for paper in failed_papers:
                logger.info(f"  - {paper}")

        return success_count, failed_papers

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Process external papers')
    parser.add_argument('--directory', type=str,
                       default='/bulk-store/random_pdfs',
                       help='Directory containing PDFs and metadata')
    parser.add_argument('--arango-password', type=str,
                       default=os.environ.get('ARANGO_PASSWORD'),
                       help='ArangoDB password')

    args = parser.parse_args()

    if not args.arango_password:
        logger.error("ArangoDB password required (--arango-password or ARANGO_PASSWORD env)")
        sys.exit(1)

    directory = Path(args.directory)
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)

    # Process papers
    processor = ExternalPaperProcessor(args.arango_password)
    success_count, failed_papers = processor.process_directory(directory)

    # Check results in database
    logger.info("\n" + "="*60)
    logger.info("Database status:")

    cursor = processor.db.aql.execute('''
        FOR p IN external_papers
        COLLECT WITH COUNT INTO total
        RETURN total
    ''')
    total_papers = list(cursor)[0] if cursor else 0

    cursor = processor.db.aql.execute('''
        FOR e IN external_embeddings
        COLLECT paper_id = e.paper_id WITH COUNT INTO chunks
        COLLECT AGGREGATE total_papers = LENGTH(1)
        RETURN total_papers
    ''')
    papers_with_embeddings = list(cursor)[0] if cursor else 0

    logger.info(f"  Total external papers: {total_papers}")
    logger.info(f"  Papers with embeddings: {papers_with_embeddings}")

if __name__ == "__main__":
    main()
