#!/usr/bin/env python3
"""
Generate Jina v4 embeddings for the extracted word2vec evolution papers.
Saves embeddings locally for experiment analysis.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import Jina embedder
from core.framework.embedders import JinaV4Embedder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 8192, overlap: int = 512) -> List[str]:
    """
    Chunk text with overlap for context preservation.
    
    Args:
        text: Full text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= text_len:
            break
            
        # Move start with overlap
        start = end - overlap
    
    return chunks

def process_paper(paper_path: Path, embedder: JinaV4Embedder) -> Dict[str, Any]:
    """
    Process a single paper and generate embeddings.
    
    Args:
        paper_path: Path to extracted JSON
        embedder: Jina v4 embedder instance
        
    Returns:
        Dictionary with paper ID, chunks, and embeddings
    """
    logger.info(f"Processing {paper_path.name}")
    
    # Load extracted data
    with open(paper_path, 'r') as f:
        data = json.load(f)
    
    arxiv_id = data['arxiv_id']
    full_text = data['full_text']
    
    # Get metadata
    metadata = data.get('metadata', {})
    title = metadata.get('title', 'Unknown')
    
    logger.info(f"  Title: {title[:60]}...")
    logger.info(f"  Text length: {len(full_text):,} chars")
    
    # Chunk the text with late chunking approach (larger chunks)
    chunks = chunk_text(full_text, chunk_size=8192, overlap=512)
    logger.info(f"  Created {len(chunks)} chunks")
    
    # Generate embeddings
    logger.info("  Generating embeddings...")
    start_time = datetime.now()
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        # Add context for late chunking
        context = f"Paper: {title}\nChunk {i+1}/{len(chunks)}\n\n{chunk}"
        
        # Generate embedding (returns numpy array)
        embedding = embedder.embed_texts([context])[0]  # embed_texts expects list, returns array
        embeddings.append(embedding.tolist())  # Convert to list for JSON serialization
        
        if (i + 1) % 10 == 0:
            logger.info(f"    Processed {i+1}/{len(chunks)} chunks")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"  Embeddings generated in {elapsed:.1f}s")
    
    # Calculate paper-level embedding (mean of all chunks)
    paper_embedding = np.mean(embeddings, axis=0).tolist()
    
    return {
        'arxiv_id': arxiv_id,
        'title': title,
        'num_chunks': len(chunks),
        'chunks': chunks,
        'chunk_embeddings': embeddings,
        'paper_embedding': paper_embedding,
        'embedding_dim': len(embeddings[0]) if embeddings else 0,
        'processing_time': elapsed,
        'timestamp': datetime.now().isoformat()
    }

def main():
    # Setup paths
    extracted_dir = Path(__file__).parent / 'extracted_papers'
    embeddings_dir = Path(__file__).parent / 'embeddings'
    embeddings_dir.mkdir(exist_ok=True)
    
    # Our three papers
    papers = [
        '1301.3781.json',  # word2vec
        '1405.4053.json',  # doc2vec
        '1803.09473.json'  # code2vec
    ]
    
    logger.info("=" * 80)
    logger.info("GENERATING JINA V4 EMBEDDINGS")
    logger.info("=" * 80)
    
    # Initialize embedder
    logger.info("Initializing Jina v4 embedder...")
    embedder = JinaV4Embedder(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_fp16=True
    )
    logger.info(f"Using device: {embedder.device}")
    logger.info("Embedding dimensions: 2048 (Jina v4)")
    
    # Process each paper
    results = []
    total_start = datetime.now()
    
    for paper_file in papers:
        paper_path = extracted_dir / paper_file
        if not paper_path.exists():
            logger.error(f"File not found: {paper_path}")
            continue
        
        logger.info("-" * 80)
        result = process_paper(paper_path, embedder)
        results.append(result)
        
        # Save individual paper embeddings
        output_file = embeddings_dir / f"{result['arxiv_id']}_embeddings.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Saved to: {output_file}")
    
    # Save summary
    total_elapsed = (datetime.now() - total_start).total_seconds()
    
    summary = {
        'papers_processed': len(results),
        'total_chunks': sum(r['num_chunks'] for r in results),
        'embedding_dim': results[0]['embedding_dim'] if results else 0,
        'total_time': total_elapsed,
        'papers': [
            {
                'arxiv_id': r['arxiv_id'],
                'title': r['title'],
                'num_chunks': r['num_chunks']
            }
            for r in results
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = embeddings_dir / 'embeddings_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ EMBEDDING GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Papers processed: {summary['papers_processed']}")
    logger.info(f"Total chunks: {summary['total_chunks']}")
    logger.info(f"Embedding dimensions: {summary['embedding_dim']}")
    logger.info(f"Total time: {total_elapsed:.1f}s")
    logger.info(f"Summary saved to: {summary_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())