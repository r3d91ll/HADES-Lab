#!/usr/bin/env python3
"""
Universal Similarity Edge Builder

A flexible module that can build similarity edges from ANY embedding type.
FAISS doesn't care what the embeddings represent - just needs vectors.

This replaces separate keyword/abstract builders with one configurable builder.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import faiss
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from arango import ArangoClient
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UniversalSimilarityBuilder:
    """
    Build similarity edges from any embedding type.
    
    Can handle:
    - Keyword embeddings
    - Abstract embeddings
    - Title embeddings
    - Full text embeddings
    - Any future embedding type
    """
    
    def __init__(self, 
                 embedding_field: str,
                 edge_collection: str,
                 threshold: float = 0.7,
                 batch_size: int = 50000,
                 use_gpu: bool = True,
                 top_k: int = 50):
        """
        Initialize universal similarity builder.
        
        Args:
            embedding_field: Field name in arxiv_embeddings collection 
                           (e.g., 'keyword_embedding', 'abstract_embedding')
            edge_collection: Name for the edge collection to create
                           (e.g., 'keyword_similarity', 'abstract_similarity')
            threshold: Similarity threshold for creating edges
            batch_size: Papers to process in each batch
            use_gpu: Whether to use GPU acceleration
            top_k: Maximum edges per node (prevent super-nodes)
        """
        self.embedding_field = embedding_field
        self.edge_collection = edge_collection
        self.threshold = threshold
        self.batch_size = batch_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.top_k = top_k
        
        # Connect to database
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Ensure edge collection exists
        if not self.db.has_collection(edge_collection):
            self.db.create_edge_collection(edge_collection)
            logger.info(f"Created edge collection: {edge_collection}")
        
        self.stats = {
            'papers_processed': 0,
            'edges_created': 0,
            'start_time': time.time(),
            'embedding_field': embedding_field,
            'edge_collection': edge_collection
        }
    
    def load_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Load all embeddings of the specified type."""
        logger.info(f"Loading {self.embedding_field} embeddings...")
        
        # Determine collection name based on embedding field
        if 'keyword' in self.embedding_field:
            collection_name = 'arxiv_keyword_embeddings'
        elif 'abstract' in self.embedding_field:
            collection_name = 'arxiv_abstract_embeddings'
        else:
            # Generic case - assume collection is named after the field
            collection_name = f"arxiv_{self.embedding_field}s"
        
        # Get all embeddings from the appropriate collection
        query = f"""
        FOR e IN {collection_name}
        RETURN {{
            paper_id: e.paper_id,
            embedding: e.embedding
        }}
        """
        
        paper_ids = []
        embeddings = []
        
        for doc in tqdm(self.db.aql.execute(query), desc="Loading embeddings"):
            paper_ids.append(doc['paper_id'])
            embeddings.append(doc['embedding'])
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        logger.info(f"Loaded {len(embeddings):,} embeddings from {collection_name}")
        logger.info(f"Embedding dimensions: {embeddings.shape}")
        
        return paper_ids, embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for similarity search."""
        logger.info("Building FAISS index...")
        
        d = embeddings.shape[1]  # Dimension
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        if self.use_gpu:
            logger.info("Using GPU for FAISS")
            # Build GPU index
            res = faiss.StandardGpuResources()
            
            # Use IVF index for large datasets
            if len(embeddings) > 100000:
                nlist = int(np.sqrt(len(embeddings)))  # Number of clusters
                quantizer = faiss.IndexFlatIP(d)  # Inner product = cosine after normalization
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                
                # Move to GPU
                index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
                
                # Train index
                logger.info(f"Training IVF index with {nlist} clusters...")
                index_gpu.train(embeddings)
                index_gpu.add(embeddings)
            else:
                # For smaller datasets, use flat index
                index = faiss.IndexFlatIP(d)
                index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
                index_gpu.add(embeddings)
        else:
            logger.info("Using CPU for FAISS")
            # Build CPU index
            if len(embeddings) > 100000:
                # Use IVF with PQ for very large datasets on CPU
                nlist = int(np.sqrt(len(embeddings)))
                m = 8  # Number of subquantizers
                quantizer = faiss.IndexFlatIP(d)
                index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
                
                logger.info(f"Training IVF-PQ index with {nlist} clusters...")
                index.train(embeddings)
                index.add(embeddings)
            else:
                # Simple flat index for smaller datasets
                index = faiss.IndexFlatIP(d)
                index.add(embeddings)
        
        logger.info(f"FAISS index built with {index.ntotal:,} vectors")
        return index if not self.use_gpu else index_gpu
    
    def find_similar_papers(self, index: faiss.Index, 
                           embeddings: np.ndarray,
                           batch_indices: List[int]) -> List[Tuple[int, int, float]]:
        """Find similar papers for a batch using FAISS."""
        batch_embeddings = embeddings[batch_indices]
        
        # Normalize for cosine similarity
        faiss.normalize_L2(batch_embeddings)
        
        # Search for top-k similar papers
        # We search for top_k + 1 because the paper itself will be included
        similarities, indices = index.search(batch_embeddings, self.top_k + 1)
        
        edges = []
        for i, batch_idx in enumerate(batch_indices):
            for j, (neighbor_idx, similarity) in enumerate(zip(indices[i], similarities[i])):
                # Skip self-connections and below-threshold similarities
                if neighbor_idx != batch_idx and similarity >= self.threshold:
                    # Store as (from, to, weight)
                    # Use min/max to ensure consistent edge direction
                    from_idx = min(batch_idx, neighbor_idx)
                    to_idx = max(batch_idx, neighbor_idx)
                    edges.append((from_idx, to_idx, float(similarity)))
        
        return edges
    
    def build_edges_faiss(self) -> int:
        """Build similarity edges using FAISS."""
        logger.info("="*70)
        logger.info(f"Building {self.edge_collection} edges using {self.embedding_field}")
        logger.info("="*70)
        
        # Load embeddings
        paper_ids, embeddings = self.load_embeddings()
        
        if len(embeddings) == 0:
            logger.warning(f"No {self.embedding_field} embeddings found!")
            return 0
        
        # Build FAISS index
        index = self.build_faiss_index(embeddings)
        
        # Process in batches
        logger.info(f"Finding similar papers (threshold={self.threshold}, top_k={self.top_k})...")
        
        edge_collection = self.db.collection(self.edge_collection)
        total_edges = 0
        edge_set = set()  # Track unique edges
        
        for batch_start in tqdm(range(0, len(embeddings), self.batch_size), 
                               desc="Processing batches"):
            batch_end = min(batch_start + self.batch_size, len(embeddings))
            batch_indices = list(range(batch_start, batch_end))
            
            # Find similar papers
            edges = self.find_similar_papers(index, embeddings, batch_indices)
            
            # Convert to ArangoDB edge documents
            edge_docs = []
            for from_idx, to_idx, weight in edges:
                # Create unique edge key
                edge_key = (from_idx, to_idx)
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edge_docs.append({
                        '_from': f"arxiv_papers/{paper_ids[from_idx]}",
                        '_to': f"arxiv_papers/{paper_ids[to_idx]}",
                        'weight': weight,
                        'similarity_type': self.embedding_field.replace('_embedding', ''),
                        'created_at': time.time()
                    })
            
            # Batch insert
            if edge_docs:
                edge_collection.insert_many(edge_docs)
                total_edges += len(edge_docs)
            
            # Clear memory periodically
            if batch_start % (self.batch_size * 10) == 0:
                gc.collect()
                if self.use_gpu:
                    torch.cuda.empty_cache()
        
        self.stats['edges_created'] = total_edges
        self.stats['papers_processed'] = len(embeddings)
        self.stats['duration_seconds'] = time.time() - self.stats['start_time']
        
        logger.info("="*70)
        logger.info("EDGE BUILDING COMPLETE")
        logger.info("="*70)
        logger.info(f"Papers processed: {self.stats['papers_processed']:,}")
        logger.info(f"Edges created: {self.stats['edges_created']:,}")
        logger.info(f"Time: {self.stats['duration_seconds']/60:.1f} minutes")
        logger.info(f"Rate: {self.stats['edges_created']/self.stats['duration_seconds']:.1f} edges/sec")
        
        return total_edges
    
    def build_edges_gpu_direct(self) -> int:
        """Build edges using direct GPU computation (for smaller datasets)."""
        logger.info("="*70)
        logger.info(f"Building {self.edge_collection} edges using direct GPU")
        logger.info("="*70)
        
        if not torch.cuda.is_available():
            logger.warning("GPU not available, falling back to FAISS")
            return self.build_edges_faiss()
        
        # Load embeddings
        paper_ids, embeddings = self.load_embeddings()
        
        if len(embeddings) == 0:
            logger.warning(f"No {self.embedding_field} embeddings found!")
            return 0
        
        # Check if we can fit in GPU memory
        memory_needed = embeddings.shape[0] * embeddings.shape[1] * 4 * 2  # float32, need 2x for similarity matrix
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        
        if memory_needed > gpu_memory * 0.8:  # Use 80% of GPU memory max
            logger.info("Dataset too large for direct GPU, using FAISS")
            return self.build_edges_faiss()
        
        logger.info(f"Using direct GPU computation for {len(embeddings):,} embeddings")
        
        # Move to GPU
        embeddings_gpu = torch.from_numpy(embeddings).cuda()
        
        # Normalize for cosine similarity
        embeddings_gpu = embeddings_gpu / embeddings_gpu.norm(dim=1, keepdim=True)
        
        # Compute similarity matrix
        logger.info("Computing similarity matrix...")
        similarities = torch.mm(embeddings_gpu, embeddings_gpu.t())
        
        # Apply threshold and extract edges
        logger.info(f"Extracting edges (threshold={self.threshold})...")
        edge_collection = self.db.collection(self.edge_collection)
        
        edges = []
        for i in tqdm(range(len(similarities)), desc="Processing similarities"):
            # Get top-k similar papers for this paper
            row_similarities, indices = torch.topk(similarities[i], min(self.top_k + 1, len(similarities)))
            
            for j, sim in zip(indices.cpu().numpy(), row_similarities.cpu().numpy()):
                if j != i and sim >= self.threshold:
                    from_idx = min(i, j)
                    to_idx = max(i, j)
                    edges.append({
                        '_from': f"arxiv_papers/{paper_ids[from_idx]}",
                        '_to': f"arxiv_papers/{paper_ids[to_idx]}",
                        'weight': float(sim),
                        'similarity_type': self.embedding_field.replace('_embedding', ''),
                        'created_at': time.time()
                    })
            
            # Batch insert
            if len(edges) >= 10000:
                edge_collection.insert_many(edges)
                edges = []
        
        # Insert remaining edges
        if edges:
            edge_collection.insert_many(edges)
        
        total_edges = edge_collection.count()
        
        # Clean up GPU memory
        del embeddings_gpu
        del similarities
        torch.cuda.empty_cache()
        
        self.stats['edges_created'] = total_edges
        self.stats['papers_processed'] = len(embeddings)
        self.stats['duration_seconds'] = time.time() - self.stats['start_time']
        
        logger.info(f"Created {total_edges:,} edges")
        return total_edges


def build_keyword_similarity():
    """Build keyword similarity edges."""
    builder = UniversalSimilarityBuilder(
        embedding_field='keyword',  # Will use arxiv_keyword_embeddings collection
        edge_collection='keyword_similarity',
        threshold=0.65,
        use_gpu=True
    )
    return builder.build_edges_faiss()


def build_abstract_similarity():
    """Build abstract similarity edges."""
    builder = UniversalSimilarityBuilder(
        embedding_field='abstract',  # Will use arxiv_abstract_embeddings collection
        edge_collection='abstract_similarity',
        threshold=0.75,
        use_gpu=True
    )
    return builder.build_edges_faiss()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        embedding_type = sys.argv[1]
        if embedding_type == 'keyword':
            build_keyword_similarity()
        elif embedding_type == 'abstract':
            build_abstract_similarity()
        else:
            print(f"Unknown embedding type: {embedding_type}")
            print("Usage: python build_similarity_edges_universal.py [keyword|abstract]")
    else:
        print("Usage: python build_similarity_edges_universal.py [keyword|abstract]")
        print("\nOr use the universal builder directly:")
        print("  builder = UniversalSimilarityBuilder(")
        print("      embedding_field='your_embedding_field',")
        print("      edge_collection='your_edge_collection',")
        print("      threshold=0.7")
        print("  )")
        print("  builder.build_edges_faiss()")