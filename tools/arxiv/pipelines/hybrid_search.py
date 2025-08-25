#!/usr/bin/env python3
"""
Hybrid Search Across Dual Embedding Collections
==============================================

Infrastructure for searching across both PDF and LaTeX embedding collections.
Implements unified search interface for the dual embedding strategy.

Following Actor-Network Theory: This system acts as a translator between
different document representations (PDF vs LaTeX), combining their strengths
to optimize conveyance C = (W·R·H)/T · Ctx^α.
"""

import os
import sys
import json
import yaml
import logging
import numpy as np
import psycopg2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from arango import ArangoClient

# Add HADES root to path
hades_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(hades_root))

from core.framework.embedders import JinaV4Embedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Single search result from either PDF or LaTeX embeddings.
    
    Following Information Reconstructionism: Each result represents
    a specific manifestation of information through one actant.
    """
    arxiv_id: str
    title: str
    similarity_score: float
    source_type: str  # 'pdf' or 'latex'
    chunk_text: str
    chunk_index: int
    total_chunks: int
    chunk_type: str  # 'abstract' or 'full_text'
    paper_categories: str
    
    def __post_init__(self):
        """Ensure similarity score is properly bounded."""
        self.similarity_score = max(0.0, min(1.0, float(self.similarity_score)))


@dataclass
class HybridSearchResult:
    """
    Combined search result merging PDF and LaTeX sources.
    
    This represents the enhanced information conveyance when both
    sources are available for a paper.
    """
    arxiv_id: str
    title: str
    best_similarity: float
    pdf_similarity: Optional[float]
    latex_similarity: Optional[float]
    combined_score: float  # Weighted combination
    source_types: List[str]  # ['pdf'] or ['latex'] or ['pdf', 'latex']
    best_chunk_text: str
    paper_categories: str
    has_both_sources: bool


class HybridSearchSystem:
    """
    Unified search interface across dual embedding collections.
    
    This system implements the infrastructure that makes our dual embedding
    strategy functional - it can query both PDF and LaTeX embeddings and
    intelligently combine results to maximize conveyance.
    """
    
    def __init__(self, config_path: str, pg_password: str, arango_password: str):
        """
        Initialize the hybrid search system.
        
        Args:
            config_path: Path to arxiv_hybrid.yaml config
            pg_password: PostgreSQL password
            arango_password: ArangoDB password
        """
        self.config_path = config_path
        self.pg_password = pg_password
        self.arango_password = arango_password
        
        # Load configuration
        self._load_config()
        
        # Initialize database connections
        self._init_postgresql()
        self._init_arangodb()
        
        # Initialize embedder for query processing
        embedder_config = hades_root / 'configs' / 'embedder.yaml'
        self.embedder = JinaV4Embedder(config_path=str(embedder_config))
        
        logger.info("HybridSearchSystem initialized")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract collection names
        collections = self.config['arangodb']['collections']
        self.pdf_collection = collections['pdf_embeddings']
        self.latex_collection = collections['latex_embeddings']
        
        logger.info(f"Using collections: PDF={self.pdf_collection}, LaTeX={self.latex_collection}")
    
    def _init_postgresql(self):
        """Initialize PostgreSQL connection for metadata."""
        pg_config = self.config['postgresql']
        self.pg_conn = psycopg2.connect(
            host=pg_config['host'],
            port=pg_config['port'],
            database=pg_config['database'],
            user=pg_config['user'],
            password=self.pg_password
        )
        self.pg_cur = self.pg_conn.cursor()
        logger.info("Connected to PostgreSQL for metadata")
    
    def _init_arangodb(self):
        """Initialize ArangoDB connection for embeddings."""
        arango_config = self.config['arangodb']
        client = ArangoClient(hosts=f"http://{arango_config['host']}:{arango_config['port']}")
        self.arango_db = client.db(
            arango_config['database'],
            username=arango_config['username'],
            password=self.arango_password
        )
        logger.info("Connected to ArangoDB for embeddings")
    
    def _get_paper_metadata(self, arxiv_ids: List[str]) -> Dict[str, Dict]:
        """
        Get metadata for papers from PostgreSQL.
        
        Args:
            arxiv_ids: List of ArXiv IDs to get metadata for
            
        Returns:
            Dictionary mapping arxiv_id to metadata
        """
        if not arxiv_ids:
            return {}
        
        # Create placeholders for IN clause
        placeholders = ','.join(['%s'] * len(arxiv_ids))
        query = f"""
            SELECT id, title, categories
            FROM arxiv_papers
            WHERE id IN ({placeholders})
        """
        
        self.pg_cur.execute(query, arxiv_ids)
        
        metadata = {}
        for row in self.pg_cur.fetchall():
            metadata[row[0]] = {
                'title': row[1],
                'categories': row[2] or ''
            }
        
        return metadata
    
    def _search_collection(self, query_embedding: np.ndarray, 
                          collection_name: str, limit: int = 10) -> List[SearchResult]:
        """
        Search a single embedding collection.
        
        Args:
            query_embedding: Query vector to search for
            collection_name: Name of ArangoDB collection to search
            limit: Maximum results to return
            
        Returns:
            List of SearchResult objects
        """
        # Convert numpy array to list for AQL
        query_vector = query_embedding.tolist()
        
        # AQL query for cosine similarity search
        aql_query = f"""
            FOR doc IN {collection_name}
            FOR chunk IN doc.chunk_embeddings || []
            LET similarity = COSINE_SIMILARITY(@query_vector, chunk.embedding)
            FILTER similarity > 0.3
            SORT similarity DESC
            LIMIT @limit
            RETURN {{
                arxiv_id: doc.arxiv_id,
                similarity: similarity,
                chunk_text: chunk.text,
                chunk_index: chunk.chunk_index,
                total_chunks: chunk.total_chunks,
                chunk_type: chunk.chunk_type || 'full_text'
            }}
        """
        
        try:
            cursor = self.arango_db.aql.execute(
                aql_query,
                bind_vars={
                    'query_vector': query_vector,
                    'limit': limit
                }
            )
            
            results = list(cursor)
            
            # Get metadata for all papers
            arxiv_ids = [r['arxiv_id'] for r in results]
            metadata = self._get_paper_metadata(arxiv_ids)
            
            # Convert to SearchResult objects
            search_results = []
            source_type = 'pdf' if 'pdf' in collection_name else 'latex'
            
            for result in results:
                arxiv_id = result['arxiv_id']
                paper_meta = metadata.get(arxiv_id, {})
                
                search_results.append(SearchResult(
                    arxiv_id=arxiv_id,
                    title=paper_meta.get('title', 'Unknown Title'),
                    similarity_score=result['similarity'],
                    source_type=source_type,
                    chunk_text=result['chunk_text'],
                    chunk_index=result['chunk_index'],
                    total_chunks=result['total_chunks'],
                    chunk_type=result['chunk_type'],
                    paper_categories=paper_meta.get('categories', '')
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching {collection_name}: {e}")
            return []
    
    def _combine_results(self, pdf_results: List[SearchResult], 
                        latex_results: List[SearchResult]) -> List[HybridSearchResult]:
        """
        Combine results from PDF and LaTeX searches intelligently.
        
        Args:
            pdf_results: Results from PDF embedding search
            latex_results: Results from LaTeX embedding search
            
        Returns:
            List of combined HybridSearchResult objects
        """
        # Group results by arxiv_id
        paper_results = {}
        
        # Process PDF results
        for result in pdf_results:
            arxiv_id = result.arxiv_id
            if arxiv_id not in paper_results:
                paper_results[arxiv_id] = {
                    'title': result.title,
                    'categories': result.paper_categories,
                    'pdf_result': None,
                    'latex_result': None
                }
            
            # Keep the best PDF result for this paper
            if (paper_results[arxiv_id]['pdf_result'] is None or 
                result.similarity_score > paper_results[arxiv_id]['pdf_result'].similarity_score):
                paper_results[arxiv_id]['pdf_result'] = result
        
        # Process LaTeX results
        for result in latex_results:
            arxiv_id = result.arxiv_id
            if arxiv_id not in paper_results:
                paper_results[arxiv_id] = {
                    'title': result.title,
                    'categories': result.paper_categories,
                    'pdf_result': None,
                    'latex_result': None
                }
            
            # Keep the best LaTeX result for this paper
            if (paper_results[arxiv_id]['latex_result'] is None or 
                result.similarity_score > paper_results[arxiv_id]['latex_result'].similarity_score):
                paper_results[arxiv_id]['latex_result'] = result
        
        # Create combined results
        combined_results = []
        
        for arxiv_id, data in paper_results.items():
            pdf_result = data['pdf_result']
            latex_result = data['latex_result']
            
            # Determine best similarity and source
            pdf_sim = pdf_result.similarity_score if pdf_result else 0.0
            latex_sim = latex_result.similarity_score if latex_result else 0.0
            
            best_similarity = max(pdf_sim, latex_sim)
            
            # Calculate combined score - use alignment metadata for better weighting
            if pdf_sim > 0 and latex_sim > 0:
                # Both sources available - use intelligent weighting based on alignment
                alignment_bonus = self._calculate_alignment_bonus(arxiv_id, pdf_result, latex_result)
                
                # Base weighted combination (LaTeX gets 60% weight for structured content)
                combined_score = 0.4 * pdf_sim + 0.6 * latex_sim
                
                # Apply alignment bonus to boost papers with good cross-source alignment
                combined_score = combined_score * (1.0 + alignment_bonus)
                
                source_types = ['pdf', 'latex']
                has_both = True
            elif latex_sim > 0:
                combined_score = latex_sim
                source_types = ['latex']
                has_both = False
            else:
                combined_score = pdf_sim
                source_types = ['pdf']
                has_both = False
            
            # Choose best chunk text
            if latex_result and latex_sim >= pdf_sim:
                best_chunk_text = latex_result.chunk_text
            elif pdf_result:
                best_chunk_text = pdf_result.chunk_text
            else:
                best_chunk_text = "No content available"
            
            combined_results.append(HybridSearchResult(
                arxiv_id=arxiv_id,
                title=data['title'],
                best_similarity=best_similarity,
                pdf_similarity=pdf_sim if pdf_sim > 0 else None,
                latex_similarity=latex_sim if latex_sim > 0 else None,
                combined_score=combined_score,
                source_types=source_types,
                best_chunk_text=best_chunk_text,
                paper_categories=data['categories'],
                has_both_sources=has_both
            ))
        
        # Sort by combined score (highest first)
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return combined_results
    
    def search(self, query: str, limit: int = 10, 
               search_mode: str = 'hybrid') -> List[HybridSearchResult]:
        """
        Perform hybrid search across both embedding collections.
        
        Args:
            query: Search query string
            limit: Maximum results to return
            search_mode: 'hybrid', 'pdf_only', or 'latex_only'
            
        Returns:
            List of HybridSearchResult objects
        """
        logger.info(f"Searching: '{query}' (mode: {search_mode}, limit: {limit})")
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_texts([query], task="retrieval")[0]
            
            pdf_results = []
            latex_results = []
            
            # Search collections based on mode
            if search_mode in ['hybrid', 'pdf_only']:
                logger.debug("Searching PDF embeddings...")
                pdf_results = self._search_collection(
                    query_embedding, self.pdf_collection, limit
                )
                logger.debug(f"Found {len(pdf_results)} PDF results")
            
            if search_mode in ['hybrid', 'latex_only']:
                logger.debug("Searching LaTeX embeddings...")
                latex_results = self._search_collection(
                    query_embedding, self.latex_collection, limit
                )
                logger.debug(f"Found {len(latex_results)} LaTeX results")
            
            # Combine results intelligently
            combined_results = self._combine_results(pdf_results, latex_results)
            
            # Limit to requested number
            final_results = combined_results[:limit]
            
            logger.info(f"Returning {len(final_results)} combined results")
            return final_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_similar_papers(self, arxiv_id: str, limit: int = 10) -> List[HybridSearchResult]:
        """
        Find papers similar to a given ArXiv paper.
        
        Args:
            arxiv_id: ArXiv ID of the reference paper
            limit: Maximum results to return
            
        Returns:
            List of similar papers
        """
        logger.info(f"Finding papers similar to {arxiv_id}")
        
        # Get the paper's title to use as query
        metadata = self._get_paper_metadata([arxiv_id])
        if arxiv_id not in metadata:
            logger.error(f"Paper {arxiv_id} not found in metadata")
            return []
        
        title = metadata[arxiv_id]['title']
        
        # Use title as search query
        results = self.search(title, limit + 5)  # Get more to filter out self
        
        # Filter out the original paper
        filtered_results = [r for r in results if r.arxiv_id != arxiv_id]
        
        return filtered_results[:limit]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding collections.
        
        Returns:
            Dictionary with collection statistics
        """
        stats = {}
        
        try:
            # PDF collection stats
            pdf_stats = list(self.arango_db.aql.execute(f"""
                FOR doc IN {self.pdf_collection}
                COLLECT WITH COUNT INTO total
                RETURN {{
                    total_papers: total,
                    total_chunks: SUM(
                        FOR doc IN {self.pdf_collection}
                        RETURN LENGTH(doc.chunk_embeddings || [])
                    )[0]
                }}
            """))
            
            if pdf_stats:
                stats['pdf_embeddings'] = pdf_stats[0]
            
            # LaTeX collection stats
            latex_stats = list(self.arango_db.aql.execute(f"""
                FOR doc IN {self.latex_collection}
                COLLECT WITH COUNT INTO total
                RETURN {{
                    total_papers: total,
                    total_chunks: SUM(
                        FOR doc IN {self.latex_collection}
                        RETURN LENGTH(doc.chunk_embeddings || [])
                    )[0]
                }}
            """))
            
            if latex_stats:
                stats['latex_embeddings'] = latex_stats[0]
            
            # Papers with both embeddings
            both_stats = list(self.arango_db.aql.execute(f"""
                LET pdf_papers = (FOR doc IN {self.pdf_collection} RETURN doc.arxiv_id)
                LET latex_papers = (FOR doc IN {self.latex_collection} RETURN doc.arxiv_id)
                RETURN {{
                    papers_with_both: LENGTH(INTERSECTION(pdf_papers, latex_papers))
                }}
            """))
            
            if both_stats:
                stats['dual_embeddings'] = both_stats[0]
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
        
        return stats
    
    def _calculate_alignment_bonus(self, arxiv_id: str, pdf_result: SearchResult, latex_result: SearchResult) -> float:
        """
        Calculate alignment bonus based on cross-source context hints.
        
        This uses the alignment metadata stored with embeddings to boost papers
        where PDF and LaTeX chunks show strong correspondence, helping overcome
        the context linking problem.
        
        Args:
            arxiv_id: ArXiv paper ID
            pdf_result: Best PDF search result for this paper
            latex_result: Best LaTeX search result for this paper
            
        Returns:
            Alignment bonus factor (0.0 to 0.3 typically)
        """
        try:
            # Retrieve alignment metadata from both collections
            pdf_doc = None
            latex_doc = None
            
            # Get PDF document with alignment metadata
            try:
                sanitized_id = arxiv_id.replace('/', '_').replace('.', '_')
                pdf_doc = self.arango_db.collection(self.pdf_collection).get(sanitized_id)
            except:
                pass
            
            # Get LaTeX document with alignment metadata
            try:
                latex_doc = self.arango_db.collection(self.latex_collection).get(sanitized_id)
            except:
                pass
            
            if not pdf_doc or not latex_doc:
                return 0.0  # No alignment metadata available
            
            # Extract alignment metadata
            pdf_alignment = pdf_doc.get('alignment_metadata', {})
            latex_alignment = latex_doc.get('alignment_metadata', {})
            
            if not pdf_alignment.get('has_both_sources'):
                return 0.0  # Not a dual-source paper
            
            bonus = 0.0
            
            # Bonus for alignment anchors (shared elements)
            anchors = pdf_alignment.get('alignment_anchors', [])
            if anchors:
                # Weight high-confidence anchors more
                anchor_bonus = 0.0
                for anchor in anchors:
                    confidence = anchor.get('confidence', 'low')
                    if confidence == 'high':
                        anchor_bonus += 0.05
                    elif confidence == 'medium':
                        anchor_bonus += 0.02
                
                bonus += min(anchor_bonus, 0.15)  # Cap at 15%
            
            # Bonus for chunk-level alignment hints
            pdf_chunk = None
            latex_chunk = None
            
            # Find the matching chunks from the search results
            if pdf_doc.get('chunk_embeddings'):
                for chunk in pdf_doc['chunk_embeddings']:
                    if chunk.get('chunk_index') == pdf_result.chunk_index:
                        pdf_chunk = chunk
                        break
            
            if latex_doc.get('chunk_embeddings'):
                for chunk in latex_doc['chunk_embeddings']:
                    if chunk.get('chunk_index') == latex_result.chunk_index:
                        latex_chunk = chunk
                        break
            
            # Compare alignment hints between matched chunks
            if pdf_chunk and latex_chunk:
                pdf_hints = pdf_chunk.get('alignment_hints', [])
                latex_hints = latex_chunk.get('alignment_hints', [])
                
                # Bonus for similar hint types in both chunks
                pdf_hint_types = {hint.get('type') for hint in pdf_hints}
                latex_hint_types = {hint.get('type') for hint in latex_hints}
                
                common_hints = pdf_hint_types.intersection(latex_hint_types)
                if common_hints:
                    # Bonus based on strength of shared hints
                    hint_bonus = 0.0
                    if 'section_header' in common_hints:
                        hint_bonus += 0.10  # Strong alignment
                    if 'contains_equations' in common_hints:
                        hint_bonus += 0.08  # Good alignment
                    if 'contains_tables' in common_hints:
                        hint_bonus += 0.05  # Moderate alignment
                    
                    bonus += min(hint_bonus, 0.15)  # Cap at 15%
            
            return min(bonus, 0.3)  # Cap total bonus at 30%
            
        except Exception as e:
            logger.debug(f"Alignment bonus calculation failed for {arxiv_id}: {e}")
            return 0.0  # Safe fallback


def main():
    """Command-line interface for hybrid search."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Search System')
    parser.add_argument('--config', required=True, help='Path to arxiv_hybrid.yaml')
    parser.add_argument('--pg-password', required=True, help='PostgreSQL password')
    parser.add_argument('--arango-password', required=True, help='ArangoDB password')
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--similar-to', help='Find papers similar to this ArXiv ID')
    parser.add_argument('--limit', type=int, default=10, help='Maximum results')
    parser.add_argument('--mode', choices=['hybrid', 'pdf_only', 'latex_only'], 
                       default='hybrid', help='Search mode')
    parser.add_argument('--stats', action='store_true', help='Show collection statistics')
    
    args = parser.parse_args()
    
    # Initialize search system
    search_system = HybridSearchSystem(
        config_path=args.config,
        pg_password=args.pg_password,
        arango_password=args.arango_password
    )
    
    if args.stats:
        # Show statistics
        stats = search_system.get_collection_stats()
        print("\n" + "="*60)
        print("COLLECTION STATISTICS")
        print("="*60)
        
        for collection, data in stats.items():
            print(f"\n{collection.upper()}:")
            for key, value in data.items():
                print(f"  {key}: {value:,}")
    
    elif args.similar_to:
        # Find similar papers
        results = search_system.search_similar_papers(args.similar_to, args.limit)
        
        print(f"\n" + "="*60)
        print(f"PAPERS SIMILAR TO {args.similar_to}")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title}")
            print(f"   ArXiv ID: {result.arxiv_id}")
            print(f"   Combined Score: {result.combined_score:.3f}")
            print(f"   Sources: {', '.join(result.source_types)}")
            if result.has_both_sources:
                print(f"   PDF: {result.pdf_similarity:.3f}, LaTeX: {result.latex_similarity:.3f}")
            print(f"   Categories: {result.paper_categories}")
    
    elif args.query:
        # Perform search
        results = search_system.search(args.query, args.limit, args.mode)
        
        print(f"\n" + "="*60)
        print(f"SEARCH RESULTS: '{args.query}'")
        print(f"Mode: {args.mode}, Found: {len(results)} results")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title}")
            print(f"   ArXiv ID: {result.arxiv_id}")
            print(f"   Combined Score: {result.combined_score:.3f}")
            print(f"   Sources: {', '.join(result.source_types)}")
            
            if result.has_both_sources:
                print(f"   Individual Scores - PDF: {result.pdf_similarity:.3f}, LaTeX: {result.latex_similarity:.3f}")
            
            # Show snippet of best matching text
            if result.best_chunk_text:
                snippet = result.best_chunk_text[:200]
                if len(result.best_chunk_text) > 200:
                    snippet += "..."
                print(f"   Text: {snippet}")
            else:
                print("   Text: [No text content available]")
            print(f"   Categories: {result.paper_categories}")
    
    else:
        print("Please provide either --query, --similar-to, or --stats")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()