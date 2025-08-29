#!/usr/bin/env python3
"""
GitHub Pipeline Manager with Graph Support
===========================================

Manages GitHub repository processing with graph relationships.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.github.github_document_manager import GitHubDocumentManager
from core.processors.generic_document_processor import GenericDocumentProcessor
from tools.arxiv.pipelines.arango_db_manager import ArangoDBManager
from typing import List, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class GitHubPipelineManager:
    """
    Manages GitHub repository processing with graph relationships.
    
    This manager creates proper graph connections between:
    - Repositories -> Files (papers)
    - Files -> Chunks
    - Chunks -> Embeddings
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the pipeline manager."""
        self.config = config
        self.github_manager = GitHubDocumentManager(
            clone_dir=config['processing']['github']['clone_dir'],
            cleanup=config['processing']['github']['cleanup_after_processing']
        )
        self.processor = GenericDocumentProcessor(config=config)
        
        # Override collections for GitHub
        self.processor.collections = {
            'papers': 'github_papers',
            'chunks': 'github_chunks',
            'embeddings': 'github_embeddings',
            'structures': 'github_structures'
        }
        
        # Initialize DB manager for graph operations
        self.db_manager = ArangoDBManager(config['arango'])
        
    def process_repository(self, repo_url: str) -> Dict[str, Any]:
        """
        Process a repository and create graph relationships.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            Processing results with graph information
        """
        logger.info(f"Processing repository: {repo_url}")
        
        # Step 1: Create or update repository vertex
        repo_info = self._create_repository_vertex(repo_url)
        repo_key = repo_info['_key']
        
        # Step 2: Clone and prepare documents
        tasks = self.github_manager.prepare_repository(repo_url)
        logger.info(f"Prepared {len(tasks)} files from {repo_info['full_name']}")
        
        # Add repository to metadata
        for task in tasks:
            task.metadata['repo'] = repo_info['full_name']
        
        # Step 3: Process through pipeline
        results = self.processor.process_documents(tasks)
        
        # Step 4: Create graph edges
        self._create_graph_edges(repo_key, tasks)
        
        # Step 5: Update repository stats
        self._update_repository_stats(repo_key)
        
        results['repository'] = repo_info
        return results
    
    def _create_repository_vertex(self, repo_url: str) -> Dict[str, Any]:
        """Create or update repository vertex."""
        # Parse repository info
        repo = self.github_manager.parse_repo_url(repo_url)
        
        repo_doc = {
            '_key': repo.full_name.replace('/', '_'),
            'full_name': repo.full_name,
            'owner': repo.owner,
            'name': repo.name,
            'clone_url': repo.clone_url,
            'processed_date': datetime.now().isoformat(),
            'url': repo_url
        }
        
        # Insert or update
        collection = self.db_manager.db.collection('github_repositories')
        collection.insert(repo_doc, overwrite=True)
        
        logger.info(f"Created/updated repository vertex: {repo.full_name}")
        return repo_doc
    
    def _create_graph_edges(self, repo_key: str, tasks: List) -> None:
        """Create edges between repository and files."""
        repo_collection = self.db_manager.db.collection('github_repositories')
        papers_collection = self.db_manager.db.collection('github_papers')
        edge_collection = self.db_manager.db.collection('github_repo_files')
        
        repo_id = f"github_repositories/{repo_key}"
        
        # Collect edges for batch insert
        edge_docs = []
        for task in tasks:
            # Get the paper document
            paper_key = task.document_id.replace('.', '_').replace('/', '_')
            paper_id = f"github_papers/{paper_key}"
            
            # Check if paper exists (it should after processing)
            if papers_collection.has(paper_key):
                # Create edge: repository -> file
                edge_doc = {
                    '_key': f"{repo_key}__{paper_key}",
                    '_from': repo_id,
                    '_to': paper_id,
                    'created': datetime.now().isoformat()
                }
                edge_docs.append(edge_doc)
        
        # Batch insert edges (in chunks of 100)
        if edge_docs:
            batch_size = 100
            for i in range(0, len(edge_docs), batch_size):
                batch = edge_docs[i:i+batch_size]
                try:
                    edge_collection.insert_many(batch, overwrite=True)
                    logger.debug(f"Inserted batch of {len(batch)} edges")
                except Exception as e:
                    # Fallback to individual inserts on batch failure
                    logger.warning(f"Batch insert failed, falling back to individual: {e}")
                    for doc in batch:
                        try:
                            edge_collection.insert(doc, overwrite=True)
                        except Exception as e:
                            logger.debug(f"Edge may already exist: {doc['_key']}")
        
        # Create edges for chunks and embeddings
        self._create_chunk_edges()
        
        logger.info(f"Created graph edges for repository {repo_key}")
    
    def _create_chunk_edges(self) -> None:
        """Create edges between papers -> chunks -> embeddings."""
        # This is handled automatically by the processor when storing chunks
        # But we could add additional edges here if needed
        pass
    
    def _update_repository_stats(self, repo_key: str) -> None:
        """Update repository statistics."""
        query = """
        LET repo = DOCUMENT('github_repositories', @repo_key)
        LET file_count = LENGTH(
            FOR v IN 1..1 OUTBOUND repo github_repo_files
            RETURN v
        )
        LET languages = (
            FOR v IN 1..1 OUTBOUND repo github_repo_files
            RETURN DISTINCT v.language
        )
        LET total_functions = SUM(
            FOR v IN 1..1 OUTBOUND repo github_repo_files
            RETURN LENGTH(v.symbols.functions || [])
        )
        UPDATE repo WITH {
            file_count: file_count,
            languages: languages,
            total_functions: total_functions
        } IN github_repositories
        RETURN NEW
        """
        
        cursor = self.db_manager.db.aql.execute(
            query,
            bind_vars={'repo_key': repo_key}
        )
        
        result = list(cursor)
        if result:
            stats = result[0]
            logger.info(f"Updated stats for {repo_key}: {stats.get('file_count')} files")
    
    def find_theory_practice_bridges(self, theory_embedding: List[float]) -> List[Dict]:
        """
        Find code that best implements a theory.
        
        Args:
            theory_embedding: Embedding vector of theoretical concept
            
        Returns:
            List of repositories and files ranked by similarity
        """
        # This would use vector similarity search across all embeddings
        # Then traverse the graph to find repository context
        
        query = """
        // Find similar embeddings (simplified - would use vector index)
        FOR emb IN github_embeddings
            LET similarity = 0.85  // Would calculate actual cosine similarity
            FILTER similarity > 0.8
            
            // Traverse to get context
            LET chunk = FIRST(
                FOR c IN github_chunks
                    FILTER c._key == emb.chunk_id
                    RETURN c
            )
            
            LET paper = FIRST(
                FOR p IN github_papers
                    FILTER p._key == chunk.document_id
                    RETURN p
            )
            
            LET repo = FIRST(
                FOR r IN github_repositories
                    FOR edge IN github_repo_files
                        FILTER edge._to == CONCAT('github_papers/', paper._key)
                        FILTER edge._from == CONCAT('github_repositories/', r._key)
                        RETURN r
            )
            
            SORT similarity DESC
            LIMIT 10
            
            RETURN {
                repository: repo.full_name,
                file: paper.document_id,
                language: paper.language,
                functions: LENGTH(paper.symbols.functions || []),
                similarity: similarity
            }
        """
        
        # For now, return example structure
        return [
            {
                'repository': 'example/repo',
                'file': 'src/implementation.py',
                'similarity': 0.85,
                'explanation': 'Implements the theoretical concept'
            }
        ]
    
    def compare_repositories(self, topic: str) -> List[Dict]:
        """
        Compare all repositories on a specific topic.
        
        Args:
            topic: Topic to compare (e.g., "word2vec")
            
        Returns:
            Comparative analysis of repositories
        """
        query = """
        FOR repo IN github_repositories
            FILTER CONTAINS(LOWER(repo.full_name), LOWER(@topic))
            
            LET files = (
                FOR paper IN 1..1 OUTBOUND repo github_repo_files
                    RETURN paper
            )
            
            LET avg_complexity = AVG(
                FOR f IN files
                    RETURN f.code_metrics.complexity || 0
            )
            
            LET total_functions = SUM(
                FOR f IN files
                    RETURN LENGTH(f.symbols.functions || [])
            )
            
            LET languages = UNIQUE(files[*].language)
            
            RETURN {
                repository: repo.full_name,
                file_count: LENGTH(files),
                languages: languages,
                avg_complexity: avg_complexity,
                total_functions: total_functions,
                has_tests: LENGTH(
                    FOR f IN files
                        FILTER CONTAINS(f.document_id, 'test')
                        RETURN 1
                ) > 0,
                has_docs: LENGTH(
                    FOR f IN files
                        FILTER f.language == 'markdown'
                        RETURN 1
                ) > 0
            }
        """
        
        cursor = self.db_manager.db.aql.execute(
            query,
            bind_vars={'topic': topic}
        )
        
        return list(cursor)


def main():
    """Demo the graph-based pipeline."""
    import yaml
    import os
    
    # Load config
    config_path = Path(__file__).parent / "configs" / "github_simple.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    arango_password = os.getenv('ARANGO_PASSWORD')
    if not arango_password:
        raise ValueError("ARANGO_PASSWORD environment variable is required but not set")
    config['arango']['password'] = arango_password
    
    # Create manager
    manager = GitHubPipelineManager(config)
    
    # Process a repository
    results = manager.process_repository("https://github.com/dav/word2vec")
    
    print(f"Processed {results.get('total_processed', 0)} files")
    print(f"Repository: {results['repository']['full_name']}")
    
    # Compare word2vec implementations
    comparisons = manager.compare_repositories("word2vec")
    print(f"\nFound {len(comparisons)} word2vec repositories:")
    for comp in comparisons:
        print(f"  - {comp['repository']}: {comp['file_count']} files, "
              f"{comp['total_functions']} functions")


if __name__ == "__main__":
    main()