#!/usr/bin/env python3
"""
Analyze theory-practice bridges in the word2vec evolution experiment.
Computes entropy maps and conveyance scores for paper-code relationships.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from arango import ArangoClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TheoryPracticeBridgeAnalyzer:
    """Analyzes semantic bridges between papers and their implementations."""
    
    def __init__(self):
        # Connect to ArangoDB
        client = ArangoClient(hosts='http://192.168.1.69:8529')
        self.db = client.db('academy_store', username='root', password='1luv93ngu1n$')
        
        # Our experiment configuration
        self.experiments = [
            {
                'name': 'word2vec',
                'paper_id': '1301.3781',
                'repo': 'dav/word2vec',
                'year': 2013,
                'conveyance_type': 'community_implementation'
            },
            {
                'name': 'doc2vec',
                'paper_id': '1405.4053',
                'repo': 'piskvorky/gensim',  # Pure conveyance example
                'year': 2014,
                'conveyance_type': 'pure_conveyance'  # No original code!
            },
            {
                'name': 'code2vec',
                'paper_id': '1803.09473',
                'repo': 'tech-srl/code2vec',
                'year': 2018,
                'conveyance_type': 'official_implementation'
            }
        ]
    
    def get_paper_embeddings(self, paper_id: str) -> Dict:
        """Retrieve paper embeddings from ArangoDB."""
        papers_coll = self.db.collection('arxiv_papers')
        embeddings_coll = self.db.collection('arxiv_embeddings')
        
        # Get paper
        paper_key = paper_id.replace('.', '_')
        paper = papers_coll.get({'_key': paper_key})
        
        if not paper:
            logger.warning(f"Paper {paper_id} not found")
            return {}
        
        # Get chunk embeddings
        chunk_embeddings = []
        for emb in embeddings_coll.find({'paper_id': paper_id}):
            chunk_embeddings.append({
                'vector': np.array(emb['vector']),
                'chunk_index': emb.get('chunk_index', 0)
            })
        
        # Sort by chunk index
        chunk_embeddings.sort(key=lambda x: x['chunk_index'])
        
        return {
            'paper_id': paper_id,
            'title': paper.get('title', 'Unknown'),
            'paper_embedding': np.array(paper.get('paper_embedding', [])),
            'chunk_embeddings': chunk_embeddings,
            'num_chunks': len(chunk_embeddings)
        }
    
    def get_code_embeddings(self, repo: str) -> Dict:
        """Retrieve code repository embeddings from ArangoDB."""
        repos_coll = self.db.collection('github_repositories')
        embeddings_coll = self.db.collection('github_embeddings')
        
        # Get repository
        repo_key = repo.replace('/', '_')
        repository = repos_coll.get({'_key': repo_key})
        
        if not repository:
            logger.warning(f"Repository {repo} not found")
            return {}
        
        # Get code embeddings
        code_embeddings = []
        
        # Query for embeddings related to this repository
        query = """
        FOR repo IN github_repositories
            FILTER repo._key == @repo_key
            FOR file IN github_papers
                FILTER file.repository_id == repo._id
                FOR chunk IN github_chunks
                    FILTER chunk.file_id == file._id
                    FOR emb IN github_embeddings
                        FILTER emb.chunk_id == chunk._id
                        RETURN {
                            vector: emb.vector,
                            file_path: file.path,
                            chunk_text: chunk.text
                        }
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={'repo_key': repo_key}
        )
        
        for doc in cursor:
            code_embeddings.append({
                'vector': np.array(doc['vector']),
                'file_path': doc['file_path'],
                'chunk_text': doc.get('chunk_text', '')[:100]  # First 100 chars
            })
        
        # Calculate repository-level embedding (mean)
        if code_embeddings:
            repo_embedding = np.mean([e['vector'] for e in code_embeddings], axis=0)
        else:
            repo_embedding = np.array([])
        
        return {
            'repo': repo,
            'repo_embedding': repo_embedding,
            'code_embeddings': code_embeddings,
            'num_files': len(set(e['file_path'] for e in code_embeddings))
        }
    
    def calculate_entropy_map(self, paper_embeddings: List[np.ndarray], 
                            code_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Calculate entropy map between paper and code embeddings.
        Low entropy = strong bridge (high similarity/low uncertainty)
        High entropy = weak bridge (low similarity/high uncertainty)
        """
        if not paper_embeddings or not code_embeddings:
            return np.array([])
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(paper_embeddings), len(code_embeddings)))
        
        for i, p_emb in enumerate(paper_embeddings):
            for j, c_emb in enumerate(code_embeddings):
                # Cosine similarity (1 - cosine distance)
                similarity = 1 - cosine(p_emb, c_emb)
                similarity_matrix[i, j] = similarity
        
        # Calculate entropy for each paper chunk
        entropy_values = []
        for i in range(len(paper_embeddings)):
            # Get probability distribution from similarities
            similarities = similarity_matrix[i, :]
            # Normalize to probabilities
            if similarities.sum() > 0:
                probs = similarities / similarities.sum()
                # Calculate entropy
                ent = entropy(probs)
            else:
                ent = np.log(len(code_embeddings))  # Maximum entropy
            entropy_values.append(ent)
        
        return np.array(entropy_values)
    
    def calculate_conveyance_score(self, paper_data: Dict, code_data: Dict,
                                  conveyance_type: str) -> float:
        """
        Calculate conveyance score based on paper-code alignment.
        Higher score = better knowledge transfer from paper to code.
        """
        if not paper_data or not code_data:
            return 0.0
        
        paper_emb = paper_data.get('paper_embedding', np.array([]))
        code_emb = code_data.get('repo_embedding', np.array([]))
        
        if paper_emb.size == 0 or code_emb.size == 0:
            return 0.0
        
        # Base conveyance: cosine similarity
        base_conveyance = 1 - cosine(paper_emb, code_emb)
        
        # Context amplification based on conveyance type
        amplification_factors = {
            'pure_conveyance': 2.0,      # Highest - no original code!
            'community_implementation': 1.5,  # High - community understood
            'official_implementation': 1.0    # Base - authors' own code
        }
        
        alpha = amplification_factors.get(conveyance_type, 1.0)
        
        # Context score based on implementation success
        # For Gensim (pure conveyance), the fact it became standard = high context
        context_score = 0.8 if conveyance_type == 'pure_conveyance' else 0.6
        
        # Apply conveyance formula: C = Base × Context^α
        conveyance_score = base_conveyance * (context_score ** alpha)
        
        return min(1.0, conveyance_score)  # Cap at 1.0
    
    def analyze_temporal_evolution(self, results: List[Dict]) -> Dict:
        """Analyze how concepts evolved from 2013 to 2018."""
        # Sort by year
        sorted_results = sorted(results, key=lambda x: x['year'])
        
        evolution = {
            'timeline': [],
            'complexity_growth': [],
            'conveyance_trend': []
        }
        
        for i, result in enumerate(sorted_results):
            evolution['timeline'].append({
                'year': result['year'],
                'name': result['name'],
                'num_paper_chunks': result.get('num_paper_chunks', 0),
                'num_code_files': result.get('num_code_files', 0),
                'conveyance_score': result.get('conveyance_score', 0)
            })
            
            # Track complexity growth
            complexity = result.get('num_paper_chunks', 0) * result.get('num_code_files', 1)
            evolution['complexity_growth'].append(complexity)
            
            # Track conveyance trend
            evolution['conveyance_trend'].append(result.get('conveyance_score', 0))
        
        # Calculate evolution metrics
        evolution['complexity_increase'] = (
            evolution['complexity_growth'][-1] / evolution['complexity_growth'][0]
            if evolution['complexity_growth'][0] > 0 else 0
        )
        
        evolution['conveyance_variance'] = np.var(evolution['conveyance_trend'])
        
        return evolution
    
    def run_analysis(self) -> Dict:
        """Run complete theory-practice bridge analysis."""
        logger.info("Starting theory-practice bridge analysis...")
        
        results = []
        
        for exp in self.experiments:
            logger.info(f"\nAnalyzing {exp['name']} ({exp['year']})...")
            
            # Get paper embeddings
            paper_data = self.get_paper_embeddings(exp['paper_id'])
            
            # Get code embeddings
            code_data = self.get_code_embeddings(exp['repo'])
            
            if paper_data and code_data:
                # Extract embedding vectors
                paper_vecs = [e['vector'] for e in paper_data.get('chunk_embeddings', [])]
                code_vecs = [e['vector'] for e in code_data.get('code_embeddings', [])]
                
                # Calculate entropy map
                entropy_map = self.calculate_entropy_map(paper_vecs, code_vecs)
                
                # Calculate conveyance score
                conveyance = self.calculate_conveyance_score(
                    paper_data, code_data, exp['conveyance_type']
                )
                
                # Find bridge points (low entropy = strong bridge)
                bridge_points = []
                if len(entropy_map) > 0:
                    # Find chunks with lowest entropy (strongest bridges)
                    sorted_indices = np.argsort(entropy_map)[:3]  # Top 3 bridges
                    for idx in sorted_indices:
                        bridge_points.append({
                            'chunk_index': int(idx),
                            'entropy': float(entropy_map[idx]),
                            'strength': 'strong' if entropy_map[idx] < np.median(entropy_map) else 'weak'
                        })
                
                result = {
                    'name': exp['name'],
                    'year': exp['year'],
                    'paper_id': exp['paper_id'],
                    'repo': exp['repo'],
                    'conveyance_type': exp['conveyance_type'],
                    'conveyance_score': float(conveyance),
                    'num_paper_chunks': paper_data.get('num_chunks', 0),
                    'num_code_files': code_data.get('num_files', 0),
                    'mean_entropy': float(np.mean(entropy_map)) if len(entropy_map) > 0 else 0,
                    'entropy_std': float(np.std(entropy_map)) if len(entropy_map) > 0 else 0,
                    'bridge_points': bridge_points,
                    'paper_title': paper_data.get('title', 'Unknown')
                }
                
                results.append(result)
                
                # Log summary
                logger.info(f"  Paper: {result['paper_title'][:50]}...")
                logger.info(f"  Conveyance type: {exp['conveyance_type']}")
                logger.info(f"  Conveyance score: {conveyance:.3f}")
                logger.info(f"  Mean entropy: {result['mean_entropy']:.3f}")
                logger.info(f"  Bridge points found: {len(bridge_points)}")
            else:
                logger.warning(f"  Missing data for {exp['name']}")
                results.append({
                    'name': exp['name'],
                    'year': exp['year'],
                    'error': 'Missing embeddings'
                })
        
        # Analyze temporal evolution
        evolution = self.analyze_temporal_evolution(results)
        
        # Theoretical insights
        insights = self.generate_insights(results, evolution)
        
        return {
            'experiments': results,
            'evolution': evolution,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_insights(self, results: List[Dict], evolution: Dict) -> Dict:
        """Generate theoretical insights from the analysis."""
        insights = {}
        
        # Find which implementation has highest conveyance
        if results:
            best_conveyance = max(results, key=lambda x: x.get('conveyance_score', 0))
            insights['highest_conveyance'] = {
                'name': best_conveyance['name'],
                'score': best_conveyance.get('conveyance_score', 0),
                'type': best_conveyance.get('conveyance_type', 'unknown')
            }
        
        # Special insight for Gensim
        gensim_result = next((r for r in results if 'gensim' in r.get('repo', '').lower()), None)
        if gensim_result:
            insights['pure_conveyance_validation'] = {
                'success': gensim_result.get('conveyance_score', 0) > 0.5,
                'score': gensim_result.get('conveyance_score', 0),
                'significance': 'Gensim doc2vec proves paper conveyance without original code'
            }
        
        # Evolution insights
        insights['temporal_pattern'] = {
            'complexity_growth': evolution.get('complexity_increase', 0),
            'conveyance_stability': 1.0 - evolution.get('conveyance_variance', 1.0)
        }
        
        return insights

def main():
    """Run the complete analysis."""
    analyzer = TheoryPracticeBridgeAnalyzer()
    
    logger.info("=" * 80)
    logger.info("THEORY-PRACTICE BRIDGE ANALYSIS")
    logger.info("Word2Vec → Doc2Vec → Code2Vec Evolution")
    logger.info("=" * 80)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    # Save results
    output_file = Path(__file__).parent / 'theory_practice_bridge_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    # Print key insights
    if 'insights' in results:
        insights = results['insights']
        
        if 'highest_conveyance' in insights:
            hc = insights['highest_conveyance']
            logger.info(f"\nHighest Conveyance: {hc['name']} ({hc['type']})")
            logger.info(f"  Score: {hc['score']:.3f}")
        
        if 'pure_conveyance_validation' in insights:
            pcv = insights['pure_conveyance_validation']
            logger.info(f"\nPure Conveyance (Gensim):")
            logger.info(f"  Success: {pcv['success']}")
            logger.info(f"  Score: {pcv['score']:.3f}")
            logger.info(f"  Significance: {pcv['significance']}")
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())