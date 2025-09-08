#!/usr/bin/env python3
"""
Theory-Practice Bridge Discovery using GraphSAGE.

Discovers connections between theoretical papers (ArXiv) and practical
implementations (GitHub code). Uses GraphSAGE embeddings to find
non-obvious bridges across heterogeneous graph structures.

From Information Reconstructionism: CONVEYANCE emerges at the intersection
of theory (WHAT) and practice (HOW), with GraphSAGE discovering the
latent bridges that traditional hierarchical methods miss.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import time

from core.framework.graph_embedders import GraphSAGEEmbedder, GraphSAGEConfig
from core.framework.memory_store import GraphMemoryStore


@dataclass
class Bridge:
    """Represents a theory-practice bridge."""
    paper_id: str
    code_id: str
    confidence: float
    bridge_type: str  # 'direct', 'semantic', 'structural', 'latent'
    evidence: Dict[str, any]  # Supporting evidence for the bridge


@dataclass
class BridgeDiscoveryConfig:
    """Configuration for bridge discovery."""
    similarity_threshold: float = 0.8  # Minimum similarity for bridge
    max_bridges_per_paper: int = 5  # Maximum bridges to find per paper
    bridge_types: List[str] = None  # Types of bridges to discover
    use_embeddings: bool = True  # Use GraphSAGE embeddings
    use_structure: bool = True  # Use graph structure
    use_content: bool = True  # Use content similarity
    
    def __post_init__(self):
        if self.bridge_types is None:
            self.bridge_types = ['direct', 'semantic', 'structural', 'latent']


class TheoryPracticeFinder:
    """
    Discovers bridges between theoretical papers and practical code.
    
    Uses GraphSAGE to learn cross-domain patterns that connect
    academic research with real-world implementations.
    """
    
    def __init__(self, graph_store: GraphMemoryStore, 
                 graphsage_model: GraphSAGEEmbedder,
                 config: BridgeDiscoveryConfig):
        """
        Initialize the bridge finder.
        
        Args:
            graph_store: Graph memory store
            graphsage_model: Trained GraphSAGE model
            config: Bridge discovery configuration
        """
        self.graph_store = graph_store
        self.graphsage_model = graphsage_model
        self.config = config
        
        # Separate nodes by type
        self._categorize_nodes()
        
        # Precompute embeddings if using
        if config.use_embeddings:
            self._precompute_embeddings()
    
    def _categorize_nodes(self):
        """Categorize nodes into papers and code."""
        self.paper_nodes = []
        self.code_nodes = []
        
        for node_id, node_idx in self.graph_store.node_ids.items():
            node_type = self.graph_store.node_types.get(node_idx, "")
            
            if node_type in ['paper', 'chunk']:
                self.paper_nodes.append(node_idx)
            elif node_type in ['code', 'repo']:
                self.code_nodes.append(node_idx)
        
        print(f"Categorized {len(self.paper_nodes)} paper nodes, {len(self.code_nodes)} code nodes")
    
    def _precompute_embeddings(self):
        """Precompute GraphSAGE embeddings for all nodes."""
        print("Computing GraphSAGE embeddings...")
        
        # Generate embeddings
        self.embeddings = self.graphsage_model.generate_embeddings()
        
        # Separate paper and code embeddings for faster search
        self.paper_embeddings = self.embeddings[self.paper_nodes]
        self.code_embeddings = self.embeddings[self.code_nodes]
        
        print(f"Computed embeddings: shape {self.embeddings.shape}")
    
    def find_direct_bridges(self) -> List[Bridge]:
        """
        Find direct bridges (explicit citations/references).
        
        Returns:
            List of direct bridges
        """
        bridges = []
        
        # Look for explicit edges between papers and code
        for paper_idx in self.paper_nodes:
            paper_id = self.graph_store.index_to_node[paper_idx]
            
            # Check neighbors
            neighbors = self.graph_store.adjacency.get(paper_idx, set())
            
            for neighbor_idx in neighbors:
                neighbor_type = self.graph_store.node_types.get(neighbor_idx, "")
                
                if neighbor_type in ['code', 'repo']:
                    code_id = self.graph_store.index_to_node[neighbor_idx]
                    
                    bridge = Bridge(
                        paper_id=paper_id,
                        code_id=code_id,
                        confidence=1.0,  # Direct connection
                        bridge_type='direct',
                        evidence={'edge_exists': True}
                    )
                    bridges.append(bridge)
        
        return bridges
    
    def find_semantic_bridges(self, paper_idx: Optional[int] = None) -> List[Bridge]:
        """
        Find semantic bridges using embedding similarity.
        
        Args:
            paper_idx: Specific paper to find bridges for (None = all)
            
        Returns:
            List of semantic bridges
        """
        if not self.config.use_embeddings:
            return []
        
        bridges = []
        
        # Determine which papers to process
        if paper_idx is not None:
            paper_indices = [paper_idx]
        else:
            paper_indices = self.paper_nodes[:100]  # Limit for efficiency
        
        for idx, paper_idx in enumerate(paper_indices):
            paper_id = self.graph_store.index_to_node[paper_idx]
            
            # Get paper embedding
            paper_emb = self.embeddings[paper_idx].reshape(1, -1)
            
            # Compute similarities with all code
            similarities = cosine_similarity(paper_emb, self.code_embeddings)[0]
            
            # Find top matches
            top_indices = np.argsort(similarities)[-self.config.max_bridges_per_paper:][::-1]
            
            for code_array_idx in top_indices:
                similarity = similarities[code_array_idx]
                
                if similarity >= self.config.similarity_threshold:
                    code_idx = self.code_nodes[code_array_idx]
                    code_id = self.graph_store.index_to_node[code_idx]
                    
                    bridge = Bridge(
                        paper_id=paper_id,
                        code_id=code_id,
                        confidence=float(similarity),
                        bridge_type='semantic',
                        evidence={'embedding_similarity': float(similarity)}
                    )
                    bridges.append(bridge)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(paper_indices)} papers")
        
        return bridges
    
    def find_structural_bridges(self) -> List[Bridge]:
        """
        Find structural bridges using graph patterns.
        
        Returns:
            List of structural bridges
        """
        if not self.config.use_structure:
            return []
        
        bridges = []
        
        # Find papers and code that share common neighbors
        paper_neighborhoods = {}
        code_neighborhoods = {}
        
        # Get 2-hop neighborhoods
        for paper_idx in self.paper_nodes[:50]:  # Limit for efficiency
            neighbors = self.graph_store.get_k_hop_neighbors(paper_idx, k=2)
            paper_neighborhoods[paper_idx] = set()
            for hop_neighbors in neighbors.values():
                paper_neighborhoods[paper_idx].update(hop_neighbors)
        
        for code_idx in self.code_nodes[:50]:
            neighbors = self.graph_store.get_k_hop_neighbors(code_idx, k=2)
            code_neighborhoods[code_idx] = set()
            for hop_neighbors in neighbors.values():
                code_neighborhoods[code_idx].update(hop_neighbors)
        
        # Find overlapping neighborhoods
        for paper_idx, paper_neighbors in paper_neighborhoods.items():
            paper_id = self.graph_store.index_to_node[paper_idx]
            
            for code_idx, code_neighbors in code_neighborhoods.items():
                code_id = self.graph_store.index_to_node[code_idx]
                
                # Compute Jaccard similarity of neighborhoods
                intersection = len(paper_neighbors & code_neighbors)
                union = len(paper_neighbors | code_neighbors)
                
                if union > 0:
                    jaccard = intersection / union
                    
                    if jaccard >= 0.1:  # Lower threshold for structural
                        bridge = Bridge(
                            paper_id=paper_id,
                            code_id=code_id,
                            confidence=jaccard,
                            bridge_type='structural',
                            evidence={
                                'shared_neighbors': intersection,
                                'jaccard_similarity': jaccard
                            }
                        )
                        bridges.append(bridge)
        
        return bridges
    
    def find_latent_bridges(self) -> List[Bridge]:
        """
        Find latent bridges using GraphSAGE's learned representations.
        
        These are bridges that GraphSAGE discovers through
        neighborhood aggregation that aren't obvious from
        direct similarity or structure alone.
        
        Returns:
            List of latent bridges
        """
        if not self.config.use_embeddings:
            return []
        
        bridges = []
        
        # Use GraphSAGE to predict links
        # Sample some paper-code pairs
        num_samples = min(100, len(self.paper_nodes))
        sampled_papers = np.random.choice(self.paper_nodes, num_samples, replace=False)
        sampled_codes = np.random.choice(self.code_nodes, 
                                        min(50, len(self.code_nodes)), 
                                        replace=False)
        
        for paper_idx in sampled_papers:
            paper_id = self.graph_store.index_to_node[paper_idx]
            
            # Get GraphSAGE embedding with neighborhood context
            paper_emb = self.graphsage_model.forward([paper_idx]).detach().cpu().numpy()
            
            # Score against code nodes
            code_scores = []
            for code_idx in sampled_codes:
                code_emb = self.graphsage_model.forward([code_idx]).detach().cpu().numpy()
                
                # Compute similarity in GraphSAGE space
                similarity = cosine_similarity(paper_emb, code_emb)[0, 0]
                code_scores.append((code_idx, similarity))
            
            # Sort by score
            code_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top matches
            for code_idx, score in code_scores[:self.config.max_bridges_per_paper]:
                if score >= self.config.similarity_threshold * 0.8:  # Slightly lower threshold
                    code_id = self.graph_store.index_to_node[code_idx]
                    
                    bridge = Bridge(
                        paper_id=paper_id,
                        code_id=code_id,
                        confidence=float(score),
                        bridge_type='latent',
                        evidence={
                            'graphsage_similarity': float(score),
                            'discovered_through': 'neighborhood_aggregation'
                        }
                    )
                    bridges.append(bridge)
        
        return bridges
    
    def discover_all_bridges(self) -> Dict[str, List[Bridge]]:
        """
        Discover all types of bridges.
        
        Returns:
            Dictionary mapping bridge type to list of bridges
        """
        all_bridges = {}
        
        print("Discovering theory-practice bridges...")
        
        # Direct bridges
        if 'direct' in self.config.bridge_types:
            print("Finding direct bridges...")
            start = time.time()
            all_bridges['direct'] = self.find_direct_bridges()
            print(f"  Found {len(all_bridges['direct'])} direct bridges in {time.time() - start:.2f}s")
        
        # Semantic bridges
        if 'semantic' in self.config.bridge_types:
            print("Finding semantic bridges...")
            start = time.time()
            all_bridges['semantic'] = self.find_semantic_bridges()
            print(f"  Found {len(all_bridges['semantic'])} semantic bridges in {time.time() - start:.2f}s")
        
        # Structural bridges
        if 'structural' in self.config.bridge_types:
            print("Finding structural bridges...")
            start = time.time()
            all_bridges['structural'] = self.find_structural_bridges()
            print(f"  Found {len(all_bridges['structural'])} structural bridges in {time.time() - start:.2f}s")
        
        # Latent bridges
        if 'latent' in self.config.bridge_types:
            print("Finding latent bridges...")
            start = time.time()
            all_bridges['latent'] = self.find_latent_bridges()
            print(f"  Found {len(all_bridges['latent'])} latent bridges in {time.time() - start:.2f}s")
        
        return all_bridges
    
    def rank_bridges(self, bridges: List[Bridge]) -> List[Bridge]:
        """
        Rank bridges by importance using conveyance scoring.
        
        Args:
            bridges: List of bridges to rank
            
        Returns:
            Sorted list of bridges
        """
        # Score each bridge
        for bridge in bridges:
            # Conveyance components
            W = bridge.confidence  # Semantic quality
            R = 1.0 if bridge.bridge_type == 'direct' else 0.5  # Relational strength
            H = 1.0  # Model capability (constant for now)
            T = 1.0  # Time (constant for now)
            
            # Context components (default weights)
            L = bridge.evidence.get('local_coherence', 0.5)
            I = bridge.evidence.get('instruction_fit', 0.5)
            A = bridge.evidence.get('actionability', 0.8)  # High for code
            G = bridge.evidence.get('grounding', 0.7)
            
            Ctx = 0.25 * (L + I + A + G)
            alpha = 1.6  # From empirical studies
            
            # Conveyance score
            C = (W * R * H / T) * (Ctx ** alpha)
            bridge.conveyance_score = C
        
        # Sort by conveyance score
        bridges.sort(key=lambda b: b.conveyance_score, reverse=True)
        
        return bridges
    
    def evaluate_bridge(self, bridge: Bridge) -> Dict[str, float]:
        """
        Evaluate the quality of a discovered bridge.
        
        Args:
            bridge: Bridge to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Confidence score
        metrics['confidence'] = bridge.confidence
        
        # Type-specific metrics
        if bridge.bridge_type == 'direct':
            metrics['reliability'] = 1.0  # Direct edges are most reliable
        elif bridge.bridge_type == 'semantic':
            metrics['reliability'] = 0.8 * bridge.confidence
        elif bridge.bridge_type == 'structural':
            metrics['reliability'] = 0.6 * bridge.confidence
        else:  # latent
            metrics['reliability'] = 0.5 * bridge.confidence
        
        # Novelty (inverse of how obvious the bridge is)
        if bridge.bridge_type == 'latent':
            metrics['novelty'] = 0.9  # Most novel
        elif bridge.bridge_type == 'structural':
            metrics['novelty'] = 0.6
        elif bridge.bridge_type == 'semantic':
            metrics['novelty'] = 0.3
        else:  # direct
            metrics['novelty'] = 0.1  # Least novel
        
        # Overall quality
        metrics['quality'] = (
            0.4 * metrics['confidence'] +
            0.4 * metrics['reliability'] +
            0.2 * metrics['novelty']
        )
        
        return metrics