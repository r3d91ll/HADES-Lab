#!/usr/bin/env python3
"""
HiRAG Three-Level Retrieval Engine (HiRetrieval)
Implements Local ↔ Bridge ↔ Global retrieval as specified in PRD Issue #19

This module enables the core HiRAG functionality:
- Local: Entity-level semantic matching
- Global: Cluster-level expansion and summarization  
- Bridge: Weighted shortest paths between local and global knowledge
"""

import os
import sys
import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timezone
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Use absolute imports - no sys.path manipulation needed

from arango import ArangoClient
from arango.database import StandardDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """
    Query context for HiRAG retrieval operations.
    
    In anthropological terms, this represents the interpretive framework
    through which users approach knowledge - their situated perspective
    that shapes what constitutes relevant information.
    """
    query_text: str
    query_type: str = "hi"  # hi, hi_local, hi_global, hi_bridge, naive
    top_k_entities: int = 20
    top_m_clusters: int = 10
    max_bridge_hops: int = 4
    similarity_threshold: float = 0.7
    include_context: bool = True
    response_format: str = "comprehensive"


@dataclass
class RetrievalResult:
    """Results from HiRAG retrieval operation."""
    query_context: QueryContext
    local_entities: List[Dict]
    global_clusters: List[Dict]
    bridge_paths: List[Dict]
    answer_synthesis: str
    conveyance_score: float
    retrieval_time_ms: int
    metadata: Dict


class HiRetrievalEngine:
    """
    Three-level hierarchical retrieval engine for HiRAG.
    
    Following Actor-Network Theory, this engine serves as a translator
    between user queries and the hierarchical knowledge network,
    enabling navigation across different levels of abstraction while
    preserving the relational context that gives information its meaning.
    """
    
    def __init__(self, host: str = None, port: int = 8529, 
                 username: str = "root", password: str = None):
        """Initialize the HiRAG retrieval engine."""
        # Load configuration from environment
        self.host = host or os.getenv('ARANGO_HOST', 'localhost')
        self.password = password or os.getenv('ARANGO_PASSWORD')
        
        if not self.password:
            raise ValueError(
                "ArangoDB password required. Set ARANGO_PASSWORD environment variable "
                "or pass password parameter."
            )
            
        self.client = ArangoClient(hosts=f"http://{self.host}:{port}")
        self.db: Optional[StandardDatabase] = None
        
        # Query processing components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.entity_embeddings: Optional[np.ndarray] = None
        self.entity_index: Optional[Dict] = None
        self.entity_index_reverse: Optional[Dict] = None
        
        # Performance tracking
        self.query_cache: Dict = {}
        self.performance_metrics: Dict = {
            'total_queries': 0,
            'avg_latency_ms': 0,
            'cache_hit_rate': 0.0
        }
    
    async def connect(self, database_name: str = "academy_store") -> bool:
        """Connect to ArangoDB and initialize retrieval components."""
        try:
            self.db = self.client.db(database_name, username="root", password=self.password)
            logger.info(f"Connected to database: {database_name}")
            
            # Initialize retrieval components
            await self._initialize_retrieval_components()
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def _initialize_retrieval_components(self):
        """Initialize embeddings and indexes for fast retrieval."""
        logger.info("Initializing retrieval components...")
        
        # Load entities for embedding generation
        entities = await self._load_entities_for_retrieval()
        
        if entities:
            # Create TF-IDF embeddings for semantic matching
            texts = []
            entity_keys = []
            
            for entity in entities:
                # Combine name, type, and available context
                text_parts = [
                    entity.get('name', ''),
                    entity.get('type', ''),
                    f"freq_{entity.get('frequency', 1)}"
                ]
                
                # Add category context for research areas
                if entity.get('category_code'):
                    text_parts.append(entity['category_code'])
                
                texts.append(' '.join(text_parts))
                entity_keys.append(entity['_key'])
            
            # Create vectorizer and embeddings
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            self.entity_embeddings = self.vectorizer.fit_transform(texts).toarray()
            
            # Create entity index for fast lookup (bidirectional)
            self.entity_index = {key: i for i, key in enumerate(entity_keys)}
            self.entity_index_reverse = {i: key for key, i in self.entity_index.items()}
            
            logger.info(f"✅ Initialized embeddings for {len(entities)} entities")
        else:
            logger.warning("No entities found for embedding initialization")
    
    async def _load_entities_for_retrieval(self) -> List[Dict]:
        """Load entities optimized for retrieval operations."""
        try:
            query = """
            FOR entity IN entities
                FILTER entity.layer == 0
                RETURN {
                    _key: entity._key,
                    name: entity.name,
                    type: entity.type,
                    frequency: entity.frequency,
                    confidence: entity.confidence,
                    category_code: entity.category_code,
                    source_papers: entity.source_papers
                }
            """
            
            cursor = self.db.aql.execute(query)
            entities = list(cursor)
            return entities
            
        except Exception as e:
            logger.error(f"Failed to load entity list for embedding initialization: {e.__class__.__name__}")
            raise RuntimeError("Entity loading failed during initialization") from e
    
    async def retrieve_local_entities(self, query_context: QueryContext) -> List[Dict]:
        """
        Local retrieval: Entity-level semantic matching.
        
        This implements the foundational level of HiRAG - finding entities
        that semantically match the user's query through direct similarity
        matching in the embedding space.
        """
        if not self.vectorizer or self.entity_embeddings is None:
            logger.error("Retrieval components not initialized")
            return []
        
        try:
            # Transform query to embedding space
            query_embedding = self.vectorizer.transform([query_context.query_text]).toarray()
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.entity_embeddings)[0]
            
            # Get top-k most similar entities
            top_indices = np.argsort(similarities)[::-1][:query_context.top_k_entities]
            
            # Fetch detailed entity information
            local_entities = []
            for idx in top_indices:
                if similarities[idx] >= query_context.similarity_threshold:
                    entity_key = self.entity_index_reverse[idx]
                    entity_details = await self._get_entity_details(entity_key)
                    
                    if entity_details:
                        entity_details['similarity_score'] = float(similarities[idx])
                        entity_details['retrieval_source'] = 'local_semantic'
                        local_entities.append(entity_details)
            
            logger.info(f"Retrieved {len(local_entities)} local entities")
            return local_entities
            
        except Exception as e:
            logger.error(f"Local retrieval failed: {e}")
            return []
    
    async def _get_entity_details(self, entity_key: str) -> Optional[Dict]:
        """Get detailed entity information including relationships."""
        try:
            query = """
            FOR entity IN entities
                FILTER entity._key == @entity_key
                RETURN {
                    _key: entity._key,
                    name: entity.name,
                    type: entity.type,
                    frequency: entity.frequency,
                    confidence: entity.confidence,
                    source_papers: entity.source_papers,
                    category_code: entity.category_code,
                    extraction_sources: entity.extraction_sources
                }
            """
            
            cursor = self.db.aql.execute(query, bind_vars={'entity_key': entity_key})
            result = list(cursor)
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Failed to get entity details for {entity_key}: {e}")
            return None
    
    async def retrieve_global_clusters(self, local_entities: List[Dict], 
                                     query_context: QueryContext) -> List[Dict]:
        """
        Global retrieval: Cluster-level expansion and summarization.
        
        This implements the global perspective of HiRAG - finding clusters
        that contain or are related to the local entities, providing
        higher-level thematic context and broader knowledge organization.
        """
        if not local_entities:
            return []
        
        try:
            # Get clusters containing the local entities
            local_entity_keys = [entity['_key'] for entity in local_entities]
            
            query = """
            FOR entity_key IN @entity_keys
                FOR cluster IN clusters
                    FILTER entity_key IN cluster.members
                    RETURN DISTINCT {
                        _key: cluster._key,
                        name: cluster.name,
                        layer: cluster.layer,
                        member_count: cluster.member_count,
                        members: cluster.members,
                        summary: cluster.summary,
                        key_concepts: cluster.key_concepts,
                        cohesion_score: cluster.cohesion_score,
                        cluster_type: cluster.cluster_type,
                        category_family: cluster.category_family
                    }
            """
            
            cursor = self.db.aql.execute(query, bind_vars={'entity_keys': local_entity_keys})
            clusters = list(cursor)
            
            # Enrich clusters with member overlap information
            enriched_clusters = []
            for cluster in clusters:
                # Calculate overlap with query entities
                member_overlap = len(set(cluster.get('members', [])) & set(local_entity_keys))
                overlap_ratio = member_overlap / len(local_entity_keys) if local_entity_keys else 0
                
                cluster['member_overlap'] = member_overlap
                cluster['overlap_ratio'] = overlap_ratio
                cluster['retrieval_source'] = 'global_expansion'
                
                enriched_clusters.append(cluster)
            
            # Sort by relevance (overlap ratio * cohesion score)
            enriched_clusters.sort(
                key=lambda x: x['overlap_ratio'] * x.get('cohesion_score', 0.5), 
                reverse=True
            )
            
            # Return top-m clusters
            top_clusters = enriched_clusters[:query_context.top_m_clusters]
            
            logger.info(f"Retrieved {len(top_clusters)} global clusters")
            return top_clusters
            
        except Exception as e:
            logger.error(f"Global retrieval failed: {e}")
            return []
    
    async def retrieve_bridge_paths(self, local_entities: List[Dict], 
                                  global_clusters: List[Dict],
                                  query_context: QueryContext) -> List[Dict]:
        """
        Bridge retrieval: Weighted shortest paths between local and global knowledge.
        
        This implements the bridge mechanism of HiRAG - finding meaningful
        connections between specific entities and broader clusters through
        the hierarchical graph structure, revealing how local knowledge
        connects to global patterns.
        """
        if not local_entities or not global_clusters:
            return []
        
        try:
            bridge_paths = []
            
            # For each local entity, find paths to global clusters
            for local_entity in local_entities:
                for global_cluster in global_clusters:
                    # Check if direct membership exists
                    if local_entity['_key'] in global_cluster.get('members', []):
                        path = {
                            'from_entity': local_entity['_key'],
                            'to_cluster': global_cluster['_key'],
                            'path_type': 'direct_membership',
                            'path_length': 1,
                            'bridge_strength': global_cluster.get('cohesion_score', 0.5),
                            'explanation': f"'{local_entity['name']}' is a direct member of cluster '{global_cluster['name']}'",
                            'local_context': local_entity,
                            'global_context': global_cluster
                        }
                        bridge_paths.append(path)
                    
                    # Look for indirect paths through related entities
                    indirect_paths = await self._find_indirect_bridge_paths(
                        local_entity, global_cluster, query_context.max_bridge_hops
                    )
                    bridge_paths.extend(indirect_paths)
            
            # Sort paths by bridge strength
            bridge_paths.sort(key=lambda x: x['bridge_strength'], reverse=True)
            
            # Deduplicate and return top paths
            unique_paths = self._deduplicate_bridge_paths(bridge_paths)
            top_paths = unique_paths[:20]  # Limit to top 20 bridge paths
            
            logger.info(f"Found {len(top_paths)} bridge paths")
            return top_paths
            
        except Exception as e:
            logger.error(f"Bridge retrieval failed: {e}")
            return []
    
    async def _find_indirect_bridge_paths(self, local_entity: Dict, 
                                        global_cluster: Dict, 
                                        max_hops: int) -> List[Dict]:
        """Find indirect paths between entities and clusters."""
        try:
            # Query for entities that are similar or related to the local entity
            # and are members of the target cluster
            query = """
            FOR target_member IN @cluster_members
                FOR related_entity IN entities
                    FILTER related_entity._key == target_member
                    FILTER related_entity.type == @local_type
                    LET semantic_similarity = (
                        related_entity.name LIKE CONCAT('%', @search_term, '%') OR
                        @search_term LIKE CONCAT('%', related_entity.name, '%') ? 0.8 : 0.3
                    )
                    FILTER semantic_similarity > 0.4
                    RETURN {
                        entity: related_entity,
                        similarity: semantic_similarity
                    }
            """
            
            search_term = local_entity['name'].lower()
            cursor = self.db.aql.execute(query, bind_vars={
                'cluster_members': global_cluster.get('members', []),
                'local_type': local_entity['type'],
                'search_term': search_term
            })
            
            related_entities = list(cursor)
            
            # Create bridge paths for related entities
            indirect_paths = []
            for related in related_entities:
                if related['similarity'] > 0.4:
                    path = {
                        'from_entity': local_entity['_key'],
                        'to_cluster': global_cluster['_key'],
                        'path_type': 'semantic_bridge',
                        'path_length': 2,
                        'bridge_strength': related['similarity'] * global_cluster.get('cohesion_score', 0.5),
                        'explanation': f"'{local_entity['name']}' connects to cluster '{global_cluster['name']}' through similar entity '{related['entity']['name']}'",
                        'intermediate_entity': related['entity'],
                        'local_context': local_entity,
                        'global_context': global_cluster
                    }
                    indirect_paths.append(path)
            
            return indirect_paths[:5]  # Limit indirect paths per entity-cluster pair
            
        except Exception as e:
            logger.error(f"Failed to find indirect bridge paths: {e}")
            return []
    
    def _deduplicate_bridge_paths(self, bridge_paths: List[Dict]) -> List[Dict]:
        """Remove duplicate bridge paths."""
        seen_paths = set()
        unique_paths = []
        
        for path in bridge_paths:
            path_signature = (path['from_entity'], path['to_cluster'], path['path_type'])
            if path_signature not in seen_paths:
                seen_paths.add(path_signature)
                unique_paths.append(path)
        
        return unique_paths
    
    async def synthesize_answer(self, local_entities: List[Dict], 
                              global_clusters: List[Dict],
                              bridge_paths: List[Dict],
                              query_context: QueryContext) -> str:
        """
        Synthesize a comprehensive answer from retrieval results.
        
        This represents the final translation step - converting the
        structured hierarchical knowledge back into natural language
        that addresses the user's original information need.
        """
        try:
            answer_parts = []
            
            # Introduction based on query type
            if query_context.query_type == "hi_local":
                answer_parts.append("Based on local entity analysis:")
            elif query_context.query_type == "hi_global":
                answer_parts.append("From a global cluster perspective:")
            elif query_context.query_type == "hi_bridge":
                answer_parts.append("Through bridge connections:")
            else:
                answer_parts.append("From hierarchical analysis:")
            
            # Local entity insights
            if local_entities:
                top_local = local_entities[:5]
                local_summary = []
                
                for entity in top_local:
                    entity_desc = f"**{entity['name']}** ({entity['type']})"
                    if entity.get('category_code'):
                        entity_desc += f" in {entity['category_code']}"
                    if entity.get('frequency'):
                        entity_desc += f" - appears in {entity['frequency']} papers"
                    local_summary.append(entity_desc)
                
                answer_parts.append(f"\n**Key Entities Found:**\n" + "\n".join(f"• {desc}" for desc in local_summary))
            
            # Global cluster insights
            if global_clusters:
                top_clusters = global_clusters[:3]
                cluster_summary = []
                
                for cluster in top_clusters:
                    cluster_desc = f"**{cluster['name']}** ({cluster['member_count']} entities)"
                    if cluster.get('category_family'):
                        cluster_desc += f" in {cluster['category_family']} domain"
                    cluster_summary.append(f"{cluster_desc}: {cluster.get('summary', 'Research cluster')}")
                
                answer_parts.append(f"\n**Related Research Areas:**\n" + "\n".join(f"• {desc}" for desc in cluster_summary))
            
            # Bridge path insights
            if bridge_paths:
                top_bridges = bridge_paths[:3]
                bridge_summary = []
                
                for bridge in top_bridges:
                    bridge_summary.append(f"• {bridge.get('explanation', 'Connection found')}")
                
                answer_parts.append(f"\n**Knowledge Connections:**\n" + "\n".join(bridge_summary))
            
            # Synthesis conclusion
            if local_entities and global_clusters:
                research_areas = set()
                for cluster in global_clusters[:3]:
                    if cluster.get('category_family'):
                        research_areas.add(cluster['category_family'].upper())
                
                if research_areas:
                    areas_text = ", ".join(sorted(research_areas))
                    answer_parts.append(f"\n**Research Context:** This query spans {areas_text} domains, "
                                      f"involving {len(local_entities)} specific entities across "
                                      f"{len(global_clusters)} thematic clusters.")
            
            return "\n".join(answer_parts)
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return f"Analysis found {len(local_entities)} entities and {len(global_clusters)} clusters related to your query."
    
    def calculate_conveyance_score(self, local_entities: List[Dict], 
                                 global_clusters: List[Dict],
                                 bridge_paths: List[Dict],
                                 retrieval_time_ms: int) -> float:
        """
        Calculate conveyance score: C = (W·R·H/T)·Ctx^α
        
        This implements the core Information Reconstructionism formula
        measuring the actionability and coherence of retrieved knowledge.
        """
        try:
            # W: Semantic Quality (0-1)
            if local_entities:
                similarities = [e.get('similarity_score', 0.5) for e in local_entities]
                confidences = [e.get('confidence', 0.5) for e in local_entities]
                
                avg_similarity = np.mean(similarities) if similarities else 0.5
                avg_confidence = np.mean(confidences) if confidences else 0.5
                
                W = (avg_similarity + avg_confidence) / 2
            else:
                W = 0.0
            
            # R: Graph Connectivity (0-1)  
            if bridge_paths:
                avg_bridge_strength = np.mean([p.get('bridge_strength', 0.5) for p in bridge_paths])
                path_diversity = len(set(p['path_type'] for p in bridge_paths)) / 3  # Max 3 types
                R = (avg_bridge_strength + path_diversity) / 2
            else:
                R = 0.0
            
            # H: Model Capability (fixed at 0.8 for baseline)
            H = 0.8
            
            # T: Time efficiency (inverse of retrieval time, normalized)
            T = max(0.1, 1000 / max(retrieval_time_ms, 100))  # Normalize around 1000ms
            T = min(T, 2.0)  # Cap at 2.0
            
            # Context = (L + I + A + G) / 4
            # L: Local coherence
            L = min(1.0, len(local_entities) / 20) if local_entities else 0.0
            
            # I: Instruction fit (semantic match quality)
            I = W  # Use semantic quality as instruction fit proxy
            
            # A: Actionability (presence of methods and concepts)
            actionable_types = {'method', 'concept', 'algorithm'}
            actionable_entities = [e for e in local_entities if e.get('type') in actionable_types]
            A = min(1.0, len(actionable_entities) / 10) if actionable_entities else 0.0
            
            # G: Grounding (research area coverage)
            research_areas = set(c.get('category_family') for c in global_clusters if c.get('category_family'))
            G = min(1.0, len(research_areas) / 5) if research_areas else 0.0
            
            Ctx = (L + I + A + G) / 4
            
            # α: Context amplification factor
            α = 1.6
            
            # Final conveyance score
            if W == 0 or R == 0 or H == 0 or T == 0:
                conveyance_score = 0.0  # Zero propagation
            else:
                conveyance_score = (W * R * H / T) * (Ctx ** α)
            
            return float(min(conveyance_score, 1.0))  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Conveyance calculation failed: {e}")
            return 0.0
    
    async def hierarchical_query(self, query_text: str, 
                               query_type: str = "hi",
                               **kwargs) -> RetrievalResult:
        """
        Execute hierarchical query with three-level retrieval.
        
        This is the main entry point for HiRAG retrieval operations,
        orchestrating the Local → Bridge → Global knowledge discovery process.
        """
        start_time = datetime.now()
        
        try:
            # Create query context
            query_context = QueryContext(
                query_text=query_text,
                query_type=query_type,
                top_k_entities=kwargs.get('top_k', 20),
                top_m_clusters=kwargs.get('top_m', 10),
                similarity_threshold=kwargs.get('threshold', 0.2),
                max_bridge_hops=kwargs.get('max_hops', 4)
            )
            
            logger.info(f"Processing {query_type} query: '{query_text}'")
            
            # Initialize result containers
            local_entities = []
            global_clusters = []  
            bridge_paths = []
            
            # Execute retrieval based on query type
            if query_type in ["hi", "hi_local"]:
                local_entities = await self.retrieve_local_entities(query_context)
            
            if query_type in ["hi", "hi_global"] and local_entities:
                global_clusters = await self.retrieve_global_clusters(local_entities, query_context)
            
            if query_type in ["hi", "hi_bridge"] and local_entities and global_clusters:
                bridge_paths = await self.retrieve_bridge_paths(local_entities, global_clusters, query_context)
            
            # Synthesize answer
            answer = await self.synthesize_answer(local_entities, global_clusters, bridge_paths, query_context)
            
            # Calculate performance metrics
            end_time = datetime.now()
            retrieval_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Calculate conveyance score
            conveyance_score = self.calculate_conveyance_score(
                local_entities, global_clusters, bridge_paths, retrieval_time_ms
            )
            
            # Update performance tracking
            self.performance_metrics['total_queries'] += 1
            
            # Create result
            result = RetrievalResult(
                query_context=query_context,
                local_entities=local_entities,
                global_clusters=global_clusters,
                bridge_paths=bridge_paths,
                answer_synthesis=answer,
                conveyance_score=conveyance_score,
                retrieval_time_ms=retrieval_time_ms,
                metadata={
                    'local_count': len(local_entities),
                    'global_count': len(global_clusters),
                    'bridge_count': len(bridge_paths),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            logger.info(f"✅ Query completed in {retrieval_time_ms}ms - Conveyance: {conveyance_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Hierarchical query failed: {e}")
            
            # Return empty result on failure
            return RetrievalResult(
                query_context=QueryContext(query_text=query_text, query_type=query_type),
                local_entities=[],
                global_clusters=[],
                bridge_paths=[],
                answer_synthesis=f"Query processing failed: {str(e)}",
                conveyance_score=0.0,
                retrieval_time_ms=0,
                metadata={'error': str(e)}
            )


async def main():
    """Test the HiRAG retrieval engine."""
    password = os.getenv('ARANGO_PASSWORD')
    if not password:
        print("Please set ARANGO_PASSWORD environment variable")
        return
    
    # Initialize engine
    engine = HiRetrievalEngine(password=password)
    
    if not await engine.connect():
        return
    
    # Test queries
    test_queries = [
        ("machine learning algorithms", "hi"),
        ("neural network architectures", "hi_local"),
        ("computer vision applications", "hi_global"),
        ("deep learning optimization", "hi_bridge")
    ]
    
    print("\n🤖 Testing HiRAG Retrieval Engine\n" + "="*50)
    
    for query_text, query_type in test_queries:
        print(f"\n🔍 Query: '{query_text}' (mode: {query_type})")
        print("-" * 60)
        
        result = await engine.hierarchical_query(query_text, query_type)
        
        print(f"📊 Results: {result.metadata['local_count']} entities, "
              f"{result.metadata['global_count']} clusters, "
              f"{result.metadata['bridge_count']} bridges")
        print(f"⚡ Performance: {result.retrieval_time_ms}ms, "
              f"Conveyance: {result.conveyance_score:.3f}")
        print(f"\n📝 Answer:\n{result.answer_synthesis}")
    
    print(f"\n✅ HiRAG retrieval engine testing completed!")


if __name__ == "__main__":
    asyncio.run(main())