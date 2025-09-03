#!/usr/bin/env python3
"""
ArXiv Metadata-Driven HiRAG Clustering
Uses structured ArXiv metadata (titles, abstracts, categories, authors) 
instead of raw NER extraction for better semantic clustering
"""

import os
import sys
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging
from datetime import datetime, timezone
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import hashlib
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from arango import ArangoClient
from arango.database import StandardDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArXivMetadataClustering:
    """
    Leverage ArXiv's structured metadata for semantic clustering.
    
    This approach uses the taxonomical work already done by ArXiv:
    - Categories (cs.AI, cs.LG, math.CO, etc.)
    - Paper titles (semantic content)
    - Author collaborations
    - Publication patterns
    
    Following anthropological principles, we're using the existing
    institutional classifications rather than imposing our own.
    """
    
    def __init__(self, host: str = "192.168.1.69", port: int = 8529, 
                 username: str = "root", password: str = None):
        """Initialize metadata-driven clustering."""
        self.password = password or os.getenv('ARANGO_PASSWORD', 'root_password')
        self.client = ArangoClient(hosts=f"http://{host}:{port}")
        self.db: Optional[StandardDatabase] = None
        
        # ArXiv category mapping
        self.arxiv_categories = {
            # Computer Science
            'cs.AI': 'Artificial Intelligence',
            'cs.LG': 'Machine Learning', 
            'cs.CV': 'Computer Vision',
            'cs.CL': 'Natural Language Processing',
            'cs.NE': 'Neural Networks',
            'cs.IR': 'Information Retrieval',
            'cs.RO': 'Robotics',
            'cs.DB': 'Databases',
            'cs.DS': 'Data Structures',
            'cs.DC': 'Distributed Computing',
            'cs.CR': 'Cryptography',
            'cs.SE': 'Software Engineering',
            # Mathematics  
            'math.CO': 'Combinatorics',
            'math.PR': 'Probability',
            'math.ST': 'Statistics',
            'math.OC': 'Optimization',
            'math.NA': 'Numerical Analysis',
            # Statistics
            'stat.ML': 'Machine Learning (Stats)',
            'stat.ME': 'Methodology',
            # Physics
            'cond-mat': 'Condensed Matter',
            'quant-ph': 'Quantum Physics'
        }
    
    async def connect(self, database_name: str = "academy_store") -> bool:
        """Connect to ArangoDB."""
        try:
            self.db = self.client.db(database_name, username="root", password=self.password)
            logger.info(f"Connected to database: {database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def load_arxiv_papers_metadata(self) -> List[Dict]:
        """Load ArXiv papers with their structured metadata."""
        try:
            query = """
            FOR paper IN arxiv_papers
                FILTER paper.title != null AND paper.title != ""
                RETURN {
                    _key: paper._key,
                    title: paper.title,
                    abstract: paper.abstract,
                    categories: paper.categories,
                    authors: paper.authors,
                    date: paper.date
                }
            """
            
            cursor = self.db.aql.execute(query)
            papers = list(cursor)
            logger.info(f"Loaded {len(papers)} papers with metadata")
            return papers
            
        except Exception as e:
            logger.error(f"Failed to load papers: {e}")
            return []
    
    def extract_semantic_features(self, papers: List[Dict]) -> List[Dict]:
        """
        Extract rich semantic features from ArXiv metadata.
        
        This creates structured entities from the taxonomical work
        ArXiv has already done for us.
        """
        entities = []
        
        for paper in papers:
            paper_id = paper['_key']
            
            # 1. Extract category-based entities
            if paper.get('categories'):
                categories = paper['categories']
                if isinstance(categories, str):
                    # Parse category string like "cs.AI cs.LG"
                    cat_list = categories.strip().split()
                    for cat in cat_list:
                        if cat in self.arxiv_categories:
                            entities.append({
                                'name': self.arxiv_categories[cat],
                                'type': 'research_area',
                                'source': 'arxiv_category',
                                'category_code': cat,
                                'paper_id': paper_id,
                                'confidence': 0.95  # High confidence for official categories
                            })
            
            # 2. Extract title-based concepts
            if paper.get('title'):
                title_concepts = self._extract_concepts_from_title(paper['title'])
                for concept in title_concepts:
                    entities.append({
                        'name': concept,
                        'type': 'concept',
                        'source': 'paper_title',
                        'paper_id': paper_id,
                        'confidence': 0.8
                    })
            
            # 3. Extract author entities with collaboration info
            if paper.get('authors'):
                authors = self._parse_authors(paper['authors'])
                for author in authors:
                    entities.append({
                        'name': author,
                        'type': 'researcher',
                        'source': 'paper_author',
                        'paper_id': paper_id,
                        'confidence': 0.9
                    })
            
            # 4. Extract abstract concepts (if available)
            if paper.get('abstract'):
                abstract_concepts = self._extract_concepts_from_abstract(paper['abstract'])
                for concept in abstract_concepts:
                    entities.append({
                        'name': concept,
                        'type': 'method',
                        'source': 'paper_abstract',
                        'paper_id': paper_id,
                        'confidence': 0.75
                    })
        
        return self._aggregate_semantic_entities(entities)
    
    def _extract_concepts_from_title(self, title: str) -> List[str]:
        """Extract key concepts from paper titles."""
        # ML/AI-specific patterns in titles
        ml_patterns = [
            r'\b(neural networks?|deep learning|machine learning|artificial intelligence)\b',
            r'\b(transformer|attention|BERT|GPT|LSTM|GRU|CNN|RNN)\b',
            r'\b(classification|regression|clustering|reinforcement learning)\b',
            r'\b(computer vision|natural language processing|NLP)\b',
            r'\b(graph neural networks?|GNN|node2vec|word2vec)\b',
            r'\b(generative adversarial networks?|GAN|VAE)\b',
            r'\b(optimization|gradient descent|backpropagation)\b',
            r'\b(embedding|representation|feature learning)\b'
        ]
        
        concepts = []
        for pattern in ml_patterns:
            matches = re.findall(pattern, title, re.IGNORECASE)
            concepts.extend([match.lower() for match in matches])
        
        # Extract capitalized technical terms
        tech_terms = re.findall(r'\b[A-Z][a-z]*[A-Z][A-Za-z]*\b', title)  # CamelCase
        concepts.extend([term.lower() for term in tech_terms])
        
        return list(set(concepts))  # Deduplicate
    
    def _extract_concepts_from_abstract(self, abstract: str) -> List[str]:
        """Extract methods and techniques from abstracts."""
        if not abstract or len(abstract) < 50:
            return []
        
        method_patterns = [
            r'\b(algorithm|method|approach|technique|framework)\b',
            r'\b(model|architecture|network|system)\b',
            r'\b(training|learning|inference|prediction)\b',
            r'\b(dataset|benchmark|evaluation|metric)\b'
        ]
        
        methods = []
        for pattern in method_patterns:
            # Get surrounding context for method mentions
            matches = re.finditer(pattern, abstract, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 20)
                end = min(len(abstract), match.end() + 20)
                context = abstract[start:end]
                
                # Extract meaningful phrases
                words = context.split()
                if len(words) >= 3:
                    methods.append(' '.join(words[1:3]))  # Get surrounding words
        
        return list(set(methods))[:5]  # Top 5 unique methods
    
    def _parse_authors(self, authors: str) -> List[str]:
        """Parse author string into individual names."""
        if not authors:
            return []
        
        # Handle different author formats
        if ' and ' in authors:
            author_list = authors.split(' and ')
        elif ', ' in authors:
            author_list = authors.split(', ')
        else:
            author_list = [authors]
        
        # Clean and standardize names
        cleaned_authors = []
        for author in author_list:
            # Remove extra whitespace and common suffixes
            clean_name = author.strip()
            clean_name = re.sub(r'\s+', ' ', clean_name)  # Normalize whitespace
            
            # Skip very short names or initials-only
            if len(clean_name) > 3 and not re.match(r'^[A-Z]\.\s*[A-Z]\.$', clean_name):
                cleaned_authors.append(clean_name)
        
        return cleaned_authors[:10]  # Max 10 authors per paper
    
    def _aggregate_semantic_entities(self, entities: List[Dict]) -> List[Dict]:
        """Aggregate entities by name and type."""
        entity_dict = {}
        
        for entity in entities:
            key = (entity['name'].lower(), entity['type'])
            
            if key in entity_dict:
                existing = entity_dict[key]
                existing['frequency'] += 1
                existing['paper_ids'].append(entity['paper_id'])
                existing['sources'].add(entity['source'])
                existing['confidence'] = max(existing['confidence'], entity['confidence'])
            else:
                entity_dict[key] = {
                    'name': entity['name'],
                    'type': entity['type'],
                    'frequency': 1,
                    'paper_ids': [entity['paper_id']],
                    'sources': {entity['source']},
                    'confidence': entity['confidence'],
                    'category_code': entity.get('category_code')  # For research areas
                }
        
        # Convert to list and add IDs
        result = []
        for (name, etype), data in entity_dict.items():
            data['_key'] = self._generate_entity_id(name, etype)
            data['paper_ids'] = list(set(data['paper_ids']))  # Deduplicate
            data['sources'] = list(data['sources'])
            result.append(data)
        
        return result
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate consistent entity ID."""
        content = f"{name.lower().strip()}_{entity_type}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def create_metadata_based_clusters(self, entities: List[Dict]) -> List[Dict]:
        """Create clusters based on ArXiv metadata patterns."""
        logger.info("Creating metadata-based clusters...")
        
        # Separate entities by type for better clustering
        entities_by_type = {}
        for entity in entities:
            etype = entity['type']
            if etype not in entities_by_type:
                entities_by_type[etype] = []
            entities_by_type[etype].append(entity)
        
        all_clusters = []
        
        # 1. Research area clusters (from ArXiv categories)
        if 'research_area' in entities_by_type:
            area_clusters = self._create_research_area_clusters(entities_by_type['research_area'])
            all_clusters.extend(area_clusters)
        
        # 2. Concept clusters (from titles and abstracts)  
        concept_entities = []
        for etype in ['concept', 'method']:
            if etype in entities_by_type:
                concept_entities.extend(entities_by_type[etype])
        
        if concept_entities:
            concept_clusters = self._create_concept_clusters(concept_entities)
            all_clusters.extend(concept_clusters)
        
        # 3. Collaboration clusters (from authors)
        if 'researcher' in entities_by_type:
            collab_clusters = self._create_collaboration_clusters(entities_by_type['researcher'])
            all_clusters.extend(collab_clusters)
        
        logger.info(f"Created {len(all_clusters)} metadata-based clusters")
        return all_clusters
    
    def _create_research_area_clusters(self, research_areas: List[Dict]) -> List[Dict]:
        """Create clusters for research areas based on ArXiv categories."""
        # Group by category family (cs.*, math.*, etc.)
        family_groups = {}
        
        for area in research_areas:
            category_code = area.get('category_code', '')
            if '.' in category_code:
                family = category_code.split('.')[0]  # cs, math, stat, etc.
            else:
                family = 'other'
            
            if family not in family_groups:
                family_groups[family] = []
            family_groups[family].append(area)
        
        clusters = []
        for family, areas in family_groups.items():
            if len(areas) < 2:
                continue
                
            cluster = {
                '_key': self._generate_cluster_id(f"{family}_research", 1),
                'name': f"{family.upper()} Research Areas",
                'layer': 1,
                'members': [area['_key'] for area in areas],
                'member_count': len(areas),
                'summary': f"Research areas in {family} domain with {sum(a['frequency'] for a in areas)} papers",
                'key_concepts': [area['name'] for area in areas[:5]],
                'cohesion_score': 0.9,  # High cohesion for official categories
                'cluster_type': 'research_area',
                'created': datetime.now(timezone.utc).isoformat()
            }
            clusters.append(cluster)
        
        return clusters
    
    def _create_concept_clusters(self, concepts: List[Dict]) -> List[Dict]:
        """Create concept clusters using TF-IDF similarity."""
        if len(concepts) < 10:  # Need minimum concepts for clustering
            return []
        
        # Create text representations
        texts = []
        for concept in concepts:
            text = f"{concept['name']} {' '.join(concept['sources'])} freq_{concept['frequency']}"
            texts.append(text)
        
        # TF-IDF clustering
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        embeddings = vectorizer.fit_transform(texts).toarray()
        
        # Use smaller number of clusters for concepts
        n_clusters = min(15, max(3, len(concepts) // 20))
        
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(embeddings)
        
        # Create concept clusters
        clusters = []
        for cluster_id in range(n_clusters):
            member_indices = np.where(labels == cluster_id)[0]
            if len(member_indices) < 3:
                continue
            
            cluster_concepts = [concepts[i] for i in member_indices]
            
            cluster = {
                '_key': self._generate_cluster_id(f"concepts_{cluster_id}", 1),
                'name': f"Concept Cluster {cluster_id + 1}",
                'layer': 1,
                'members': [c['_key'] for c in cluster_concepts],
                'member_count': len(cluster_concepts),
                'summary': f"Conceptual cluster with {len(cluster_concepts)} related concepts",
                'key_concepts': [c['name'] for c in cluster_concepts[:5]],
                'cohesion_score': 0.7,
                'cluster_type': 'concept',
                'created': datetime.now(timezone.utc).isoformat()
            }
            clusters.append(cluster)
        
        return clusters
    
    def _create_collaboration_clusters(self, researchers: List[Dict]) -> List[Dict]:
        """Create collaboration clusters based on co-authorship patterns."""
        # For now, create geographic/institutional clusters based on names
        # This is a simplified approach - real collaboration would need co-authorship analysis
        
        # Group by common name patterns (rough geographic clustering)
        name_groups = {
            'chinese': [],
            'western': [],
            'indian': [],
            'other': []
        }
        
        chinese_patterns = ['Wang', 'Li', 'Zhang', 'Chen', 'Liu', 'Yang', 'Huang', 'Zhao', 'Wu', 'Zhou']
        indian_patterns = ['Kumar', 'Singh', 'Sharma', 'Gupta', 'Agarwal']
        
        for researcher in researchers:
            name = researcher['name']
            if any(pattern in name for pattern in chinese_patterns):
                name_groups['chinese'].append(researcher)
            elif any(pattern in name for pattern in indian_patterns):
                name_groups['indian'].append(researcher)
            elif re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', name):  # Western naming pattern
                name_groups['western'].append(researcher)
            else:
                name_groups['other'].append(researcher)
        
        clusters = []
        for group_name, group_researchers in name_groups.items():
            if len(group_researchers) < 10:  # Skip small groups
                continue
            
            cluster = {
                '_key': self._generate_cluster_id(f"researchers_{group_name}", 1),
                'name': f"{group_name.title()} Researchers",
                'layer': 1,
                'members': [r['_key'] for r in group_researchers],
                'member_count': len(group_researchers),
                'summary': f"Research collaboration cluster: {len(group_researchers)} {group_name} researchers",
                'key_concepts': [r['name'] for r in sorted(group_researchers, key=lambda x: x['frequency'], reverse=True)[:5]],
                'cohesion_score': 0.6,
                'cluster_type': 'collaboration',
                'created': datetime.now(timezone.utc).isoformat()
            }
            clusters.append(cluster)
        
        return clusters
    
    def _generate_cluster_id(self, name: str, layer: int) -> str:
        """Generate cluster ID."""
        content = f"cluster_L{layer}_{name.lower().replace(' ', '_')}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def store_metadata_entities_and_clusters(self, entities: List[Dict], clusters: List[Dict]) -> bool:
        """Store metadata-based entities and clusters."""
        try:
            # Clear existing entities and clusters
            self.db.collection("entities").truncate()
            self.db.collection("clusters").truncate() 
            self.db.collection("cluster_edges").truncate()
            self.db.collection("paper_entities").truncate()
            
            logger.info("Cleared existing data")
            
            # Store entities
            entities_collection = self.db.collection("entities")
            entity_docs = []
            
            for entity in entities:
                entity_doc = {
                    '_key': entity['_key'],
                    'name': entity['name'],
                    'type': entity['type'],
                    'layer': 0,  # Base entities
                    'frequency': entity['frequency'],
                    'confidence': entity['confidence'],
                    'source_papers': entity['paper_ids'],
                    'extraction_sources': entity['sources'],
                    'category_code': entity.get('category_code'),
                    'created': datetime.now(timezone.utc).isoformat()
                }
                entity_docs.append(entity_doc)
            
            # Batch insert entities
            batch_size = 1000
            for i in range(0, len(entity_docs), batch_size):
                batch = entity_docs[i:i + batch_size]
                entities_collection.insert_many(batch, overwrite=True)
            
            logger.info(f"✅ Stored {len(entity_docs)} metadata-based entities")
            
            # Store clusters
            clusters_collection = self.db.collection("clusters")
            clusters_collection.insert_many(clusters, overwrite=True)
            
            logger.info(f"✅ Stored {len(clusters)} metadata-based clusters")
            
            # Create cluster edges
            cluster_edges = []
            for cluster in clusters:
                for member_id in cluster['members']:
                    cluster_edges.append({
                        '_from': f"entities/{member_id}",
                        '_to': f"clusters/{cluster['_key']}",
                        'type': 'member_of',
                        'membership_score': 0.9,
                        'created': datetime.now(timezone.utc).isoformat()
                    })
            
            # Store edges
            edges_collection = self.db.collection("cluster_edges")
            for i in range(0, len(cluster_edges), batch_size):
                batch = cluster_edges[i:i + batch_size]
                edges_collection.insert_many(batch, overwrite=True)
            
            logger.info(f"✅ Created {len(cluster_edges)} cluster edges")
            
            # Create paper-entity relationships
            paper_entities = []
            for entity in entities:
                for paper_id in entity['paper_ids']:
                    paper_entities.append({
                        '_from': f"arxiv_papers/{paper_id}",
                        '_to': f"entities/{entity['_key']}",
                        'type': 'mentions',
                        'confidence': entity['confidence'],
                        'extraction_source': 'arxiv_metadata'
                    })
            
            paper_entities_collection = self.db.collection("paper_entities")
            for i in range(0, len(paper_entities), batch_size):
                batch = paper_entities[i:i + batch_size]
                paper_entities_collection.insert_many(batch, overwrite=True)
            
            logger.info(f"✅ Created {len(paper_entities)} paper-entity relationships")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store metadata entities and clusters: {e}")
            return False
    
    async def build_arxiv_metadata_clusters(self) -> bool:
        """Build complete metadata-driven clustering."""
        logger.info("Starting ArXiv metadata-driven clustering...")
        
        # Load papers
        papers = await self.load_arxiv_papers_metadata()
        if not papers:
            return False
        
        # Extract semantic features
        entities = self.extract_semantic_features(papers)
        logger.info(f"Extracted {len(entities)} semantic entities from metadata")
        
        # Create clusters
        clusters = self.create_metadata_based_clusters(entities)
        logger.info(f"Created {len(clusters)} metadata-based clusters")
        
        # Store everything
        success = await self.store_metadata_entities_and_clusters(entities, clusters)
        
        if success:
            logger.info("✅ ArXiv metadata clustering completed!")
            return True
        else:
            return False


async def main():
    """Main execution function."""
    password = os.getenv('ARANGO_PASSWORD')
    if not password:
        print("Please set ARANGO_PASSWORD environment variable")
        return
    
    clustering = ArXivMetadataClustering(password=password)
    
    if not await clustering.connect():
        return
    
    success = await clustering.build_arxiv_metadata_clusters()
    
    if success:
        print("\n🎉 ArXiv metadata-driven clustering completed!")
        print("Now using structured ArXiv taxonomies and metadata!")
    else:
        print("\n❌ Metadata clustering failed!")


if __name__ == "__main__":
    asyncio.run(main())