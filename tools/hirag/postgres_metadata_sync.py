#!/usr/bin/env python3
"""
PostgreSQL to ArangoDB Metadata Synchronization
Updates ArangoDB papers with structured ArXiv metadata from PostgreSQL
"""

import os
import sys
import asyncio
import psycopg2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timezone
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from arango import ArangoClient
from arango.database import StandardDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostgreSQLMetadataSync:
    """
    Synchronize ArXiv metadata from PostgreSQL to ArangoDB for HiRAG processing.
    
    This represents the bureaucratic data transfer between institutional systems -
    the PostgreSQL archive becomes the authoritative source of truth,
    while ArangoDB serves as the operational graph database.
    """
    
    def __init__(self, 
                 arango_host: str = "192.168.1.69", 
                 postgres_host: str = "localhost",
                 arango_password: str = None,
                 postgres_password: str = None):
        """Initialize dual-database synchronization."""
        self.arango_password = arango_password or os.getenv('ARANGO_PASSWORD', 'root_password') 
        self.postgres_password = postgres_password or os.getenv('PGPASSWORD', '1luv93ngu1n$')
        
        # ArangoDB connection
        self.arango_client = ArangoClient(hosts=f"http://{arango_host}:8529")
        self.arango_db: Optional[StandardDatabase] = None
        
        # PostgreSQL connection
        self.postgres_conn = None
        self.postgres_host = postgres_host
        
        # ArXiv category mappings
        self.category_descriptions = {
            # Computer Science
            'cs.AI': 'Artificial Intelligence',
            'cs.LG': 'Machine Learning',
            'cs.CV': 'Computer Vision and Pattern Recognition',
            'cs.CL': 'Computation and Language',
            'cs.NE': 'Neural and Evolutionary Computing',
            'cs.IR': 'Information Retrieval',
            'cs.RO': 'Robotics',
            'cs.DB': 'Databases',
            'cs.DS': 'Data Structures and Algorithms',
            'cs.DC': 'Distributed, Parallel, and Cluster Computing',
            'cs.CR': 'Cryptography and Security',
            'cs.SE': 'Software Engineering',
            'cs.ET': 'Emerging Technologies',
            'cs.GT': 'Computer Science and Game Theory',
            'cs.HC': 'Human-Computer Interaction',
            'cs.IT': 'Information Theory',
            'cs.MA': 'Multiagent Systems',
            'cs.MM': 'Multimedia',
            'cs.NI': 'Networking and Internet Architecture',
            'cs.OH': 'Other Computer Science',
            'cs.OS': 'Operating Systems',
            'cs.PF': 'Performance',
            'cs.PL': 'Programming Languages',
            'cs.SC': 'Symbolic Computation',
            'cs.SD': 'Sound',
            'cs.SY': 'Systems and Control',
            
            # Mathematics
            'math.CO': 'Combinatorics',
            'math.PR': 'Probability',
            'math.ST': 'Statistics Theory',
            'math.OC': 'Optimization and Control',
            'math.NA': 'Numerical Analysis',
            'math.RT': 'Representation Theory',
            'math.AG': 'Algebraic Geometry',
            'math.AT': 'Algebraic Topology',
            'math.CA': 'Classical Analysis and ODEs',
            'math.CV': 'Complex Variables',
            'math.DG': 'Differential Geometry',
            'math.DS': 'Dynamical Systems',
            'math.FA': 'Functional Analysis',
            'math.GM': 'General Mathematics',
            'math.GR': 'Group Theory',
            'math.GT': 'Geometric Topology',
            'math.HO': 'History and Overview',
            'math.IT': 'Information Theory',
            'math.KT': 'K-Theory and Homology',
            'math.LO': 'Logic',
            'math.MP': 'Mathematical Physics',
            'math.MG': 'Metric Geometry',
            'math.NT': 'Number Theory',
            'math.QA': 'Quantum Algebra',
            'math.RA': 'Rings and Algebras',
            'math.SG': 'Symplectic Geometry',
            'math.SP': 'Spectral Theory',
            
            # Statistics  
            'stat.AP': 'Applications',
            'stat.CO': 'Computation',
            'stat.ML': 'Machine Learning',
            'stat.ME': 'Methodology',
            'stat.OT': 'Other Statistics',
            'stat.TH': 'Statistics Theory',
            
            # Physics
            'physics.gen-ph': 'General Physics',
            'physics.bio-ph': 'Biological Physics',
            'physics.comp-ph': 'Computational Physics',
            'physics.data-an': 'Data Analysis, Statistics and Probability',
            'physics.flu-dyn': 'Fluid Dynamics',
            'physics.med-ph': 'Medical Physics',
            'physics.optics': 'Optics',
            'physics.soc-ph': 'Physics and Society',
            'cond-mat': 'Condensed Matter',
            'cond-mat.stat-mech': 'Statistical Mechanics',
            'quant-ph': 'Quantum Physics',
            'astro-ph': 'Astrophysics',
            'hep-ph': 'High Energy Physics - Phenomenology',
            'hep-th': 'High Energy Physics - Theory',
            'gr-qc': 'General Relativity and Quantum Cosmology',
            'nucl-th': 'Nuclear Theory',
            'nucl-ex': 'Nuclear Experiment',
            
            # Biology
            'q-bio.BM': 'Biomolecules',
            'q-bio.CB': 'Cell Behavior',
            'q-bio.GN': 'Genomics',
            'q-bio.MN': 'Molecular Networks',
            'q-bio.NC': 'Neurons and Cognition',
            'q-bio.OT': 'Other Quantitative Biology',
            'q-bio.PE': 'Populations and Evolution',
            'q-bio.QM': 'Quantitative Methods',
            'q-bio.SC': 'Subcellular Processes',
            'q-bio.TO': 'Tissues and Organs',
            
            # Economics
            'econ.EM': 'Econometrics',
            'econ.GN': 'General Economics',
            'econ.TH': 'Theoretical Economics',
        }
    
    async def connect_arango(self, database_name: str = "academy_store") -> bool:
        """Connect to ArangoDB."""
        try:
            self.arango_db = self.arango_client.db(database_name, username="root", password=self.arango_password)
            logger.info(f"Connected to ArangoDB: {database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            return False
    
    def connect_postgres(self) -> bool:
        """Connect to PostgreSQL."""
        try:
            self.postgres_conn = psycopg2.connect(
                host=self.postgres_host,
                database="arxiv",
                user="postgres", 
                password=self.postgres_password
            )
            logger.info("Connected to PostgreSQL arxiv database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def get_arxiv_papers_with_categories(self, limit: Optional[int] = None) -> List[Dict]:
        """Fetch ArXiv papers with categories from PostgreSQL."""
        cursor = self.postgres_conn.cursor()
        
        # Query to get papers with all their categories
        query = """
        SELECT 
            p.arxiv_id,
            p.title,
            p.abstract,
            p.primary_category,
            p.published_at,
            p.year,
            p.month,
            array_agg(pc.category ORDER BY pc.category) as all_categories
        FROM papers p
        LEFT JOIN paper_categories pc ON p.arxiv_id = pc.arxiv_id
        WHERE p.title IS NOT NULL 
        AND p.abstract IS NOT NULL
        AND LENGTH(p.abstract) > 50
        GROUP BY p.arxiv_id, p.title, p.abstract, p.primary_category, p.published_at, p.year, p.month
        ORDER BY p.published_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            
            papers = []
            for row in rows:
                arxiv_id, title, abstract, primary_category, published_at, year, month, categories = row
                
                # Clean categories (remove None values)
                categories = [cat for cat in (categories or []) if cat]
                
                papers.append({
                    'arxiv_id': arxiv_id,
                    'title': title,
                    'abstract': abstract,
                    'primary_category': primary_category,
                    'categories': categories,
                    'published_at': published_at.isoformat() if published_at else None,
                    'year': year,
                    'month': month
                })
            
            logger.info(f"Fetched {len(papers)} papers with metadata from PostgreSQL")
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            return []
        finally:
            cursor.close()
    
    async def update_arxiv_papers_in_arango(self, papers: List[Dict]) -> bool:
        """Update ArangoDB papers with PostgreSQL metadata."""
        try:
            papers_collection = self.arango_db.collection("arxiv_papers")
            
            # Convert ArXiv IDs to our format (replace . with _)
            updated_count = 0
            not_found_count = 0
            
            for paper in papers:
                # Convert arxiv_id format: "1301.0001" -> "1301_0001"
                arango_key = paper['arxiv_id'].replace('.', '_')
                
                # Check if paper exists in ArangoDB
                try:
                    existing_paper = papers_collection.get(arango_key)
                    if existing_paper:
                        # Update with metadata
                        update_data = {
                            'title': paper['title'],
                            'abstract': paper['abstract'],
                            'categories': ' '.join(paper['categories']) if paper['categories'] else None,
                            'primary_category': paper['primary_category'],
                            'published_at': paper['published_at'],
                            'year': paper['year'],
                            'month': paper['month'],
                            'category_list': paper['categories'],
                            'metadata_source': 'postgresql',
                            'metadata_updated': datetime.now(timezone.utc).isoformat()
                        }
                        
                        papers_collection.update(arango_key, update_data)
                        updated_count += 1
                        
                        if updated_count % 100 == 0:
                            logger.info(f"Updated {updated_count} papers...")
                    else:
                        not_found_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error updating paper {arango_key}: {e}")
                    not_found_count += 1
            
            logger.info(f"✅ Updated {updated_count} papers in ArangoDB")
            logger.info(f"📊 {not_found_count} papers not found in ArangoDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update ArangoDB papers: {e}")
            return False
    
    def extract_entities_from_metadata(self, papers: List[Dict]) -> List[Dict]:
        """Extract structured entities from PostgreSQL metadata."""
        entities = []
        
        for paper in papers:
            paper_id = paper['arxiv_id'].replace('.', '_')  # Convert to ArangoDB format
            
            # 1. Research area entities from categories
            for category in paper.get('categories', []):
                if category in self.category_descriptions:
                    entities.append({
                        'name': self.category_descriptions[category],
                        'type': 'research_area',
                        'source': 'arxiv_category',
                        'category_code': category,
                        'paper_id': paper_id,
                        'confidence': 0.95
                    })
                else:
                    # Create entity for unknown categories too
                    entities.append({
                        'name': category.replace('-', ' ').replace('.', ' ').title(),
                        'type': 'research_area',
                        'source': 'arxiv_category',
                        'category_code': category,
                        'paper_id': paper_id,
                        'confidence': 0.8
                    })
            
            # 2. Extract concepts from titles and abstracts
            title_concepts = self._extract_technical_terms(paper['title'])
            for concept in title_concepts:
                entities.append({
                    'name': concept,
                    'type': 'concept',
                    'source': 'paper_title',
                    'paper_id': paper_id,
                    'confidence': 0.85
                })
            
            abstract_methods = self._extract_methods_from_abstract(paper.get('abstract', ''))
            for method in abstract_methods:
                entities.append({
                    'name': method,
                    'type': 'method',
                    'source': 'paper_abstract',
                    'paper_id': paper_id,
                    'confidence': 0.8
                })
        
        return self._aggregate_entities(entities)
    
    def _extract_technical_terms(self, title: str) -> List[str]:
        """Extract technical terms from paper titles."""
        import re
        
        if not title:
            return []
        
        # Technical patterns
        patterns = [
            # ML/AI terms
            r'\b(neural network|deep learning|machine learning|artificial intelligence|AI)\b',
            r'\b(transformer|attention|BERT|GPT|LSTM|GRU|CNN|RNN|VAE|GAN)\b',
            r'\b(classification|regression|clustering|reinforcement learning|RL)\b',
            r'\b(computer vision|natural language processing|NLP|CV)\b',
            r'\b(graph neural network|GNN|node2vec|word2vec|embedding)\b',
            r'\b(optimization|gradient descent|backpropagation|training)\b',
            
            # General technical terms
            r'\b(algorithm|method|approach|framework|model|architecture)\b',
            r'\b(system|network|protocol|database|software|application)\b',
            r'\b(analysis|evaluation|benchmark|performance|metric)\b',
            
            # Math/Physics terms  
            r'\b(theorem|lemma|proof|equation|formula|matrix|tensor)\b',
            r'\b(probability|statistics|stochastic|bayesian|markov)\b',
            r'\b(quantum|physics|mechanics|dynamics|field|theory)\b'
        ]
        
        concepts = set()
        for pattern in patterns:
            matches = re.findall(pattern, title, re.IGNORECASE)
            for match in matches:
                concepts.add(match.lower())
        
        return list(concepts)
    
    def _extract_methods_from_abstract(self, abstract: str) -> List[str]:
        """Extract methods and techniques from abstracts."""
        import re
        
        if not abstract or len(abstract) < 50:
            return []
        
        # Method patterns
        method_patterns = [
            r'\b(we propose|we present|we introduce|we develop)\s+([a-zA-Z\s]{5,30}?)\s+(?:that|which|to)',
            r'\b(novel|new)\s+([a-zA-Z\s]{5,20}?)\s+(?:for|to|that)',
            r'\b(method|approach|technique|algorithm|framework|model)\s+(?:called|named)?\s*([A-Z][a-zA-Z\s]{3,20})',
            r'\b([A-Z][a-zA-Z]*[A-Z][a-zA-Z]*)\b',  # CamelCase terms
        ]
        
        methods = set()
        for pattern in method_patterns:
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    for part in match:
                        if len(part.strip()) > 3:
                            methods.add(part.strip().lower())
                elif len(match.strip()) > 3:
                    methods.add(match.strip().lower())
        
        return list(methods)[:5]  # Top 5
    
    def _aggregate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Aggregate entities by name and type."""
        entity_dict = {}
        
        for entity in entities:
            key = (entity['name'].lower().strip(), entity['type'])
            
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
                    'category_code': entity.get('category_code')
                }
        
        # Convert to list with IDs
        result = []
        for (name, etype), data in entity_dict.items():
            data['_key'] = self._generate_entity_id(name, etype)
            data['paper_ids'] = list(set(data['paper_ids']))
            data['sources'] = list(data['sources'])
            result.append(data)
        
        return result
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate consistent entity ID."""
        content = f"{name.lower().strip()}_{entity_type}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def create_category_based_clusters(self, entities: List[Dict]) -> List[Dict]:
        """Create clusters based on ArXiv categories."""
        # Group research areas by category family
        category_families = {}
        research_areas = [e for e in entities if e['type'] == 'research_area']
        
        for entity in research_areas:
            category_code = entity.get('category_code', '')
            if '.' in category_code:
                family = category_code.split('.')[0]  # cs, math, stat, etc.
            elif '-' in category_code:
                family = category_code.split('-')[0]  # physics families
            else:
                family = 'other'
            
            if family not in category_families:
                category_families[family] = []
            category_families[family].append(entity)
        
        clusters = []
        for family, areas in category_families.items():
            if len(areas) < 2:
                continue
            
            cluster = {
                '_key': self._generate_cluster_id(f"{family}_research", 1),
                'name': f"{family.upper()} Research Areas",
                'layer': 1,
                'members': [area['_key'] for area in areas],
                'member_count': len(areas),
                'summary': f"Research areas in {family} domain covering {sum(a['frequency'] for a in areas)} papers",
                'key_concepts': [area['name'] for area in areas[:7]],
                'cohesion_score': 0.9,
                'cluster_type': 'research_domain',
                'category_family': family,
                'created': datetime.now(timezone.utc).isoformat()
            }
            clusters.append(cluster)
        
        logger.info(f"Created {len(clusters)} category-based clusters")
        return clusters
    
    def _generate_cluster_id(self, name: str, layer: int) -> str:
        """Generate cluster ID."""
        content = f"cluster_L{layer}_{name.lower().replace(' ', '_')}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def store_metadata_entities_and_clusters(self, entities: List[Dict], clusters: List[Dict]) -> bool:
        """Store entities and clusters in ArangoDB."""
        try:
            # Clear existing
            self.arango_db.collection("entities").truncate()
            self.arango_db.collection("clusters").truncate()
            self.arango_db.collection("cluster_edges").truncate()
            self.arango_db.collection("paper_entities").truncate()
            
            logger.info("Cleared existing HiRAG data")
            
            # Store entities
            entity_docs = []
            for entity in entities:
                entity_doc = {
                    '_key': entity['_key'],
                    'name': entity['name'],
                    'type': entity['type'],
                    'layer': 0,
                    'frequency': entity['frequency'],
                    'confidence': entity['confidence'],
                    'source_papers': entity['paper_ids'],
                    'extraction_sources': entity['sources'],
                    'category_code': entity.get('category_code'),
                    'created': datetime.now(timezone.utc).isoformat(),
                    'metadata_source': 'postgresql'
                }
                entity_docs.append(entity_doc)
            
            # Batch insert entities
            entities_collection = self.arango_db.collection("entities")
            batch_size = 1000
            for i in range(0, len(entity_docs), batch_size):
                batch = entity_docs[i:i + batch_size]
                entities_collection.insert_many(batch, overwrite=True)
            
            logger.info(f"✅ Stored {len(entity_docs)} entities")
            
            # Store clusters
            if clusters:
                clusters_collection = self.arango_db.collection("clusters")
                clusters_collection.insert_many(clusters, overwrite=True)
                logger.info(f"✅ Stored {len(clusters)} clusters")
                
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
                
                edges_collection = self.arango_db.collection("cluster_edges")
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
                        'extraction_source': 'postgresql_metadata',
                        'created': datetime.now(timezone.utc).isoformat()
                    })
            
            paper_entities_collection = self.arango_db.collection("paper_entities")
            for i in range(0, len(paper_entities), batch_size):
                batch = paper_entities[i:i + batch_size]
                paper_entities_collection.insert_many(batch, overwrite=True)
            
            logger.info(f"✅ Created {len(paper_entities)} paper-entity relationships")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            return False
    
    async def full_metadata_sync(self, sample_size: Optional[int] = None) -> bool:
        """Execute complete PostgreSQL to ArangoDB metadata synchronization."""
        logger.info("Starting PostgreSQL → ArangoDB metadata synchronization...")
        
        # Get papers from PostgreSQL
        papers = self.get_arxiv_papers_with_categories(limit=sample_size)
        if not papers:
            return False
        
        logger.info(f"Processing {len(papers)} papers...")
        
        # Update ArangoDB papers with metadata
        await self.update_arxiv_papers_in_arango(papers)
        
        # Extract entities from metadata
        entities = self.extract_entities_from_metadata(papers)
        logger.info(f"Extracted {len(entities)} entities from metadata")
        
        # Create category-based clusters
        clusters = self.create_category_based_clusters(entities)
        
        # Store everything
        success = await self.store_metadata_entities_and_clusters(entities, clusters)
        
        if success:
            logger.info("✅ PostgreSQL metadata synchronization completed!")
            return True
        else:
            return False
    
    def close_connections(self):
        """Close database connections."""
        if self.postgres_conn:
            self.postgres_conn.close()
            logger.info("Closed PostgreSQL connection")


async def main():
    """Main execution function."""
    arango_password = os.getenv('ARANGO_PASSWORD')
    postgres_password = os.getenv('PGPASSWORD')
    
    if not arango_password or not postgres_password:
        print("Please set ARANGO_PASSWORD and PGPASSWORD environment variables")
        return
    
    sync = PostgreSQLMetadataSync(
        arango_password=arango_password,
        postgres_password=postgres_password
    )
    
    try:
        # Connect to both databases
        if not await sync.connect_arango():
            return
            
        if not sync.connect_postgres():
            return
        
        # Run synchronization (start with sample for testing)
        sample_size = 10000  # Process 10K papers to start
        success = await sync.full_metadata_sync(sample_size=sample_size)
        
        if success:
            print(f"\n🎉 Successfully synchronized {sample_size} papers with metadata!")
            print("Ready for metadata-driven HiRAG clustering!")
        else:
            print("\n❌ Metadata synchronization failed!")
            
    finally:
        sync.close_connections()


if __name__ == "__main__":
    asyncio.run(main())