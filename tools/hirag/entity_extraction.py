#!/usr/bin/env python3
"""
HiRAG Entity Extraction Pipeline
Extracts entities from existing ArXiv chunks in the database
Based on PRD Issue #19 requirements

This module implements the first phase of HiRAG construction:
extracting meaningful entities (concepts, methods, algorithms, people)
from the existing corpus of 42,554 ArXiv chunks.
"""

import os
import sys
import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from arango import ArangoClient
from arango.database import StandardDatabase
import spacy
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    """
    Represents an extracted entity from text.
    
    In anthropological terms, this is a boundary object - something that
    exists coherently across different interpretive communities while
    remaining plastic enough to adapt to local needs.
    """
    name: str
    type: str  # concept, method, algorithm, person, dataset
    description: str
    confidence: float
    source_chunks: List[str]
    source_papers: List[str]
    context_snippets: List[str]
    frequency: int = 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for ArangoDB storage."""
        entity_id = self.generate_id()
        return {
            "_key": entity_id,
            "name": self.name,
            "type": self.type,
            "layer": 0,  # Base entities are layer 0
            "cluster_id": None,  # Will be assigned during clustering
            "description": self.description,
            "semantic_embedding": None,  # Will be computed separately
            "source_papers": self.source_papers,
            "source_chunks": self.source_chunks,
            "frequency": self.frequency,
            "importance_score": None,  # Will be computed during graph analysis
            "confidence": self.confidence,
            "context_snippets": self.context_snippets[:3],  # Keep top 3 contexts
            "created": datetime.utcnow().isoformat() + "Z",
            "updated": datetime.utcnow().isoformat() + "Z",
            "extraction_method": "spacy_ner_domain_rules"
        }
    
    def generate_id(self) -> str:
        """Generate consistent entity ID based on name and type."""
        content = f"{self.name.lower().strip()}_{self.type}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class HiRAGEntityExtractor:
    """
    Extracts entities from ArXiv paper chunks using hybrid NER + domain rules.
    
    Following Actor-Network Theory, this extractor serves as a translator -
    it enrolls textual actors into our institutional network by categorizing
    and standardizing them according to our domain ontology.
    """
    
    def __init__(self, host: str = "192.168.1.69", port: int = 8529, 
                 username: str = "root", password: str = None):
        """Initialize the entity extractor."""
        self.password = password or os.getenv('ARANGO_PASSWORD', 'root_password')
        self.client = ArangoClient(hosts=f"http://{host}:{port}")
        self.db: Optional[StandardDatabase] = None
        
        # NLP models
        self.nlp = None
        self.embedder = None
        
        # Domain-specific patterns
        self.ml_patterns = self._compile_ml_patterns()
        self.person_patterns = self._compile_person_patterns()
        
    def _compile_ml_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Compile domain-specific ML/AI patterns.
        
        These patterns embody the tacit knowledge of our domain -
        the implicit categories that practitioners use to organize
        their conceptual landscape.
        """
        return {
            "algorithm": [
                re.compile(r'\b(?:gradient descent|backpropagation|attention mechanism|transformer|LSTM|GRU|CNN|RNN)\b', re.I),
                re.compile(r'\b(?:clustering|classification|regression|reinforcement learning|supervised learning)\b', re.I),
                re.compile(r'\b(?:neural network|deep learning|machine learning|artificial intelligence)\b', re.I),
            ],
            "concept": [
                re.compile(r'\b(?:embedding|representation|feature|activation|layer|neuron)\b', re.I),
                re.compile(r'\b(?:loss function|optimizer|learning rate|batch size|epoch)\b', re.I),
                re.compile(r'\b(?:overfitting|underfitting|generalization|regularization)\b', re.I),
                re.compile(r'\b(?:precision|recall|F1 score|accuracy|AUC)\b', re.I),
            ],
            "method": [
                re.compile(r'\b(?:fine-tuning|transfer learning|data augmentation|dropout)\b', re.I),
                re.compile(r'\b(?:cross-validation|grid search|hyperparameter tuning)\b', re.I),
                re.compile(r'\b(?:preprocessing|normalization|standardization|tokenization)\b', re.I),
            ],
            "dataset": [
                re.compile(r'\b(?:ImageNet|COCO|MNIST|CIFAR|Pascal VOC|MS COCO)\b', re.I),
                re.compile(r'\b(?:GLUE|SQuAD|WMT|Penn Treebank|CoNLL)\b', re.I),
                re.compile(r'\b(?:training set|test set|validation set|benchmark)\b', re.I),
            ]
        }
    
    def _compile_person_patterns(self) -> List[re.Pattern]:
        """Compile robust patterns for person name extraction with Unicode support."""
        return [
            # Title + Full names with middle initials/names, hyphens, apostrophes
            re.compile(r'\b(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?)\s+[\p{Lu}][\p{Ll}\-\']*(?:\s+[\p{Lu}]\.?\s+)?[\p{Lu}][\p{Ll}\-\']+(?:\s+[\p{Lu}][\p{Ll}\-\']+)*\b', re.UNICODE),
            # Full names (2+ capitalized words, allowing hyphens/apostrophes)  
            re.compile(r'\b[\p{Lu}][\p{Ll}\-\']+(?:\s+[\p{Lu}][\p{Ll}\-\']+)+\b', re.UNICODE),
            # Initials + Last name (F. Last, F.M. Last)
            re.compile(r'\b[\p{Lu}]\.(?:\s*[\p{Lu}]\.)?\s+[\p{Lu}][\p{Ll}\-\']+\b', re.UNICODE),
            # Et al. patterns
            re.compile(r'\b[\p{Lu}][\p{Ll}\-\']+(?:\s+[\p{Lu}][\p{Ll}\-\']+)*\s+et\s+al\.?\b', re.UNICODE),
        ]
    
    def _filter_person_names(self, candidates: List[str]) -> List[str]:
        """Filter person name candidates to remove false positives."""
        # Common stopwords and false positives
        stopwords = {
            'The', 'This', 'That', 'These', 'Those', 'Our', 'We', 'They', 'It', 'Is', 'Are',
            'Figure', 'Table', 'Section', 'Chapter', 'Appendix', 'References', 'Abstract',
            'Neural Network', 'Machine Learning', 'Deep Learning', 'Computer Vision',
            'Natural Language', 'Data Science', 'Artificial Intelligence', 'Big Data'
        }
        
        filtered = []
        for name in candidates:
            # Skip if in stopwords
            if name in stopwords:
                continue
                
            # Skip single tokens (too ambiguous)
            if len(name.split()) < 2:
                continue
                
            # Skip if all tokens are short (likely abbreviations)
            tokens = name.split()
            if all(len(token) <= 2 for token in tokens):
                continue
                
            filtered.append(name)
            
        return filtered
    
    async def connect(self, database_name: str = "academy_store") -> bool:
        """Connect to the ArangoDB database."""
        try:
            self.db = self.client.db(database_name, username="root", password=self.password)
            logger.info(f"Connected to database: {database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def load_models(self):
        """
        Load NLP models for entity extraction.
        
        This represents the moment when external technological actors
        (pre-trained models) become enrolled in our local network,
        bringing their own agencies and capabilities.
        """
        try:
            # Load spaCy model for NER
            logger.info("Loading spaCy model for NER...")
            self.nlp = spacy.load("en_core_web_sm")
            
            # Add custom entity ruler for domain-specific terms
            if "entity_ruler" not in self.nlp.pipe_names:
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                patterns = self._create_custom_patterns()
                ruler.add_patterns(patterns)
            
            logger.info("✅ NLP models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}")
            raise
    
    def _create_custom_patterns(self) -> List[Dict]:
        """Create custom entity patterns for domain-specific terms."""
        patterns = []
        
        # ML algorithms
        ml_terms = [
            "gradient descent", "backpropagation", "attention mechanism", 
            "transformer", "LSTM", "GRU", "CNN", "RNN", "VAE", "GAN",
            "reinforcement learning", "supervised learning", "unsupervised learning",
            "deep learning", "neural network", "machine learning"
        ]
        
        for term in ml_terms:
            patterns.append({"label": "ALGORITHM", "pattern": term})
            
        # Evaluation metrics
        metrics = ["precision", "recall", "F1 score", "accuracy", "AUC", "BLEU", "ROUGE"]
        for metric in metrics:
            patterns.append({"label": "METRIC", "pattern": metric})
            
        return patterns
    
    def extract_entities_from_text(self, text: str, chunk_id: str, paper_id: str) -> List[ExtractedEntity]:
        """
        Extract entities from a single text chunk.
        
        This method performs the translation work - converting raw text into
        structured entities that can participate in our knowledge network.
        """
        entities = []
        
        # spaCy NER extraction
        doc = self.nlp(text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "ALGORITHM", "METRIC"]:
                entity_type = self._map_spacy_label(ent.label_)
                entities.append(ExtractedEntity(
                    name=ent.text.strip(),
                    type=entity_type,
                    description=f"{entity_type.title()} mentioned in {paper_id}",
                    confidence=0.8,  # Base confidence for spaCy
                    source_chunks=[chunk_id],
                    source_papers=[paper_id],
                    context_snippets=[self._extract_context(text, ent.start_char, ent.end_char)]
                ))
        
        # Pattern-based extraction for domain-specific terms
        for entity_type, patterns in self.ml_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    entities.append(ExtractedEntity(
                        name=match.group().strip(),
                        type=entity_type,
                        description=f"{entity_type.title()} term from {paper_id}",
                        confidence=0.9,  # Higher confidence for pattern matching
                        source_chunks=[chunk_id],
                        source_papers=[paper_id],
                        context_snippets=[self._extract_context(text, match.start(), match.end())]
                    ))
        
        return self._deduplicate_entities(entities)
    
    def _map_spacy_label(self, label: str) -> str:
        """Map spaCy entity labels to our domain types."""
        mapping = {
            "PERSON": "person",
            "ORG": "organization", 
            "ALGORITHM": "algorithm",
            "METRIC": "concept"
        }
        return mapping.get(label, "concept")
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around an entity mention."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """
        Deduplicate entities within a single text chunk.
        
        This represents the standardization process - ensuring that multiple
        mentions of the same concept are recognized as referring to the
        same actor in our network.
        """
        entity_dict = {}
        
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key in entity_dict:
                # Merge with existing entity
                existing = entity_dict[key]
                existing.frequency += 1
                existing.context_snippets.extend(entity.context_snippets)
                existing.confidence = max(existing.confidence, entity.confidence)
            else:
                entity_dict[key] = entity
                
        return list(entity_dict.values())
    
    async def process_chunks_batch(self, chunks: List[Dict], batch_size: int = 100) -> List[ExtractedEntity]:
        """
        Process a batch of chunks for entity extraction.
        
        This implements the bureaucratic processing logic - systematic,
        standardized procedures applied uniformly across our document corpus.
        """
        all_entities = []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            for chunk in chunks:
                if chunk.get('content'):  # Skip chunks with no content
                    future = executor.submit(
                        self.extract_entities_from_text,
                        chunk['content'],
                        chunk['_key'],
                        chunk['paper_id']
                    )
                    futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    chunk_entities = future.result()
                    all_entities.extend(chunk_entities)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
        
        return self._aggregate_entities(all_entities)
    
    def _aggregate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """
        Aggregate entities across all chunks.
        
        This creates the canonical representation of each entity -
        the authoritative version that will serve as the obligatory
        passage point for all future references.
        """
        entity_dict = {}
        
        for entity in entities:
            key = entity.generate_id()
            
            if key in entity_dict:
                # Merge with existing
                existing = entity_dict[key]
                existing.frequency += entity.frequency
                existing.source_chunks.extend(entity.source_chunks)
                existing.source_papers.extend(entity.source_papers)
                existing.context_snippets.extend(entity.context_snippets)
                existing.confidence = max(existing.confidence, entity.confidence)
                
                # Deduplicate lists
                existing.source_chunks = list(set(existing.source_chunks))
                existing.source_papers = list(set(existing.source_papers))
                existing.context_snippets = existing.context_snippets[:10]  # Keep top 10
                
            else:
                entity_dict[key] = entity
        
        return list(entity_dict.values())
    
    async def extract_all_entities(self, limit: Optional[int] = None) -> bool:
        """
        Extract entities from all ArXiv chunks in the database.
        
        This represents the institutional processing moment - the systematic
        transformation of raw textual material into structured knowledge
        objects that can participate in our hierarchical retrieval system.
        """
        try:
            # Query all chunks
            logger.info("Querying ArXiv chunks from database...")
            
            if limit:
                query = f"""
                FOR chunk IN arxiv_chunks
                    FILTER chunk.text != null AND chunk.text != ""
                    LIMIT {limit}
                    RETURN {{
                        _key: chunk._key,
                        paper_id: chunk.paper_id,
                        content: chunk.text,
                        chunk_index: chunk.chunk_index
                    }}
                """
            else:
                query = """
                FOR chunk IN arxiv_chunks
                    FILTER chunk.text != null AND chunk.text != ""
                    RETURN {
                        _key: chunk._key,
                        paper_id: chunk.paper_id,
                        content: chunk.text,
                        chunk_index: chunk.chunk_index
                    }
                """
            
            cursor = self.db.aql.execute(query)
            chunks = list(cursor)
            logger.info(f"Found {len(chunks)} chunks to process")
            
            if not chunks:
                logger.warning("No chunks found with content!")
                return False
            
            # Process in batches
            batch_size = 500
            all_entities = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                batch_entities = await self.process_chunks_batch(batch)
                all_entities.extend(batch_entities)
                
                logger.info(f"Extracted {len(batch_entities)} entities from batch")
            
            # Final aggregation across all batches
            logger.info("Aggregating entities across all batches...")
            final_entities = self._aggregate_entities(all_entities)
            logger.info(f"Total unique entities extracted: {len(final_entities)}")
            
            # Store entities in database
            return await self.store_entities(final_entities)
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return False
    
    async def store_entities(self, entities: List[ExtractedEntity]) -> bool:
        """
        Store extracted entities in the ArangoDB entities collection.
        
        This completes the enrollment process - the entities are now
        full participants in our knowledge network, ready to form
        relationships and participate in hierarchical clustering.
        """
        try:
            entities_collection = self.db.collection("entities")
            
            # Convert to documents
            entity_docs = [entity.to_dict() for entity in entities]
            
            # Batch insert with overwrite
            batch_size = 1000
            inserted_count = 0
            
            for i in range(0, len(entity_docs), batch_size):
                batch = entity_docs[i:i + batch_size]
                result = entities_collection.insert_many(batch, overwrite=True)
                inserted_count += len([r for r in result if not r.get('error')])
                
                logger.info(f"Stored batch {i//batch_size + 1}: {inserted_count} entities")
            
            logger.info(f"✅ Successfully stored {inserted_count} entities")
            
            # Create paper-entity relationships
            await self.create_paper_entity_relationships(entities)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store entities: {e}")
            return False
    
    async def create_paper_entity_relationships(self, entities: List[ExtractedEntity]) -> bool:
        """Create paper-to-entity relationship edges."""
        try:
            paper_entities_collection = self.db.collection("paper_entities")
            relationships = []
            
            for entity in entities:
                for paper_id in entity.source_papers:
                    rel_doc = {
                        "_from": f"arxiv_papers/{paper_id}",
                        "_to": f"entities/{entity.generate_id()}",
                        "type": "mentions",
                        "mention_count": entity.source_papers.count(paper_id),
                        "confidence": entity.confidence,
                        "context_chunks": entity.source_chunks,
                        "extraction_method": "spacy_ner_domain_rules",
                        "created": datetime.utcnow().isoformat() + "Z"
                    }
                    relationships.append(rel_doc)
            
            # Insert relationships
            if relationships:
                result = paper_entities_collection.insert_many(relationships, overwrite=True)
                success_count = len([r for r in result if not r.get('error')])
                logger.info(f"✅ Created {success_count} paper-entity relationships")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to create paper-entity relationships: {e}")
            return False


async def main():
    """Main execution function."""
    # Get password from environment
    password = os.getenv('ARANGO_PASSWORD')
    if not password:
        print("Please set ARANGO_PASSWORD environment variable")
        return
    
    # Create extractor
    extractor = HiRAGEntityExtractor(password=password)
    
    # Connect to database
    if not await extractor.connect():
        return
    
    # Load models
    extractor.load_models()
    
    # Extract entities (limit for testing)
    logger.info("Starting entity extraction pipeline...")
    
    # Process FULL corpus - all 42,554 chunks
    success = await extractor.extract_all_entities(limit=None)
    
    if success:
        print("\n🎉 Entity extraction completed successfully!")
        print("Next steps:")
        print("1. Generate embeddings for entities")
        print("2. Run hierarchical clustering")
        print("3. Build retrieval engine")
    else:
        print("\n❌ Entity extraction failed!")


if __name__ == "__main__":
    asyncio.run(main())