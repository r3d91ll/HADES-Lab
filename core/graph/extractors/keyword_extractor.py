"""
Keyword extraction and embedding for academic papers.

Implements the WHAT dimension of Information Reconstructionism - extracting
semantic content indicators that enable interdisciplinary bridge discovery.
Keywords represent the conceptual DNA that can migrate across field boundaries.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    Extract and embed keywords from academic papers.
    
    In ANT terms, keywords are the "immutable mobiles" - concepts that
    maintain identity while crossing disciplinary boundaries.
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        use_gpu: bool = True,
        batch_size: int = 32
    ):
        """Initialize keyword extractor with SciBERT model."""
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing keyword extractor on {self.device}")
        
        # Load SciBERT for scientific text embeddings
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        
        # Load spaCy for NER and POS tagging
        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except:
            logger.warning("Scientific spaCy model not found, using general model")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Common stop words in scientific text
        self.stop_words = set([
            'paper', 'study', 'work', 'approach', 'method', 'result',
            'show', 'find', 'use', 'using', 'based', 'propose', 'proposed',
            'present', 'presented', 'discuss', 'discussed', 'consider',
            'considered', 'describe', 'described', 'analyze', 'analyzed'
        ])
    
    def extract_keywords_from_text(
        self, 
        text: str, 
        max_keywords: int = 20
    ) -> List[str]:
        """
        Extract keywords using multiple strategies.
        
        Combines:
        1. Named Entity Recognition (for specific concepts)
        2. TF-IDF (for statistically significant terms)
        3. Noun phrases (for composite concepts)
        """
        if not text or len(text) < 50:
            return []
        
        keywords = set()
        
        # Process with spaCy
        doc = self.nlp(text[:1000000])  # Limit to 1M chars for memory
        
        # 1. Extract named entities (focus on scientific ones)
        scientific_ent_types = {'ORG', 'PRODUCT', 'LAW', 'LANGUAGE', 'WORK_OF_ART'}
        for ent in doc.ents:
            if ent.label_ in scientific_ent_types:
                clean_ent = self._clean_keyword(ent.text)
                if clean_ent:
                    keywords.add(clean_ent)
        
        # 2. Extract noun phrases
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 4:  # Multi-word concepts
                clean_chunk = self._clean_keyword(chunk.text)
                if clean_chunk and clean_chunk not in self.stop_words:
                    keywords.add(clean_chunk)
        
        # 3. Extract technical terms (capitalized or with numbers)
        technical_pattern = re.compile(r'\b[A-Z][A-Za-z0-9-]*[A-Za-z0-9]\b|\b[A-Za-z]+[0-9]+\b')
        for match in technical_pattern.findall(text):
            clean_match = self._clean_keyword(match)
            if clean_match and len(clean_match) > 2:
                keywords.add(clean_match)
        
        # 4. Use TF-IDF for important terms
        try:
            # Split into sentences for TF-IDF
            sentences = [sent.text for sent in doc.sents][:100]  # Limit sentences
            if len(sentences) > 3:
                vectorizer = TfidfVectorizer(
                    max_features=max_keywords,
                    stop_words='english',
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=0.8
                )
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top TF-IDF terms
                scores = tfidf_matrix.sum(axis=0).A1
                top_indices = scores.argsort()[-max_keywords:][::-1]
                
                for idx in top_indices:
                    term = feature_names[idx]
                    clean_term = self._clean_keyword(term)
                    if clean_term and clean_term not in self.stop_words:
                        keywords.add(clean_term)
        except Exception as e:
            logger.debug(f"TF-IDF extraction failed: {e}")
        
        # Convert to list and limit
        keyword_list = list(keywords)
        return keyword_list[:max_keywords]
    
    def _clean_keyword(self, keyword: str) -> Optional[str]:
        """Clean and validate keyword."""
        # Remove extra whitespace and lowercase
        keyword = ' '.join(keyword.split()).lower()
        
        # Remove special characters at edges
        keyword = keyword.strip('.,;:!?()[]{}"\'-')
        
        # Validate
        if len(keyword) < 2 or len(keyword) > 50:
            return None
        
        # Must contain at least one letter
        if not any(c.isalpha() for c in keyword):
            return None
        
        return keyword
    
    def extract_keywords_from_paper(
        self,
        paper: Dict[str, Any],
        use_title: bool = True,
        use_abstract: bool = True,
        use_categories: bool = True
    ) -> List[str]:
        """
        Extract keywords from paper metadata.
        
        Combines title, abstract, and category information for
        comprehensive keyword extraction.
        """
        text_parts = []
        
        # Add title (weighted higher by repetition)
        if use_title and paper.get('title'):
            text_parts.extend([paper['title']] * 3)
        
        # Add abstract
        if use_abstract and paper.get('abstract'):
            text_parts.append(paper['abstract'])
        
        # Add categories as keywords
        category_keywords = []
        if use_categories and paper.get('categories'):
            for cat in paper['categories']:
                # Convert category codes to readable form
                cat_parts = cat.replace('-', ' ').replace('.', ' ').split()
                category_keywords.extend(cat_parts)
        
        # Combine all text
        full_text = ' '.join(text_parts)
        
        # Extract keywords
        keywords = self.extract_keywords_from_text(full_text)
        
        # Add category keywords (ensure they're included)
        for cat_kw in category_keywords:
            clean_kw = self._clean_keyword(cat_kw)
            if clean_kw and clean_kw not in keywords:
                keywords.append(clean_kw)
        
        return keywords[:25]  # Limit total keywords
    
    def embed_keywords(
        self,
        keywords: List[str]
    ) -> Optional[np.ndarray]:
        """
        Create embedding from keywords.
        
        Returns 768-dimensional SciBERT embedding representing
        the semantic centroid of the keyword space.
        """
        if not keywords:
            return None
        
        try:
            # Join keywords into a single text
            keyword_text = ' '.join(keywords)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model.encode(
                    keyword_text,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed keywords: {e}")
            return None
    
    def batch_process_papers(
        self,
        papers: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Tuple[List[str], Optional[np.ndarray]]]:
        """
        Process multiple papers in batch.
        
        Returns list of (keywords, embedding) tuples.
        """
        results = []
        
        iterator = tqdm(papers, desc="Extracting keywords") if show_progress else papers
        
        for paper in iterator:
            # Extract keywords
            keywords = self.extract_keywords_from_paper(paper)
            
            # Create embedding
            embedding = self.embed_keywords(keywords) if keywords else None
            
            results.append((keywords, embedding))
        
        return results
    
    def find_interdisciplinary_keywords(
        self,
        papers_by_field: Dict[str, List[Dict[str, Any]]],
        min_fields: int = 2,
        min_occurrences: int = 5
    ) -> Dict[str, List[str]]:
        """
        Find keywords that appear across multiple fields.
        
        These "boundary objects" indicate potential interdisciplinary
        bridges per Actor-Network Theory.
        """
        # Collect keywords by field
        field_keywords = {}
        
        for field, papers in papers_by_field.items():
            field_keywords[field] = Counter()
            
            for paper in papers:
                keywords = self.extract_keywords_from_paper(paper)
                field_keywords[field].update(keywords)
        
        # Find keywords appearing in multiple fields
        interdisciplinary = {}
        
        all_keywords = set()
        for field_kws in field_keywords.values():
            all_keywords.update(field_kws.keys())
        
        for keyword in all_keywords:
            fields_with_keyword = []
            total_count = 0
            
            for field, kw_counter in field_keywords.items():
                if keyword in kw_counter:
                    fields_with_keyword.append(field)
                    total_count += kw_counter[keyword]
            
            if len(fields_with_keyword) >= min_fields and total_count >= min_occurrences:
                interdisciplinary[keyword] = fields_with_keyword
        
        return interdisciplinary


def main():
    """Test keyword extraction."""
    import os
    from arango import ArangoClient
    
    # Connect to database
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('academy_store', username='root', password=os.environ.get('ARANGO_PASSWORD'))
    
    # Initialize extractor
    extractor = KeywordExtractor()
    
    # Get sample papers
    papers = list(db.aql.execute(
        "FOR p IN arxiv_papers LIMIT 10 RETURN p"
    ))
    
    # Process papers
    for paper in papers:
        keywords = extractor.extract_keywords_from_paper(paper)
        embedding = extractor.embed_keywords(keywords)
        
        print(f"\nPaper: {paper.get('title', 'Unknown')[:80]}")
        print(f"Categories: {paper.get('categories', [])}")
        print(f"Keywords ({len(keywords)}): {', '.join(keywords[:10])}")
        if embedding is not None:
            print(f"Embedding shape: {embedding.shape}")


if __name__ == "__main__":
    main()