#!/usr/bin/env python3
"""
Extract keywords from papers and generate embeddings for keyword similarity.

Uses titles and abstracts to extract meaningful keywords, then embeds them
for fast similarity search during graph construction.
"""

import os
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from arango import ArangoClient
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List, Dict
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordExtractor:
    """Extract keywords from academic papers."""
    
    def __init__(self, batch_size: int = 32):
        """Initialize the keyword extractor."""
        self.client = ArangoClient(hosts='http://localhost:8529')
        self.db = self.client.db(
            'academy_store',
            username='root',
            password=os.environ.get('ARANGO_PASSWORD')
        )
        
        # Load a lightweight model for keyword embeddings
        logger.info("Loading embedding model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
        self.model.eval()
        
        self.batch_size = batch_size
        
        # Common stop words to filter
        self.stop_words = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'we', 'our', 'is', 'was',
            'are', 'been', 'has', 'had', 'were', 'their', 'they', 'an',
            'will', 'can', 'may', 'should', 'could', 'would', 'must'
        ])
    
    def extract_keywords_from_text(self, title: str, abstract: str, max_keywords: int = 20) -> List[str]:
        """Extract keywords from title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Combine title and abstract
        text = f"{title} {abstract}".lower()
        
        # Extract technical terms and phrases
        keywords = []
        
        # Pattern for technical terms (multi-word or single)
        # Matches: neural network, deep learning, GPU, LSTM, etc.
        technical_pattern = r'\b(?:[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*|[a-z]+(?:\s+[a-z]+){0,2})\b'
        
        # Find all potential keywords
        words = re.findall(r'\b[a-z]+(?:[-_][a-z]+)*\b', text)
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Filter stop words and short words
        filtered_words = [
            word for word, count in word_freq.most_common(100)
            if word not in self.stop_words and len(word) > 2 and count > 1
        ]
        
        # Also extract important multi-word phrases from abstract
        # Look for recurring bigrams and trigrams
        words_list = text.split()
        for i in range(len(words_list) - 1):
            bigram = f"{words_list[i]} {words_list[i+1]}"
            if (words_list[i] not in self.stop_words and 
                words_list[i+1] not in self.stop_words and
                text.count(bigram) > 1):
                keywords.append(bigram)
        
        # Add top single words
        keywords.extend(filtered_words[:max_keywords])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:max_keywords]
    
    def embed_keywords(self, keywords_list: List[List[str]]) -> np.ndarray:
        """Generate embeddings for keyword lists.
        
        Args:
            keywords_list: List of keyword lists (one per paper)
            
        Returns:
            Array of embeddings
        """
        # Convert keyword lists to strings
        texts = [' '.join(keywords) if keywords else 'unknown' for keywords in keywords_list]
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                # Normalize
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
            embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def process_papers(self, limit: int = None):
        """Process papers to extract keywords and generate embeddings.
        
        Args:
            limit: Maximum number of papers to process (None for all)
        """
        logger.info("Processing papers for keyword extraction...")
        
        # Get papers without keywords
        query = """
        FOR paper IN arxiv_papers
            FILTER paper.keywords == null
            LIMIT @limit
            RETURN paper
        """
        
        if limit is None:
            query = query.replace("LIMIT @limit", "")
            papers = list(self.db.aql.execute(query))
        else:
            papers = list(self.db.aql.execute(query, bind_vars={'limit': limit}))
        
        logger.info(f"Found {len(papers):,} papers to process")
        
        if not papers:
            logger.info("No papers to process")
            return
        
        # Process in batches
        batch_size = 1000
        total_processed = 0
        
        for batch_start in tqdm(range(0, len(papers), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(papers))
            batch = papers[batch_start:batch_end]
            
            # Extract keywords for batch
            keywords_batch = []
            paper_updates = []
            
            for paper in batch:
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                if title or abstract:
                    keywords = self.extract_keywords_from_text(title, abstract)
                    keywords_batch.append(keywords)
                    
                    # Prepare update
                    paper_updates.append({
                        '_key': paper['_key'],
                        'keywords': keywords
                    })
                else:
                    keywords_batch.append([])
            
            # Generate embeddings for batch
            if keywords_batch:
                embeddings = self.embed_keywords(keywords_batch)
                
                # Update papers with keywords
                if paper_updates:
                    for update in paper_updates:
                        self.db.collection('arxiv_papers').update(update)
                
                # Store embeddings
                for i, (paper, embedding) in enumerate(zip(batch, embeddings)):
                    if len(keywords_batch[i]) > 0:  # Only store if keywords exist
                        # Check if embedding document exists
                        embed_key = paper['_key']
                        embed_doc = self.db.collection('arxiv_embeddings').get(embed_key)
                        
                        if embed_doc:
                            # Update existing document
                            self.db.collection('arxiv_embeddings').update({
                                '_key': embed_key,
                                'keyword_embedding': embedding.tolist()
                            })
                        else:
                            # Create new document
                            self.db.collection('arxiv_embeddings').insert({
                                '_key': embed_key,
                                'keyword_embedding': embedding.tolist()
                            })
                
                total_processed += len(batch)
                logger.info(f"Processed {total_processed:,} papers so far...")
        
        logger.info(f"Completed processing {total_processed:,} papers")
    
    def verify_keywords(self):
        """Verify keyword extraction results."""
        # Count papers with keywords
        query = "FOR paper IN arxiv_papers FILTER paper.keywords != null RETURN 1"
        count = len(list(self.db.aql.execute(query)))
        logger.info(f"Papers with keywords: {count:,}")
        
        # Count keyword embeddings
        query = "FOR embed IN arxiv_embeddings FILTER embed.keyword_embedding != null RETURN 1"
        embed_count = len(list(self.db.aql.execute(query)))
        logger.info(f"Keyword embeddings: {embed_count:,}")
        
        # Sample some keywords
        query = """
        FOR paper IN arxiv_papers
            FILTER paper.keywords != null
            LIMIT 5
            RETURN {title: paper.title, keywords: paper.keywords}
        """
        samples = list(self.db.aql.execute(query))
        
        logger.info("\nSample keywords:")
        for sample in samples:
            logger.info(f"  Title: {sample['title'][:60]}...")
            logger.info(f"  Keywords: {', '.join(sample['keywords'][:10])}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract keywords from papers")
    parser.add_argument('--limit', type=int, help='Limit number of papers to process')
    parser.add_argument('--verify', action='store_true', help='Just verify existing keywords')
    args = parser.parse_args()
    
    extractor = KeywordExtractor()
    
    if args.verify:
        extractor.verify_keywords()
    else:
        extractor.process_papers(limit=args.limit)
        extractor.verify_keywords()