#!/usr/bin/env python3
"""
Extended ArXiv Paper Collection - Scale to 20,000 Papers
=========================================================

Expands on existing collection to reach 20,000 papers for comprehensive testing.
Preserves existing collection and adds new papers through broader search strategies.

Targets: 20,000 total papers (expanding from 4,126)
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Set
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtendedArXivCollector:
    """Extended collector to reach 20,000 papers."""
    
    def __init__(self, output_dir: str = "data/arxiv_collections"):
        self.base_url = "http://export.arxiv.org/api/query"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_ids = set()
        self.existing_collection = {}
        
    def load_existing_collection(self, id_file: str = None):
        """Load existing ArXiv IDs from previous collection."""
        if id_file is None:
            # Look for most recent ID file
            id_files = list(self.output_dir.glob("arxiv_ids_*.txt"))
            if id_files:
                id_file = sorted(id_files)[-1]
                logger.info(f"Loading existing IDs from {id_file}")
            else:
                logger.warning("No existing ID file found")
                return
        else:
            id_file = self.output_dir / id_file
        
        if id_file.exists():
            with open(id_file, 'r') as f:
                for line in f:
                    arxiv_id = line.strip()
                    if arxiv_id:
                        self.collected_ids.add(arxiv_id)
            logger.info(f"Loaded {len(self.collected_ids)} existing ArXiv IDs")
        
        # Also try to load the JSON collection if it exists
        json_files = list(self.output_dir.glob("arxiv_ai_collection_*.json"))
        if json_files:
            json_file = sorted(json_files)[-1]
            logger.info(f"Loading existing collection from {json_file}")
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'papers' in data:
                    self.existing_collection = data['papers']
    
    def search_arxiv(self, query: str, max_results: int = 1000, start: int = 0) -> List[Dict]:
        """Search ArXiv API for papers matching query."""
        params = {
            'search_query': query,
            'start': start,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            logger.info(f"Searching ArXiv: {query[:50]}... (start={start}, max={max_results})")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Extract namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            papers = []
            entries = root.findall('atom:entry', ns)
            
            for entry in entries:
                paper = self._parse_entry(entry, ns)
                if paper and paper['arxiv_id'] not in self.collected_ids:
                    papers.append(paper)
                    self.collected_ids.add(paper['arxiv_id'])
            
            logger.info(f"Found {len(papers)} new papers (total: {len(self.collected_ids)})")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def _parse_entry(self, entry, ns) -> Dict:
        """Parse a single entry from ArXiv XML."""
        try:
            # Extract ArXiv ID from the id URL
            id_url = entry.find('atom:id', ns).text
            arxiv_id = id_url.split('/')[-1]
            
            # Extract version if present
            if 'v' in arxiv_id:
                arxiv_id = arxiv_id.split('v')[0]
            
            paper = {
                'arxiv_id': arxiv_id,
                'title': entry.find('atom:title', ns).text.strip().replace('\n', ' '),
                'summary': entry.find('atom:summary', ns).text.strip(),
                'authors': [author.find('atom:name', ns).text 
                           for author in entry.findall('atom:author', ns)],
                'published': entry.find('atom:published', ns).text,
                'updated': entry.find('atom:updated', ns).text,
                'categories': [],
                'primary_category': None,
                'pdf_url': None
            }
            
            # Get categories
            categories = entry.findall('atom:category', ns)
            if categories:
                paper['categories'] = [cat.get('term') for cat in categories]
                paper['primary_category'] = categories[0].get('term')
            
            # Get PDF link
            links = entry.findall('atom:link', ns)
            for link in links:
                if link.get('type') == 'application/pdf':
                    paper['pdf_url'] = link.get('href')
                    break
            
            return paper
            
        except Exception as e:
            logger.error(f"Error parsing entry: {e}")
            return None
    
    def collect_extended_papers(self) -> Dict[str, List[Dict]]:
        """
        Collect papers using extended search strategies to reach 20,000.
        
        Returns:
            Dictionary mapping topics to lists of papers
        """
        # Extended queries - broader searches and additional topics
        extended_queries = {
            'ml_fundamentals': [
                # Machine Learning fundamentals
                'cat:cs.LG',  # Machine Learning category
                'cat:stat.ML',  # Statistics Machine Learning
                'all:"deep learning"',
                'all:"neural network"',
                'all:"supervised learning"',
                'all:"unsupervised learning"',
                'all:"reinforcement learning"',
                'all:"transfer learning"',
                'all:"meta learning"',
                'all:"few shot learning"',
                'all:"zero shot"',
                'all:"continual learning"',
                'all:"federated learning"',
                'all:"active learning"',
                'all:"semi supervised"',
                'all:"self supervised"',
                'all:"representation learning"',
                'all:"feature learning"',
            ],
            'nlp_extended': [
                # Natural Language Processing
                'cat:cs.CL',  # Computation and Language
                'all:"natural language processing"',
                'all:"computational linguistics"',
                'all:"text mining"',
                'all:"sentiment analysis"',
                'all:"named entity recognition"',
                'all:"part of speech"',
                'all:"dependency parsing"',
                'all:"machine translation"',
                'all:"text generation"',
                'all:"text summarization"',
                'all:"question answering"',
                'all:"information extraction"',
                'all:"dialogue system"',
                'all:"chatbot"',
                'all:"language understanding"',
            ],
            'cv_extended': [
                # Computer Vision
                'cat:cs.CV',  # Computer Vision
                'all:"computer vision"',
                'all:"image recognition"',
                'all:"object detection"',
                'all:"image segmentation"',
                'all:"face recognition"',
                'all:"pose estimation"',
                'all:"scene understanding"',
                'all:"visual tracking"',
                'all:"image generation"',
                'all:"style transfer"',
                'all:"video analysis"',
                'all:"optical flow"',
                'all:"visual transformer"',
                'all:"vision language"',
            ],
            'transformers_extended': [
                # Transformer architectures
                'all:transformer AND all:attention',
                'all:"vision transformer"',
                'all:"transformer model"',
                'all:BERT OR all:RoBERTa OR all:ALBERT',
                'all:GPT OR all:"generative pre"',
                'all:T5 OR all:BART OR all:mBART',
                'all:XLNet OR all:ELECTRA',
                'all:"masked language model"',
                'all:"autoregressive model"',
                'all:"encoder decoder"',
                'all:"cross attention"',
                'all:"multi head attention"',
                'all:"positional encoding"',
                'all:"layer normalization"',
            ],
            'ai_applications': [
                # AI Applications
                'all:"artificial intelligence" AND all:application',
                'all:AI AND all:healthcare',
                'all:AI AND all:finance',
                'all:AI AND all:robotics',
                'all:AI AND all:autonomous',
                'all:AI AND all:education',
                'all:AI AND all:security',
                'all:AI AND all:manufacturing',
                'all:AI AND all:agriculture',
                'all:AI AND all:climate',
                'all:AI AND all:science',
                'all:"expert system"',
                'all:"decision support"',
                'all:"intelligent agent"',
                'all:"cognitive computing"',
            ],
            'optimization_methods': [
                # Optimization and Training
                'all:"gradient descent"',
                'all:"stochastic gradient"',
                'all:"Adam optimizer"',
                'all:"learning rate"',
                'all:"batch normalization"',
                'all:"dropout" AND all:regularization',
                'all:"weight decay"',
                'all:"early stopping"',
                'all:"hyperparameter optimization"',
                'all:"neural architecture search"',
                'all:"automl"',
                'all:"model compression"',
                'all:"knowledge distillation"',
                'all:"pruning" AND all:neural',
                'all:"quantization" AND all:model',
            ],
            'data_science': [
                # Data Science and Analytics
                'cat:cs.DB',  # Databases
                'cat:cs.IR',  # Information Retrieval
                'all:"data science"',
                'all:"data mining"',
                'all:"big data"',
                'all:"data analytics"',
                'all:"data visualization"',
                'all:"exploratory data analysis"',
                'all:"feature engineering"',
                'all:"data preprocessing"',
                'all:"data augmentation"',
                'all:"synthetic data"',
                'all:"data quality"',
                'all:"data governance"',
                'all:"data pipeline"',
            ],
            'ethics_fairness': [
                # AI Ethics and Fairness
                'all:"AI ethics"',
                'all:"algorithmic fairness"',
                'all:"bias" AND all:"machine learning"',
                'all:"explainable AI"',
                'all:"interpretable machine learning"',
                'all:"AI safety"',
                'all:"AI alignment"',
                'all:"trustworthy AI"',
                'all:"responsible AI"',
                'all:"AI governance"',
                'all:"privacy preserving"',
                'all:"differential privacy"',
                'all:"federated learning" AND all:privacy',
            ],
            'graph_ml': [
                # Graph Machine Learning
                'all:"graph neural network"',
                'all:"graph convolutional"',
                'all:"graph attention"',
                'all:"knowledge graph"',
                'all:"graph embedding"',
                'all:"network embedding"',
                'all:"graph representation"',
                'all:"node classification"',
                'all:"link prediction"',
                'all:"graph generation"',
                'all:"molecular graph"',
                'all:"social network" AND all:learning',
                'all:"heterogeneous graph"',
            ],
            'generative_models': [
                # Generative Models
                'all:"generative model"',
                'all:"generative adversarial"',
                'all:GAN OR all:VAE',
                'all:"diffusion model"',
                'all:"flow model"',
                'all:"autoencoder"',
                'all:"variational autoencoder"',
                'all:"stable diffusion"',
                'all:"DALL-E" OR all:"midjourney"',
                'all:"image synthesis"',
                'all:"text to image"',
                'all:"neural rendering"',
                'all:"deepfake"',
            ],
            'robotics_ai': [
                # Robotics and Embodied AI
                'cat:cs.RO',  # Robotics
                'all:"robot learning"',
                'all:"robotic manipulation"',
                'all:"path planning"',
                'all:"motion planning"',
                'all:"SLAM" AND all:robot',
                'all:"visual servoing"',
                'all:"imitation learning"',
                'all:"sim to real"',
                'all:"embodied AI"',
                'all:"autonomous navigation"',
                'all:"multi robot"',
                'all:"human robot interaction"',
            ],
            'theoretical_ml': [
                # Theoretical Machine Learning
                'cat:cs.LG AND cat:math',
                'all:"learning theory"',
                'all:"statistical learning"',
                'all:"PAC learning"',
                'all:"VC dimension"',
                'all:"sample complexity"',
                'all:"generalization bound"',
                'all:"optimization theory" AND all:learning',
                'all:"convergence analysis"',
                'all:"approximation theory"',
                'all:"information theory" AND all:learning',
                'all:"kernel method"',
                'all:"reproducing kernel"',
            ]
        }
        
        all_papers = {}
        
        # Copy existing collection
        if self.existing_collection:
            all_papers = self.existing_collection.copy()
            logger.info(f"Starting with {sum(len(papers) for papers in all_papers.values())} existing papers")
        
        # Target: 20,000 total papers
        target_total = 20000
        papers_needed = target_total - len(self.collected_ids)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Extended Collection Phase")
        logger.info(f"Current papers: {len(self.collected_ids)}")
        logger.info(f"Target total: {target_total}")
        logger.info(f"Papers needed: {papers_needed}")
        logger.info(f"{'='*60}")
        
        for topic, topic_queries in extended_queries.items():
            if len(self.collected_ids) >= target_total:
                logger.info(f"Reached target of {target_total} papers!")
                break
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting papers for topic: {topic}")
            logger.info(f"{'='*60}")
            
            topic_papers = []
            
            for query in topic_queries:
                if len(self.collected_ids) >= target_total:
                    break
                
                # Search with pagination
                for start_idx in range(0, 2000, 100):  # Get up to 2000 per query
                    papers = self.search_arxiv(query, max_results=100, start=start_idx)
                    
                    if not papers:
                        break
                    
                    topic_papers.extend(papers)
                    
                    # Rate limiting - ArXiv API requires 3 second delay
                    time.sleep(3)
                    
                    # Check if we've reached target
                    if len(self.collected_ids) >= target_total:
                        logger.info(f"✅ Reached target of {target_total} papers!")
                        break
                    
                    # Stop if we have enough papers for this query
                    if len(papers) < 100:
                        break
            
            if topic not in all_papers:
                all_papers[topic] = []
            all_papers[topic].extend(topic_papers)
            
            logger.info(f"Collected {len(topic_papers)} papers for {topic}")
            logger.info(f"Total unique papers so far: {len(self.collected_ids)}")
        
        return all_papers
    
    def save_extended_collection(self, papers: Dict[str, List[Dict]], filename: str = None):
        """Save extended collection to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arxiv_extended_collection_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Create summary statistics
        stats = {
            'collection_date': datetime.now().isoformat(),
            'total_papers': len(self.collected_ids),
            'topics': {topic: len(papers) for topic, papers in papers.items()},
            'papers': papers
        }
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved collection to {output_path}")
        return output_path
    
    def create_extended_paper_list(self, papers: Dict[str, List[Dict]], output_file: str = None):
        """Create extended list of ArXiv IDs for processing."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"arxiv_ids_extended_{timestamp}.txt"
        
        output_path = self.output_dir / output_file
        
        # Write all unique IDs
        sorted_ids = sorted(self.collected_ids)
        
        with open(output_path, 'w') as f:
            for arxiv_id in sorted_ids:
                f.write(f"{arxiv_id}\n")
        
        logger.info(f"Saved {len(sorted_ids)} ArXiv IDs to {output_path}")
        return output_path


def main():
    """Main execution function."""
    logger.info("Starting Extended ArXiv Paper Collection (Target: 20,000 papers)")
    
    collector = ExtendedArXivCollector()
    
    # Load existing collection
    collector.load_existing_collection("arxiv_ids_20250829_174450.txt")
    
    # Collect additional papers
    papers = collector.collect_extended_papers()
    
    # Print statistics
    print("\n" + "="*60)
    print("EXTENDED COLLECTION STATISTICS")
    print("="*60)
    
    total_papers = len(collector.collected_ids)
    for topic, topic_papers in papers.items():
        if topic_papers:  # Only show topics with papers
            print(f"{topic:20s}: {len(topic_papers):5d} papers")
    
    print(f"{'TOTAL UNIQUE':20s}: {total_papers:5d} papers")
    
    if total_papers >= 20000:
        print(f"\n✅ Target of 20,000 papers achieved!")
    else:
        print(f"\n⚠️  Collected {total_papers} papers (target: 20,000)")
        print("Consider running with additional queries or date ranges")
    
    # Save results
    json_path = collector.save_extended_collection(papers)
    txt_path = collector.create_extended_paper_list(papers)
    
    print(f"\n📁 Output files:")
    print(f"  - Full data: {json_path}")
    print(f"  - ID list:   {txt_path}")
    
    print(f"\n🎯 Ready for weekend test run!")
    print(f"  - Total papers available: {total_papers}")
    print(f"  - Recommended test size: 15,000 papers")
    print(f"  - Estimated runtime: ~22 hours @ 11.3 papers/minute")
    
    return papers, collector.collected_ids


if __name__ == "__main__":
    papers, ids = main()