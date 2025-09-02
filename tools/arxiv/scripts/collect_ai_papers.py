#!/usr/bin/env python3
"""
Collect ArXiv Papers for AI/RAG/LLM/ANT Research
================================================

Searches ArXiv for papers on:
- Artificial Intelligence (AI)
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Actor Network Theory (ANT)

Targets: 5000+ papers for comprehensive testing
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


class ArXivCollector:
    """Collect papers from ArXiv API based on search queries."""
    
    def __init__(self, output_dir: str = "data/arxiv_collections"):
        self.base_url = "http://export.arxiv.org/api/query"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_ids = set()
        
    def search_arxiv(self, query: str, max_results: int = 1000, start: int = 0) -> List[Dict]:
        """
        Search ArXiv API for papers matching query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to fetch
            start: Starting index for pagination
            
        Returns:
            List of paper metadata dictionaries
        """
        params = {
            'search_query': query,
            'start': start,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            logger.info(f"Searching ArXiv: {query} (start={start}, max={max_results})")
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
            
            logger.info(f"Found {len(papers)} new papers")
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
    
    def collect_papers_by_topic(self) -> Dict[str, List[Dict]]:
        """
        Collect papers for all target topics.
        
        Returns:
            Dictionary mapping topics to lists of papers
        """
        # Define comprehensive search queries for each topic
        queries = {
            'llm': [
                # Core LLM queries
                'all:"large language model"',
                'all:"large language models"',
                'all:LLM AND (all:transformer OR all:attention)',
                'all:GPT OR all:BERT OR all:T5 OR all:LLaMA',
                'all:"language model" AND all:billion',
                'all:"foundation model" AND all:language',
                'all:"pretrained language model"',
                'all:"autoregressive language model"',
                'ti:"language model" AND (all:large OR all:massive)',
                'all:ChatGPT OR all:Claude OR all:Gemini OR all:PaLM',
                # LLM techniques
                'all:"instruction tuning" OR all:"instruction following"',
                'all:"chain of thought" OR all:"few shot learning"',
                'all:"in context learning" AND all:"language model"',
                'all:"prompt engineering" AND all:LLM',
                'all:RLHF OR all:"reinforcement learning from human feedback"',
            ],
            'rag': [
                # RAG core concepts
                'all:"retrieval augmented generation"',
                'all:"retrieval-augmented generation"',
                'all:RAG AND (all:language OR all:generation)',
                'all:"retrieval enhanced" AND all:generation',
                'all:"retrieve and generate"',
                'all:"neural retrieval" AND all:generation',
                # RAG techniques
                'all:"dense retrieval" AND all:generation',
                'all:"hybrid retrieval" AND all:LLM',
                'all:"vector database" AND all:"language model"',
                'all:"semantic search" AND all:generation',
                'all:"knowledge grounding" AND all:retrieval',
                'all:"external knowledge" AND all:"language model"',
                'all:RETRO OR all:REALM OR all:FiD',
                'all:"retrieval" AND all:"augmented" AND all:"QA"',
            ],
            'ai_general': [
                # General AI
                'cat:cs.AI',
                'cat:cs.LG',
                'cat:cs.CL',
                'cat:cs.NE',
                'all:"artificial intelligence" AND all:survey',
                'all:"machine learning" AND all:"deep learning"',
                'all:"neural network" AND all:architecture',
                # AI subfields
                'all:"computer vision" AND all:"transformer"',
                'all:"natural language processing" AND all:neural',
                'all:"reinforcement learning" AND all:agent',
                'all:"multi agent" AND all:system',
                'all:"knowledge graph" AND all:reasoning',
                'all:"graph neural network"',
                'all:"diffusion model" OR all:"generative model"',
                'all:"contrastive learning" OR all:"self supervised"',
            ],
            'ant_theory': [
                # Actor Network Theory
                'all:"actor network theory"',
                'all:"actor-network theory"',
                'all:ANT AND all:theory AND all:network',
                'all:Latour AND all:"network theory"',
                'all:Callon AND all:actor AND all:network',
                'all:"science and technology studies" AND all:network',
                'all:STS AND all:"actor network"',
                'all:"sociotechnical" AND all:network',
                'all:"material semiotics"',
                'all:"sociology of translation"',
                # ANT in computing contexts
                'all:"actor network" AND (all:computing OR all:software)',
                'all:"actor network" AND all:algorithm',
                'all:"actor network" AND all:AI',
                'all:"actor network" AND all:data',
                'all:"assemblage theory" AND all:technology',
            ],
            'related_topics': [
                # Related important topics
                'all:"information theory" AND all:neural',
                'all:"attention mechanism" AND all:transformer',
                'all:"knowledge representation" AND all:learning',
                'all:"semantic embedding" OR all:"word embedding"',
                'all:"vector space" AND all:semantic',
                'all:"neural information retrieval"',
                'all:"question answering" AND all:neural',
                'all:"dialogue system" OR all:"conversational AI"',
                'all:"multimodal" AND all:"language model"',
                'all:"code generation" AND all:LLM',
                'all:"reasoning" AND all:"language model"',
                'all:"hallucination" AND all:LLM',
                'all:"alignment" AND all:"language model"',
            ]
        }
        
        all_papers = {}
        
        for topic, topic_queries in queries.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting papers for topic: {topic}")
            logger.info(f"{'='*60}")
            
            topic_papers = []
            
            for query in topic_queries:
                # Search with pagination to get more results
                for start_idx in range(0, 500, 100):  # Get up to 500 per query
                    papers = self.search_arxiv(query, max_results=100, start=start_idx)
                    
                    if not papers:
                        break
                    
                    topic_papers.extend(papers)
                    
                    # Rate limiting - ArXiv API requires 3 second delay
                    time.sleep(3)
                    
                    # Stop if we have enough papers for this query
                    if len(papers) < 100:
                        break
            
            all_papers[topic] = topic_papers
            logger.info(f"Collected {len(topic_papers)} papers for {topic}")
            logger.info(f"Total unique papers so far: {len(self.collected_ids)}")
        
        return all_papers
    
    def save_collection(self, papers: Dict[str, List[Dict]], filename: str = None):
        """Save collected papers to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arxiv_ai_collection_{timestamp}.json"
        
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
    
    def create_paper_list(self, papers: Dict[str, List[Dict]], output_file: str = None):
        """
        Create a simple list of ArXiv IDs for processing.
        
        Args:
            papers: Dictionary of papers by topic
            output_file: Output filename for the ID list
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"arxiv_ids_{timestamp}.txt"
        
        output_path = self.output_dir / output_file
        
        # Collect all unique IDs
        all_ids = set()
        for topic_papers in papers.values():
            for paper in topic_papers:
                all_ids.add(paper['arxiv_id'])
        
        # Sort IDs for consistent ordering
        sorted_ids = sorted(all_ids)
        
        # Write to file
        with open(output_path, 'w') as f:
            for arxiv_id in sorted_ids:
                f.write(f"{arxiv_id}\n")
        
        logger.info(f"Saved {len(sorted_ids)} ArXiv IDs to {output_path}")
        return output_path


def main():
    """Main execution function."""
    logger.info("Starting ArXiv paper collection for AI/RAG/LLM/ANT topics")
    
    collector = ArXivCollector()
    
    # Collect papers
    papers = collector.collect_papers_by_topic()
    
    # Print statistics
    print("\n" + "="*60)
    print("COLLECTION STATISTICS")
    print("="*60)
    
    total_papers = len(collector.collected_ids)
    for topic, topic_papers in papers.items():
        print(f"{topic:20s}: {len(topic_papers):5d} papers")
    
    print(f"{'TOTAL UNIQUE':20s}: {total_papers:5d} papers")
    
    if total_papers >= 5000:
        print(f"\n✅ Target of 5000+ papers achieved!")
    else:
        print(f"\n⚠️  Collected {total_papers} papers (target: 5000+)")
        print("Consider running with extended queries or date ranges")
    
    # Save results
    json_path = collector.save_collection(papers)
    txt_path = collector.create_paper_list(papers)
    
    print(f"\n📁 Output files:")
    print(f"  - Full data: {json_path}")
    print(f"  - ID list:   {txt_path}")
    
    return papers, collector.collected_ids


if __name__ == "__main__":
    papers, ids = main()