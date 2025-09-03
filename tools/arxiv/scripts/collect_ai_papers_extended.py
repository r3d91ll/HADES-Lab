#!/usr/bin/env python3
"""
Extended ArXiv Paper Collection (SQL-backed)
===========================================

Adapted to use the Postgres-backed metadata service for fast, reproducible ID
list generation. Retains the old API-based code path for reference, but the
default mode now queries Postgres with FTS/category filters and optional caps.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
import math
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
    
    def __init__(self, output_dir: str = "tools/arxiv/scripts/data/arxiv_collections/", 
                 pdf_dir: str = "/bulk-store/arxiv-data/pdf",
                 metadata_path: str = "/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json"):
        self.base_url = "http://export.arxiv.org/api/query"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_ids = set()
        self.existing_collection = {}
        self.pdf_dir = Path(pdf_dir)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.metadata_cache = {}  # Cache for metadata snapshot
        self.pdf_availability = {}  # Track which papers have PDFs

        # SQL exporter imports (lazy optional use)
        try:
            from tools.arxiv.db.export_ids import (
                build_query as _build_query,
                build_where as _build_where,
                collect_ids as _collect_ids,
                collect_stats as _collect_stats,
                write_outputs as _write_outputs,
            )
            from tools.arxiv.db.config import load_config as _load_cfg
            from tools.arxiv.db.pg import get_connection as _get_conn
            self._sql = {
                'build_query': _build_query,
                'build_where': _build_where,
                'collect_ids': _collect_ids,
                'collect_stats': _collect_stats,
                'write_outputs': _write_outputs,
                'load_config': _load_cfg,
                'get_connection': _get_conn,
            }
        except Exception as e:  # pragma: no cover
            logger.debug("sql_modules_unavailable", error=str(e))
            self._sql = None
        
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
                    # Merge paper IDs from existing collection into dedupe set
                    for paper in self.existing_collection:
                        # Check for different possible ID field names
                        paper_id = paper.get('id') or paper.get('arxiv_id')
                        if paper_id:
                            self.collected_ids.add(paper_id)
                    logger.info(f"Added {len(self.existing_collection)} papers from existing collection to dedupe set")
        
        # Check PDF availability for loaded IDs
        self._check_pdf_availability()
    
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
    
    def _check_pdf_availability(self):
        """Check which collected papers have PDFs available locally."""
        logger.info("Checking PDF availability...")
        available_count = 0
        
        for arxiv_id in self.collected_ids:
            pdf_path = self._get_pdf_path(arxiv_id)
            if pdf_path and pdf_path.exists():
                self.pdf_availability[arxiv_id] = str(pdf_path)
                available_count += 1
            else:
                self.pdf_availability[arxiv_id] = None
        
        logger.info(f"PDF availability: {available_count}/{len(self.collected_ids)} papers have local PDFs")
        return available_count
    
    def _get_pdf_path(self, arxiv_id: str) -> Optional[Path]:
        """Get the expected PDF path for an ArXiv ID."""
        # Handle different ID formats
        if '.' in arxiv_id:
            # Format: YYMM.NNNNN or YY.NNNNN
            parts = arxiv_id.split('.')
            yymm = parts[0]
            if len(yymm) == 2:  # YY format, need to prepend century
                # Assume 20YY for 00-99
                yymm = yymm.zfill(4)
        else:
            # Old format without dot, extract YYMM
            yymm = arxiv_id[:4] if len(arxiv_id) >= 4 else None
        
        if yymm:
            # Check for PDF in expected location
            pdf_path = self.pdf_dir / yymm / f"{arxiv_id}.pdf"
            return pdf_path
        return None

    # --------------------------
    # SQL-backed collection path
    # --------------------------
    def _default_keywords_tsquery(self) -> str:
        """Default FTS tsquery capturing common AI/ML/NLP/CV terms.

        Tuned to approximate the broad coverage previously achieved via API queries.
        """
        return (
            "attention | transformer | \"neural network\" | \"deep learning\" |"
            " diffusion | VAE | GAN | \"vision transformer\" | BERT | GPT | T5 | XLNet | ELECTRA |"
            " \"language model\" | \"encoder & decoder\" | \"cross attention\" | \"multi head attention\" |"
            " \"positional encoding\" | \"knowledge distillation\" | quantization | pruning |"
            " \"neural architecture search\" | hyperparameter | \"few shot\" | \"zero shot\" |"
            " \"federated learning\" | \"meta learning\" | \"representation learning\" |"
            " \"graph neural\" | GNN | \"graph network\" | \"image segmentation\" | \"object detection\" |"
            " \"pose estimation\" | SLAM | \"text summarization\" | \"machine translation\""
        )

    def collect_via_sql(
        self,
        start_year: int = 2010,
        end_year: int = 2025,
        categories: Optional[list[str]] = None,
        keywords_tsquery: Optional[str] = None,
        target_total: Optional[int] = None,
        cap_mode: str = "per-year",  # or "per-month"
        with_pdf: bool = False,
        missing_pdf: bool = False,
        write_monthly_lists: bool = True,
        config_path: str = "tools/arxiv/configs/db.yaml",
    ) -> Tuple[Path, Path, Dict]:
        """Generate lists via SQL exporter functions and write outputs.

        Returns (stats_json_path, master_list_path, stats_dict)
        """
        if not self._sql:
            raise RuntimeError("SQL exporter is unavailable. Ensure dependencies are installed.")

        build_query = self._sql['build_query']
        build_where = self._sql['build_where']
        collect_ids = self._sql['collect_ids']
        collect_stats = self._sql['collect_stats']
        write_outputs = self._sql['write_outputs']
        load_config = self._sql['load_config']
        get_connection = self._sql['get_connection']

        cfg = load_config(config_path)

        months = None
        yymm_range = None
        if categories is None:
            categories = ["cs.LG", "cs.CL", "cs.CV", "cs.AI", "stat.ML"]
        if keywords_tsquery is None:
            keywords_tsquery = self._default_keywords_tsquery()

        per_year_cap = None
        per_month_cap = None
        if target_total:
            years = max(1, end_year - start_year + 1)
            if cap_mode == "per-month":
                months_count = years * 12
                per_month_cap = math.ceil(target_total / months_count)
            else:
                per_year_cap = math.ceil(target_total / years)

        sql, params, monthly_sql = build_query(
            start_year,
            end_year,
            months,
            yymm_range,  # type: ignore
            categories,
            keywords_tsquery,
            with_pdf,
            missing_pdf,
            per_year_cap,
            per_month_cap,
        )

        ids = collect_ids(cfg, sql, params)
        where_sql, where_params = build_where(
            start_year,
            end_year,
            months,
            yymm_range,  # type: ignore
            categories,
            keywords_tsquery,
            with_pdf,
            missing_pdf,
            None,
        )
        stats = collect_stats(cfg, where_sql, where_params)

        monthly: Dict[Tuple[int, int], list[str]] | None = None
        if write_monthly_lists:
            monthly = {}
            with get_connection(cfg.postgres) as conn:
                cur = conn.cursor()
                try:
                    cur.execute(monthly_sql, params)
                    for aid, yr, mo, _yymm in cur.fetchall():
                        if yr is None or mo is None:
                            continue
                        monthly.setdefault((int(yr), int(mo)), []).append(aid)
                finally:
                    try:
                        cur.close()
                    except Exception:
                        pass

        prefix = "arxiv_ids"
        stamp = f"{start_year}_{end_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        write_outputs(self.output_dir, prefix, ids, with_pdf, missing_pdf, stats, monthly, suffix=stamp, update_symlinks=True)
        stats_path = self.output_dir / f"{prefix}_stats.json"
        master_path = self.output_dir / f"{prefix}_sweep.txt"
        return stats_path, master_path, stats
    
    def load_metadata_snapshot(self):
        """Load ArXiv metadata from JSON Lines snapshot."""
        if not self.metadata_path or not self.metadata_path.exists():
            logger.warning(f"Metadata snapshot not found: {self.metadata_path}")
            return
        
        logger.info(f"Loading metadata from {self.metadata_path}...")
        count = 0
        
        try:
            with open(self.metadata_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            doc = json.loads(line)
                            arxiv_id = doc.get('id', '').replace('http://arxiv.org/abs/', '')
                            if arxiv_id:
                                self.metadata_cache[arxiv_id] = doc
                                count += 1
                                if count % 100000 == 0:
                                    logger.info(f"  Loaded {count:,} metadata entries...")
                        except json.JSONDecodeError:
                            continue
            
            logger.info(f"Loaded {len(self.metadata_cache):,} metadata entries")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
    
    def collect_extended_papers(self, target_count: int = 20000) -> Dict[str, List[Dict]]:
        """
        Collect papers using extended search strategies to reach target count.
        
        Args:
            target_count: Target number of papers to collect
        
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
        
        papers_needed = target_count - len(self.collected_ids)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Extended Collection Phase")
        logger.info(f"Current papers: {len(self.collected_ids)}")
        logger.info(f"Target total: {target_count}")
        logger.info(f"Papers needed: {papers_needed}")
        logger.info(f"{'='*60}")
        
        for topic, topic_queries in extended_queries.items():
            if len(self.collected_ids) >= target_count:
                logger.info(f"Reached target of {target_count} papers!")
                break
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting papers for topic: {topic}")
            logger.info(f"{'='*60}")
            
            topic_papers = []
            
            for query in topic_queries:
                if len(self.collected_ids) >= target_count:
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
                    if len(self.collected_ids) >= target_count:
                        logger.info(f"✅ Reached target of {target_count} papers!")
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
    
    def create_extended_paper_list(self, papers: Dict[str, List[Dict]], output_file: str = None,
                                  filter_by_pdf: bool = False):
        """Create extended list of ArXiv IDs for processing.
        
        Args:
            papers: Dictionary of collected papers
            output_file: Output filename (auto-generated if None)
            filter_by_pdf: If True, only include papers with local PDFs
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_with_pdfs" if filter_by_pdf else "_extended"
            output_file = f"arxiv_ids{suffix}_{timestamp}.txt"
        
        output_path = self.output_dir / output_file
        
        # Filter IDs based on PDF availability if requested
        if filter_by_pdf:
            # Re-check PDF availability
            self._check_pdf_availability()
            sorted_ids = sorted([arxiv_id for arxiv_id in self.collected_ids 
                               if self.pdf_availability.get(arxiv_id)])
            logger.info(f"Filtered to {len(sorted_ids)} papers with local PDFs")
        else:
            sorted_ids = sorted(self.collected_ids)
        
        with open(output_path, 'w') as f:
            for arxiv_id in sorted_ids:
                f.write(f"{arxiv_id}\n")
        
        logger.info(f"Saved {len(sorted_ids)} ArXiv IDs to {output_path}")
        return output_path
    
    def create_pdf_availability_report(self) -> Tuple[Dict, Path]:
        """Create a detailed report of PDF availability."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"pdf_availability_report_{timestamp}.json"
        
        # Re-check availability
        self._check_pdf_availability()
        
        # Categorize papers
        with_pdf = []
        without_pdf = []
        
        for arxiv_id in sorted(self.collected_ids):
            pdf_path = self.pdf_availability.get(arxiv_id)
            if pdf_path:
                with_pdf.append({
                    'arxiv_id': arxiv_id,
                    'pdf_path': pdf_path,
                    'has_metadata': arxiv_id in self.metadata_cache
                })
            else:
                without_pdf.append({
                    'arxiv_id': arxiv_id,
                    'expected_path': str(self._get_pdf_path(arxiv_id)) if self._get_pdf_path(arxiv_id) else None,
                    'has_metadata': arxiv_id in self.metadata_cache
                })
        
        report = {
            'generated': datetime.now().isoformat(),
            'total_papers': len(self.collected_ids),
            'papers_with_pdf': len(with_pdf),
            'papers_without_pdf': len(without_pdf),
            'pdf_availability_rate': len(with_pdf) / len(self.collected_ids) if self.collected_ids else 0,
            'with_pdf': with_pdf,
            'without_pdf': without_pdf
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"PDF availability report saved to {report_path}")
        return report, report_path


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect ArXiv papers for AI/RAG/LLM research (SQL-backed)")
    parser.add_argument('--mode', choices=['sql', 'api'], default='sql', help='Use SQL-backed exporter or legacy API mode')
    parser.add_argument('--target', type=int, default=20000,
                       help='Target number of papers (used to compute caps)')
    parser.add_argument('--existing-file', type=str, default="arxiv_ids_20250829_174450.txt",
                       help='Existing ID file to load (default: arxiv_ids_20250829_174450.txt)')
    parser.add_argument('--no-load-existing', action='store_true',
                       help='Start fresh without loading existing collection')
    parser.add_argument('--filter-by-pdf', action='store_true',
                       help='Only include papers with local PDFs in output list')
    parser.add_argument('--load-metadata', action='store_true',
                       help='Load metadata from snapshot file')
    parser.add_argument('--pdf-report', action='store_true',
                       help='Generate detailed PDF availability report')
    # SQL mode params
    parser.add_argument('--start-year', type=int, default=2010, help='Start year inclusive')
    parser.add_argument('--end-year', type=int, default=2025, help='End year inclusive')
    parser.add_argument('--categories', nargs='*', default=["cs.LG","cs.CL","cs.CV","cs.AI","stat.ML"], help='Categories filter')
    parser.add_argument('--keywords', type=str, default=None, help='Postgres to_tsquery string for FTS')
    parser.add_argument('--cap-mode', choices=['per-year','per-month'], default='per-year', help='Cap distribution mode when --target is set')
    parser.add_argument('--config', type=str, default='tools/arxiv/configs/db.yaml', help='DB config YAML')
    parser.add_argument('--no-monthly', action='store_true', help='Do not write monthly lists')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Extended ArXiv Paper Collection (mode={args.mode})")

    collector = ExtendedArXivCollector()
    
    # Load existing collection unless told not to
    if not args.no_load_existing:
        collector.load_existing_collection(args.existing_file)
    
    if args.mode == 'sql':
        # SQL-backed generation
        stats_path, master_path, stats = collector.collect_via_sql(
            start_year=args.start_year,
            end_year=args.end_year,
            categories=args.categories,
            keywords_tsquery=args.keywords,
            target_total=args.target if args.target and args.target > 0 else None,
            cap_mode=args.cap_mode,
            with_pdf=args.filter_by_pdf,
            missing_pdf=False,
            write_monthly_lists=not args.no_monthly,
            config_path=args.config,
        )

        # Print stats summary
        print("\n" + "="*60)
        print("SQL EXPORT STATISTICS")
        print("="*60)
        print(f"Total IDs: {stats.get('total', 0):,}")
        by_year = stats.get('by_year', {})
        if by_year:
            years_sample = sorted(by_year.items())[:5]
            print("By year (sample):", years_sample)
        print(f"\n📁 Output files:")
        print(f"  - Stats: {stats_path}")
        print(f"  - IDs:   {master_path}")

        # We also produce old-style outputs (optional) for compatibility
        if args.pdf_report:
            # Stats already reflect has_pdf filter; PDF report uses filesystem scan if needed
            collector.collected_ids = set(Path(master_path).read_text().splitlines())
            report, report_path = collector.create_pdf_availability_report()
            print(f"\n📊 PDF Availability Report:")
            print(f"  - Papers with PDFs: {report['papers_with_pdf']:,} ({report['pdf_availability_rate']*100:.1f}%)")
            print(f"  - Papers without PDFs: {report['papers_without_pdf']:,}")
            print(f"  - Report saved to: {report_path}")

        return {}, collector.collected_ids

    # Legacy API mode
    # Load metadata if requested
    if args.load_metadata:
        collector.load_metadata_snapshot()

    papers = collector.collect_extended_papers(target_count=args.target)
    
    # Print statistics
    print("\n" + "="*60)
    print("EXTENDED COLLECTION STATISTICS")
    print("="*60)
    
    total_papers = len(collector.collected_ids)
    for topic, topic_papers in papers.items():
        if topic_papers:  # Only show topics with papers
            print(f"{topic:20s}: {len(topic_papers):5d} papers")
    
    print(f"{'TOTAL UNIQUE':20s}: {total_papers:5d} papers")
    
    if total_papers >= args.target:
        print(f"\n✅ Target of {args.target:,} papers achieved!")
    else:
        print(f"\n⚠️  Collected {total_papers:,} papers (target: {args.target:,})")
        print("Consider running with additional queries or date ranges")
    
    # Save results
    json_path = collector.save_extended_collection(papers)
    txt_path = collector.create_extended_paper_list(papers, filter_by_pdf=args.filter_by_pdf)
    
    # Generate PDF availability report if requested
    if args.pdf_report:
        report, report_path = collector.create_pdf_availability_report()
        print(f"\n📊 PDF Availability Report:")
        print(f"  - Papers with PDFs: {report['papers_with_pdf']:,} ({report['pdf_availability_rate']*100:.1f}%)")
        print(f"  - Papers without PDFs: {report['papers_without_pdf']:,}")
        print(f"  - Report saved to: {report_path}")
    
    print(f"\n📁 Output files:")
    print(f"  - Full data: {json_path}")
    print(f"  - ID list:   {txt_path}")
    
    # Show PDF availability summary
    if collector.pdf_availability:
        available = sum(1 for v in collector.pdf_availability.values() if v)
        print(f"\n📄 PDF Availability:")
        print(f"  - Local PDFs found: {available:,}/{len(collector.collected_ids):,} ({available/len(collector.collected_ids)*100:.1f}%)")
        if args.filter_by_pdf:
            print(f"  - Output list contains only papers with PDFs")
    
    if args.target <= 1000:
        print(f"\n🧪 Ready for test run!")
        print(f"  - Total papers available: {total_papers:,}")
        print(f"  - Estimated runtime: ~{total_papers / 11.3:.0f} minutes @ 11.3 papers/minute")
    else:
        print(f"\n🎯 Ready for large-scale test!")
        print(f"  - Total papers available: {total_papers:,}")
        print(f"  - Estimated runtime: ~{total_papers / 11.3 / 60:.1f} hours @ 11.3 papers/minute")
    
    return papers, collector.collected_ids


if __name__ == "__main__":
    papers, ids = main()
