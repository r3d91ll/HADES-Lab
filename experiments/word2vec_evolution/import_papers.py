#!/usr/bin/env python3
"""
Import word2vec, doc2vec, and code2vec papers for experiment.
Downloads PDFs from ArXiv and stores them for processing.
"""

import os
import sys
import requests
from pathlib import Path
from typing import Dict, List
import yaml
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.arxiv.db.config import load_config
from tools.arxiv.db.pg import get_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAPERS = {
    "word2vec": {
        "arxiv_id": "1301.3781",
        "title": "Efficient Estimation of Word Representations in Vector Space",
        "year": 2013,
        "authors": ["Tomas Mikolov", "Kai Chen", "Greg Corrado", "Jeffrey Dean"],
    },
    "doc2vec": {
        "arxiv_id": "1405.4053", 
        "title": "Distributed Representations of Sentences and Documents",
        "year": 2014,
        "authors": ["Quoc Le", "Tomas Mikolov"],
    },
    "code2vec": {
        "arxiv_id": "1803.09473",
        "title": "code2vec: Learning Distributed Representations of Code",
        "year": 2018,
        "authors": ["Uri Alon", "Meital Zilberstein", "Omer Levy", "Eran Yahav"],
    }
}

def download_paper(arxiv_id: str, output_dir: Path) -> Path:
    """Download paper PDF from ArXiv."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    output_path = output_dir / f"{arxiv_id.replace('.', '_')}.pdf"
    
    if output_path.exists():
        logger.info(f"Paper {arxiv_id} already downloaded")
        return output_path
    
    logger.info(f"Downloading {arxiv_id} from {pdf_url}")
    response = requests.get(pdf_url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Downloaded {arxiv_id} to {output_path}")
    return output_path

def check_database(arxiv_ids: List[str]) -> Dict[str, bool]:
    """Check which papers are already in the database."""
    config = load_config('tools/arxiv/configs/db.yaml')
    
    results = {}
    with get_connection(config.postgres) as conn:
        cur = conn.cursor()
        for arxiv_id in arxiv_ids:
            cur.execute(
                "SELECT arxiv_id, has_pdf FROM papers WHERE arxiv_id = %s",
                (arxiv_id,)
            )
            row = cur.fetchone()
            results[arxiv_id] = bool(row)
            if row:
                logger.info(f"Paper {arxiv_id} exists in database (has_pdf={row[1]})")
            else:
                logger.info(f"Paper {arxiv_id} NOT in database")
    
    return results

def main():
    # Set environment
    os.environ['POSTGRES_USER'] = 'postgres'
    os.environ['PGPASSWORD'] = '1luv93ngu1n$'
    
    # Create output directory
    output_dir = Path("papers")
    output_dir.mkdir(exist_ok=True)
    
    # Check what's in database
    arxiv_ids = [paper["arxiv_id"] for paper in PAPERS.values()]
    db_status = check_database(arxiv_ids)
    
    # Download papers
    downloaded = {}
    for name, info in PAPERS.items():
        arxiv_id = info["arxiv_id"]
        
        # Download PDF
        pdf_path = download_paper(arxiv_id, output_dir)
        downloaded[name] = {
            "arxiv_id": arxiv_id,
            "pdf_path": str(pdf_path),
            "in_database": db_status.get(arxiv_id, False),
            **info
        }
    
    # Save manifest
    manifest_path = output_dir / "manifest.yaml"
    with open(manifest_path, 'w') as f:
        yaml.dump(downloaded, f, default_flow_style=False)
    
    logger.info(f"Saved manifest to {manifest_path}")
    
    # Report status
    print("\n" + "="*60)
    print("PAPER IMPORT STATUS")
    print("="*60)
    for name, info in downloaded.items():
        status = "✓ IN DB" if info["in_database"] else "✗ NOT IN DB"
        print(f"{name:12} | {info['arxiv_id']:12} | {status}")
    print("="*60)
    
    # Next steps
    print("\nNext steps:")
    print("1. Process papers through ACID pipeline if not in database")
    print("2. Clone GitHub repositories")
    print("3. Generate embeddings for semantic analysis")
    
    return 0

if __name__ == "__main__":
    exit(main())