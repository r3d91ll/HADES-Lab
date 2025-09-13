#!/usr/bin/env python3
"""
Process cited papers directly using the correct PDF paths.
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.arxiv.pipelines.arxiv_pipeline import AcidPipeline
from tools.arxiv.pipelines.arxiv_pipeline import ProcessingTask
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Read the arxiv IDs
    with open('cited_papers_downloaded.txt', 'r') as f:
        arxiv_ids = [line.strip() for line in f if line.strip()]

    logger.info(f"Found {len(arxiv_ids)} ArXiv IDs to process")

    # Build tasks with correct paths
    tasks = []
    pdf_dir = Path('/bulk-store/arxiv-data/pdf')

    for arxiv_id in arxiv_ids:
        # Convert ID to filename format (remove dots)
        clean_id = arxiv_id.replace('.', '')
        pdf_path = pdf_dir / f"{clean_id}.pdf"

        if pdf_path.exists():
            tasks.append(ProcessingTask(
                arxiv_id=arxiv_id,
                pdf_path=str(pdf_path),
                latex_path=None
            ))
            logger.info(f"Found PDF for {arxiv_id}: {pdf_path}")
        else:
            logger.warning(f"PDF not found for {arxiv_id}: {pdf_path}")

    logger.info(f"Created {len(tasks)} tasks for processing")

    if not tasks:
        logger.error("No tasks to process!")
        return

    # Load config
    with open('cited_papers_ready_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize pipeline
    pipeline = AcidPipeline(config)

    # Process in phases
    logger.info("Starting PHASE 1: Extraction")
    extraction_results = pipeline.phase_extraction(tasks)

    logger.info("Starting PHASE 2: Embedding")
    embedding_results = pipeline.phase_embedding(tasks)

    logger.info("Processing complete!")
    logger.info(f"Extraction results: {extraction_results}")
    logger.info(f"Embedding results: {embedding_results}")

if __name__ == "__main__":
    main()
