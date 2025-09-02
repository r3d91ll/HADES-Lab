#!/usr/bin/env python3
"""
Process the three word2vec evolution papers through ACID pipeline.
Uses the standard pipeline tools with optimized configuration.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Our three papers
    papers = [
        {
            'arxiv_id': '1301.3781',
            'pdf_path': '/bulk-store/arxiv-data/pdf/1301/1301.3781.pdf',
            'title': 'word2vec'
        },
        {
            'arxiv_id': '1405.4053', 
            'pdf_path': '/bulk-store/arxiv-data/pdf/1405/1405.4053.pdf',
            'title': 'doc2vec'
        },
        {
            'arxiv_id': '1803.09473',
            'pdf_path': '/bulk-store/arxiv-data/pdf/1803/1803.09473.pdf',
            'title': 'code2vec'
        }
    ]
    
    # Create a file list for the pipeline
    list_file = Path(__file__).parent / 'paper_list.txt'
    with open(list_file, 'w') as f:
        for paper in papers:
            # Just write the arxiv IDs
            f.write(f"{paper['arxiv_id']}\n")
    
    logger.info("=" * 80)
    logger.info("WORD2VEC EVOLUTION EXPERIMENT - ACID PROCESSING")
    logger.info("=" * 80)
    logger.info(f"Processing {len(papers)} papers with optimized settings")
    for paper in papers:
        logger.info(f"  • {paper['arxiv_id']}: {paper['title']} - {paper['pdf_path']}")
    logger.info("=" * 80)
    
    # Verify files exist
    missing = []
    for paper in papers:
        if not Path(paper['pdf_path']).exists():
            missing.append(paper['arxiv_id'])
    
    if missing:
        logger.error(f"Missing PDFs: {missing}")
        return 1
    
    # Set environment variables
    os.environ['ARANGO_PASSWORD'] = 'root_password'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use both GPUs
    
    # Path to pipeline script
    pipeline_script = Path(__file__).parent.parent.parent / 'tools' / 'arxiv' / 'pipelines' / 'arxiv_pipeline.py'
    config_file = Path(__file__).parent / 'pipeline_config.yaml'
    
    # Command to run
    cmd = [
        sys.executable,
        str(pipeline_script),
        '--config', str(config_file),
        '--source', 'specific_list',
        '--count', '3',
        '--arango-password', 'root_password'
    ]
    
    logger.info("\nStarting ACID pipeline...")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("-" * 80)
    
    start_time = datetime.now()
    
    try:
        # Run the pipeline
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(pipeline_script.parent)
        )
        
        # Log output
        if result.stdout:
            logger.info("Pipeline output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        if result.stderr:
            logger.warning("Pipeline warnings/errors:")
            for line in result.stderr.split('\n'):
                if line.strip():
                    logger.warning(f"  {line}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if result.returncode == 0:
            logger.info("\n" + "=" * 80)
            logger.info(f"✓ PROCESSING COMPLETE in {elapsed:.1f} seconds")
            logger.info("=" * 80)
            
            # Report performance
            papers_per_minute = (len(papers) / elapsed) * 60
            logger.info(f"Processing rate: {papers_per_minute:.1f} papers/minute")
            logger.info(f"Average time per paper: {elapsed/len(papers):.1f} seconds")
            
            # Save completion marker
            completion_file = Path(__file__).parent / 'acid_processing_complete.json'
            with open(completion_file, 'w') as f:
                json.dump({
                    'completed': True,
                    'timestamp': datetime.now().isoformat(),
                    'elapsed_seconds': elapsed,
                    'papers_processed': len(papers),
                    'papers': papers
                }, f, indent=2)
            
            logger.info(f"\nCompletion marker saved to: {completion_file}")
            return 0
        else:
            logger.error(f"\n✗ Pipeline failed with return code: {result.returncode}")
            return result.returncode
            
    except Exception as e:
        logger.error(f"\n✗ Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())