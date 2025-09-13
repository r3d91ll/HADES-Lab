#!/usr/bin/env python3
"""
Download and process papers cited by our target papers.

This script:
1. Reads the extracted bibliographies
2. Downloads PDFs for cited ArXiv papers
3. Processes them through the ACID pipeline to add embeddings
"""

import os
import sys
import json
import re
import yaml
from pathlib import Path
from typing import List, Set

def clean_arxiv_ids(ids: List[str]) -> List[str]:
    """Clean and validate ArXiv IDs."""
    valid_ids = []

    for arxiv_id in ids:
        # Remove any 'v' versions
        clean_id = re.sub(r'v\d+$', '', arxiv_id)

        # Check if it looks like a valid ArXiv ID
        if re.match(r'^\d{4}\.\d{4,5}$', clean_id):
            # Check year is reasonable (07-25 for 2007-2025)
            year = int(clean_id[:2])
            if 7 <= year <= 25 or 90 <= year <= 99:
                valid_ids.append(clean_id)
        elif re.match(r'^[a-z\-]+/\d{7}$', clean_id):
            valid_ids.append(clean_id)

    return valid_ids

def main():
    # Load extracted bibliographies
    with open('bibliographies_extracted.json', 'r') as f:
        data = json.load(f)

    # Get valid ArXiv IDs
    all_arxiv_ids = clean_arxiv_ids(data['all_cited_arxiv_ids'])

    print(f"=== Processing Cited Papers ===\n")
    print(f"Found {len(all_arxiv_ids)} valid ArXiv IDs to process")

    if not all_arxiv_ids:
        print("No valid ArXiv IDs found.")
        return

    # Save the list of papers to process
    cited_papers_file = 'cited_papers_to_process.txt'
    with open(cited_papers_file, 'w') as f:
        for arxiv_id in all_arxiv_ids:
            f.write(f"{arxiv_id}\n")

    print(f"Saved list to: {cited_papers_file}")
    print(f"\nFirst 10 papers to process:")
    for arxiv_id in all_arxiv_ids[:10]:
        print(f"  - {arxiv_id}")

    # Create config for processing
    config = {
        'mode': 'phased',
        'phases': {
            'extraction': {
                'workers': 16,
                'batch_size': 12,
                'timeout_seconds': 300
            },
            'embedding': {
                'workers': 4,
                'batch_size': 8,
                'use_fp16': True
            }
        },
        'arxiv_ids_file': cited_papers_file,
        'checkpoint_interval': 10,
        'error_handling': {
            'max_retries': 2,
            'retry_delay': 5,
            'continue_on_error': True
        }
    }

    # Save config
    config_file = 'cited_papers_config.yaml'
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nCreated config: {config_file}")
    print("\nTo process these papers, run:")
    print(f"  cd {os.getcwd()}")
    print(f"  python ../pipelines/arxiv_pipeline.py --config {config_file} --arango-password $ARANGO_PASSWORD")

    # Also create a simple list for checking availability
    print(f"\n=== Checking PDF Availability ===")

    pdf_base = Path('/bulk-store/arxiv-data/pdf')
    available = []
    missing = []

    for arxiv_id in all_arxiv_ids:
        # Convert ID to path format
        clean_id = arxiv_id.replace('.', '')
        pdf_path = pdf_base / f"{clean_id}.pdf"

        if pdf_path.exists():
            available.append(arxiv_id)
        else:
            missing.append(arxiv_id)

    print(f"PDFs available: {len(available)}/{len(all_arxiv_ids)}")
    print(f"PDFs missing: {len(missing)}")

    if missing:
        print(f"\nFirst 10 missing PDFs:")
        for arxiv_id in missing[:10]:
            print(f"  - {arxiv_id}")

        # Save missing list
        with open('cited_papers_missing.txt', 'w') as f:
            for arxiv_id in missing:
                f.write(f"{arxiv_id}\n")
        print(f"\nMissing papers saved to: cited_papers_missing.txt")

    # Save available list for immediate processing
    if available:
        with open('cited_papers_available.txt', 'w') as f:
            for arxiv_id in available:
                f.write(f"{arxiv_id}\n")
        print(f"Available papers saved to: cited_papers_available.txt")

        # Update config to use available papers
        config['arxiv_ids_file'] = 'cited_papers_available.txt'
        config_available_file = 'cited_papers_available_config.yaml'
        with open(config_available_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"\nTo process available papers immediately:")
        print(f"  python ../pipelines/arxiv_pipeline.py --config {config_available_file} --arango-password $ARANGO_PASSWORD")

if __name__ == "__main__":
    main()
