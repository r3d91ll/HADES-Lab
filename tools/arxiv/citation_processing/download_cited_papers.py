#!/usr/bin/env python3
"""
Download cited papers from ArXiv.
"""

import os
import time
import requests
from pathlib import Path
from typing import List
import concurrent.futures
from tqdm import tqdm

def download_arxiv_pdf(arxiv_id: str, output_dir: Path) -> bool:
    """Download a single ArXiv PDF."""
    # Clean the ID
    clean_id = arxiv_id.replace('.', '')

    # Create output path
    pdf_path = output_dir / f"{clean_id}.pdf"

    # Skip if already exists
    if pdf_path.exists():
        return True

    # Build ArXiv PDF URL
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        # Download with timeout
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save PDF
        pdf_path.write_bytes(response.content)
        return True

    except Exception as e:
        print(f"Failed to download {arxiv_id}: {e}")
        return False

def download_papers(arxiv_ids: List[str], output_dir: Path, max_workers: int = 5):
    """Download multiple papers with rate limiting."""

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    success = []
    failed = []

    print(f"Downloading {len(arxiv_ids)} papers to {output_dir}")

    # Download with progress bar
    with tqdm(total=len(arxiv_ids)) as pbar:
        for i, arxiv_id in enumerate(arxiv_ids):
            # Rate limiting - ArXiv recommends max 3-4 requests per second
            if i > 0:
                time.sleep(0.5)  # Be polite to ArXiv servers

            if download_arxiv_pdf(arxiv_id, output_dir):
                success.append(arxiv_id)
            else:
                failed.append(arxiv_id)

            pbar.update(1)
            pbar.set_description(f"Success: {len(success)}, Failed: {len(failed)}")

    return success, failed

def main():
    # Read the list of papers to download
    with open('cited_papers_missing.txt', 'r') as f:
        arxiv_ids = [line.strip() for line in f if line.strip()]

    print(f"=== Downloading Cited Papers ===\n")
    print(f"Papers to download: {len(arxiv_ids)}")

    # Set output directory
    output_dir = Path('/bulk-store/arxiv-data/pdf')

    # Download papers
    success, failed = download_papers(arxiv_ids, output_dir)

    print(f"\n=== Download Summary ===")
    print(f"Successfully downloaded: {len(success)}/{len(arxiv_ids)}")
    print(f"Failed downloads: {len(failed)}")

    if success:
        # Save successful downloads
        with open('cited_papers_downloaded.txt', 'w') as f:
            for arxiv_id in success:
                f.write(f"{arxiv_id}\n")
        print(f"\nDownloaded papers saved to: cited_papers_downloaded.txt")

        # Create config for processing downloaded papers
        import yaml
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
            'arxiv_ids_file': 'cited_papers_downloaded.txt',
            'checkpoint_interval': 10,
            'error_handling': {
                'max_retries': 2,
                'retry_delay': 5,
                'continue_on_error': True
            }
        }

        config_file = 'cited_papers_downloaded_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Created config: {config_file}")
        print("\nTo process downloaded papers, run:")
        print(f"  python ../pipelines/arxiv_pipeline.py --config {config_file} --arango-password $ARANGO_PASSWORD")

    if failed:
        print(f"\nFailed to download:")
        for arxiv_id in failed[:10]:
            print(f"  - {arxiv_id}")

        with open('cited_papers_failed.txt', 'w') as f:
            for arxiv_id in failed:
                f.write(f"{arxiv_id}\n")
        print(f"\nFailed downloads saved to: cited_papers_failed.txt")

if __name__ == "__main__":
    main()
