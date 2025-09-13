#!/usr/bin/env python3
"""
Run the ACID pipeline on cited papers with correct paths.
"""

import subprocess
import sys
from pathlib import Path

# Get list of PDFs that exist
pdf_dir = Path('/bulk-store/arxiv-data/pdf')
arxiv_ids = []

with open('cited_papers_downloaded.txt', 'r') as f:
    for line in f:
        arxiv_id = line.strip()
        if arxiv_id:
            # Check if PDF exists with clean ID
            clean_id = arxiv_id.replace('.', '')
            pdf_path = pdf_dir / f"{clean_id}.pdf"
            if pdf_path.exists():
                arxiv_ids.append(arxiv_id)
                print(f"Found: {arxiv_id} -> {pdf_path}")

print(f"\nTotal papers to process: {len(arxiv_ids)}")

# Save the list of arxiv IDs with confirmed PDFs
with open('cited_papers_confirmed.txt', 'w') as f:
    for arxiv_id in arxiv_ids:
        # Write the path directly
        clean_id = arxiv_id.replace('.', '')
        pdf_path = pdf_dir / f"{clean_id}.pdf"
        f.write(f"{pdf_path}\n")

print(f"Saved confirmed PDF paths to: cited_papers_confirmed.txt")

# Now run the pipeline with local source
cmd = [
    'python', '../pipelines/arxiv_pipeline.py',
    '--config', '../configs/acid_pipeline_phased.yaml',
    '--source', 'local',
    '--count', str(len(arxiv_ids)),
    '--arango-password', sys.argv[1] if len(sys.argv) > 1 else input("Enter ArangoDB password: ")
]

print(f"\nRunning command: {' '.join(cmd)}")
subprocess.run(cmd)
