#!/usr/bin/env python3
"""
Process cited papers in batch using direct PDF paths.
"""

import os
import sys
import subprocess
from pathlib import Path

# Build list of PDF files to process
pdf_dir = Path('/bulk-store/arxiv-data/pdf')
pdf_files = []

# Read downloaded arxiv IDs
with open('cited_papers_downloaded.txt', 'r') as f:
    arxiv_ids = [line.strip() for line in f if line.strip()]

print(f"Processing {len(arxiv_ids)} cited papers...")

# Create a file list with actual PDF paths
pdf_list_file = 'cited_papers_pdf_list.txt'
with open(pdf_list_file, 'w') as f:
    for arxiv_id in arxiv_ids:
        clean_id = arxiv_id.replace('.', '')
        pdf_path = pdf_dir / f"{clean_id}.pdf"
        if pdf_path.exists():
            f.write(f"{pdf_path}\n")
            pdf_files.append(str(pdf_path))
            print(f"  ✓ {arxiv_id}")
        else:
            print(f"  ✗ {arxiv_id} (PDF not found)")

print(f"\nFound {len(pdf_files)} PDF files")

if pdf_files:
    # Process each PDF directly
    print("\nProcessing PDFs through ACID pipeline...")

    # Call the pipeline with the first 5 PDFs as a test
    test_count = min(5, len(pdf_files))

    for i, pdf_path in enumerate(pdf_files[:test_count]):
        print(f"\n[{i+1}/{test_count}] Processing {Path(pdf_path).stem}...")

        # Extract arxiv_id from filename
        stem = Path(pdf_path).stem
        # Try to reconstruct the arxiv ID (add dot back)
        if len(stem) >= 8:
            arxiv_id = f"{stem[:4]}.{stem[4:]}"
        else:
            arxiv_id = stem

        # Call a simple processing script for each paper
        cmd = [
            'python', '-c',
            f'''
import sys
import os
sys.path.append("/home/todd/olympus/HADES-Lab")
os.environ["ARANGO_PASSWORD"] = "{os.environ.get('ARANGO_PASSWORD', '')}"

from pathlib import Path
from core.framework.extractors import DoclingExtractor
from core.framework.embedders import JinaV4Embedder
from core.framework.storage import ArangoStorage

# Initialize components
extractor = DoclingExtractor(use_gpu=True, batch_size=1)
embedder = JinaV4Embedder(device="cuda", use_fp16=True)
storage = ArangoStorage(
    host="http://localhost:8529",
    database="academy_store",
    username="root",
    password=os.environ.get("ARANGO_PASSWORD")
)

# Process the paper
pdf_path = "{pdf_path}"
arxiv_id = "{arxiv_id}"

print(f"Extracting: {{arxiv_id}}")
chunks = extractor.extract(pdf_path)

if chunks:
    print(f"Embedding {{len(chunks)}} chunks...")
    embeddings = embedder.embed_batch([c["text"] for c in chunks])

    print(f"Storing in ArangoDB...")
    for chunk, embedding in zip(chunks, embeddings):
        chunk["paper_id"] = arxiv_id
        chunk["embedding"] = embedding
        storage.store_chunk(chunk)

    print(f"✓ Processed {{arxiv_id}}: {{len(chunks)}} chunks")
else:
    print(f"✗ No chunks extracted from {{arxiv_id}}")
'''
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("Timeout - skipping")
        except Exception as e:
            print(f"Error: {e}")

    print(f"\n{'='*60}")
    print(f"Test processing complete for {test_count} papers")
    print(f"To process all {len(pdf_files)} papers, modify the script")
else:
    print("No PDFs found to process")
