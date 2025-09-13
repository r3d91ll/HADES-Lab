#!/usr/bin/env python3
"""
Process cited papers through the ACID pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get list of papers to process
    with open('cited_papers_downloaded.txt', 'r') as f:
        arxiv_ids = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(arxiv_ids)} cited papers...")

    # Process in batches
    batch_size = 10
    for i in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(arxiv_ids) + batch_size - 1) // batch_size

        print(f"\n{'='*60}")
        print(f"Batch {batch_num}/{total_batches}: Processing {len(batch)} papers")
        print(f"Papers: {', '.join(batch)}")
        print('='*60)

        # Write batch to temp file
        batch_file = f'batch_{batch_num}_arxiv_ids.txt'
        with open(batch_file, 'w') as f:
            for arxiv_id in batch:
                f.write(f"{arxiv_id}\n")

        # Create batch config
        import yaml

        config = {
            'arango': {
                'host': 'http://localhost:8529',
                'database': 'academy_store',
                'username': 'root'
            },
            'phases': {
                'extraction': {
                    'workers': 8,
                    'memory_per_worker_gb': 4,
                    'timeout_seconds': 120,
                    'gpu_devices': [0, 1],
                    'workers_per_gpu': 4,
                    'docling': {
                        'use_ocr': False,
                        'extract_tables': True,
                        'extract_equations': True,
                        'extract_images': True,
                        'use_fallback': True,
                        'max_file_size_mb': 50,
                        'batch_size': 8,
                        'use_gpu': True
                    }
                },
                'embedding': {
                    'workers': 4,
                    'gpu_devices': [0, 1],
                    'workers_per_gpu': 2,
                    'jina': {
                        'model_name': 'jinaai/jina-embeddings-v4',
                        'device': 'cuda',
                        'use_fp16': True,
                        'chunk_size_tokens': 1000,
                        'chunk_overlap_tokens': 200,
                        'max_context_length': 32768,
                        'batch_size': 8
                    }
                }
            },
            'staging': {
                'directory': '/dev/shm/acid_staging',
                'cleanup_on_success': True,
                'cleanup_on_error': False
            },
            'processing': {
                'local': {
                    'pdf_dir': '/bulk-store/arxiv-data/pdf',
                    'pattern': '*.pdf'
                },
                'specific_list': {
                    'arxiv_ids': batch
                },
                'checkpoint_interval': 5,
                'max_retries': 2,
                'retry_delay': 5,
                'continue_on_error': True
            },
            'logging': {
                'level': 'INFO',
                'file': f'../logs/batch_{batch_num}.log'
            }
        }

        config_file = f'batch_{batch_num}_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Process batch
        cmd = [
            'poetry', 'run', 'python', '../pipelines/arxiv_pipeline.py',
            '--config', config_file,
            '--source', 'specific_list',
            '--count', str(len(batch)),
            '--arango-password', os.environ.get('ARANGO_PASSWORD', '')
        ]

        print(f"Running: {' '.join(cmd[:5])}...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Parse output for success/failure
            if 'successfully' in result.stdout.lower() or 'complete' in result.stdout.lower():
                print(f"✓ Batch {batch_num} completed successfully")
            else:
                print(f"⚠ Batch {batch_num} completed with warnings")

            # Show last few lines of output
            output_lines = result.stdout.split('\n')
            print("Last output:")
            for line in output_lines[-5:]:
                if line.strip():
                    print(f"  {line}")

        except subprocess.TimeoutExpired:
            print(f"✗ Batch {batch_num} timed out")
        except Exception as e:
            print(f"✗ Batch {batch_num} error: {e}")

    print(f"\n{'='*60}")
    print("All batches processed!")

    # Check results
    print("\nChecking database for processed papers...")
    check_cmd = ['poetry', 'run', 'python', '-c', '''
from arango import ArangoClient
import os

client = ArangoClient(hosts="http://localhost:8529")
db = client.db("academy_store", username="root", password=os.environ.get("ARANGO_PASSWORD"))

cursor = db.aql.execute("""
    FOR e IN arxiv_embeddings
    COLLECT paper_id = e.paper_id WITH COUNT INTO chunks
    RETURN {paper_id: paper_id, chunks: chunks}
""")

papers = list(cursor)
print(f"Papers with embeddings: {len(papers)}")
for p in papers[:10]:
    print(f"  - {p['paper_id']}: {p['chunks']} chunks")
''']

    subprocess.run(check_cmd)

if __name__ == "__main__":
    main()
