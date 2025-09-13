#!/usr/bin/env python3
"""
Create a proper pipeline config with arxiv_ids loaded from file.
"""

import yaml

# Read the arxiv IDs from file
with open('cited_papers_downloaded.txt', 'r') as f:
    arxiv_ids = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(arxiv_ids)} ArXiv IDs")

# Create the config
config = {
    'arango': {
        'host': 'http://localhost:8529',
        'database': 'academy_store',
        'username': 'root'
    },
    'phases': {
        'extraction': {
            'workers': 16,
            'memory_per_worker_gb': 4,
            'timeout_seconds': 120,
            'gpu_devices': [0, 1],
            'workers_per_gpu': 8,
            'docling': {
                'use_ocr': False,
                'extract_tables': True,
                'extract_equations': True,
                'extract_images': True,
                'use_fallback': True,
                'max_file_size_mb': 50,
                'batch_size': 12,
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
            'arxiv_ids': arxiv_ids  # Direct list of IDs
        },
        'checkpoint_interval': 5,
        'max_retries': 2,
        'retry_delay': 5,
        'continue_on_error': True
    },
    'error_handling': {
        'max_retries': 2,
        'retry_delay': 5,
        'continue_on_error': True,
        'log_errors': True
    },
    'monitoring': {
        'log_level': 'INFO',
        'progress_interval': 10,
        'show_memory_usage': True
    }
}

# Save the config
output_file = 'cited_papers_ready_config.yaml'
with open(output_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Created config: {output_file}")
print(f"First 5 ArXiv IDs in config:")
for arxiv_id in arxiv_ids[:5]:
    print(f"  - {arxiv_id}")

print(f"\nTo run the pipeline:")
print(f"  cd /home/todd/olympus/HADES-Lab/tools/arxiv/pipelines")
print(f"  poetry run python arxiv_pipeline.py --config ../utils/{output_file} --arango-password $ARANGO_PASSWORD --source specific_list --count {len(arxiv_ids)}")
