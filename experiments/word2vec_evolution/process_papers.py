#!/usr/bin/env python3
"""
Process the three word2vec evolution papers through ACID pipeline.
Optimized for speed with 3 papers using multiple workers.
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set environment
# Env configuration (do not hard-code secrets)
if 'ARANGO_PASSWORD' not in os.environ:
    print("Warning: ARANGO_PASSWORD not set; configure via environment or secure config.", file=sys.stderr)
# Respect existing GPU selection
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # Use both GPUs if not predefined

# Import pipeline
from tools.arxiv.pipelines.arxiv_pipeline import ACIDPipeline, ProcessingTask

def main():
    # Our three papers with their actual paths
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
    
    # Create config optimized for 3 papers with fast processing
    config = {
        'pipeline': {
            'staging_directory': '/dev/shm/acid_staging',
            'use_gpu': True,
            'checkpoint_interval': 1,
            'batch_processing': True,
            'atomic_transactions': True,
            'max_retries': 3,
            'retry_delay_seconds': 5
        },
        'phases': {
            'extraction': {
                'workers': 3,  # One worker per paper for parallelism
                'batch_size': 1,  # Process each paper separately
                'timeout_seconds': 300,
                'gpu_memory_fraction': 0.3  # Lower since we have 3 workers
            },
            'embedding': {
                'workers': 3,  # Parallel embedding
                'batch_size': 1,
                'timeout_seconds': 600,
                'use_fp16': True,
                'max_chunk_size': 8192
            }
        },
        'storage': {
            'arango': {
                'host': '192.168.1.69',
                'port': 8529,
                'database': 'academy_store',
                'collections': {
                    'papers': 'arxiv_papers',
                    'chunks': 'arxiv_chunks',
                    'embeddings': 'arxiv_embeddings',
                    'structures': 'arxiv_structures'
                },
                'batch_size': 100
            }
        },
        'logging': {
            'level': 'INFO',
            'log_file': 'experiments/word2vec_evolution/processing.log'
        }
    }
    
    # Initialize pipeline
    print("=" * 80)
    print("WORD2VEC EVOLUTION EXPERIMENT - ACID PROCESSING")
    print("=" * 80)
    print(f"Processing {len(papers)} papers with optimized settings:")
    print(f"  - Extraction workers: {config['phases']['extraction']['workers']}")
    print(f"  - Embedding workers: {config['phases']['embedding']['workers']}")
    print(f"  - GPU acceleration: {config['pipeline']['use_gpu']}")
    print("=" * 80)
    
    pipeline = ACIDPipeline(config)
    
    # Create tasks
    tasks = []
    for paper in papers:
        task = ProcessingTask(
            arxiv_id=paper['arxiv_id'],
            pdf_path=paper['pdf_path'],
            latex_path=None
        )
        tasks.append(task)
        print(f"  • {paper['arxiv_id']}: {paper['title']}")
    
    print("\nStarting processing...")
    print("-" * 80)
    
    # Process through both phases
    try:
        # Phase 1: Extraction
        print("\n[PHASE 1] EXTRACTION")
        extraction_results = pipeline.phase_manager.run_extraction_phase(tasks)
        print(f"Extraction complete: {len([r for r in extraction_results if r['success']])} successful")
        
        # Phase 2: Embedding
        print("\n[PHASE 2] EMBEDDING")
        embedding_results = pipeline.phase_manager.run_embedding_phase()
        print(f"Embedding complete: {len([r for r in embedding_results if r['success']])} successful")
        
        # Save checkpoint (consider exposing a public method)
        pipeline._save_checkpoint({
            "tasks": [t.to_dict() for t in tasks],
            "extraction": extraction_results,
            "embedding": embedding_results
        })
        print("\n" + "=" * 80)
        print("✓ PROCESSING COMPLETE")
        print("=" * 80)
        
        # Report results
        for result in extraction_results:
            if result['success']:
                print(f"  ✓ {result['arxiv_id']}: Extracted")
            else:
                print(f"  ✗ {result['arxiv_id']}: Failed - {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())