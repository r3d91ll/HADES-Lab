#!/usr/bin/env python3
"""
ArXiv Pipeline Validation Script
=================================

Quick validation that the ArXiv pipeline is ready for large-scale testing.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, List

# Fix Python path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent  # Go up to HADES-Lab root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir.parent.parent))  # Add tools/arxiv to path


def check_environment() -> Tuple[bool, List[str]]:
    """Check if the environment is properly configured."""
    issues = []
    
    # Check ARANGO_PASSWORD
    if not os.getenv('ARANGO_PASSWORD'):
        issues.append("ARANGO_PASSWORD environment variable not set")
    
    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available - GPU acceleration will not work")
        else:
            gpu_count = torch.cuda.device_count()
            print(f"✅ Found {gpu_count} GPU(s)")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    except ImportError:
        issues.append("PyTorch not installed or not working")
    
    # Check disk space
    staging_path = Path("/dev/shm")
    if staging_path.exists():
        stats = os.statvfs(staging_path)
        free_gb = (stats.f_bavail * stats.f_frsize) / (1024**3)
        if free_gb < 10:
            issues.append(f"Low RAM disk space: {free_gb:.1f} GB free in /dev/shm")
        else:
            print(f"✅ RAM disk: {free_gb:.1f} GB free")
    else:
        issues.append("/dev/shm not available - staging will be slower")
    
    # Check PDF directory
    pdf_dir = Path("/bulk-store/arxiv-data/pdf")
    if not pdf_dir.exists():
        issues.append(f"PDF directory not found: {pdf_dir}")
    else:
        # Count available PDFs
        pdf_count = len(list(pdf_dir.glob("*/*.pdf")))
        print(f"✅ Found {pdf_count:,} PDFs in {pdf_dir}")
    
    return len(issues) == 0, issues


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if all required dependencies are installed."""
    issues = []
    
    # Package name to import name mapping
    packages_to_check = [
        ('arango', 'arango'),
        ('docling', 'docling'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('yaml', 'yaml'),  # PyYAML imports as yaml
        ('tqdm', 'tqdm')
    ]
    
    for display_name, import_name in packages_to_check:
        try:
            __import__(import_name)
            print(f"✅ {display_name} installed")
        except ImportError:
            issues.append(f"Required package not installed: {display_name}")
    
    # Check Jina
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
        print("✅ Jina embeddings model available")
    except Exception as e:
        issues.append(f"Cannot load Jina model: {e}")
    
    return len(issues) == 0, issues


def check_database() -> Tuple[bool, List[str]]:
    """Check database connectivity."""
    issues = []
    
    try:
        from core.database.arango.arango_client import ArangoDBManager
        
        config = {
            'host': os.getenv('ARANGO_HOST', 'http://192.168.1.69:8529'),
            'database': 'academy_store',
            'username': 'root',
            'password': os.getenv('ARANGO_PASSWORD')
        }
        
        if not config['password']:
            issues.append("Cannot test database - ARANGO_PASSWORD not set")
            return False, issues
        
        db_manager = ArangoDBManager(config)
        db = db_manager.db
        
        # Check collections
        required_collections = [
            'arxiv_papers',
            'arxiv_chunks', 
            'arxiv_embeddings',
            'arxiv_structures'
        ]
        
        for collection in required_collections:
            if db.has_collection(collection):
                count = db.collection(collection).count()
                print(f"✅ {collection}: {count:,} documents")
            else:
                issues.append(f"Missing collection: {collection}")
        
    except Exception as e:
        issues.append(f"Database connection failed: {e}")
    
    return len(issues) == 0, issues


def check_pipeline() -> Tuple[bool, List[str]]:
    """Check if the pipeline can be imported and initialized."""
    issues = []
    
    try:
        # Import the pipeline - we're in tools/arxiv/tests, need to go up one level
        import sys
        from pathlib import Path
        arxiv_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(arxiv_dir))
        
        from pipelines.arxiv_pipeline import ACIDPhasedPipeline
        
        # Check config file
        config_path = Path(__file__).parent.parent / "configs" / "acid_pipeline_phased.yaml"
        if not config_path.exists():
            issues.append(f"Pipeline config not found: {config_path}")
        else:
            print(f"✅ Pipeline config found: {config_path}")
            
            # Try to initialize pipeline (just check if it can be created)
            # We don't actually initialize it fully to avoid side effects
            print("✅ Pipeline class imported successfully")
                
    except ImportError as e:
        issues.append(f"Cannot import pipeline: {e}")
    
    return len(issues) == 0, issues


def main():
    """
    Run all validation checks and print a human-readable summary.
    
    Executes the Environment, Dependencies, Database, and Pipeline readiness checks in sequence, collects any reported issues, prints per-check results and a final summary to stdout, and suggests next steps when all checks pass.
    
    Returns:
        int: Exit code 0 when all checks pass; 1 when any check fails or an exception occurs during checking.
    """
    print("="*60)
    print("ArXiv Pipeline Validation")
    print("="*60)
    
    all_good = True
    all_issues = []
    
    # Run checks
    checks = [
        ("Environment", check_environment),
        ("Dependencies", check_dependencies),
        ("Database", check_database),
        ("Pipeline", check_pipeline)
    ]
    
    for check_name, check_func in checks:
        print(f"\n{check_name} Check:")
        print("-" * 40)
        
        try:
            success, issues = check_func()
            
            if success:
                print(f"✅ {check_name} check passed")
            else:
                print(f"❌ {check_name} check failed")
                for issue in issues:
                    print(f"   - {issue}")
                all_issues.extend(issues)
                all_good = False
                
        except Exception as e:
            print(f"❌ {check_name} check crashed: {e}")
            all_good = False
            all_issues.append(f"{check_name} check crashed: {e}")
    
    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✅ ALL CHECKS PASSED - Pipeline ready for testing!")
        print("\nNext steps:")
        print("1. Run: cd tools/arxiv/utils")
        print("2. Run: python lifecycle.py process [arxiv_id]")
        print("3. Run: cd ../tests")
        print("4. Run: ./run_large_scale_test.sh")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print(f"\nFound {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
        print("\nPlease fix these issues before running tests.")
        return 1


if __name__ == "__main__":
    sys.exit(main())