#!/usr/bin/env python3
"""
Run ArXiv Pipeline from Paper List
===================================

Process a specific number of papers from an existing ArXiv ID list.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import random

def run_pipeline_from_list(
    paper_list: str,
    count: int,
    config_file: str = None,
    shuffle: bool = False,
    start_from: int = 0
):
    """
    Run the ArXiv pipeline using papers from a list file.
    
    Args:
        paper_list: Path to file containing ArXiv IDs (one per line)
        count: Number of papers to process
        config_file: Configuration file to use
        shuffle: Whether to shuffle the paper list before processing
        start_from: Index to start from in the list
    """
    # Default config if not specified
    if config_file is None:
        config_file = "../configs/acid_pipeline_phased.yaml"
    
    # Get ArXiv password
    arango_password = os.getenv('ARANGO_PASSWORD')
    if not arango_password:
        print("❌ ARANGO_PASSWORD not set")
        print("Please set: export ARANGO_PASSWORD='your-password'")
        return False
    
    # Check paper list exists
    list_path = Path(paper_list)
    if not list_path.exists():
        print(f"❌ Paper list not found: {list_path}")
        print("\nAvailable lists:")
        data_dir = Path("data/arxiv_collections")
        if data_dir.exists():
            for f in data_dir.glob("arxiv_ids*.txt"):
                print(f"  - {f}")
        return False
    
    # Load paper IDs
    with open(list_path, 'r') as f:
        all_paper_ids = [line.strip() for line in f if line.strip()]
    
    total_available = len(all_paper_ids)
    print(f"📄 Loaded {total_available:,} paper IDs from {list_path.name}")
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(all_paper_ids)
        print("🔀 Shuffled paper list")
    
    # Select papers to process
    end_idx = min(start_from + count, total_available)
    selected_ids = all_paper_ids[start_from:end_idx]
    actual_count = len(selected_ids)
    
    if actual_count < count:
        print(f"⚠️  Only {actual_count} papers available (requested {count})")
    
    # Create temporary file with selected IDs
    temp_list = Path(f"/tmp/arxiv_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(temp_list, 'w') as f:
        for paper_id in selected_ids:
            f.write(f"{paper_id}\n")
    
    print(f"📝 Created temporary list: {temp_list}")
    print(f"   Papers {start_from+1} to {end_idx} of {total_available}")
    
    # Check config exists
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    print("\n" + "=" * 60)
    print(f"ArXiv Pipeline Test - {actual_count} Papers")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate estimated time
    estimated_minutes = actual_count / 11.3
    if estimated_minutes < 60:
        print(f"Estimated runtime: {estimated_minutes:.1f} minutes")
    else:
        print(f"Estimated runtime: {estimated_minutes/60:.1f} hours")
    print()
    
    # Run the pipeline
    pipeline_dir = Path(__file__).parent.parent / "pipelines"
    pipeline_script = pipeline_dir / "arxiv_pipeline.py"
    
    if not pipeline_script.exists():
        print(f"❌ Pipeline script not found: {pipeline_script}")
        return False
    
    cmd = [
        sys.executable,
        str(pipeline_script),
        "--config", str(config_path),
        "--paper-list", str(temp_list),
        "--count", str(actual_count),
        "--arango-password", arango_password
    ]
    
    print(f"Running command:")
    print(f"  python arxiv_pipeline.py")
    print(f"    --config {config_path.name}")
    print(f"    --paper-list {temp_list.name}")
    print(f"    --count {actual_count}")
    print(f"    --arango-password [HIDDEN]")
    print()
    
    # Start timer
    start_time = time.time()
    
    try:
        # Run pipeline
        result = subprocess.run(
            cmd,
            cwd=str(pipeline_dir),
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        # Calculate runtime
        elapsed = time.time() - start_time
        elapsed_minutes = elapsed / 60
        
        print()
        print("=" * 60)
        print("Test Complete")
        print("=" * 60)
        print(f"Papers processed: {actual_count}")
        print(f"Actual runtime: {elapsed_minutes:.1f} minutes")
        if elapsed_minutes > 0:
            print(f"Processing rate: {actual_count / elapsed_minutes:.1f} papers/minute")
        
        # Clean up temp file
        if temp_list.exists():
            temp_list.unlink()
            print(f"🧹 Cleaned up temporary list")
        
        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")
            return True
        else:
            print(f"❌ Pipeline failed with exit code: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        elapsed = time.time() - start_time
        print(f"Runtime before interruption: {elapsed/60:.1f} minutes")
        
        # Clean up temp file
        if temp_list.exists():
            temp_list.unlink()
            print(f"🧹 Cleaned up temporary list")
        
        return False
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        
        # Clean up temp file
        if temp_list.exists():
            temp_list.unlink()
            print(f"🧹 Cleaned up temporary list")
        
        return False


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Run ArXiv pipeline with papers from a list file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 100 papers from the latest collection
  %(prog)s 100
  
  # Process 1000 papers from a specific list
  %(prog)s 1000 --list data/arxiv_collections/arxiv_ids_extended_20250829_190526.txt
  
  # Process 500 papers starting from paper 1000
  %(prog)s 500 --start-from 1000
  
  # Process 100 random papers
  %(prog)s 100 --shuffle
  
  # Use custom config
  %(prog)s 100 --config ../configs/weekend_test_15k.yaml
        """
    )
    
    parser.add_argument('count', type=int, 
                       help='Number of papers to process')
    parser.add_argument('--list', type=str, dest='paper_list',
                       help='Path to paper ID list file (default: latest in data/arxiv_collections/)')
    parser.add_argument('--config', type=str,
                       help='Configuration file (default: acid_pipeline_phased.yaml)')
    parser.add_argument('--shuffle', action='store_true',
                       help='Shuffle papers before processing')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start from this index in the list (default: 0)')
    
    args = parser.parse_args()
    
    # Find latest paper list if not specified
    if args.paper_list is None:
        data_dir = Path("data/arxiv_collections")
        if data_dir.exists():
            # Find the most recent arxiv_ids file
            id_files = sorted(data_dir.glob("arxiv_ids*.txt"))
            if id_files:
                args.paper_list = str(id_files[-1])
                print(f"📂 Using latest paper list: {id_files[-1].name}")
            else:
                print("❌ No paper lists found in data/arxiv_collections/")
                print("   Please run: python collect_ai_papers_extended.py")
                return
        else:
            print("❌ Data directory not found: data/arxiv_collections/")
            return
    
    # Run the pipeline
    success = run_pipeline_from_list(
        paper_list=args.paper_list,
        count=args.count,
        config_file=args.config,
        shuffle=args.shuffle,
        start_from=args.start_from
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()