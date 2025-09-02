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
import yaml

def run_pipeline_from_list(
    paper_list: str,
    count: int,
    config_file: str = None,
    shuffle: bool = False,
    start_from: int = 0,
    skip_processed: bool = True
):
    """
    Run the ArXiv pipeline using papers from a list file.
    
    Args:
        paper_list: Path to file containing ArXiv IDs (one per line)
        count: Number of papers to process
        config_file: Configuration file to use
        shuffle: Whether to shuffle the paper list before processing
        start_from: Index to start from in the list
        skip_processed: Whether to skip papers already in ArangoDB (default: True)
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
    
    # Check for already processed papers if requested
    if skip_processed:
        print("🔍 Checking for already processed papers...")
        
        # Import check_existing_papers function
        sys.path.insert(0, str(Path(__file__).parent))
        from check_existing_papers import check_existing_papers
        
        # Check existing papers
        check_result = check_existing_papers(paper_list)
        
        if check_result and 'unprocessed_ids' in check_result:
            unprocessed_ids = check_result['unprocessed_ids']
            processed_count = check_result.get('processed', 0)
            
            if processed_count > 0:
                print(f"⚠️  Found {processed_count:,} already processed papers - skipping them")
                all_paper_ids = unprocessed_ids
                total_available = len(all_paper_ids)
                print(f"📄 Using {total_available:,} unprocessed papers")
        else:
            print("⚠️  Could not check for processed papers - continuing with full list")
    
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
    
    # Load the base config file
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with selected paper IDs
    config['processing']['source'] = 'specific_list'
    config['processing']['specific_list']['arxiv_ids'] = selected_ids
    config['processing']['count'] = actual_count
    
    # Create temporary config file
    temp_config = Path(f"/tmp/arxiv_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📝 Created temporary config: {temp_config}")
    print(f"   Papers {start_from+1} to {end_idx} of {total_available}")
    print(f"   Selected {actual_count} paper IDs")
    
    print("\n" + "=" * 60)
    print(f"ArXiv Pipeline Test - {actual_count} Papers")
    print("=" * 60)
    print(f"Base Config: {config_path}")
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
        "--config", str(temp_config),
        "--source", "specific_list",
        "--count", str(actual_count),
        "--arango-password", arango_password
    ]
    
    print(f"Running command:")
    print(f"  python arxiv_pipeline.py")
    print(f"    --config {temp_config.name}")
    print(f"    --source specific_list")
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
        if temp_config.exists():
            temp_config.unlink()
            print(f"🧹 Cleaned up temporary config")
        
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
        if temp_config.exists():
            temp_config.unlink()
            print(f"🧹 Cleaned up temporary config")
        
        return False
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        
        # Clean up temp file
        if temp_config.exists():
            temp_config.unlink()
            print(f"🧹 Cleaned up temporary config")
        
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
    parser.add_argument('--no-skip', action='store_true',
                       help='Do not skip already processed papers (default: skip them)')
    
    args = parser.parse_args()
    
    # Find latest paper list if not specified
    if args.paper_list is None:
        # Search in both legacy and new SQL-backed output locations
        candidates = [
            Path("data/arxiv_collections"),
            Path("tools/arxiv/scripts/data/arxiv_collections"),
        ]
        id_files = []
        for data_dir in candidates:
            if data_dir.exists():
                id_files.extend(sorted(data_dir.glob("arxiv_ids*.txt")))
        if id_files:
            latest = sorted(id_files)[-1]
            args.paper_list = str(latest)
            print(f"📂 Using latest paper list: {latest}")
        else:
            print("❌ No paper lists found in:")
            for d in candidates:
                print(f"   - {d}")
            print("   Please run: tools/arxiv/scripts/collect_ai_papers_extended.py --mode sql")
            return
    
    # Run the pipeline
    success = run_pipeline_from_list(
        paper_list=args.paper_list,
        count=args.count,
        config_file=args.config,
        shuffle=args.shuffle,
        start_from=args.start_from,
        skip_processed=(not args.no_skip)  # Skip by default unless --no-skip is used
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
