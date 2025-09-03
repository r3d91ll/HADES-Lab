#!/usr/bin/env python3
"""
ArXiv Pipeline Test Runner
===========================

Run the ArXiv pipeline with configurable paper counts for testing.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

def run_pipeline_test(count: int, config_file: str = None):
    """
    Run the ArXiv pipeline with specified paper count.
    
    Args:
        count: Number of papers to process
        config_file: Configuration file to use
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
    
    # Check config exists
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    print("=" * 60)
    print(f"ArXiv Pipeline Test - {count} Papers")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate estimated time
    estimated_minutes = count / 11.3
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
        "--count", str(count),
        "--arango-password", arango_password
    ]
    
    print(f"Running command:")
    print(f"  {' '.join(cmd[:3])}")
    print(f"  {' '.join(cmd[3:5])}")
    print(f"  {' '.join(cmd[5:7])}")
    print(f"  --arango-password [HIDDEN]")
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
        print(f"Papers processed: {count}")
        print(f"Actual runtime: {elapsed_minutes:.1f} minutes")
        print(f"Processing rate: {count / elapsed_minutes:.1f} papers/minute")
        
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
        return False
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Run ArXiv pipeline tests")
    parser.add_argument('count', type=int, 
                       help='Number of papers to process')
    parser.add_argument('--config', type=str,
                       help='Configuration file (default: acid_pipeline_phased.yaml)')
    parser.add_argument('--series', action='store_true',
                       help='Run a series of tests: 10, 50, 100, 500, 1000')
    
    args = parser.parse_args()
    
    if args.series:
        # Run series of tests
        test_counts = [10, 50, 100, 500, 1000]
        print("Running test series:", test_counts)
        print()
        
        results = []
        for count in test_counts:
            print(f"\n{'='*60}")
            print(f"Test {len(results)+1}/{len(test_counts)}: {count} papers")
            print(f"{'='*60}\n")
            
            success = run_pipeline_test(count, args.config)
            results.append((count, success))
            
            if not success:
                print(f"\n⚠️  Stopping series due to failure")
                break
            
            # Pause between tests
            if count < test_counts[-1]:
                print(f"\n⏸️  Pausing 30 seconds before next test...")
                time.sleep(30)
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SERIES SUMMARY")
        print("="*60)
        for count, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{count:5d} papers: {status}")
        
    else:
        # Run single test
        run_pipeline_test(args.count, args.config)


if __name__ == "__main__":
    main()