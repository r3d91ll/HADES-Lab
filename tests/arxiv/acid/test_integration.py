"""
Integration test for ACID pipeline with real ArXiv data.

Tests the complete workflow with actual papers from our local PDF collection.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path safely
project_root = Path(__file__).resolve().parent.parent.parent.parent
if project_root.exists() and (project_root / 'tools').exists():
    sys.path.insert(0, str(project_root))
else:
    # Fallback path handling
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from tools.arxiv.utils.on_demand_processor import OnDemandProcessor
from tools.arxiv.monitoring.acid_monitoring import ArangoMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_real_paper_processing():
    """Test processing with a real ArXiv paper we know exists"""
    
    # Check password first
    arango_password = os.environ.get('ARANGO_PASSWORD')
    if not arango_password:
        logger.error("ARANGO_PASSWORD environment variable not set")
        return False
    
    # Papers we know exist in /bulk-store/arxiv-data/pdf/
    test_papers = [
        "1212.1432",  # A paper from 2012 that should exist
        "1301.0007",  # Another known paper
        "1506.01094",  # A 2015 paper
    ]
    
    config = {
        'cache_root': '/bulk-store/arxiv-data',
        'sqlite_db': '/bulk-store/arxiv-cache.db',
        'arango': {
            'host': ['http://192.168.1.69:8529'],
            'database': 'academy_store',
            'username': 'root',
            'password': arango_password
        }
    }
    
    # Initialize processor
    processor = OnDemandProcessor(config)
    monitor = ArangoMonitor(config['arango'])
    
    # Get initial status
    logger.info("Initial database status:")
    initial_metrics = monitor.get_overall_metrics()
    logger.info(f"  Papers in ArangoDB: {initial_metrics.total_papers}")
    
    # Process test papers
    logger.info(f"\nProcessing {len(test_papers)} test papers...")
    results = processor.process_papers(test_papers, force_reprocess=False)
    
    # Check results
    logger.info("\nProcessing results:")
    for paper_id, status in results.items():
        logger.info(f"  {paper_id}: {status}")
    
    # Get final status
    final_metrics = monitor.get_overall_metrics()
    logger.info(f"\nFinal status:")
    logger.info(f"  Papers in ArangoDB: {final_metrics.total_papers}")
    logger.info(f"  Papers processed: {final_metrics.processed_papers}")
    logger.info(f"  Total chunks: {final_metrics.total_chunks}")
    logger.info(f"  Total equations: {final_metrics.total_equations}")
    logger.info(f"  Total tables: {final_metrics.total_tables}")
    
    # Check health
    health = monitor.check_health()
    logger.info(f"\nSystem health: {health['status']}")
    if health['issues']:
        for issue in health['issues']:
            logger.info(f"  Issue: {issue}")
    
    return all(status in ['processed', 'already_processed'] for status in results.values())


def check_prerequisites():
    """Check if prerequisites are met"""
    issues = []
    
    # Check SQLite database exists
    if not Path('/bulk-store/arxiv-cache.db').exists():
        issues.append("SQLite cache database not found. Run: python setup_sqlite_cache.py")
    
    # Check PDF directory
    pdf_dir = Path('/bulk-store/arxiv-data/pdf')
    if not pdf_dir.exists():
        issues.append(f"PDF directory not found: {pdf_dir}")
    else:
        # Count PDFs
        pdf_count = sum(1 for _ in pdf_dir.glob('*/*.pdf'))
        logger.info(f"Found {pdf_count:,} PDF files")
    
    # Check environment variables
    if not os.environ.get('ARANGO_PASSWORD'):
        issues.append("ARANGO_PASSWORD environment variable not set")
    
    if issues:
        logger.error("Prerequisites not met:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    return True


def main():
    """Run integration tests"""
    logger.info("="*60)
    logger.info("ACID PIPELINE INTEGRATION TEST")
    logger.info("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("\nPlease resolve prerequisites before testing")
        return 1
    
    # Run tests
    try:
        success = test_real_paper_processing()
        
        if success:
            logger.info("\n✓ Integration test PASSED")
            return 0
        else:
            logger.error("\n✗ Integration test FAILED")
            return 1
            
    except Exception as e:
        logger.error(f"\n✗ Integration test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())