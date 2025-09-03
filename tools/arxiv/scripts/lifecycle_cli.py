#!/usr/bin/env python3
"""
ArXiv Lifecycle Manager CLI
===========================

Command-line interface for the ArXiv Lifecycle Manager. Provides easy access
to all lifecycle operations including single paper processing, batch operations,
status checking, and system reporting.

Usage Examples:
    # Process a single paper
    python lifecycle_cli.py process 2508.21038
    
    # Check status of a paper
    python lifecycle_cli.py status 2508.21038
    
    # Process multiple papers
    python lifecycle_cli.py batch papers.txt
    
    # Force reprocessing
    python lifecycle_cli.py process 2508.21038 --force
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List

# Add utils directory to path for imports
utils_dir = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(utils_dir))

from arxiv_lifecycle_manager import ArXivLifecycleManager, PaperStatus, LifecycleResult
from arxiv_api_client import ArXivAPIClient

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging for CLI usage"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Reduce verbosity of some libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def validate_arxiv_id(arxiv_id: str) -> bool:
    """Validate ArXiv ID format"""
    client = ArXivAPIClient()
    return client.validate_arxiv_id(arxiv_id)


def process_single_paper(args):
    """Process a single paper through the lifecycle"""
    arxiv_id = args.arxiv_id
    
    if not validate_arxiv_id(arxiv_id):
        print(f"❌ Invalid ArXiv ID format: {arxiv_id}")
        sys.exit(1)
    
    print(f"🔄 Processing paper: {arxiv_id}")
    
    manager = ArXivLifecycleManager()
    result = manager.process_paper(arxiv_id, force=args.force)
    
    # Display results
    print("\n" + "="*60)
    print("PROCESSING RESULTS")
    print("="*60)
    
    status_emoji = {
        PaperStatus.ERROR: "❌",
        PaperStatus.NOT_FOUND: "❓",
        PaperStatus.METADATA_ONLY: "📋",
        PaperStatus.DOWNLOADED: "📥",
        PaperStatus.PROCESSED: "⚙️",
        PaperStatus.HIRAG_INTEGRATED: "🎯"
    }
    
    emoji = status_emoji.get(result.status, "⚪")
    print(f"Status: {emoji} {result.status.value.upper()}")
    
    if result.error_message:
        print(f"Error: {result.error_message}")
    
    print(f"Processing time: {result.processing_time_seconds:.2f}s")
    
    # Show what was accomplished
    accomplishments = []
    if result.metadata_fetched:
        accomplishments.append("✅ Metadata fetched")
    if result.pdf_downloaded:
        accomplishments.append("✅ PDF downloaded")
    if result.latex_downloaded:
        accomplishments.append("✅ LaTeX downloaded")
    if result.processed:
        accomplishments.append("✅ ACID pipeline processed")
    if result.hirag_updated:
        accomplishments.append("✅ HiRAG system updated")
    if result.database_updated:
        accomplishments.append("✅ Database updated")
    
    if accomplishments:
        print("\nAccomplishments:")
        for item in accomplishments:
            print(f"  {item}")
    
    # Show file locations
    if result.pdf_path:
        print(f"\nPDF: {result.pdf_path}")
    if result.latex_path:
        print(f"LaTeX: {result.latex_path}")
    
    # Success/failure exit codes
    if result.status == PaperStatus.ERROR:
        sys.exit(1)
    else:
        print(f"\n🎉 Paper {arxiv_id} successfully processed!")


def check_paper_status(args):
    """Check the status of a paper in the system"""
    arxiv_id = args.arxiv_id
    
    if not validate_arxiv_id(arxiv_id):
        print(f"❌ Invalid ArXiv ID format: {arxiv_id}")
        sys.exit(1)
    
    print(f"🔍 Checking status for: {arxiv_id}")
    
    manager = ArXivLifecycleManager()
    status, details = manager.check_paper_status(arxiv_id)
    
    # Display status
    print("\n" + "="*60)
    print("PAPER STATUS")
    print("="*60)
    
    status_emoji = {
        PaperStatus.ERROR: "❌",
        PaperStatus.NOT_FOUND: "❓",
        PaperStatus.METADATA_ONLY: "📋",
        PaperStatus.DOWNLOADED: "📥", 
        PaperStatus.PROCESSED: "⚙️",
        PaperStatus.HIRAG_INTEGRATED: "🎯"
    }
    
    status_descriptions = {
        PaperStatus.NOT_FOUND: "Paper not found in system",
        PaperStatus.METADATA_ONLY: "Metadata available, files not downloaded",
        PaperStatus.DOWNLOADED: "Files downloaded, not processed",
        PaperStatus.PROCESSED: "Fully processed through ACID pipeline",
        PaperStatus.HIRAG_INTEGRATED: "Integrated into HiRAG system",
        PaperStatus.ERROR: "Error occurred during processing"
    }
    
    emoji = status_emoji.get(status, "⚪")
    description = status_descriptions.get(status, "Unknown status")
    
    print(f"Status: {emoji} {status.value.upper()}")
    print(f"Description: {description}")
    
    # Show details
    print("\nDetails:")
    print(f"  📊 In PostgreSQL: {'✅' if details.get('in_postgresql') else '❌'}")
    print(f"  📄 PDF exists: {'✅' if details.get('pdf_exists') else '❌'}")
    print(f"  🔧 LaTeX exists: {'✅' if details.get('latex_exists') else '❌'}")
    print(f"  ⚙️ Processed: {'✅' if details.get('processed') else '❌'}")
    
    if details.get('pdf_path'):
        print(f"  📁 PDF path: {details['pdf_path']}")
    if details.get('latex_path'):
        print(f"  📁 LaTeX path: {details['latex_path']}")
    
    if details.get('error'):
        print(f"  ❌ Error: {details['error']}")
    
    if args.json:
        print(f"\nJSON Output:")
        print(json.dumps({
            'arxiv_id': arxiv_id,
            'status': status.value,
            'details': details
        }, indent=2))


def process_batch_papers(args):
    """Process multiple papers from a file"""
    batch_file = Path(args.batch_file)
    
    if not batch_file.exists():
        print(f"❌ Batch file not found: {batch_file}")
        sys.exit(1)
    
    # Read ArXiv IDs from file
    arxiv_ids = []
    try:
        with open(batch_file, 'r') as f:
            for line in f:
                arxiv_id = line.strip()
                if arxiv_id and not arxiv_id.startswith('#'):
                    if validate_arxiv_id(arxiv_id):
                        arxiv_ids.append(arxiv_id)
                    else:
                        print(f"⚠️  Invalid ArXiv ID skipped: {arxiv_id}")
    except Exception as e:
        print(f"❌ Error reading batch file: {e}")
        sys.exit(1)
    
    if not arxiv_ids:
        print("❌ No valid ArXiv IDs found in batch file")
        sys.exit(1)
    
    print(f"🔄 Processing {len(arxiv_ids)} papers from {batch_file}")
    
    manager = ArXivLifecycleManager()
    results = manager.batch_process_papers(arxiv_ids, force=args.force)
    
    # Generate and display report
    report = manager.generate_report(results)
    print(report)
    
    # Save detailed results if requested
    if args.output:
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            json.dump({
                arxiv_id: result.to_dict() 
                for arxiv_id, result in results.items()
            }, f, indent=2)
        print(f"📁 Detailed results saved to: {output_file}")
    
    # Exit with error code if any papers failed
    failed_count = len([r for r in results.values() if r.status == PaperStatus.ERROR])
    if failed_count > 0:
        print(f"\n⚠️  {failed_count} papers failed processing")
        sys.exit(1)


def fetch_metadata_only(args):
    """Fetch metadata for a paper without full processing"""
    arxiv_id = args.arxiv_id
    
    if not validate_arxiv_id(arxiv_id):
        print(f"❌ Invalid ArXiv ID format: {arxiv_id}")
        sys.exit(1)
    
    print(f"📋 Fetching metadata for: {arxiv_id}")
    
    client = ArXivAPIClient()
    metadata = client.get_paper_metadata(arxiv_id)
    
    if not metadata:
        print(f"❌ Failed to fetch metadata for {arxiv_id}")
        sys.exit(1)
    
    # Display metadata
    print("\n" + "="*60)
    print("PAPER METADATA")
    print("="*60)
    
    print(f"Title: {metadata.title}")
    print(f"Authors: {', '.join(metadata.authors)}")
    print(f"Primary Category: {metadata.primary_category}")
    print(f"Categories: {', '.join(metadata.categories)}")
    print(f"Published: {metadata.published.strftime('%Y-%m-%d')}")
    print(f"Updated: {metadata.updated.strftime('%Y-%m-%d')}")
    
    if metadata.doi:
        print(f"DOI: {metadata.doi}")
    if metadata.journal_ref:
        print(f"Journal: {metadata.journal_ref}")
    
    print(f"Has LaTeX: {'✅' if metadata.has_latex else '❌'}")
    
    print(f"\nAbstract:")
    print(metadata.abstract)
    
    if args.json:
        print(f"\nJSON Output:")
        # Convert to dict for JSON serialization
        metadata_dict = {
            'arxiv_id': metadata.arxiv_id,
            'title': metadata.title,
            'abstract': metadata.abstract,
            'authors': metadata.authors,
            'categories': metadata.categories,
            'primary_category': metadata.primary_category,
            'published': metadata.published.isoformat(),
            'updated': metadata.updated.isoformat(),
            'doi': metadata.doi,
            'journal_ref': metadata.journal_ref,
            'has_latex': metadata.has_latex,
            'pdf_url': metadata.pdf_url
        }
        print(json.dumps(metadata_dict, indent=2))


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ArXiv Lifecycle Manager CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process 2508.21038
  %(prog)s status 2508.21038 --json
  %(prog)s batch papers.txt --output results.json
  %(prog)s metadata 2508.21038
        """
    )
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a paper through full lifecycle")
    process_parser.add_argument("arxiv_id", help="ArXiv paper ID")
    process_parser.add_argument("--force", action="store_true",
                               help="Force reprocessing even if already complete")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check paper status")
    status_parser.add_argument("arxiv_id", help="ArXiv paper ID") 
    status_parser.add_argument("--json", action="store_true",
                              help="Output status as JSON")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process multiple papers")
    batch_parser.add_argument("batch_file", help="File containing ArXiv IDs (one per line)")
    batch_parser.add_argument("--force", action="store_true",
                             help="Force reprocessing even if already complete")
    batch_parser.add_argument("--output", "-o", help="Save detailed results to JSON file")
    
    # Metadata command
    metadata_parser = subparsers.add_parser("metadata", help="Fetch metadata only")
    metadata_parser.add_argument("arxiv_id", help="ArXiv paper ID")
    metadata_parser.add_argument("--json", action="store_true",
                                help="Output metadata as JSON")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check required environment variables
    if args.command in ['process', 'batch']:
        if not os.getenv('ARANGO_PASSWORD'):
            print("❌ ARANGO_PASSWORD environment variable is required for processing")
            print("   Set it with: export ARANGO_PASSWORD='your-password'")
            sys.exit(1)
        
        if not os.getenv('PGPASSWORD'):
            print("❌ PGPASSWORD environment variable is required for processing")
            print("   Set it with: export PGPASSWORD='your-password'")
            sys.exit(1)
    
    # Route to appropriate handler
    try:
        if args.command == "process":
            process_single_paper(args)
        elif args.command == "status":
            check_paper_status(args)
        elif args.command == "batch":
            process_batch_papers(args)
        elif args.command == "metadata":
            fetch_metadata_only(args)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()