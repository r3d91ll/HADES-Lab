#!/usr/bin/env python3
"""
PDF Scanner for ArXiv Rebuild
=============================

Scans /bulk-store/arxiv-data/pdf/ directory structure and maps all available PDFs
by ArXiv ID. This is Phase 2.1 of the database rebuild process.

Expected structure: /bulk-store/arxiv-data/pdf/YYMM/arxiv_id.pdf
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFScanner:
    def __init__(self, pdf_base_dir: str = "/bulk-store/arxiv-data/pdf"):
        self.pdf_base_dir = Path(pdf_base_dir)
        self.pdf_map = {}
        self.scan_stats = {
            'directories_scanned': 0,
            'pdfs_found': 0,
            'invalid_filenames': 0,
            'scan_start_time': None,
            'scan_end_time': None
        }

    def extract_arxiv_id_from_filename(self, filename: str) -> str:
        """
        Extract ArXiv ID from PDF filename.

        Expected formats:
        - 2508.21038.pdf -> 2508.21038
        - 1234.5678v1.pdf -> 1234.5678
        - cs-LG-0012345.pdf -> cs-LG/0012345 (old format)
        """
        # Remove .pdf extension
        base_name = filename.replace('.pdf', '')

        # Handle version numbers (remove v1, v2, etc.)
        base_name = re.sub(r'v\d+$', '', base_name)

        # Handle old format with categories (cs-LG-0012345 -> cs-LG/0012345)
        if re.match(r'^[a-zA-Z-]+-\d+', base_name):
            parts = base_name.split('-')
            if len(parts) >= 3:
                category = '-'.join(parts[:-1])
                paper_num = parts[-1]
                return f"{category}/{paper_num}"

        # Modern format (YYMM.NNNNN)
        if re.match(r'^\d{4}\.\d{4,5}$', base_name):
            return base_name

        # If we can't parse it, return as-is and log warning
        logger.warning(f"Could not parse ArXiv ID from filename: {filename}")
        return base_name

    def scan_pdf_directory(self) -> dict[str, dict]:
        """
        Scan the PDF directory structure and build a map of available PDFs.

        Returns:
            Dictionary mapping arxiv_id -> {path, size, year_month, filename}
        """
        logger.info(f"Starting PDF scan of {self.pdf_base_dir}")
        self.scan_stats['scan_start_time'] = datetime.now()

        if not self.pdf_base_dir.exists():
            logger.error(f"PDF directory does not exist: {self.pdf_base_dir}")
            return {}

        for year_month_dir in self.pdf_base_dir.iterdir():
            if not year_month_dir.is_dir():
                continue

            year_month = year_month_dir.name
            if not re.match(r'^\d{4}$', year_month):
                logger.warning(f"Skipping non-YYMM directory: {year_month}")
                continue

            self.scan_stats['directories_scanned'] += 1
            logger.info(f"Scanning directory: {year_month}")

            for pdf_file in year_month_dir.glob("*.pdf"):
                try:
                    arxiv_id = self.extract_arxiv_id_from_filename(pdf_file.name)
                    file_size = pdf_file.stat().st_size

                    if arxiv_id in self.pdf_map:
                        logger.warning(f"Duplicate ArXiv ID found: {arxiv_id}")
                        logger.warning(f"  Existing: {self.pdf_map[arxiv_id]['path']}")
                        logger.warning(f"  New: {pdf_file}")
                        continue

                    self.pdf_map[arxiv_id] = {
                        'path': str(pdf_file),
                        'size': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'year_month': year_month,
                        'filename': pdf_file.name
                    }

                    self.scan_stats['pdfs_found'] += 1

                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")
                    self.scan_stats['invalid_filenames'] += 1

        self.scan_stats['scan_end_time'] = datetime.now()
        logger.info(f"PDF scan completed. Found {self.scan_stats['pdfs_found']} PDFs in {self.scan_stats['directories_scanned']} directories")

        return self.pdf_map

    def save_pdf_map(self, output_file: str = "pdf_scan_results.json"):
        """Save the PDF map to a JSON file for later use."""
        output_path = Path(output_file)

        # Convert datetime objects to timezone-aware ISO format strings for JSON serialization
        serializable_stats = self.scan_stats.copy()
        if serializable_stats.get('scan_start_time'):
            serializable_stats['scan_start_time'] = serializable_stats['scan_start_time'].astimezone().isoformat(timespec='seconds')
        if serializable_stats.get('scan_end_time'):
            serializable_stats['scan_end_time'] = serializable_stats['scan_end_time'].astimezone().isoformat(timespec='seconds')

        scan_results = {
            'scan_timestamp': datetime.now().astimezone().isoformat(timespec='seconds'),
            'scan_stats': serializable_stats,
            'pdf_map': self.pdf_map
        }

        with open(output_path, 'w') as f:
            json.dump(scan_results, f, indent=2)

        logger.info(f"PDF scan results saved to: {output_path}")
        return output_path

    def get_year_month_distribution(self) -> dict[str, int]:
        """Get distribution of papers by year-month."""
        distribution = {}
        for arxiv_id, info in self.pdf_map.items():
            year_month = info['year_month']
            distribution[year_month] = distribution.get(year_month, 0) + 1

        return dict(sorted(distribution.items()))

    def print_scan_summary(self):
        """Print a summary of the scan results."""
        print(f"\n{'='*60}")
        print("PDF SCAN SUMMARY")
        print(f"{'='*60}")
        print(f"Total PDFs found: {self.scan_stats['pdfs_found']:,}")
        print(f"Directories scanned: {self.scan_stats['directories_scanned']}")
        print(f"Invalid filenames: {self.scan_stats['invalid_filenames']}")

        if self.scan_stats['scan_start_time'] and self.scan_stats['scan_end_time']:
            duration = self.scan_stats['scan_end_time'] - self.scan_stats['scan_start_time']
            print(f"Scan duration: {duration.total_seconds():.2f} seconds")

        # Show distribution by year-month
        distribution = self.get_year_month_distribution()
        print("\nDistribution by Year-Month:")
        for year_month in sorted(distribution.keys()):
            print(f"  {year_month}: {distribution[year_month]:,} papers")

        # Show size statistics
        sizes = [info['size'] for info in self.pdf_map.values()]
        if sizes:
            total_size_gb = sum(sizes) / (1024 * 1024 * 1024)
            avg_size_mb = sum(size / (1024 * 1024) for size in sizes) / len(sizes)
            print("\nSize Statistics:")
            print(f"  Total size: {total_size_gb:.2f} GB")
            print(f"  Average PDF size: {avg_size_mb:.2f} MB")


def main():
    scanner = PDFScanner()

    # Scan the PDF directory
    pdf_map = scanner.scan_pdf_directory()

    # Print summary
    scanner.print_scan_summary()

    # Save results
    output_file = "/home/todd/olympus/HADES-Lab/tools/arxiv/logs/pdf_scan_results.json"
    scanner.save_pdf_map(output_file)

    print(f"\nScan results saved to: {output_file}")
    print(f"Found {len(pdf_map)} unique PDFs ready for processing")


if __name__ == "__main__":
    main()
