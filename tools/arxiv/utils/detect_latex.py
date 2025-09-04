#!/usr/bin/env python3
"""
LaTeX Status Detector
====================

Detects LaTeX availability for our PDF sample by:
1. Checking local LaTeX files in /bulk-store/arxiv-data/latex/
2. Checking for signal files (PDFs in LaTeX directory = no LaTeX upstream)
3. Querying ArXiv API for unknown cases

This is Phase 2.2 of the database rebuild process.
"""

import os
import json
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# No sys.path manipulation needed - use proper package imports

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LaTeXStatus:
    arxiv_id: str
    has_latex: Optional[bool]
    latex_path: Optional[str] = None
    status_source: str = "unknown"  # 'local_file', 'signal_file', 'api_check', 'api_unavailable'
    file_size: Optional[int] = None
    error_message: Optional[str] = None

class LaTeXDetector:
    def __init__(self, latex_base_dir: str = "/bulk-store/arxiv-data/latex"):
        self.latex_base_dir = Path(latex_base_dir)
        self.api_delay = 3.0  # Respect ArXiv API rate limits
        self.last_api_call = 0
        
    def check_local_latex_status(self, arxiv_id: str, year_month: str) -> LaTeXStatus:
        """
        Check local LaTeX status for a paper.
        
        Returns LaTeXStatus with:
        - has_latex=True if .tar.gz exists locally
        - has_latex=False if .pdf signal file exists (no LaTeX upstream)
        - has_latex=None if no local information available
        """
        latex_dir = self.latex_base_dir / year_month
        
        # Check for LaTeX archive
        latex_file = latex_dir / f"{arxiv_id}.tar.gz"
        if latex_file.exists():
            return LaTeXStatus(
                arxiv_id=arxiv_id,
                has_latex=True,
                latex_path=str(latex_file),
                status_source="local_file",
                file_size=latex_file.stat().st_size
            )
        
        # Check for signal file (PDF in LaTeX directory = no LaTeX available)
        signal_file = latex_dir / f"{arxiv_id}.pdf"
        if signal_file.exists():
            return LaTeXStatus(
                arxiv_id=arxiv_id,
                has_latex=False,
                latex_path=None,
                status_source="signal_file"
            )
        
        # No local information available
        return LaTeXStatus(
            arxiv_id=arxiv_id,
            has_latex=None,
            status_source="unknown"
        )
    
    def check_arxiv_api_latex(self, arxiv_id: str) -> LaTeXStatus:
        """
        Check ArXiv API to see if LaTeX source is available for a paper.
        """
        # Rate limiting
        now = time.time()
        if now - self.last_api_call < self.api_delay:
            time.sleep(self.api_delay - (now - self.last_api_call))
        
        try:
            # Query ArXiv API
            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            response = requests.get(url, timeout=30)
            self.last_api_call = time.time()
            
            if response.status_code != 200:
                return LaTeXStatus(
                    arxiv_id=arxiv_id,
                    has_latex=None,
                    status_source="api_unavailable",
                    error_message=f"API returned status {response.status_code}"
                )
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Check if entry exists
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            if not entries:
                return LaTeXStatus(
                    arxiv_id=arxiv_id,
                    has_latex=False,
                    status_source="api_check",
                    error_message="Paper not found in ArXiv API"
                )
            
            entry = entries[0]
            
            # Look for LaTeX source link
            links = entry.findall('{http://www.w3.org/2005/Atom}link')
            has_latex = False
            
            for link in links:
                if link.get('type') == 'application/x-eprint-tar':
                    has_latex = True
                    break
            
            return LaTeXStatus(
                arxiv_id=arxiv_id,
                has_latex=has_latex,
                status_source="api_check"
            )
            
        except Exception as e:
            logger.error(f"Error checking ArXiv API for {arxiv_id}: {e}")
            return LaTeXStatus(
                arxiv_id=arxiv_id,
                has_latex=None,
                status_source="api_unavailable",
                error_message=str(e)
            )
    
    def detect_latex_status_batch(self, pdf_list: List[Dict], max_api_calls: int = 200) -> Dict[str, LaTeXStatus]:
        """
        Detect LaTeX status for a batch of PDFs.
        
        Args:
            pdf_list: List of PDF info dicts with 'arxiv_id' and 'year_month'
            max_api_calls: Maximum number of API calls to make (for rate limiting)
        
        Returns:
            Dictionary mapping arxiv_id -> LaTeXStatus
        """
        results = {}
        api_calls_made = 0
        
        logger.info(f"Detecting LaTeX status for {len(pdf_list)} papers...")
        
        # Phase 1: Check local files first (fast)
        unknown_papers = []
        for pdf_info in pdf_list:
            arxiv_id = pdf_info['arxiv_id']
            year_month = pdf_info['year_month']
            
            status = self.check_local_latex_status(arxiv_id, year_month)
            results[arxiv_id] = status
            
            if status.status_source == "unknown":
                unknown_papers.append(pdf_info)
        
        logger.info(f"Local check complete: {len(pdf_list) - len(unknown_papers)} resolved, {len(unknown_papers)} unknown")
        
        # Phase 2: API checks for unknown papers (rate limited)
        if unknown_papers and api_calls_made < max_api_calls:
            logger.info(f"Making API calls for up to {min(len(unknown_papers), max_api_calls - api_calls_made)} papers...")
            
            for pdf_info in unknown_papers[:max_api_calls - api_calls_made]:
                arxiv_id = pdf_info['arxiv_id']
                api_status = self.check_arxiv_api_latex(arxiv_id)
                results[arxiv_id] = api_status
                api_calls_made += 1
                
                if api_calls_made % 10 == 0:
                    logger.info(f"API progress: {api_calls_made}/{min(len(unknown_papers), max_api_calls)} calls completed")
        
        logger.info(f"LaTeX detection complete: {api_calls_made} API calls made")
        return results
    
    def save_latex_status_results(self, results: Dict[str, LaTeXStatus], output_file: str):
        """Save LaTeX status results to JSON file."""
        
        # Convert LaTeXStatus objects to dict
        serializable_results = {}
        for arxiv_id, status in results.items():
            serializable_results[arxiv_id] = {
                'arxiv_id': status.arxiv_id,
                'has_latex': status.has_latex,
                'latex_path': status.latex_path,
                'status_source': status.status_source,
                'file_size': status.file_size,
                'error_message': status.error_message
            }
        
        # Calculate statistics
        stats = {
            'total_papers': len(results),
            'has_latex': len([s for s in results.values() if s.has_latex is True]),
            'no_latex': len([s for s in results.values() if s.has_latex is False]),
            'unknown': len([s for s in results.values() if s.has_latex is None]),
            'local_files': len([s for s in results.values() if s.status_source == 'local_file']),
            'signal_files': len([s for s in results.values() if s.status_source == 'signal_file']),
            'api_checked': len([s for s in results.values() if s.status_source == 'api_check']),
            'api_unavailable': len([s for s in results.values() if s.status_source == 'api_unavailable'])
        }
        
        output_data = {
            'detection_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': stats,
            'results': serializable_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"LaTeX status results saved to: {output_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("LATEX STATUS DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total papers analyzed: {stats['total_papers']:,}")
        print(f"Papers with LaTeX: {stats['has_latex']:,} ({stats['has_latex']/stats['total_papers']*100:.1f}%)")
        print(f"Papers without LaTeX: {stats['no_latex']:,} ({stats['no_latex']/stats['total_papers']*100:.1f}%)")
        print(f"Unknown status: {stats['unknown']:,} ({stats['unknown']/stats['total_papers']*100:.1f}%)")
        print(f"\nSource breakdown:")
        print(f"  Local LaTeX files: {stats['local_files']:,}")
        print(f"  Signal files (no LaTeX): {stats['signal_files']:,}")
        print(f"  API checked: {stats['api_checked']:,}")
        print(f"  API unavailable: {stats['api_unavailable']:,}")


def main():
    # Load the PDF sample
    sample_file = "/home/todd/olympus/HADES-Lab/tools/arxiv/logs/pdf_sample_2000.json"
    if not Path(sample_file).exists():
        logger.error(f"PDF sample file not found: {sample_file}")
        return
    
    with open(sample_file, 'r') as f:
        sample_data = json.load(f)
    
    pdf_list = sample_data['sample_pdfs']
    logger.info(f"Loaded {len(pdf_list)} PDFs for LaTeX detection")
    
    # Initialize detector
    detector = LaTeXDetector()
    
    # Detect LaTeX status
    results = detector.detect_latex_status_batch(pdf_list, max_api_calls=500)
    
    # Save results
    output_file = "/home/todd/olympus/HADES-Lab/tools/arxiv/logs/latex_status_results.json"
    detector.save_latex_status_results(results, output_file)
    
    print(f"\nLaTeX status detection complete!")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()