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

# Add utils directory to path for imports
utils_dir = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(utils_dir))

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
        """
        Initialize the LaTeXDetector.
        
        Parameters:
            latex_base_dir (str): Path to the root directory that contains LaTeX archives organized by year/month (default: "/bulk-store/arxiv-data/latex"). Converted to a pathlib.Path.
        
        Notes:
            - api_delay (float) is set to the number of seconds to wait between ArXiv API calls (defaults to 3.0).
            - last_api_call is initialized to 0 and used to enforce rate limiting.
        """
        self.latex_base_dir = Path(latex_base_dir)
        self.api_delay = 3.0  # Respect ArXiv API rate limits
        self.last_api_call = 0
        
    def check_local_latex_status(self, arxiv_id: str, year_month: str) -> LaTeXStatus:
        """
        Determine local LaTeX availability for a given arXiv paper.
        
        Checks the detector's latex_base_dir/year_month for two local indicators:
        - a LaTeX archive named "{arxiv_id}.tar.gz" — returns LaTeXStatus with has_latex=True, latex_path set, status_source="local_file", and file_size populated.
        - a signal file named "{arxiv_id}.pdf" — interpreted as "no LaTeX upstream"; returns has_latex=False and status_source="signal_file".
        
        If neither file is present, returns LaTeXStatus with has_latex=None and status_source="unknown".
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
        Check the arXiv API to determine whether source (LaTeX) tarball is available for the given paper.
        
        Performs a rate-limited HTTP query to the arXiv Atom API for the provided arXiv identifier, parses the returned entry (if any) and inspects its links for a MIME type of "application/x-eprint-tar". Behavior summary:
        - If a 200 response contains an entry and a link with type "application/x-eprint-tar" is present, returns has_latex=True and status_source="api_check".
        - If a 200 response contains an entry but no such link, returns has_latex=False and status_source="api_check".
        - If the paper is not found in the API response, returns has_latex=False, status_source="api_check", and an explanatory error_message.
        - If the HTTP response status is not 200 or an exception occurs (including network or XML parse errors), returns has_latex=None, status_source="api_unavailable", and an error_message describing the problem.
        
        Returns:
            LaTeXStatus: Result object containing arxiv_id, has_latex (True/False/None), status_source, and optional error_message.
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
        Detect LaTeX availability for a batch of papers by performing local checks first and optional ArXiv API queries for unresolved items.
        
        This performs a two-phase detection:
        1. Local check: inspects local LaTeX archives and signal files for each entry in pdf_list.
        2. API check: for items still unknown, queries the ArXiv API up to max_api_calls to determine LaTeX availability.
        
        Parameters:
            pdf_list (List[Dict]): Iterable of dicts describing papers. Each dict must contain:
                - 'arxiv_id' (str): the paper identifier
                - 'year_month' (str): the year/month folder name used for local lookup
            max_api_calls (int): Maximum number of ArXiv API requests to perform for unknown papers.
        
        Returns:
            Dict[str, LaTeXStatus]: Mapping from arXiv_id to the detected LaTeXStatus. Statuses come from either local checks or API checks (when performed).
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
        """
        Write LaTeX detection results to a JSON file and print a concise summary.
        
        Parameters:
            results (Dict[str, LaTeXStatus]): Mapping from arXiv identifier to LaTeXStatus objects describing availability,
                source, local path, file size, and any error message.
            output_file (str): Path to the JSON file to write.
        
        Description:
            Serializes the provided results to JSON (including a detection timestamp), computes summary statistics
            (counts of total, has_latex, no_latex, unknown, and source breakdowns), writes the JSON to `output_file`,
            and prints a human-readable summary to stdout.
        """
        
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
    """
    Main entry point for the LaTeX Status Detector script.
    
    Loads a JSON sample of PDFs, runs the two-phase LaTeX detection workflow, and writes a JSON report.
    
    Behavior:
    - Expects a sample file at /home/todd/olympus/HADES-Lab/tools/arxiv/logs/pdf_sample_2000.json containing a "sample_pdfs" list; logs an error and returns early if the file is missing.
    - Instantiates LaTeXDetector and runs detect_latex_status_batch on the loaded list (up to 500 arXiv API checks).
    - Persists results to /home/todd/olympus/HADES-Lab/tools/arxiv/logs/latex_status_results.json via LaTeXDetector.save_latex_status_results.
    - Prints completion messages to stdout.
    
    Side effects:
    - Reads from and writes to fixed filesystem paths described above.
    - May perform network requests when querying the arXiv API during detection.
    """
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