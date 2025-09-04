#!/usr/bin/env python3
"""
Quick PDF Count and Sample Extraction
====================================

Get a quick count and extract a sample of PDFs for processing.
"""

import os
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

def quick_pdf_sample(pdf_base_dir: str = "/bulk-store/arxiv-data/pdf", sample_size: int = 2000) -> dict[str, list[dict]]:
    """
    Collect a stratified random sample of PDFs found under pdf_base_dir and save the sample as JSON.
    
    Scans year-month subdirectories (names of four digits) under pdf_base_dir for files ending in .pdf, extracts metadata (arXiv id without version, file path, year_month, and size), and builds a stratified sample by year proportional to each year's share of the archive. If the stratified pass produces fewer than sample_size entries, the function fills the remainder by sampling from the remaining PDFs and finally truncates to sample_size.
    
    Parameters:
        pdf_base_dir (str): Root directory containing year-month subdirectories of PDFs.
        sample_size (int): Desired number of PDFs in the returned sample.
    
    Returns:
        List[dict]: The list of sampled PDF metadata dicts (keys: 'arxiv_id', 'path', 'year_month', 'size').
    
    Side effects:
        - Writes a JSON file with keys 'total_pdfs_available', 'sample_size', and 'sample_pdfs' to
          /home/todd/olympus/HADES-Lab/tools/arxiv/logs/pdf_sample_2000.json.
        - Prints progress and summary lines to stdout.
    
    Notes:
        - Sampling uses the random module; set the RNG seed externally for reproducible results.
        - File-system errors (e.g., permission or missing-directory errors) may propagate from the underlying file operations.
    """
    pdf_base_dir = Path(pdf_base_dir)
    all_pdfs = []
    
    print("Collecting PDF sample...")
    
    # Get all PDF files (just paths for now)
    for year_month_dir in pdf_base_dir.iterdir():
        if not year_month_dir.is_dir():
            continue
        
        year_month = year_month_dir.name
        if not year_month.isdigit() or len(year_month) != 4:
            continue
        
        for pdf_file in year_month_dir.glob("*.pdf"):
            arxiv_id = pdf_file.stem.split('v')[0]  # Remove version
            all_pdfs.append({
                'arxiv_id': arxiv_id,
                'path': str(pdf_file),
                'year_month': year_month,
                'size': pdf_file.stat().st_size
            })
    
    print(f"Total PDFs found: {len(all_pdfs):,}")
    
    # Get a stratified sample - some from each year to ensure diversity
    year_groups = {}
    for pdf in all_pdfs:
        year = pdf['year_month'][:2]  # First 2 digits for year (25 = 2025, 20 = 2020, etc.)
        if year not in year_groups:
            year_groups[year] = []
        year_groups[year].append(pdf)
    
    # Sample from each year proportionally
    sample_pdfs = []
    for year, pdfs in year_groups.items():
        year_sample_size = min(len(pdfs), max(1, int(sample_size * len(pdfs) / len(all_pdfs))))
        year_sample = random.sample(pdfs, year_sample_size)
        sample_pdfs.extend(year_sample)
        print(f"Year {year}: {len(pdfs):,} papers, sampled {len(year_sample)}")
    
    # If we need more, randomly sample from the rest
    if len(sample_pdfs) < sample_size:
        remaining_needed = sample_size - len(sample_pdfs)
        remaining_pdfs = [pdf for pdf in all_pdfs if pdf not in sample_pdfs]
        additional_sample = random.sample(remaining_pdfs, min(remaining_needed, len(remaining_pdfs)))
        sample_pdfs.extend(additional_sample)
    
    # Limit to requested sample size
    sample_pdfs = sample_pdfs[:sample_size]
    
    print(f"\nFinal sample size: {len(sample_pdfs):,} papers")
    
    # Save sample
    output_file = "/home/todd/olympus/HADES-Lab/tools/arxiv/logs/pdf_sample_2000.json"
    with open(output_file, 'w') as f:
        json.dump({
            'total_pdfs_available': len(all_pdfs),
            'sample_size': len(sample_pdfs),
            'sample_pdfs': sample_pdfs
        }, f, indent=2)
    
    print(f"Sample saved to: {output_file}")
    
    return sample_pdfs

if __name__ == "__main__":
    random.seed(42)  # For reproducible sampling
    sample = quick_pdf_sample()
    
    # Show some sample papers
    print("\nSample papers:")
    for i, pdf in enumerate(sample[:10]):
        print(f"  {i+1}. {pdf['arxiv_id']} ({pdf['year_month']}) - {pdf['size']/1024/1024:.1f} MB")