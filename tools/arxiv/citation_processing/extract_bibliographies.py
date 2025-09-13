#!/usr/bin/env python3
"""
Extract bibliographies from target papers to identify papers to download.

This script extracts the bibliography/references section from papers
we already have processed, identifies ArXiv IDs from citations, and
creates a list of papers to download and process.
"""

import os
import re
import json
from typing import List, Dict, Set
from arango import ArangoClient
from collections import defaultdict

def extract_arxiv_ids_from_text(text: str) -> Set[str]:
    """Extract ArXiv IDs from text using various patterns."""
    arxiv_ids = set()

    # Common ArXiv ID patterns
    patterns = [
        r'arxiv[:\s]+(\d{4}\.\d{4,5})',  # arxiv:1234.5678
        r'arxiv[:\s]+([a-z\-]+/\d{7})',  # arxiv:cs/0123456
        r'(\d{4}\.\d{4,5})',  # Just the number format
        r'([a-z\-]+/\d{7})',  # Old format
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        arxiv_ids.update(matches)

    # Clean up IDs
    cleaned = set()
    for id in arxiv_ids:
        # Skip obvious non-arxiv patterns
        if '2010' in id or '2011' in id or '2012' in id:
            if not id.startswith('10') and not id.startswith('11') and not id.startswith('12'):
                continue
        cleaned.add(id)

    return cleaned

def get_paper_chunks(db, paper_id: str) -> List[Dict]:
    """Get all chunks for a paper."""
    # Try both ID formats
    clean_id = paper_id.replace('.', '_')

    cursor = db.aql.execute('''
        FOR chunk IN arxiv_chunks
        FILTER chunk.paper_id == @paper_id OR chunk.paper_id == @clean_id
        SORT chunk.chunk_index
        RETURN chunk
    ''', bind_vars={'paper_id': paper_id, 'clean_id': clean_id})

    return list(cursor)

def extract_bibliography_section(chunks: List[Dict]) -> str:
    """Extract bibliography/references section from chunks."""
    bibliography_text = []
    in_references = False

    for chunk in chunks:
        text = chunk.get('text', '').lower()

        # Look for start of references section
        if any(marker in text for marker in ['references', 'bibliography', 'works cited']):
            # Check if this looks like a section header
            lines = chunk.get('text', '').split('\n')
            for i, line in enumerate(lines):
                if any(marker in line.lower() for marker in ['references', 'bibliography']):
                    # Found references section
                    in_references = True
                    # Add everything after the header
                    bibliography_text.extend(lines[i+1:])
                    break
        elif in_references:
            # Continue collecting bibliography
            bibliography_text.append(chunk.get('text', ''))

    return '\n'.join(bibliography_text)

def parse_citations(bibliography_text: str) -> List[Dict]:
    """Parse individual citations from bibliography text."""
    citations = []

    # Split by common citation patterns
    # Look for [1], 1., (1), etc.
    patterns = [
        r'\[\d+\]',  # [1], [2], etc.
        r'^\d+\.',   # 1., 2., etc. at line start
        r'\(\d+\)',  # (1), (2), etc.
    ]

    lines = bibliography_text.split('\n')
    current_citation = []

    for line in lines:
        # Check if this starts a new citation
        is_new = False
        for pattern in patterns:
            if re.match(pattern, line.strip()):
                is_new = True
                break

        if is_new and current_citation:
            # Save previous citation
            citation_text = ' '.join(current_citation)
            citations.append({
                'text': citation_text,
                'arxiv_ids': list(extract_arxiv_ids_from_text(citation_text))
            })
            current_citation = [line]
        else:
            current_citation.append(line)

    # Don't forget last citation
    if current_citation:
        citation_text = ' '.join(current_citation)
        citations.append({
            'text': citation_text,
            'arxiv_ids': list(extract_arxiv_ids_from_text(citation_text))
        })

    return citations

def main():
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('academy_store', username='root', password=os.environ.get('ARANGO_PASSWORD'))

    target_papers = [
        {'id': '1706.03762', 'title': 'Attention Is All You Need (Transformer)'},
        {'id': '1301.3781', 'title': 'Word2Vec - Efficient Estimation'},
        {'id': '1405.4053', 'title': 'Doc2Vec - Distributed Representations'},
        {'id': '1803.09473', 'title': 'Code2Vec - Learning Distributed Representations'}
    ]

    all_cited_arxiv_ids = set()
    paper_bibliographies = {}

    print("=== Extracting Bibliographies ===\n")

    for paper in target_papers:
        paper_id = paper['id']
        print(f"Processing: {paper_id} - {paper['title']}")

        # Get chunks for paper
        chunks = get_paper_chunks(db, paper_id)
        print(f"  Found {len(chunks)} chunks")

        # Extract bibliography section
        bibliography_text = extract_bibliography_section(chunks)

        if bibliography_text:
            print(f"  Extracted bibliography section ({len(bibliography_text)} chars)")

            # Parse citations
            citations = parse_citations(bibliography_text)
            print(f"  Found {len(citations)} citations")

            # Extract ArXiv IDs
            paper_arxiv_ids = set()
            for citation in citations:
                paper_arxiv_ids.update(citation['arxiv_ids'])

            print(f"  Found {len(paper_arxiv_ids)} ArXiv references")

            paper_bibliographies[paper_id] = {
                'title': paper['title'],
                'num_citations': len(citations),
                'arxiv_citations': list(paper_arxiv_ids),
                'citations': citations[:10]  # Sample of first 10
            }

            all_cited_arxiv_ids.update(paper_arxiv_ids)
        else:
            print(f"  WARNING: Could not find bibliography section")
            paper_bibliographies[paper_id] = {
                'title': paper['title'],
                'num_citations': 0,
                'arxiv_citations': [],
                'citations': []
            }

        print()

    # Save results
    output = {
        'target_papers': target_papers,
        'paper_bibliographies': paper_bibliographies,
        'all_cited_arxiv_ids': sorted(list(all_cited_arxiv_ids)),
        'total_unique_arxiv_citations': len(all_cited_arxiv_ids)
    }

    with open('bibliographies_extracted.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("=== Summary ===")
    print(f"Total unique ArXiv papers cited: {len(all_cited_arxiv_ids)}")
    print(f"Results saved to: bibliographies_extracted.json")

    if all_cited_arxiv_ids:
        print("\nSample of cited ArXiv IDs:")
        for arxiv_id in list(all_cited_arxiv_ids)[:10]:
            print(f"  - {arxiv_id}")

if __name__ == "__main__":
    main()
