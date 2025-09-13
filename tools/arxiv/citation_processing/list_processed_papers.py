#!/usr/bin/env python3
"""
List Processed ArXiv Papers
============================

Query ArangoDB for all ArXiv IDs that have been fully processed
(have complete document data including chunks and embeddings).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from arango import ArangoClient
from arango.database import Database


def connect_to_arango(password: str = None) -> Database:
    """Connect to ArangoDB."""
    if not password:
        password = os.getenv('ARANGO_PASSWORD', '')

    if not password:
        raise ValueError("ArangoDB password required (set ARANGO_PASSWORD env var)")

    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('academy_store', username='root', password=password)

    return db


def get_processed_papers(db: Database, min_chunks: int = 1) -> List[Dict[str, Any]]:
    """
    Get all fully processed ArXiv papers from the database.

    Args:
        db: ArangoDB database connection
        min_chunks: Minimum number of chunks required to consider paper as processed

    Returns:
        List of paper records with their ArXiv IDs and metadata
    """

    # Query for papers that have associated chunks and embeddings
    query = """
    FOR paper IN arxiv_papers
        // Check if paper has chunks
        LET chunk_count = LENGTH(
            FOR chunk IN arxiv_chunks
                FILTER chunk.document_id == paper.arxiv_id
                RETURN 1
        )

        // Check if paper has embeddings
        LET embedding_count = LENGTH(
            FOR embedding IN arxiv_embeddings
                FILTER embedding.document_id == paper.arxiv_id
                RETURN 1
        )

        // Only return papers with chunks and embeddings
        FILTER chunk_count >= @min_chunks
        FILTER embedding_count >= @min_chunks

        RETURN {
            arxiv_id: paper.arxiv_id,
            num_chunks: chunk_count,
            num_embeddings: embedding_count,
            processed_at: paper.processed_at,
            pdf_path: paper.pdf_path,
            has_latex: paper.has_latex_source
        }
    """

    cursor = db.aql.execute(
        query,
        bind_vars={'min_chunks': min_chunks}
    )

    papers = list(cursor)
    return papers


def get_arxiv_ids_only(db: Database) -> List[str]:
    """
    Get just the ArXiv IDs of fully processed papers.

    Args:
        db: ArangoDB database connection

    Returns:
        List of ArXiv ID strings
    """

    query = """
    FOR paper IN arxiv_papers
        // Check if paper has at least one chunk
        LET has_chunks = LENGTH(
            FOR chunk IN arxiv_chunks
                FILTER chunk.document_id == paper.arxiv_id
                LIMIT 1
                RETURN 1
        ) > 0

        // Check if paper has at least one embedding
        LET has_embeddings = LENGTH(
            FOR embedding IN arxiv_embeddings
                FILTER embedding.document_id == paper.arxiv_id
                LIMIT 1
                RETURN 1
        ) > 0

        // Only return papers with both chunks and embeddings
        FILTER has_chunks == true
        FILTER has_embeddings == true

        RETURN paper.arxiv_id
    """

    cursor = db.aql.execute(query)
    return list(cursor)


def main():
    """Main function to list processed papers."""
    parser = argparse.ArgumentParser(
        description="List all fully processed ArXiv papers in ArangoDB"
    )

    parser.add_argument(
        '--password',
        type=str,
        help='ArangoDB password (defaults to ARANGO_PASSWORD env var)'
    )

    parser.add_argument(
        '--format',
        choices=['ids', 'detailed', 'json'],
        default='ids',
        help='Output format (default: ids)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file (default: stdout)'
    )

    parser.add_argument(
        '--min-chunks',
        type=int,
        default=1,
        help='Minimum number of chunks required (default: 1)'
    )

    args = parser.parse_args()

    try:
        # Connect to database
        print("Connecting to ArangoDB...", file=sys.stderr)
        db = connect_to_arango(args.password)

        # Get processed papers
        if args.format == 'ids':
            print("Fetching ArXiv IDs...", file=sys.stderr)
            results = get_arxiv_ids_only(db)
            output = '\n'.join(results)
        else:
            print(f"Fetching detailed paper information (min chunks: {args.min_chunks})...", file=sys.stderr)
            papers = get_processed_papers(db, args.min_chunks)

            if args.format == 'json':
                output = json.dumps(papers, indent=2)
            else:  # detailed
                lines = []
                lines.append(f"Found {len(papers)} fully processed papers")
                lines.append("=" * 60)
                for paper in papers:
                    lines.append(f"ArXiv ID: {paper['arxiv_id']}")
                    lines.append(f"  Chunks: {paper['num_chunks']}")
                    lines.append(f"  Embeddings: {paper['num_embeddings']}")
                    lines.append(f"  Has LaTeX: {paper.get('has_latex', False)}")
                    lines.append(f"  Processed: {paper.get('processed_at', 'Unknown')}")
                    lines.append("")
                output = '\n'.join(lines)

        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Results written to {args.output}", file=sys.stderr)
        else:
            print(output)

        # Summary to stderr
        if args.format == 'ids':
            print(f"\nTotal processed papers: {len(results)}", file=sys.stderr)
        else:
            print(f"\nTotal processed papers: {len(papers)}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
