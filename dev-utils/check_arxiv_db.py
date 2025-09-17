#!/usr/bin/env python3
"""
ArXiv Database Verification Tool
=================================

Checks the ArangoDB collections to verify ArXiv metadata processing results.
Connects via Unix socket for lowest latency and displays comprehensive statistics.

Usage:
    python check_arxiv_db.py
    python check_arxiv_db.py --detailed
    python check_arxiv_db.py --sample-size 5
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database.database_factory import DatabaseFactory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArxivDBChecker:
    """
    Verification tool for ArXiv database collections.

    Checks metadata, chunks, and embeddings collections for data integrity
    and provides comprehensive statistics about the processed data.
    """

    def __init__(self, database: str = "academy_store"):
        """
        Create an ArxivDBChecker for the given ArangoDB database.
        
        Parameters:
            database (str): Name of the ArangoDB database to inspect (default: "academy_store").
        """
        self.database = database
        self.db = None

        # Collection names (matching the workflow configuration)
        self.metadata_collection = "arxiv_metadata"
        self.chunks_collection = "arxiv_abstract_chunks"
        self.embeddings_collection = "arxiv_abstract_embeddings"

    def connect(self):
        """
        Connect to ArangoDB using a Unix socket and store the client on self.db.
        
        Attempts to obtain an ArangoDB connection via DatabaseFactory.get_arango (root user, unix socket).
        On success sets self.db to the database client and returns True; on failure leaves self.db unset and returns False.
        """
        try:
            self.db = DatabaseFactory.get_arango(
                database=self.database,
                username="root",
                use_unix=True  # Use Unix socket for lowest latency
            )
            print("✅ Connected to ArangoDB via Unix socket")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to ArangoDB: {e}")
            print("   Make sure ARANGO_PASSWORD is set and ArangoDB is running")
            return False

    def check_collections(self) -> Dict[str, Any]:
        """
        Return basic statistics for the three ArXiv collections (metadata, chunks, embeddings).
        
        Per collection this checks existence in the connected ArangoDB instance, retrieves a document
        count and collection properties (when present), prints a short human-readable overview,
        and returns a dictionary keyed by collection name. Each value contains:
        - 'exists' (bool)
        - 'count' (int)
        - 'properties' (dict) — present only when the collection exists.
        
        The printed size estimate (when available) is derived from collection properties and is
        approximate.
        """
        stats = {}

        collections = [
            self.metadata_collection,
            self.chunks_collection,
            self.embeddings_collection
        ]

        print("\n" + "="*60)
        print("COLLECTION OVERVIEW")
        print("="*60)

        for collection_name in collections:
            if self.db.has_collection(collection_name):
                collection = self.db.collection(collection_name)
                count = collection.count()

                # Get collection properties
                properties = collection.properties()

                stats[collection_name] = {
                    'exists': True,
                    'count': count,
                    'properties': properties
                }

                print(f"\n📁 {collection_name}:")
                print(f"   Documents: {count:,}")
                print(f"   Size: ~{properties.get('figures', {}).get('datafiles', {}).get('fileSize', 0) / (1024*1024):.2f} MB")
            else:
                stats[collection_name] = {
                    'exists': False,
                    'count': 0
                }
                print(f"\n❌ {collection_name}: NOT FOUND")

        return stats

    def check_metadata_sample(self, sample_size: int = 3) -> List[Dict]:
        """
        Retrieve up to `sample_size` sample documents from the metadata collection and print a concise summary for each.
        
        If the metadata collection is missing, returns an empty list. For each retrieved document this method prints key fields (arxiv_id, title, authors, categories, has_abstract, abstract length, processed_at) and returns the list of documents.
        
        Args:
            sample_size (int): Maximum number of metadata documents to fetch and display.
        
        Returns:
            List[Dict]: The sampled metadata documents (empty if the collection does not exist).
        """
        if not self.db.has_collection(self.metadata_collection):
            return []

        print(f"\n📊 Sample Metadata Documents ({sample_size}):")
        print("-" * 40)

        # Query for sample documents
        query = f"""
        FOR doc IN {self.metadata_collection}
        LIMIT {sample_size}
        RETURN doc
        """

        samples = []
        cursor = self.db.aql.execute(query)

        for i, doc in enumerate(cursor, 1):
            samples.append(doc)
            print(f"\n{i}. ArXiv ID: {doc.get('arxiv_id', 'N/A')}")
            print(f"   Title: {doc.get('title', 'N/A')[:100]}...")
            print(f"   Authors: {doc.get('authors', 'N/A')[:100]}...")
            print(f"   Categories: {doc.get('categories', 'N/A')}")
            print(f"   Has Abstract: {doc.get('has_abstract', False)}")
            print(f"   Abstract Length: {len(doc.get('abstract', ''))}")
            print(f"   Processed At: {doc.get('processed_at', 'N/A')}")

        return samples

    def check_chunks_sample(self, sample_size: int = 3) -> List[Dict]:
        """
        Retrieve and print a small sample of chunk documents from the chunks collection.
        
        Fetches up to `sample_size` documents from the configured chunks collection, prints a concise human-readable summary of each sample (id, arXiv id, chunk index/total, text length and preview, token range, context window), and returns the raw documents as a list of dicts. If the chunks collection does not exist, returns an empty list.
        
        Parameters:
            sample_size (int): Maximum number of chunk documents to retrieve.
        
        Returns:
            List[Dict]: The retrieved chunk documents (may be empty).
        """
        if not self.db.has_collection(self.chunks_collection):
            return []

        print(f"\n📄 Sample Chunk Documents ({sample_size}):")
        print("-" * 40)

        # Query for sample chunks
        query = f"""
        FOR doc IN {self.chunks_collection}
        LIMIT {sample_size}
        RETURN doc
        """

        samples = []
        cursor = self.db.aql.execute(query)

        for i, doc in enumerate(cursor, 1):
            samples.append(doc)
            print(f"\n{i}. Chunk ID: {doc.get('_key', 'N/A')}")
            print(f"   ArXiv ID: {doc.get('arxiv_id', 'N/A')}")
            print(f"   Chunk Index: {doc.get('chunk_index', 'N/A')} / {doc.get('total_chunks', 'N/A')}")
            print(f"   Text Length: {len(doc.get('text', ''))}")
            print(f"   Text Preview: {doc.get('text', '')[:150]}...")
            print(f"   Token Range: {doc.get('start_token', 'N/A')}-{doc.get('end_token', 'N/A')}")
            print(f"   Context Window: {doc.get('context_window_used', 'N/A')}")

        return samples

    def check_embeddings_sample(self, sample_size: int = 3) -> List[Dict]:
        """
        Return a small set of embedding documents from the embeddings collection and print a brief summary for each.
        
        If the embeddings collection is not present, an empty list is returned. For each returned document this prints key fields (document key, chunk_id, arxiv_id, embedding_dim, model). If an embedding vector is present, basic statistics (mean, standard deviation, min, max) are computed on the first 100 dimensions and printed.
        
        Parameters:
            sample_size (int): Maximum number of embedding documents to retrieve.
        
        Returns:
            List[Dict]: The list of retrieved embedding documents (possibly empty if the collection is missing).
        """
        if not self.db.has_collection(self.embeddings_collection):
            return []

        print(f"\n🔢 Sample Embedding Documents ({sample_size}):")
        print("-" * 40)

        # Query for sample embeddings
        query = f"""
        FOR doc IN {self.embeddings_collection}
        LIMIT {sample_size}
        RETURN doc
        """

        samples = []
        cursor = self.db.aql.execute(query)

        for i, doc in enumerate(cursor, 1):
            samples.append(doc)
            embedding = doc.get('embedding', [])

            print(f"\n{i}. Embedding ID: {doc.get('_key', 'N/A')}")
            print(f"   Chunk ID: {doc.get('chunk_id', 'N/A')}")
            print(f"   ArXiv ID: {doc.get('arxiv_id', 'N/A')}")
            print(f"   Embedding Dim: {doc.get('embedding_dim', len(embedding))}")
            print(f"   Model: {doc.get('model', 'N/A')}")

            if embedding:
                # Calculate basic statistics
                emb_sample = embedding[:100]  # First 100 dims for stats
                print(f"   Stats (first 100 dims):")
                print(f"     Mean: {statistics.mean(emb_sample):.4f}")
                print(f"     Std:  {statistics.stdev(emb_sample) if len(emb_sample) > 1 else 0:.4f}")
                print(f"     Min:  {min(emb_sample):.4f}")
                print(f"     Max:  {max(emb_sample):.4f}")

        return samples

    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Compute and return detailed verification statistics about the ArXiv collections.
        
        Returns a dictionary of computed metrics. Possible keys:
        - papers_with_chunks (int): Number of distinct papers that have at least one chunk.
        - avg_chunks_per_paper (float): Mean number of chunks per paper (when chunks exist).
        - min_chunks_per_paper (int): Minimum chunks found for a paper.
        - max_chunks_per_paper (int): Maximum chunks found for a paper.
        - embedding_dimensions (List[int]): Up to 10 distinct embedding dimensions observed in the embeddings collection.
          Note: the code treats 2048 as the expected embedding dimension (Jina v4).
        - processing_timeline (dict): Aggregate processing timestamps from metadata documents with these fields:
            - first (str): ISO8601 timestamp of the earliest processed_at value.
            - last (str): ISO8601 timestamp of the latest processed_at value.
            - total (int): Number of metadata documents with a non-null processed_at.
        
        Only keys for which data is available are included in the returned dictionary.
        """
        stats = {}

        print("\n" + "="*60)
        print("DETAILED STATISTICS")
        print("="*60)

        # Papers with chunks
        if self.db.has_collection(self.chunks_collection):
            query = f"""
            FOR chunk IN {self.chunks_collection}
            COLLECT arxiv_id = chunk.arxiv_id WITH COUNT INTO num_chunks
            RETURN {{
                arxiv_id: arxiv_id,
                num_chunks: num_chunks
            }}
            """

            cursor = self.db.aql.execute(query)
            papers_with_chunks = list(cursor)

            if papers_with_chunks:
                chunk_counts = [p['num_chunks'] for p in papers_with_chunks]
                stats['papers_with_chunks'] = len(papers_with_chunks)
                stats['avg_chunks_per_paper'] = statistics.mean(chunk_counts)
                stats['min_chunks_per_paper'] = min(chunk_counts)
                stats['max_chunks_per_paper'] = max(chunk_counts)

                print(f"\n📈 Chunking Statistics:")
                print(f"   Papers with chunks: {stats['papers_with_chunks']:,}")
                print(f"   Avg chunks/paper: {stats['avg_chunks_per_paper']:.2f}")
                print(f"   Min chunks/paper: {stats['min_chunks_per_paper']}")
                print(f"   Max chunks/paper: {stats['max_chunks_per_paper']}")

        # Embedding dimensions check
        if self.db.has_collection(self.embeddings_collection):
            query = f"""
            FOR emb IN {self.embeddings_collection}
            LIMIT 10
            RETURN DISTINCT emb.embedding_dim
            """

            cursor = self.db.aql.execute(query)
            dimensions = list(cursor)

            if dimensions:
                stats['embedding_dimensions'] = dimensions
                expected_dim = 2048  # Jina v4 expected dimension

                print(f"\n🎯 Embedding Validation:")
                print(f"   Dimensions found: {dimensions}")
                print(f"   Expected (Jina v4): {expected_dim}")

                if all(d == expected_dim for d in dimensions):
                    print(f"   ✅ All embeddings have correct dimensions")
                else:
                    print(f"   ⚠️ Dimension mismatch detected!")

        # Processing timeline
        if self.db.has_collection(self.metadata_collection):
            query = f"""
            FOR doc IN {self.metadata_collection}
            FILTER doc.processed_at != null
            COLLECT
                first_processed = MIN(doc.processed_at),
                last_processed = MAX(doc.processed_at),
                total = COUNT(doc)
            RETURN {{
                first: first_processed,
                last: last_processed,
                total: total
            }}
            """

            cursor = self.db.aql.execute(query)
            timeline = list(cursor)

            if timeline and timeline[0]:
                t = timeline[0]
                stats['processing_timeline'] = t

                print(f"\n⏱️ Processing Timeline:")
                print(f"   First processed: {t.get('first', 'N/A')}")
                print(f"   Last processed: {t.get('last', 'N/A')}")
                print(f"   Total processed: {t.get('total', 0):,}")

                # Calculate throughput if we have timestamps
                if t.get('first') and t.get('last'):
                    try:
                        first = datetime.fromisoformat(t['first'].replace('Z', '+00:00'))
                        last = datetime.fromisoformat(t['last'].replace('Z', '+00:00'))
                        duration = (last - first).total_seconds()
                        if duration > 0:
                            throughput = t.get('total', 0) / duration
                            print(f"   Throughput: {throughput:.2f} papers/second")
                    except:
                        pass

        return stats

    def run_verification(self, detailed: bool = False, sample_size: int = 3):
        """
        Run the full verification workflow for the ArXiv database and print a human-readable report.
        
        This method orchestrates the end-to-end checks:
        - establishes a connection to ArangoDB (calls self.connect()); aborts early if connection fails,
        - inspects the three target collections and gathers counts/properties (self.check_collections()),
        - optionally prints representative samples from metadata, chunks, and embeddings (controlled by
          `detailed` and `sample_size` via self.check_metadata_sample / self.check_chunks_sample /
          self.check_embeddings_sample),
        - computes and prints aggregated statistics and consistency checks (self.calculate_statistics()),
        - prints a final verification summary comparing metadata, chunks, and embeddings and reports
          per-paper chunk ratios when applicable.
        
        Parameters:
            detailed (bool): If True, show detailed output and samples. If False, samples are still shown
                when `sample_size` > 0.
            sample_size (int): Number of example documents to fetch and display from each collection
                when samples are enabled.
        
        Side effects:
            - Prints progress, samples, statistics, and summary to standard output.
            - May return early without raising if connection fails or no documents are present.
        """
        print("\n" + "="*60)
        print("ARXIV DATABASE VERIFICATION TOOL")
        print("="*60)
        print(f"Database: {self.database}")
        print(f"Time: {datetime.now().isoformat()}")

        # Connect to database
        if not self.connect():
            return

        # Check collections
        collection_stats = self.check_collections()

        # Check if we have any data
        total_docs = sum(s.get('count', 0) for s in collection_stats.values())
        if total_docs == 0:
            print("\n⚠️ No documents found in any collection!")
            print("   Run the metadata processor first to populate the database.")
            return

        # Show samples if requested
        if detailed or sample_size > 0:
            self.check_metadata_sample(sample_size)
            self.check_chunks_sample(sample_size)
            self.check_embeddings_sample(sample_size)

        # Calculate and show statistics
        stats = self.calculate_statistics()

        # Summary
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)

        # Check data consistency
        metadata_count = collection_stats.get(self.metadata_collection, {}).get('count', 0)
        chunks_count = collection_stats.get(self.chunks_collection, {}).get('count', 0)
        embeddings_count = collection_stats.get(self.embeddings_collection, {}).get('count', 0)

        print(f"\n📊 Data Consistency Check:")
        print(f"   Metadata docs: {metadata_count:,}")
        print(f"   Chunk docs: {chunks_count:,}")
        print(f"   Embedding docs: {embeddings_count:,}")

        if chunks_count == embeddings_count:
            print(f"   ✅ Chunks and embeddings match ({chunks_count:,})")
        else:
            print(f"   ⚠️ Mismatch: {chunks_count:,} chunks vs {embeddings_count:,} embeddings")

        if metadata_count > 0:
            if chunks_count > 0:
                ratio = chunks_count / metadata_count
                print(f"   📈 Chunks per paper: {ratio:.2f}")

            print(f"\n✨ Database contains {metadata_count:,} ArXiv papers")
            print(f"   with {chunks_count:,} text chunks")
            print(f"   and {embeddings_count:,} embeddings")

        print("\n" + "="*60)
        print("Verification complete!")


def main():
    """
    Command-line entry point that parses arguments and runs the ArXiv ArangoDB verification.
    
    Parses:
      --detailed (flag): show detailed statistics and samples.
      --sample-size (int): number of sample documents to display (default 3).
      --database (str): ArangoDB database name to inspect (default "academy_store").
    
    Requires the ARANGO_PASSWORD environment variable to be set; exits with status 1 if missing.
    Instantiates ArxivDBChecker with the selected database and invokes run_verification using the parsed options.
    """
    parser = argparse.ArgumentParser(
        description="Verify ArXiv data in ArangoDB"
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed statistics and samples'
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        default=3,
        help='Number of sample documents to show (default: 3)'
    )

    parser.add_argument(
        '--database',
        default='academy_store',
        help='ArangoDB database name (default: academy_store)'
    )

    args = parser.parse_args()

    # Check for ARANGO_PASSWORD
    if not os.environ.get('ARANGO_PASSWORD'):
        print("❌ ERROR: ARANGO_PASSWORD environment variable not set")
        print("Please set: export ARANGO_PASSWORD='your-password'")
        sys.exit(1)

    # Run verification
    checker = ArxivDBChecker(database=args.database)
    checker.run_verification(
        detailed=args.detailed,
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()