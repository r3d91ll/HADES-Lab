#!/usr/bin/env python3
"""
Benchmark Unix Socket vs HTTP Performance for ArangoDB

Tests various operation types to measure the performance improvement
from using Unix sockets instead of TCP/IP network stack.
"""

import os
import time
import json
import random
import string
from typing import List, Dict, Any
import numpy as np
from arango import ArangoClient
from arango_unix_client import ArangoUnixClient


def generate_test_documents(count: int) -> List[Dict]:
    """Generate test documents with embeddings."""
    docs = []
    for i in range(count):
        docs.append({
            '_key': f'test_{i:06d}',
            'title': f'Test Document {i}',
            'abstract': ''.join(random.choices(string.ascii_letters + ' ', k=500)),
            'embedding': np.random.randn(2048).tolist(),  # Jina v4 dimension
            'categories': random.sample(['cs.AI', 'cs.LG', 'cs.CL', 'math.NA'], k=2)
        })
    return docs


class BenchmarkRunner:
    """Run comprehensive benchmarks comparing Unix socket vs HTTP."""
    
    def __init__(self, db_password: str):
        """Initialize benchmark runner."""
        self.password = db_password
        self.test_collection = 'benchmark_test'
        
        # Initialize both clients
        self.unix_client = ArangoUnixClient(
            database='academy_store',
            username='root',
            password=db_password
        )
        
        # Force HTTP client
        self.http_client = ArangoUnixClient(
            database='academy_store',
            username='root',
            password=db_password,
            socket_path='/nonexistent/force_http.sock'
        )
        
        print(f"Unix socket available: {self.unix_client.use_unix}")
        print(f"HTTP client forced: {not self.http_client.use_unix}")
        
        # Ensure test collection exists
        if not self.unix_client.has_collection(self.test_collection):
            self.unix_client.create_collection(self.test_collection)
    
    def benchmark_bulk_insert(self, doc_count: int = 1000) -> Dict[str, float]:
        """Benchmark bulk document insertion."""
        print(f"\n📊 Benchmarking bulk insert ({doc_count} documents)...")
        
        # Generate test data
        docs = generate_test_documents(doc_count)
        
        results = {}
        
        # Unix socket test
        start = time.time()
        stats = self.unix_client.insert_documents(
            collection=self.test_collection,
            documents=docs,
            batch_size=500
        )
        unix_time = time.time() - start
        results['unix_socket'] = unix_time
        print(f"  Unix socket: {unix_time:.3f}s ({doc_count/unix_time:.0f} docs/sec)")
        
        # Clear collection
        self.unix_client.execute_aql(f"FOR d IN {self.test_collection} REMOVE d IN {self.test_collection}")
        
        # HTTP test
        start = time.time()
        stats = self.http_client.insert_documents(
            collection=self.test_collection,
            documents=docs,
            batch_size=500
        )
        http_time = time.time() - start
        results['http'] = http_time
        print(f"  HTTP: {http_time:.3f}s ({doc_count/http_time:.0f} docs/sec)")
        
        improvement = (http_time/unix_time - 1) * 100 if self.unix_client.use_unix else 0
        print(f"  ✨ Improvement: {improvement:.1f}%")
        
        return results
    
    def benchmark_aql_queries(self, iterations: int = 100) -> Dict[str, float]:
        """Benchmark AQL query execution."""
        print(f"\n📊 Benchmarking AQL queries ({iterations} iterations)...")
        
        # Ensure we have data to query
        if self.unix_client.get_collection_count(self.test_collection) < 1000:
            docs = generate_test_documents(1000)
            self.unix_client.insert_documents(self.test_collection, docs)
        
        query = f"""
        FOR doc IN {self.test_collection}
        FILTER doc.title != null
        LIMIT 100
        RETURN {{
            _key: doc._key,
            title: doc.title,
            categories: doc.categories
        }}
        """
        
        results = {}
        
        # Unix socket test
        unix_times = []
        for _ in range(iterations):
            start = time.time()
            self.unix_client.execute_aql(query)
            unix_times.append(time.time() - start)
        
        avg_unix = np.mean(unix_times)
        results['unix_socket'] = avg_unix
        print(f"  Unix socket: {avg_unix*1000:.2f}ms avg")
        
        # HTTP test
        http_times = []
        for _ in range(iterations):
            start = time.time()
            self.http_client.execute_aql(query)
            http_times.append(time.time() - start)
        
        avg_http = np.mean(http_times)
        results['http'] = avg_http
        print(f"  HTTP: {avg_http*1000:.2f}ms avg")
        
        improvement = (avg_http/avg_unix - 1) * 100 if self.unix_client.use_unix else 0
        print(f"  ✨ Improvement: {improvement:.1f}%")
        
        return results
    
    def benchmark_vector_search(self, iterations: int = 50) -> Dict[str, float]:
        """Benchmark vector similarity search (simulated)."""
        print(f"\n📊 Benchmarking vector operations ({iterations} iterations)...")
        
        # Generate query vector
        query_vector = np.random.randn(2048).tolist()
        
        # This simulates what would happen in a real vector search
        query = f"""
        FOR doc IN {self.test_collection}
        FILTER doc.embedding != null
        LIMIT 50
        RETURN {{
            _key: doc._key,
            title: doc.title,
            embedding: doc.embedding
        }}
        """
        
        results = {}
        
        # Unix socket test
        unix_times = []
        for _ in range(iterations):
            start = time.time()
            docs = self.unix_client.execute_aql(query)
            # Simulate distance computation
            for doc in docs:
                if 'embedding' in doc:
                    dist = np.linalg.norm(np.array(doc['embedding']) - np.array(query_vector))
            unix_times.append(time.time() - start)
        
        avg_unix = np.mean(unix_times)
        results['unix_socket'] = avg_unix
        print(f"  Unix socket: {avg_unix*1000:.2f}ms avg")
        
        # HTTP test
        http_times = []
        for _ in range(iterations):
            start = time.time()
            docs = self.http_client.execute_aql(query)
            # Simulate distance computation
            for doc in docs:
                if 'embedding' in doc:
                    dist = np.linalg.norm(np.array(doc['embedding']) - np.array(query_vector))
            http_times.append(time.time() - start)
        
        avg_http = np.mean(http_times)
        results['http'] = avg_http
        print(f"  HTTP: {avg_http*1000:.2f}ms avg")
        
        improvement = (avg_http/avg_unix - 1) * 100 if self.unix_client.use_unix else 0
        print(f"  ✨ Improvement: {improvement:.1f}%")
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Dict]:
        """Run all benchmarks and return results."""
        print("="*70)
        print("ARANGODB UNIX SOCKET PERFORMANCE BENCHMARK")
        print("="*70)
        
        all_results = {}
        
        # Run benchmarks
        all_results['bulk_insert'] = self.benchmark_bulk_insert(doc_count=5000)
        all_results['aql_queries'] = self.benchmark_aql_queries(iterations=100)
        all_results['vector_search'] = self.benchmark_vector_search(iterations=50)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        total_unix = sum(r['unix_socket'] for r in all_results.values())
        total_http = sum(r['http'] for r in all_results.values())
        
        if self.unix_client.use_unix:
            overall_improvement = (total_http/total_unix - 1) * 100
            print(f"Overall performance improvement: {overall_improvement:.1f}%")
            print(f"Total Unix socket time: {total_unix:.2f}s")
            print(f"Total HTTP time: {total_http:.2f}s")
            print(f"Time saved: {total_http - total_unix:.2f}s")
        else:
            print("Unix socket not available - no improvement measured")
        
        # Cleanup
        self.unix_client.execute_aql(f"FOR d IN {self.test_collection} REMOVE d IN {self.test_collection}")
        
        return all_results


if __name__ == "__main__":
    # Get password from environment
    password = os.environ.get('ARANGO_PASSWORD')
    if not password:
        print("Please set ARANGO_PASSWORD environment variable")
        exit(1)
    
    # Run benchmarks
    runner = BenchmarkRunner(password)
    results = runner.run_all_benchmarks()
    
    # Save results
    with open('unix_socket_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to unix_socket_benchmark_results.json")