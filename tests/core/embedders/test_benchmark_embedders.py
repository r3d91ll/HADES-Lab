#!/usr/bin/env python3
"""
Benchmark comparison script for dual embedder implementations.

Compares performance between:
1. SentenceTransformersEmbedder (target: 48+ papers/sec)
2. JinaV4Embedder (current: 8-15 papers/sec)

Both implementations use MANDATORY late chunking.
"""

import time
import torch
import numpy as np
import logging
from typing import List, Dict, Any
import argparse
import json
from datetime import datetime
import psutil
import GPUtil

from core.embedders.embedders_sentence import SentenceTransformersEmbedder
from core.embedders.embedders_jina import JinaV4Embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbedderBenchmark:
    """Benchmark suite for comparing embedder implementations."""

    def __init__(self, device: str = "cuda", use_fp16: bool = True):
        """
        Initialize the EmbedderBenchmark.
        
        Sets the target device and FP16 flag and creates an empty results container.
        
        Parameters:
            device (str): Target compute device, e.g. "cuda" or "cpu" (default: "cuda").
            use_fp16 (bool): Whether to use FP16 precision when supported (default: True).
        """
        self.device = device
        self.use_fp16 = use_fp16
        self.results = {}

    def generate_test_data(self, num_papers: int = 100) -> Dict[str, List[str]]:
        """
        Generate synthetic text datasets used by the benchmarks.
        
        Produces three collections derived from num_papers:
        - "abstracts": num_papers medium-length abstract strings.
        - "full_papers": num_papers // 10 long texts intended to exercise late chunking.
        - "short_texts": num_papers * 2 short title/keyword strings.
        
        Parameters:
            num_papers (int): Base number used to size each collection (default 100).
        
        Returns:
            Dict[str, List[str]]: Dictionary with keys "abstracts", "full_papers", and "short_texts".
        """
        # Abstracts (medium length)
        abstracts = [
            f"Abstract {i}: " + "Information Reconstructionism demonstrates that " * 50 +
            "multiplicative dependencies between dimensions create emergent properties. " * 10
            for i in range(num_papers)
        ]

        # Full papers (long text requiring chunking)
        full_papers = [
            f"Paper {i}: Introduction. " + "The conveyance framework shows that " * 500 +
            "when any dimension equals zero, information ceases to exist. " * 200
            for i in range(num_papers // 10)  # Fewer full papers
        ]

        # Short texts (titles/keywords)
        short_texts = [
            f"Information Theory Paper {i}"
            for i in range(num_papers * 2)  # More short texts
        ]

        return {
            "abstracts": abstracts,
            "full_papers": full_papers,
            "short_texts": short_texts
        }

    def benchmark_throughput(self,
                            embedder,
                            texts: List[str],
                            batch_size: int,
                            name: str) -> Dict[str, Any]:
        """
                            Measure throughput and GPU memory usage for an embedder processing a list of texts.
                            
                            Runs an optional short warm-up, calls the embedder's batch embedding method for all provided texts using the given batch size, and records elapsed time, computed throughput (texts/sec), the embeddings tensor shape, and GPU memory metrics (when CUDA is available).
                            
                            Returns a dictionary with keys:
                            - name: benchmark name
                            - num_texts: number of input texts
                            - batch_size: batch size used for embedding
                            - elapsed_time: wall-clock time in seconds for the embedding run
                            - throughput: texts processed per second
                            - embeddings_shape: shape of the embeddings tensor returned by the embedder
                            - memory_before_gb: GPU memory allocated (GB) before the run (0 when CUDA is not available)
                            - memory_after_gb: GPU memory allocated (GB) after the run (0 when CUDA is not available)
                            - peak_memory_gb: peak GPU memory allocated (GB) during the run (0 when CUDA is not available)
                            """
        logger.info(f"\nBenchmarking {name}...")
        logger.info(f"  Texts: {len(texts)}")
        logger.info(f"  Batch size: {batch_size}")

        # Warm-up
        if len(texts) > 10:
            _ = embedder.embed_batch(texts[:10], batch_size=min(10, batch_size))
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / 1e9

        # Benchmark
        start_time = time.time()
        embeddings = embedder.embed_batch(texts, batch_size=batch_size)
        end_time = time.time()

        elapsed = end_time - start_time
        throughput = len(texts) / elapsed

        # Measure memory after
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1e9
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
        else:
            mem_after = mem_before = peak_mem = 0

        results = {
            "name": name,
            "num_texts": len(texts),
            "batch_size": batch_size,
            "elapsed_time": elapsed,
            "throughput": throughput,
            "embeddings_shape": embeddings.shape,
            "memory_before_gb": mem_before,
            "memory_after_gb": mem_after,
            "peak_memory_gb": peak_mem
        }

        logger.info(f"  Time: {elapsed:.2f}s")
        logger.info(f"  Throughput: {throughput:.2f} texts/sec")
        logger.info(f"  Peak memory: {peak_mem:.2f} GB")

        return results

    def benchmark_late_chunking(self,
                               embedder,
                               long_text: str,
                               name: str) -> Dict[str, Any]:
        """
                               Measure late-chunking performance of an embedder on a single long text.
                               
                               Runs the embedder's embed_with_late_chunking(long_text), times the call, and returns metrics about chunking throughput and basic metadata. The returned dictionary contains:
                               - name: benchmark name (str)
                               - text_length: number of characters in the input text (int)
                               - num_chunks: number of chunks produced (int)
                               - elapsed_time: wall-clock seconds taken to produce chunks (float)
                               - chunks_per_second: num_chunks / elapsed_time (float)
                               - first_chunk: (optional) metadata for the first chunk when any chunks were produced:
                                   - start_char (int), end_char (int), context_window (int or bool depending on embedder),
                                     embedding_shape (tuple)
                               
                               Parameters:
                                   long_text (str): Input long text that should trigger late chunking.
                                   name (str): Human-readable name for this benchmark run.
                               
                               Returns:
                                   Dict[str, Any]: A dictionary with the metrics and optional first-chunk metadata described above.
                               """
        logger.info(f"\nBenchmarking late chunking for {name}...")

        start_time = time.time()
        chunks = embedder.embed_with_late_chunking(long_text)
        end_time = time.time()

        elapsed = end_time - start_time

        results = {
            "name": name,
            "text_length": len(long_text),
            "num_chunks": len(chunks),
            "elapsed_time": elapsed,
            "chunks_per_second": len(chunks) / elapsed if elapsed > 0 else 0
        }

        if chunks:
            results["first_chunk"] = {
                "start_char": chunks[0].start_char,
                "end_char": chunks[0].end_char,
                "context_window": chunks[0].context_window_used,
                "embedding_shape": chunks[0].embedding.shape
            }

        logger.info(f"  Chunks created: {len(chunks)}")
        logger.info(f"  Time: {elapsed:.2f}s")
        logger.info(f"  Chunks/sec: {results['chunks_per_second']:.2f}")

        return results

    def compare_embedders(self, num_papers: int = 100) -> Dict[str, Any]:
        """
        Run the full benchmark workflow comparing two embedder implementations and return aggregated results.
        
        Generates synthetic test data, initializes a SentenceTransformers-based embedder and a Transformers-based embedder,
        runs throughput benchmarks on abstracts and short texts, runs late-chunking benchmarks on a long document (if available),
        computes a speedup metric (SentenceTransformers throughput ÷ Transformers throughput), and prints a summary.
        
        Parameters:
            num_papers (int): Number of synthetic papers to generate for the benchmarks (controls size of abstracts,
                full_papers, and short_texts sets). Defaults to 100.
        
        Returns:
            Dict[str, Any]: A results dictionary containing:
                - "timestamp": ISO timestamp of the run.
                - "device": device used ("cuda" or "cpu").
                - "use_fp16": whether FP16 was enabled.
                - "benchmarks": per-benchmark result dictionaries (abstracts_st, abstracts_tf, short_st, short_tf,
                  optionally chunking_st and chunking_tf).
                - "summary": overall summary with keys "st_throughput", "tf_throughput", "speedup", and "target_achieved".
        """
        logger.info("=" * 60)
        logger.info("DUAL EMBEDDER BENCHMARK COMPARISON")
        logger.info("=" * 60)

        # Generate test data
        test_data = self.generate_test_data(num_papers)

        # Initialize embedders
        logger.info("\nInitializing embedders...")

        # Sentence-transformers (high performance)
        st_embedder = SentenceTransformersEmbedder({
            'model_name': 'jinaai/jina-embeddings-v4',
            'device': self.device,
            'batch_size': 128,
            'use_fp16': self.use_fp16
        })

        # Transformers (sophisticated)
        tf_embedder = JinaV4Embedder({
            'model_name': 'jinaai/jina-embeddings-v4',
            'device': self.device,
            'use_fp16': self.use_fp16
        })

        results = {
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "use_fp16": self.use_fp16,
            "benchmarks": {}
        }

        # Benchmark abstracts (primary use case)
        logger.info("\n" + "=" * 40)
        logger.info("ABSTRACT PROCESSING (Primary Use Case)")
        logger.info("=" * 40)

        results["benchmarks"]["abstracts_st"] = self.benchmark_throughput(
            st_embedder,
            test_data["abstracts"],
            batch_size=128,
            name="SentenceTransformers - Abstracts"
        )

        results["benchmarks"]["abstracts_tf"] = self.benchmark_throughput(
            tf_embedder,
            test_data["abstracts"],
            batch_size=24,
            name="Transformers - Abstracts"
        )

        # Benchmark short texts
        logger.info("\n" + "=" * 40)
        logger.info("SHORT TEXT PROCESSING")
        logger.info("=" * 40)

        results["benchmarks"]["short_st"] = self.benchmark_throughput(
            st_embedder,
            test_data["short_texts"],
            batch_size=256,
            name="SentenceTransformers - Short"
        )

        results["benchmarks"]["short_tf"] = self.benchmark_throughput(
            tf_embedder,
            test_data["short_texts"],
            batch_size=32,
            name="Transformers - Short"
        )

        # Benchmark late chunking
        logger.info("\n" + "=" * 40)
        logger.info("LATE CHUNKING (Mandatory)")
        logger.info("=" * 40)

        if test_data["full_papers"]:
            long_text = test_data["full_papers"][0]

            results["benchmarks"]["chunking_st"] = self.benchmark_late_chunking(
                st_embedder,
                long_text,
                name="SentenceTransformers - Chunking"
            )

            results["benchmarks"]["chunking_tf"] = self.benchmark_late_chunking(
                tf_embedder,
                long_text,
                name="Transformers - Chunking"
            )

        # Calculate speedup
        st_throughput = results["benchmarks"]["abstracts_st"]["throughput"]
        tf_throughput = results["benchmarks"]["abstracts_tf"]["throughput"]
        speedup = st_throughput / tf_throughput if tf_throughput > 0 else 0

        results["summary"] = {
            "st_throughput": st_throughput,
            "tf_throughput": tf_throughput,
            "speedup": speedup,
            "target_achieved": st_throughput >= 48
        }

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results: Dict[str, Any]):
        """
        Log a human-readable benchmark summary of throughput, memory, and late-chunking results.
        
        Detailed: Reads the provided results dictionary and logs:
        - Abstract processing throughput for both embedders and the computed speedup.
        - Whether the SentenceTransformers throughput met the 48 papers/sec target.
        - Peak memory usage for both embedders (abstracts benchmarks).
        - Late-chunking counts for both embedders, if present.
        
        Parameters:
            results (Dict[str, Any]): Benchmark results produced by compare_embedders(). Expected keys used:
                - "summary": contains 'st_throughput', 'tf_throughput', 'speedup', 'target_achieved'.
                - "benchmarks": contains 'abstracts_st' and 'abstracts_tf' entries with 'peak_memory_gb',
                  and optional 'chunking_st' / 'chunking_tf' entries with 'num_chunks'.
        
        Returns:
            None
        """
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)

        summary = results["summary"]

        logger.info("\nAbstract Processing (Papers/Second):")
        logger.info(f"  SentenceTransformers: {summary['st_throughput']:.2f}")
        logger.info(f"  Transformers:         {summary['tf_throughput']:.2f}")
        logger.info(f"  Speedup:              {summary['speedup']:.2f}x")

        if summary["target_achieved"]:
            logger.info(f"\n✅ TARGET ACHIEVED! {summary['st_throughput']:.2f} >= 48 papers/sec")
        else:
            logger.info(f"\n⚠️ Below target: {summary['st_throughput']:.2f} < 48 papers/sec")

        # Memory comparison
        st_mem = results["benchmarks"]["abstracts_st"]["peak_memory_gb"]
        tf_mem = results["benchmarks"]["abstracts_tf"]["peak_memory_gb"]

        logger.info("\nMemory Usage (Peak):")
        logger.info(f"  SentenceTransformers: {st_mem:.2f} GB")
        logger.info(f"  Transformers:         {tf_mem:.2f} GB")

        # Late chunking comparison
        if "chunking_st" in results["benchmarks"]:
            st_chunks = results["benchmarks"]["chunking_st"]["num_chunks"]
            tf_chunks = results["benchmarks"]["chunking_tf"]["num_chunks"]

            logger.info("\nLate Chunking:")
            logger.info(f"  Both implementations use MANDATORY late chunking")
            logger.info(f"  SentenceTransformers: {st_chunks} chunks")
            logger.info(f"  Transformers:         {tf_chunks} chunks")

        logger.info("\n" + "=" * 60)

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """
        Save benchmark results to a JSON file.
        
        If `filename` is None a timestamped file named "benchmark_results_YYYYMMDD_HHMMSS.json" is created.
        Results are written with indent=2 and use `str()` for objects that are not JSON-serializable.
        
        Parameters:
            results (Dict[str, Any]): The benchmark results structure to serialize.
            filename (str, optional): Destination path. When omitted, a timestamped filename is used.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nResults saved to: {filename}")


def main():
    """
    Parse CLI arguments, run the embedder benchmarks, optionally save results, and return an exit code.
    
    This function is the command-line entry point for the benchmark script. It:
    - Parses arguments for number of papers, target device (cuda/cpu), FP16 toggle, and an optional save path.
    - Falls back to CPU if CUDA is requested but unavailable.
    - Logs basic system and CUDA/GPU information.
    - Instantiates EmbedderBenchmark with the chosen device and precision, runs compare_embedders, and optionally writes results to the provided file.
    - Returns 0 when the benchmark summary indicates the performance target was achieved, otherwise returns 1.
    
    Returns:
        int: Exit code (0 on success/target achieved, 1 otherwise).
    """
    parser = argparse.ArgumentParser(description="Benchmark dual embedder implementations")
    parser.add_argument("--papers", type=int, default=100,
                       help="Number of papers to benchmark (default: 100)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--no-fp16", action="store_true",
                       help="Disable FP16 precision")
    parser.add_argument("--save", type=str,
                       help="Save results to file")

    args = parser.parse_args()

    # Check GPU availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Print system info
    logger.info("System Information:")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Run benchmark
    benchmark = EmbedderBenchmark(
        device=args.device,
        use_fp16=not args.no_fp16
    )

    results = benchmark.compare_embedders(num_papers=args.papers)

    # Save results if requested
    if args.save:
        benchmark.save_results(results, args.save)

    # Return success if target achieved
    return 0 if results["summary"]["target_achieved"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())