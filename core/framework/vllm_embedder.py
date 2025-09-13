#!/usr/bin/env python3
"""
vLLM-based Jina v4 Embedder for high-performance inference.

Theory Connection:
vLLM provides up to 24x higher throughput than transformers through PagedAttention
and optimized memory management. This dramatically improves the TIME dimension
in our Information = WHERE × WHAT × CONVEYANCE × TIME equation by reducing
processing latency while maintaining embedding quality.

The retrieval-only model reduces memory footprint by loading only the retrieval
LoRA adapter, not all 5 adapters, optimizing resource usage.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Union
from vllm import LLM
from vllm.config import PoolerConfig
from vllm.inputs.data import TextPrompt
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class JinaV4VLLMEmbedder:
    """
    High-performance Jina v4 embedder using vLLM inference engine.
    
    Provides up to 24x faster inference than transformers while maintaining
    identical embedding quality. Optimized for production environments
    with high throughput requirements.
    """
    
    def __init__(self,
                 model_name: str = "jinaai/jina-embeddings-v4",
                 device: str = "cuda",
                 use_fp16: bool = True,
                 tensor_parallel_size: int = 1,
                 max_model_len: int = 8192,
                 config_path: Optional[str] = None):
        """
        Initialize vLLM-based Jina embedder.
        
        Args:
            model_name: Model to use (retrieval-only recommended)
            device: Device to use (cuda/cpu)
            use_fp16: Use half precision for memory efficiency
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length to process
            config_path: Optional YAML config file
        """
        # Load config if provided
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                try:
                    import yaml
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f) or {}
                    
                    # Override parameters with config values
                    model_name = config.get('model_name', model_name)
                    use_fp16 = config.get('use_fp16', use_fp16)
                    tensor_parallel_size = config.get('tensor_parallel_size', tensor_parallel_size)
                    max_model_len = config.get('max_model_len', max_model_len)
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
        
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        
        logger.info(f"Initializing vLLM with {model_name}")
        logger.info(f"Using {'fp16' if use_fp16 else 'fp32'} precision")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"Max model length: {max_model_len}")
        
        # Initialize vLLM model
        dtype = "float16" if use_fp16 else "float32"
        
        # Create pooler config for embedding generation
        # Use MEAN pooling for general purpose embeddings
        pooler_config = PoolerConfig(
            pooling_type="MEAN",  # Mean pooling for embeddings
            normalize=True        # L2 normalization built-in
        )
        
        # Initialize LLM for embedding
        self.model = LLM(
            model=model_name,
            task="embed",  # vLLM embedding task
            override_pooler_config=pooler_config,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            gpu_memory_utilization=0.9  # Use most of GPU memory for performance
        )
        
        logger.info("vLLM model initialized successfully")
    
    def embed_texts(self, 
                    texts: List[str],
                    prefix: str = "Passage: ") -> np.ndarray:
        """
        Embed texts using vLLM for high-performance inference.
        
        Args:
            texts: List of texts to embed
            prefix: Prefix for texts (Passage: for documents, Query: for queries)
            
        Returns:
            Numpy array of embeddings (N x 2048)
        """
        if not texts:
            return np.empty((0, 2048))
        
        # Add prefix to texts
        prompts = [TextPrompt(prompt=f"{prefix}{text}") for text in texts]
        
        # Generate embeddings using vLLM
        outputs = self.model.encode(prompts)
        
        # Extract embeddings from outputs
        embeddings = []
        for output in outputs:
            # vLLM returns pooled outputs directly
            embedding = output.outputs.embedding
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Already L2 normalized by vLLM pooler config
        return embeddings
    
    def embed_abstracts(self, abstracts: List[str]) -> np.ndarray:
        """
        Embed paper abstracts optimized for retrieval.
        
        Args:
            abstracts: List of paper abstracts
            
        Returns:
            L2-normalized embeddings (N x 2048)
        """
        return self.embed_texts(abstracts, prefix="Passage: ")
    
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Embed search queries optimized for retrieval.
        
        Args:
            queries: List of search queries
            
        Returns:
            L2-normalized embeddings (N x 2048)
        """
        return self.embed_texts(queries, prefix="Query: ")
    
    def embed_code(self, code_snippets: List[str]) -> np.ndarray:
        """
        Embed code snippets using code-specific prefix.
        
        Args:
            code_snippets: List of code snippets
            
        Returns:
            L2-normalized embeddings (N x 2048)
        """
        # For code, we can use a different prefix if needed
        # The retrieval model is optimized for text, but can handle code
        return self.embed_texts(code_snippets, prefix="Code: ")
    
    def __del__(self):
        """Clean up vLLM resources."""
        if hasattr(self, 'model'):
            # vLLM handles cleanup internally
            pass