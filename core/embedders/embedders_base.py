#!/usr/bin/env python3
"""
Base Embedder Interface

Defines the contract for all embedding implementations in HADES.
Following the Conveyance Framework: embedders transform WHAT (content)
into vector representations while preserving semantic relationships.

Theory Connection:
Embedders are the bridge between human-readable text and machine-processable
vectors. They preserve the WHAT dimension of information while enabling
efficient WHERE (similarity) computations in vector space.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedders."""
    model_name: str
    device: str = "cuda"
    batch_size: int = 32
    max_seq_length: int = 8192
    use_fp16: bool = True
    chunk_size_tokens: Optional[int] = None
    chunk_overlap_tokens: Optional[int] = None


class EmbedderBase(ABC):
    """
    Abstract base class for all embedders.

    Defines the interface that all embedding implementations must follow
    to ensure consistency across different models and approaches.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedder with an EmbeddingConfig.
        
        If no config is provided, a default EmbeddingConfig with model_name="default" is created and assigned.
        Parameters:
            config (Optional[EmbeddingConfig]): Configuration for the embedder; when None a default config is used.
        """
        self.config = config or EmbeddingConfig(model_name="default")

    @abstractmethod
    def embed_texts(self,
                    texts: List[str],
                    task: str = "retrieval",
                    batch_size: Optional[int] = None) -> np.ndarray:
        """
                    Embed a list of texts into dense vectors.
                    
                    Returns a 2-D NumPy array of shape (len(texts), embedding_dimension) where each row is the embedding for the corresponding input text.
                    
                    Parameters:
                        texts: List of input texts to embed.
                        task: Optional task hint that may affect embedding behavior (default "retrieval").
                        batch_size: Optional override for the embedding batch size; if omitted the embedder's configured batch size is used.
                    
                    Returns:
                        np.ndarray: 2-D array with one embedding vector per input text.
                    """
        pass

    @abstractmethod
    def embed_single(self,
                     text: str,
                     task: str = "retrieval") -> np.ndarray:
        """
                     Embed a single piece of text and return its embedding vector.
                     
                     Parameters:
                         text (str): The input text to embed.
                         task (str): Optional task hint affecting embedding behavior (default "retrieval").
                     
                     Returns:
                         numpy.ndarray: 1-D embedding vector whose length equals the embedder's embedding_dimension.
                     """
        pass

    def embed_queries(self,
                     queries: List[str],
                     batch_size: Optional[int] = None) -> np.ndarray:
        """
                     Embed a list of search queries into embedding vectors (convenience wrapper).
                     
                     This is a thin convenience method that produces embeddings for the provided queries using the model's retrieval embedding pathway.
                     
                     Parameters:
                         queries: List of query strings to embed.
                         batch_size: Optional override for the embedding batch size.
                     
                     Returns:
                         A 2D numpy array of shape (len(queries), embedding_dimension) containing the embeddings.
                     """
        return self.embed_texts(queries, task="retrieval", batch_size=batch_size)

    def embed_documents(self,
                       documents: List[str],
                       batch_size: Optional[int] = None) -> np.ndarray:
        """
                       Convenience wrapper to embed a list of documents for retrieval.
                       
                       Parameters:
                           documents (List[str]): Documents to embed.
                           batch_size (Optional[int]): Optional override for the batch size; if None uses the embedder's configured batch size.
                       
                       Returns:
                           np.ndarray: 2D array of shape (len(documents), embedding_dimension) containing the document embeddings.
                       """
        return self.embed_texts(documents, task="retrieval", batch_size=batch_size)

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Return the dimensionality of vectors produced by this embedder.
        
        This is an abstract property that concrete embedder implementations must provide;
        it indicates the length of each embedding vector (number of dimensions).
        
        Returns:
            int: Number of dimensions in each embedding vector.
        """
        pass

    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """
        Maximum input sequence length supported by the embedder.
        
        Returns:
            int: The model's maximum number of input tokens (or sequence units) accepted for a single inference. Implementations should return the hard limit used for chunking and validation.
        """
        pass

    @property
    def supports_late_chunking(self) -> bool:
        """Whether this embedder supports late chunking."""
        return False

    @property
    def supports_multimodal(self) -> bool:
        """
        Whether the embedder accepts multimodal (non-text) inputs such as images or audio.
        
        Implementations that support embedding inputs beyond plain text should override this to return True. Defaults to False.
        """
        return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return a dictionary of metadata describing the embedder and its configured model.
        
        Returns:
            dict: Metadata keys:
                - model_name (str): Name of the configured model.
                - embedding_dimension (int): Dimensionality of produced embedding vectors.
                - max_sequence_length (int): Maximum supported input sequence length (tokens).
                - supports_late_chunking (bool): Whether the embedder supports late chunking.
                - supports_multimodal (bool): Whether the embedder accepts multimodal inputs.
                - device (str): Device configured for the model (e.g., "cuda", "cpu").
                - use_fp16 (bool): Whether mixed/half precision is enabled for the model.
        """
        return {
            "model_name": self.config.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": self.max_sequence_length,
            "supports_late_chunking": self.supports_late_chunking,
            "supports_multimodal": self.supports_multimodal,
            "device": self.config.device,
            "use_fp16": self.config.use_fp16
        }