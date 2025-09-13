#!/usr/bin/env python3
"""
Embedder Factory

Factory pattern for creating embedder instances based on configuration.
Supports automatic model selection and fallback strategies.

Theory Connection:
The factory pattern enables flexible switching between embedding models
while maintaining consistent interfaces, supporting the WHO dimension
(different models as agents) without changing the WHAT processing logic.
"""

from typing import Optional, Dict, Any
import logging
from .embedders_base import EmbedderBase, EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbedderFactory:
    """
    Factory for creating embedder instances.

    Manages the instantiation of different embedder types based on
    configuration, with support for fallbacks and auto-detection.
    """

    # Registry of available embedders
    _embedders: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, embedder_class: type):
        """
        Register an embedder class under a given name in the factory registry.
        
        Adds the provided embedder class to the class-level registry mapping used by
        EmbedderFactory. If a different class is already registered under the same
        name, it will be replaced.
        """
        cls._embedders[name] = embedder_class
        logger.info(f"Registered embedder: {name}")

    @classmethod
    def create(cls,
              model_name: str = "jinaai/jina-embeddings-v4",
              config: Optional[EmbeddingConfig] = None,
              **kwargs) -> EmbedderBase:
        """
              Create and return an EmbedderBase instance for the given model.
              
              If `config` is not provided, an EmbeddingConfig is constructed from `model_name` and any extra keyword arguments. The factory determines an embedder type from `model_name`, attempts on-demand registration for that type, and instantiates the registered embedder class.
              
              Parameters:
                  model_name (str): Model identifier or path used to pick and configure the embedder.
                  config (Optional[EmbeddingConfig]): Pre-built embedding configuration. If omitted, one is created.
                  **kwargs: Additional fields forwarded to EmbeddingConfig when `config` is not supplied.
              
              Returns:
                  EmbedderBase: An instance of the selected embedder class initialized with `config`.
              
              Raises:
                  ValueError: If no embedder is registered (after auto-registration) for the determined embedder type.
              """
        # Create config if not provided
        if config is None:
            config = EmbeddingConfig(model_name=model_name, **kwargs)

        # Determine embedder type based on model name
        embedder_type = cls._determine_embedder_type(model_name)

        if embedder_type not in cls._embedders:
            # Try to import and register on-demand
            cls._auto_register(embedder_type)

        if embedder_type not in cls._embedders:
            available = list(cls._embedders.keys())
            raise ValueError(
                f"No embedder registered for type '{embedder_type}'. "
                f"Available: {available}"
            )

        embedder_class = cls._embedders[embedder_type]
        logger.info(f"Creating {embedder_type} embedder for model: {model_name}")

        return embedder_class(config)

    @classmethod
    def _determine_embedder_type(cls, model_name: str) -> str:
        """
        Determine the embedder type from a model name.
        
        Performs a case-insensitive substring check against the provided model_name and returns one of the supported embedder type keys: "jina", "sentence", "openai", or "cohere". If no known marker is found, defaults to "jina".
        
        Parameters:
            model_name: Model identifier or path used to infer the embedder type.
        
        Returns:
            A string key identifying the embedder type: "jina", "sentence", "openai", or "cohere".
        """
        model_lower = model_name.lower()

        if "jina" in model_lower:
            return "jina"
        elif "sentence-transformers" in model_lower or "st-" in model_lower:
            return "sentence"
        elif "openai" in model_lower or "text-embedding" in model_lower:
            return "openai"
        elif "cohere" in model_lower:
            return "cohere"
        else:
            # Default to Jina for unknown models (transformers-based)
            return "jina"

    @classmethod
    def _auto_register(cls, embedder_type: str):
        """
        Auto-registers a known embedder implementation for the given embedder type.
        
        If embedder_type == "jina", attempts to import JinaV4Embedder and register it under the name "jina".
        If embedder_type == "sentence", emits a warning that the sentence-transformers embedder is not yet migrated.
        For any other embedder_type, emits a warning about the unknown type.
        
        Side effects:
        - May call cls.register(...) to add an embedder to the factory registry.
        - Logs warnings for unimplemented or unknown types.
        - Logs an error if an ImportError occurs while attempting to import a backend (the exception is not propagated).
        """
        try:
            if embedder_type == "jina":
                from .embedders_jina import JinaV4Embedder
                cls.register("jina", JinaV4Embedder)
            elif embedder_type == "sentence":
                # Will be implemented when we migrate sentence_embedder.py
                logger.warning(f"Sentence-transformers embedder not yet migrated")
            else:
                logger.warning(f"Unknown embedder type: {embedder_type}")
        except ImportError as e:
            logger.error(f"Failed to import {embedder_type} embedder: {e}")

    @classmethod
    def list_available(cls) -> Dict[str, Any]:
        """
        Return a mapping of registered embedder names to their basic metadata.
        
        For each embedder registered in the factory registry, the mapping contains either:
        - a dict with keys `"class"` (the embedder class name) and `"module"` (the class' module path),
        or
        - a dict with key `"error"` containing a string message if retrieving the info failed.
        
        Returns:
            Dict[str, Any]: Mapping from embedder registry name to metadata or error information.
        """
        available = {}
        for name, embedder_class in cls._embedders.items():
            try:
                # Try to get class-level info without instantiation
                available[name] = {
                    "class": embedder_class.__name__,
                    "module": embedder_class.__module__
                }
            except Exception as e:
                logger.warning(f"Failed to get info for {name}: {e}")
                available[name] = {"error": str(e)}

        return available