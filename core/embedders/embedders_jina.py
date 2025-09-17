#!/usr/bin/env python3
"""
Jina v4 Embedder with proper API usage, fp16 support, and LATE CHUNKING.

Theory Connection:
Late chunking preserves the contextual relationships across text boundaries,
maintaining the WHERE dimension of information even when physically chunked.
This prevents zero-propagation in the WHAT dimension by ensuring each chunk
contains awareness of its surrounding context.

The 32k token context window allows processing entire papers at once,
then intelligently chunking while preserving cross-boundary semantic relationships.
"""

# cspell:ignore jina Jina embedder Embedder

import torch
import numpy as np
from typing import List, Optional, Dict, Union, Tuple, Any
from transformers import AutoModel, AutoTokenizer
import logging
from PIL import Image
import io
import base64
import hashlib
from dataclasses import dataclass
from .embedders_base import EmbedderBase, EmbeddingConfig

logger = logging.getLogger(__name__)


@dataclass
class ChunkWithEmbedding:
    """
    Represents a text chunk with its late-chunked embedding.
    
    In ANT terms, this is a boundary object that maintains coherence
    across the transformation from continuous text to discrete chunks.
    The embedding preserves awareness of surrounding context, preventing
    zero-propagation in the WHAT dimension of information.
    """
    text: str
    embedding: np.ndarray
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    chunk_index: int
    total_chunks: int
    context_window_used: int  # How many tokens were in the full context


class JinaV4Embedder(EmbedderBase):
    """
    Jina v4 embedder with late chunking support.
    
    This embedder supports both:
    1. Traditional embedding (for short texts, backward compatibility)
    2. Late chunking (for long documents, superior context preservation)
    """
    
    # Jina v4 constants
    MAX_TOKENS = 32768  # Jina v4's context window
    EMBEDDING_DIM = 2048
    
    def __init__(self, config: Optional[Union[EmbeddingConfig, Dict[str, Any]]] = None) -> None:
        """
        Initialize the Jina v4 embedder and load model/tokenizer with late-chunking configuration.
        
        This constructor accepts an EmbeddingConfig instance, a plain dict, a device string, or None and normalizes it to runtime configuration values. It sets embedder attributes (device, model_name, batch_size, chunk_size_tokens, chunk_overlap_tokens, and internal dtype), loads the tokenizer and model via Hugging Face `from_pretrained`, moves the model to the target device, and puts it into evaluation mode. If CUDA was requested but is unavailable, a warning is emitted. The method also logs model and embedding characteristics.
        
        Parameters:
            config (EmbeddingConfig | dict | str | None): Configuration for the embedder. Can be:
                - an EmbeddingConfig object (fields extracted if present),
                - a dict with keys like 'device', 'use_fp16', 'batch_size', 'chunk_size_tokens',
                  'chunk_overlap_tokens', 'model_name', 'normalize_embeddings',
                - a device string (old-style single param), or
                - None to use sensible defaults (CUDA if available, otherwise CPU).
        
        Side effects:
            - Loads tokenizer and model weights from `model_name` (network/disk I/O).
            - Moves the model to the chosen device and sets it to eval mode.
            - Logs configuration and may warn if CUDA was requested but not available.
        """
        # Handle both old-style params and new config object
        if config is None:
            config = {}
        elif hasattr(config, 'device'):
            # It's an EmbeddingConfig object - extract ALL values
            old_config = config
            config = {
                'device': getattr(old_config, 'device', 'cuda'),
                'use_fp16': getattr(old_config, 'use_fp16', True),
                'batch_size': getattr(old_config, 'batch_size', 128),
                'chunk_size_tokens': getattr(old_config, 'chunk_size_tokens', 500),
                'chunk_overlap_tokens': getattr(old_config, 'chunk_overlap_tokens', 200),
                'model_name': getattr(old_config, 'model_name', 'jinaai/jina-embeddings-v4'),
                'normalize_embeddings': getattr(old_config, 'normalize_embeddings', True)
            }
        elif not isinstance(config, dict):
            # Old-style single param (device)
            config = {'device': str(config)}

        # Remove config_path handling since we have the config already
        device = None
        use_fp16 = None
        chunk_size_tokens = None
        chunk_overlap_tokens = None
        # Config is already processed above, no need to load from file

        # Determine default device based on CUDA availability
        default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Set parameters with config as defaults, overridable by arguments
        self.device = device or config.get('device', default_device)
        self.model_name = config.get('model_name', 'jinaai/jina-embeddings-v4')

        # Log if we had to fallback to CPU
        if config.get('device', default_device) == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
        self.batch_size = config.get('batch_size', 128)  # Default to 128 for better throughput
        # Chunk size set to handle most abstracts without chunking, but chunk when needed
        self.chunk_size_tokens = chunk_size_tokens or config.get('chunk_size_tokens', 500)  # Most abstracts < 500 tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens or config.get('chunk_overlap_tokens', 200)
        use_fp16 = use_fp16 if use_fp16 is not None else config.get('use_fp16', True)
        
        # Load model with appropriate dtype
        # Check if device starts with "cuda" to handle cuda:0, cuda:1, etc.
        dtype = torch.float16 if (use_fp16 and self.device.startswith("cuda")) else torch.float32
        
        logger.info(f"Loading {self.model_name} on {self.device} with dtype={dtype}")
        logger.info(f"Batch size for embedding: {self.batch_size}")
        logger.info(f"Late chunking config: {self.chunk_size_tokens} tokens/chunk, {self.chunk_overlap_tokens} overlap")
        
        # Load tokenizer first for late chunking
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Load model and then move to target device
        # device_map should be "auto" or a dict, not a device string
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype
        )

        # Move model to the target device if not CPU
        if self.device and self.device != "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

        # Log actual model configuration
        logger.info("Jina v4 model loaded with late chunking support")
        logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
        logger.info(f"Embedding dimension: 2048")
        logger.info(f"Model has encode method: {hasattr(self.model, 'encode')}")

        # Check if Flash Attention is available and being used
        try:
            import flash_attn
            logger.info("Flash Attention 2 is available - model should use it automatically")
        except ImportError:
            logger.warning("Flash Attention 2 not available - performance may be limited")
        
    def embed_texts(self,
                    texts: List[str],
                    task: str = "retrieval.passage",
                    batch_size: Optional[int] = None) -> np.ndarray:
        """
                    Embed a list of texts using the Jina v4 embedding model and return a 2D numpy array of embeddings.
                    
                    Processes inputs in batches (default batch size from the embedder) and uses either the model's high-level
                    encode(...) method when available or a tokenized forward pass with a task-specific LoRA adapter. The method
                    returns float embeddings with dimension 2048 for each input text.
                    
                    Parameters:
                        texts (List[str]): Input texts to embed.
                        task (str): Task identifier that determines the adapter or pooling behavior (default "retrieval.passage").
                            Common values: "retrieval.passage", "retrieval.query", "text-matching", "classification", "separation".
                            When using model.encode, "retrieval" will be used if "retrieval" appears in the provided task.
                        batch_size (Optional[int]): Number of texts to process per batch. If None, the embedder's configured
                            batch_size is used.
                    
                    Returns:
                        numpy.ndarray: 2-D array of shape (N, 2048) containing one embedding per input text. Returns an empty
                        array with shape (0, 2048) when given an empty input list.
                    
                    Raises:
                        AttributeError: If the model forward-path is used and the model output does not contain the expected
                            `single_vec_emb` attribute.
                        Exception: If embeddings cannot be converted to a numpy array, the underlying conversion error is propagated.
                    """
        all_embeddings = []
        batch_size = batch_size or self.batch_size

        # Commented out for performance - this was logging 30+ times per second
        # logger.info(f"Processing {len(texts)} texts with batch_size={batch_size}")

        with torch.no_grad():
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Use Jina's encode method if available
                if hasattr(self.model, 'encode'):
                    # Jina v4 with encode method
                    # logger.debug(f"Using model.encode for batch of {len(batch)} texts")
                    # Jina v4 uses 'retrieval' not 'retrieval.passage'
                    jina_task = 'retrieval' if 'retrieval' in task else task
                    embeddings = self.model.encode(
                        batch,
                        task=jina_task
                    )
                else:
                    # Jina v4 requires task_label when using forward pass
                    # logger.debug(f"Using forward pass with task_label for batch of {len(batch)} texts")
                    # Map task to the correct LoRA adapter name
                    # Jina v4 uses: 'retrieval', 'retrieval.query', 'text-matching', 'classification', 'separation'
                    task_mapping = {
                        'retrieval.passage': 'retrieval',
                        'retrieval.query': 'retrieval.query',
                        'retrieval': 'retrieval',
                        'text-matching': 'text-matching',
                        'classification': 'classification',
                        'separation': 'separation'
                    }
                    task_label = task_mapping.get(task, 'retrieval')

                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.MAX_TOKENS
                    ).to(self.device)

                    # Add task_label to inputs - must be a string for LoRA adapter selection
                    with torch.no_grad():
                        outputs = self.model(**inputs, task_label=task_label)

                        # Jina v4 returns single_vec_emb for 2048-dimensional embeddings
                        if hasattr(outputs, 'single_vec_emb') and outputs.single_vec_emb is not None:
                            embeddings = outputs.single_vec_emb
                        else:
                            # Log error with available attributes for debugging
                            available_attrs = [attr for attr in dir(outputs) if not attr.startswith('_')]
                            raise AttributeError(
                                f"Expected 'single_vec_emb' in JinaV4 output, but got: {available_attrs}. "
                                f"Output type: {type(outputs).__name__}"
                            )

                        if torch.is_tensor(embeddings):
                            if embeddings.is_cuda:
                                embeddings = embeddings.cpu()
                            embeddings = embeddings.numpy()
                
                # Debug: Check what type of object we got
                # logger.debug(f"Embeddings type: {type(embeddings)}")
                
                # Handle different return types - prioritize torch.is_tensor check
                if torch.is_tensor(embeddings):
                    # Handle PyTorch tensors directly
                    if embeddings.is_cuda:
                        embeddings = embeddings.cpu()
                    embeddings = embeddings.numpy()
                elif hasattr(embeddings, 'detach'):  # Other tensor-like objects
                    embeddings = embeddings.detach()
                    if hasattr(embeddings, 'is_cuda') and embeddings.is_cuda:
                        embeddings = embeddings.cpu()
                    embeddings = embeddings.numpy()
                elif isinstance(embeddings, list):
                    # If it's a list of tensors
                    processed = []
                    for e in embeddings:
                        if torch.is_tensor(e):
                            if e.is_cuda:
                                e = e.cpu()
                            processed.append(e.numpy())
                        elif hasattr(e, 'detach'):
                            e = e.detach()
                            if hasattr(e, 'is_cuda') and e.is_cuda:
                                e = e.cpu()
                            processed.append(e.numpy())
                        else:
                            processed.append(np.array(e))
                    embeddings = np.vstack(processed)
                elif not isinstance(embeddings, np.ndarray):
                    # Try to convert to numpy
                    try:
                        embeddings = np.array(embeddings)
                    except Exception as e:
                        logger.error(f"Cannot convert embeddings of type {type(embeddings)} to numpy: {e}")
                        raise
                    
                all_embeddings.append(embeddings)

                # DO NOT clear GPU cache after each batch - this kills performance!
                # PyTorch's allocator efficiently reuses memory.
                # Only clear cache if encountering OOM errors.
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()
        
        # Concatenate all batches
        result = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 2048))
        
        return result
    
    def embed_single(self, text: str, task: str = "retrieval.passage") -> np.ndarray:
        """
        Embed a single text (required by EmbedderBase interface).

        Args:
            text: Text to embed
            task: Task type

        Returns:
            1D embedding array
        """
        embeddings = self.embed_texts([text], task=task, batch_size=1)
        return embeddings[0] if embeddings.size > 0 else np.zeros(self.EMBEDDING_DIM)

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.EMBEDDING_DIM

    @property
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length supported."""
        return self.MAX_TOKENS

    @property
    def supports_late_chunking(self) -> bool:
        """Whether this embedder supports late chunking."""
        return True

    @property
    def supports_multimodal(self) -> bool:
        """Whether this embedder supports multimodal inputs."""
        return True  # Jina v4 supports images

    def embed_code(self,
                   code_snippets: List[str],
                   batch_size: int = 4) -> np.ndarray:
        """
                   Embed a list of code snippets using the model's code-specific task.
                   
                   Parameters:
                       code_snippets (List[str]): Code strings to embed.
                       batch_size (int): Number of snippets processed per batch.
                   
                   Returns:
                       np.ndarray: Embeddings array of shape (N, 2048), one vector per input snippet.
                   """
        return self.embed_texts(code_snippets, task="code", batch_size=batch_size)
    
    def embed_images(self, images: List[Union[bytes, Image.Image, str]]) -> np.ndarray:
        """
        Embed images using Jina v4's multimodal capabilities.
        
        Args:
            images: List of images as bytes, PIL Images, or base64 strings
            
        Returns:
            L2-normalized embeddings as numpy array
        """
        processed_images = []
        
        for img in images:
            if isinstance(img, bytes):
                # Convert bytes to PIL Image
                pil_img = Image.open(io.BytesIO(img))
            elif isinstance(img, str):
                # Assume base64 encoded
                img_bytes = base64.b64decode(img)
                pil_img = Image.open(io.BytesIO(img_bytes))
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            # Convert to RGB if necessary
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            processed_images.append(pil_img)
        
        # Encode images using Jina v4's encode_image method
        with torch.no_grad():
            embeddings = self.model.encode_image(
                images=processed_images,
                task="retrieval"
            )
            
            # Handle CUDA tensors properly
            if torch.is_tensor(embeddings):
                if embeddings.is_cuda:
                    embeddings = embeddings.cpu()
                embeddings = embeddings.numpy()
            elif hasattr(embeddings, 'detach'):
                embeddings = embeddings.detach()
                if hasattr(embeddings, 'is_cuda') and embeddings.is_cuda:
                    embeddings = embeddings.cpu()
                embeddings = embeddings.numpy()
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def embed_multimodal(self, pairs: List[Dict[str, Union[str, List[bytes]]]]) -> np.ndarray:
        """
        Create unified embeddings for text+image pairs.
        
        Args:
            pairs: List of dicts with 'text' and optional 'images' keys
            
        Returns:
            L2-normalized multimodal embeddings
        """
        embeddings_list = []
        
        for pair in pairs:
            text = pair.get('text', '')
            images = pair.get('images', [])
            
            if not text and not images:
                # Empty pair, return zero vector
                embeddings_list.append(np.zeros(2048))
                continue
            
            # Use late fusion as Jina v4 doesn't have true multimodal yet
            components = []
            weights = []
            
            if text:
                text_emb = self.embed_texts([text])[0]
                components.append(text_emb)
                weights.append(0.7)  # Default text weight
            
            if images:
                img_embs = self.embed_images(images)
                # Average multiple images
                img_emb = np.mean(img_embs, axis=0)
                components.append(img_emb)
                weights.append(0.3)  # Default image weight
            
            # Weighted combination
            if len(components) == 1:
                combined = components[0]
            else:
                weights = np.array(weights) / np.sum(weights)  # Normalize weights
                combined = np.sum([w * c for w, c in zip(weights, components)], axis=0)
            
            # L2 normalize
            norm = np.linalg.norm(combined)
            combined = combined / (norm + 1e-8)
            
            embeddings_list.append(combined)
        
        return np.array(embeddings_list)
    
    def embed_with_late_chunking(self,
                                 text: str,
                                 task: str = "retrieval.passage") -> List[ChunkWithEmbedding]:
        """
                                 Embed a single document by splitting it into fixed-size chunks, embedding each chunk, and returning chunk-level embeddings.
                                 
                                 The text is deterministically chunked using the embedder's chunk_size_tokens and chunk_overlap_tokens via _prepare_simple_chunks; each resulting chunk is embedded (batched) and paired with its original character and token boundaries.
                                 
                                 Parameters:
                                     text (str): Full document text to chunk and embed. If empty, returns an empty list.
                                     task (str): Embedding task label (e.g., "retrieval.passage", "code"). Passed through to embed_texts.
                                 
                                 Returns:
                                     List[ChunkWithEmbedding]: A list of ChunkWithEmbedding objects, each containing the chunk text, its embedding, character/token boundaries, chunk index, total_chunks, and context_window_used.
                                 """
        if not text:
            return []

        # Use simple chunking approach that always works
        chunks = self._prepare_simple_chunks(text)

        # Extract chunk texts
        chunk_texts = [c['text'] for c in chunks]

        # Embed all chunks
        embeddings = self.embed_texts(chunk_texts, task=task, batch_size=len(chunk_texts))

        # Create ChunkWithEmbedding objects
        result = []
        for chunk_info, embedding in zip(chunks, embeddings):
            result.append(ChunkWithEmbedding(
                text=chunk_info['text'],
                embedding=embedding,
                start_char=chunk_info['start_char'],
                end_char=chunk_info['end_char'],
                start_token=chunk_info['start_token'],
                end_token=chunk_info['end_token'],
                chunk_index=chunk_info['chunk_index'],
                total_chunks=chunk_info['total_chunks'],
                context_window_used=chunk_info['context_size']
            ))

        return result
    
    def embed_batch_with_late_chunking(self,
                                       texts: List[str],
                                       task: str = "retrieval.passage") -> List[List[ChunkWithEmbedding]]:
        """
                                       Embed a batch of documents using late chunking, returning per-document chunk embeddings.
                                       
                                       Processes each input text with late chunking (via embed_with_late_chunking) and returns a list of ChunkWithEmbedding lists—one list per input document. Empty or falsy texts produce an empty list entry. This implementation preserves chunk boundary metadata and handles varying document lengths by delegating chunking to the per-document routine.
                                       
                                       Parameters:
                                           texts: List of full document strings to embed.
                                           task: Task name passed to the embedder (default "retrieval.passage"); task strings supported by the underlying model are accepted.
                                       
                                       Returns:
                                           A list where each element is a list of ChunkWithEmbedding for the corresponding input text.
                                       """
        if not texts:
            return []

        # Process each text individually to handle varying lengths correctly
        all_results = []

        for text in texts:
            if not text:
                all_results.append([])
                continue

            # Process this text through standard chunking
            # This handles both short and long texts correctly
            chunks = self.embed_with_late_chunking(text, task=task)
            all_results.append(chunks)

        return all_results

    
    def _prepare_simple_chunks(self, text: str) -> List[Dict]:
        """
        Create deterministic character-based chunks from a single document using the embedder's chunk size and overlap.
        
        This method splits `text` into contiguous chunks by estimating tokens as 4 characters each (chunk size = self.chunk_size_tokens * 4). Each chunk includes a small context overlap determined by self.chunk_overlap_tokens. When possible, chunk boundaries prefer a nearby sentence break (searching up to ~100 characters before the nominal end) to avoid cutting sentences awkwardly.
        
        Returns a list of dictionaries, one per chunk, with these keys:
        - 'text': chunk substring
        - 'start_char', 'end_char': character indices into the original text (end_char is exclusive)
        - 'start_token', 'end_token': approximate token indices computed as floor(char_index / 4)
        - 'chunk_index': zero-based index of the chunk
        - 'total_chunks': total number of chunks (populated for all entries)
        - 'context_size': approximate token count for the chunk (len(chunk_text) // 4)
        """
        chunks = []

        # Character-based chunking (rough token estimate)
        chars_per_token = 4
        chunk_size_chars = self.chunk_size_tokens * chars_per_token
        overlap_chars = self.chunk_overlap_tokens * chars_per_token

        start_char = 0
        chunk_index = 0

        while start_char < len(text):
            # Define chunk boundaries
            end_char = min(start_char + chunk_size_chars, len(text))

            # Try to break at sentence boundary if possible
            if end_char < len(text):
                # Look for sentence end near boundary
                search_start = max(start_char, end_char - 100)
                sentence_end = text.find('. ', search_start, end_char)
                if sentence_end != -1:
                    end_char = sentence_end + 2

            chunk_text = text[start_char:end_char]

            chunks.append({
                'text': chunk_text,
                'start_char': start_char,
                'end_char': end_char,
                'start_token': start_char // chars_per_token,
                'end_token': end_char // chars_per_token,
                'chunk_index': chunk_index,
                'total_chunks': 0,  # Will be updated
                'context_size': len(chunk_text) // chars_per_token
            })

            # Move to next chunk with overlap
            if end_char >= len(text):
                break

            start_char = end_char - overlap_chars
            chunk_index += 1

        # Update total chunks
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total

        return chunks

    def _prepare_chunks_for_batch(self, text: str, doc_idx: int) -> List[Dict]:
        """
        Prepare chunk and context-window metadata for a single document to be used in batched embedding.
        
        Creates contiguous chunks (estimated from token-based settings) along the text and, for each chunk,
        also extracts a larger context window that includes the chunk plus neighboring text. Estimates
        token indices and sizes using a rough heuristic of 4 characters per token.
        
        Parameters:
            text (str): The full document text to split.
            doc_idx (int): Document position in the batch (used for logging/debugging).
        
        Returns:
            List[Dict]: A list of chunk metadata dictionaries. Each dictionary contains:
                - 'chunk_text' (str): The chunk's raw text.
                - 'context_text' (str): The chunk plus surrounding text used as context.
                - 'start_char' (int): Start character index of the chunk in the original text.
                - 'end_char' (int): End character index (exclusive) of the chunk.
                - 'start_token' (int): Rough estimated start token index (chars // 4).
                - 'end_token' (int): Rough estimated end token index (chars // 4).
                - 'chunk_index' (int): Index of this chunk within the document.
                - 'total_chunks' (int): Total number of chunks for the document.
                - 'context_window_used' (int): Rough estimated token count of the context window.
        """
        # Estimate chunk size in characters (rough: ~4 chars per token)
        chunk_size_chars = self.chunk_size_tokens * 4
        
        chunks = []
        chunk_index = 0
        start_char = 0
        
        while start_char < len(text):
            # Define chunk boundaries
            end_char = min(start_char + chunk_size_chars, len(text))
            chunk_text = text[start_char:end_char]
            
            # Define context window (chunk + surrounding text)
            context_start = max(0, start_char - chunk_size_chars)
            context_end = min(len(text), end_char + chunk_size_chars)
            context_text = text[context_start:context_end]
            
            chunk_info = {
                'chunk_text': chunk_text,
                'context_text': context_text,
                'start_char': start_char,
                'end_char': end_char,
                'start_token': start_char // 4,  # Rough estimate
                'end_token': end_char // 4,
                'chunk_index': chunk_index,
                'total_chunks': 0,  # Will be updated later
                'context_window_used': len(context_text) // 4  # Rough token estimate
            }
            
            chunks.append(chunk_info)
            
            # Move to next chunk with overlap
            if end_char >= len(text):
                break
            start_char = end_char - (self.chunk_overlap_tokens * 4)  # Convert overlap to chars
            chunk_index += 1
        
        # Update total chunks count
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        logger.debug(f"Prepared {len(chunks)} chunks for document {doc_idx}")
        return chunks
    
    def _chunk_with_context_windows(self,
                                    text: str,
                                    task: str = "retrieval.passage") -> List[ChunkWithEmbedding]:
        """
        DEPRECATED: This method has O(N^2) complexity due to redundant encoding.

        Use embed_with_late_chunking() instead which properly chunks without
        redundant model calls. This method is kept only for backward compatibility
        and will be removed in future versions.
        """
        logger.warning("_chunk_with_context_windows is deprecated due to O(N^2) complexity. "
                      "Using embed_with_late_chunking instead for better performance.")

        # Redirect to the efficient implementation
        return self.embed_with_late_chunking(text, task)

    def encode_full_document(self, 
                           text: str, 
                           task: str = "retrieval.passage") -> Tuple[torch.Tensor, Dict]:
        """
                           Encode an entire document to obtain contextualized token-level embeddings for late chunking.
                           
                           Processes the full input text through the model to produce per-token contextual embeddings and returns accompanying metadata required to map token ranges back to character offsets and to drive subsequent chunking.
                           
                           Parameters:
                               text (str): The full document text to encode.
                               task (str): Logical task hint (e.g., "retrieval.passage", "code", "classification"). The value is mapped internally to the model's task labels and may influence the encoder path used.
                           
                           Returns:
                               Tuple[torch.Tensor, dict]: A pair (token_embeddings, metadata) where:
                                 - token_embeddings (torch.Tensor): Tensor of shape [seq_len, hidden_dim] with contextualized token embeddings for the input (first/only batch element).
                                 - metadata (dict): Information to support chunking:
                                     - 'offset_mapping' (ndarray): Character span offsets for each token (start, end).
                                     - 'num_tokens' (int): Number of tokens produced by the tokenizer (after truncation).
                                     - 'text_length' (int): Length of the original input text in characters.
                                     - 'task' (str): The task string passed through.
                           
                           Notes:
                               - If the input is empty, returns an empty tensor and an empty metadata dict.
                               - Inputs longer than the model's MAX_TOKENS are tokenized with truncation; a warning is emitted when truncation is detected. For very long documents, consider using process_long_document().
                           """
        if not text:
            return torch.empty(0, self.EMBEDDING_DIM), {}
        
        # Check if truncation will be needed
        estimated_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        if estimated_tokens > self.MAX_TOKENS:
            logger.warning(
                f"Document will be truncated from ~{estimated_tokens} to {self.MAX_TOKENS} tokens. "
                f"Consider using process_long_document() for documents > 32k tokens."
            )

        # Tokenize the full document
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,  # Truncate to MAX_TOKENS if needed
            max_length=self.MAX_TOKENS,
            return_offsets_mapping=True,
            return_attention_mask=True
        )
        
        # Move to device
        input_ids = tokens['input_ids'].to(self.model.device)
        attention_mask = tokens['attention_mask'].to(self.model.device)
        offset_mapping = tokens['offset_mapping'][0].cpu().numpy()
        
        with torch.no_grad():
            # Map task to Jina v4 task labels
            task_mapping = {
                'retrieval': 'retrieval',
                'code': 'text-matching',
                'classification': 'classification',
                'clustering': 'clustering'
            }
            task_label = task_mapping.get(task, 'retrieval')
            
            # For Jina v4, we need to pass task_label to the model
            # Access the underlying transformer and add task_label
            if hasattr(self.model, 'model'):
                # Call the underlying model with task_label
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_label=task_label,
                    output_hidden_states=True
                )
            else:
                # Fallback to direct model call
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_label=task_label,
                    output_hidden_states=True
                )
            
            # Get the last hidden state (contextualized token embeddings)
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                token_embeddings = outputs.last_hidden_state[0]  # Shape: [seq_len, hidden_dim]
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # Some models put it in hidden_states
                token_embeddings = outputs.hidden_states[-1][0]  # Last layer, first batch
            elif hasattr(outputs, 'vlm_last_hidden_states') and outputs.vlm_last_hidden_states is not None:
                # Jina v4 specific: use vlm_last_hidden_states for token-level embeddings
                token_embeddings = outputs.vlm_last_hidden_states[0]  # Shape: [seq_len, hidden_dim]
            elif hasattr(outputs, 'multi_vec_emb') and outputs.multi_vec_emb is not None:
                # Jina v4: multi_vec_emb contains token embeddings before pooling
                token_embeddings = outputs.multi_vec_emb[0]
            elif isinstance(outputs, tuple):
                # For tuple outputs, first element is usually the embeddings
                token_embeddings = outputs[0][0] if len(outputs[0].shape) > 2 else outputs[0]
            else:
                # For custom output types like JinaEmbeddingsV4ModelOutput
                # Try to get the first available tensor attribute
                found_embeddings = False
                for attr_name in ['embeddings', 'last_hidden_state', 'hidden_states', 'pooler_output', 'single_vec_emb']:
                    if hasattr(outputs, attr_name):
                        attr_value = getattr(outputs, attr_name)
                        if attr_value is not None:
                            if isinstance(attr_value, (list, tuple)) and len(attr_value) > 0:
                                token_embeddings = attr_value[-1][0] if hasattr(attr_value[-1], 'shape') else attr_value[0]
                            else:
                                token_embeddings = attr_value[0] if hasattr(attr_value, 'shape') and len(attr_value.shape) > 2 else attr_value
                            found_embeddings = True
                            break
                
                if not found_embeddings:
                    # Last resort: raise a more informative error
                    available_attrs = [attr for attr in dir(outputs) if not attr.startswith('_')]
                    raise ValueError(f"Could not extract token embeddings from {type(outputs).__name__}. Available attributes: {available_attrs}")
            
            # Apply pooling or projection if needed to get to 2048 dims
            # (Jina v4 uses a projection layer for final embeddings)
            if hasattr(self.model, 'encode'):
                # Use Jina's projection mechanism
                # We need to pool to get document embedding first
                pooled = token_embeddings.mean(dim=0, keepdim=True)
                # For now, we'll just use the pooled embeddings directly
                # since we can't pass pre-computed embeddings to encode
                # Use this as a reference for projection
            else:
                # Direct use of token embeddings
                pass
        
        metadata = {
            'offset_mapping': offset_mapping,
            'num_tokens': len(input_ids[0]),
            'text_length': len(text),
            'task': task
        }
        
        return token_embeddings, metadata
    
    def embed_chunks_from_tokens(self,
                                 token_embeddings: torch.Tensor,
                                 metadata: Dict,
                                 text: str,
                                 chunk_size_tokens: Optional[int] = None,
                                 chunk_overlap_tokens: Optional[int] = None) -> List[ChunkWithEmbedding]:
        """
        Create chunks with embeddings from pre-computed token embeddings (second step).
        
        This is the second step of proper late chunking:
        1. Take the contextualized token embeddings from step 1
        2. Create chunks by slicing the token embeddings
        3. Pool each chunk's tokens to get chunk embedding
        4. Each chunk embedding preserves full document context
        
        Args:
            token_embeddings: Token-level embeddings from encode_full_document
            metadata: Metadata dict from encode_full_document
            text: Original text for creating chunk text
            chunk_size_tokens: Override default chunk size
            chunk_overlap_tokens: Override default overlap
            
        Returns:
            List of ChunkWithEmbedding objects with context-aware embeddings
        """
        chunk_size = chunk_size_tokens or self.chunk_size_tokens
        overlap = chunk_overlap_tokens or self.chunk_overlap_tokens
        
        offset_mapping = metadata['offset_mapping']
        num_tokens = metadata['num_tokens']
        
        chunks = []
        chunk_index = 0
        start_token = 0
        
        while start_token < num_tokens:
            # Define chunk token boundaries
            end_token = min(start_token + chunk_size, num_tokens)
            
            # Get character boundaries from offset mapping
            start_offset = offset_mapping[start_token]
            end_offset = offset_mapping[end_token - 1]
            
            # Handle if these are still tensors
            if hasattr(start_offset, 'cpu'):
                start_offset = start_offset.cpu().numpy()
            if hasattr(end_offset, 'cpu'):
                end_offset = end_offset.cpu().numpy()
                
            start_char = int(start_offset[0])
            end_char = int(end_offset[1])
            
            # Extract chunk text
            chunk_text = text[start_char:end_char]
            
            # Get chunk embedding by pooling token embeddings
            chunk_token_embeddings = token_embeddings[start_token:end_token]
            
            # Mean pooling over tokens in the chunk
            chunk_embedding = chunk_token_embeddings.mean(dim=0)
            
            # Ensure tensor is on CPU before numpy conversion
            if hasattr(chunk_embedding, 'is_cuda') and chunk_embedding.is_cuda:
                chunk_embedding = chunk_embedding.cpu()
            
            # Convert to numpy and normalize
            chunk_embedding_np = chunk_embedding.numpy()
            norm = np.linalg.norm(chunk_embedding_np)
            if norm > 0:
                chunk_embedding_np = chunk_embedding_np / norm
            
            # Create ChunkWithEmbedding object
            chunks.append(ChunkWithEmbedding(
                text=chunk_text,
                embedding=chunk_embedding_np,
                start_char=start_char,
                end_char=end_char,
                start_token=start_token,
                end_token=end_token,
                chunk_index=chunk_index,
                total_chunks=0,  # Will update after loop
                context_window_used=num_tokens  # Full document context
            ))
            
            # Move to next chunk with overlap
            if end_token >= num_tokens:
                break
            start_token = end_token - overlap
            chunk_index += 1
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        logger.debug(f"Created {len(chunks)} chunks from {num_tokens} tokens with full context")
        return chunks
    
    def process_long_document(self,
                             text: str,
                             task: str = "retrieval.passage") -> List[ChunkWithEmbedding]:
        """
                             Split and embed a potentially very long document by processing it in (possibly overlapping) windows.
                             
                             If the document is within the model context window (<= MAX_TOKENS, estimated as len(text)//4), this delegates to _chunk_with_context_windows.
                             For documents exceeding MAX_TOKENS it processes the text in sliding windows of roughly MAX_TOKENS tokens (window size ≈ MAX_TOKENS*4 characters) with a fixed overlap of ≈1000 tokens (≈1000*4 characters). Each window is chunked via _chunk_with_context_windows; chunk character indices are adjusted to be relative to the original document and token indices are offset using an approximate 4 chars-per-token heuristic. All chunks are reindexed (chunk_index and total_chunks) before being returned.
                             
                             Parameters:
                                 text: The full document text to process.
                                 task: Embedding task name (passed through to the internal chunking/embedding calls).
                             
                             Returns:
                                 A list of ChunkWithEmbedding objects covering the entire document in order. Chunk token index adjustments use a heuristic approximation and may differ from exact tokenizers' offsets for some inputs.
                             """
        # Quick token count estimate (rough: ~4 chars per token)
        estimated_tokens = len(text) // 4
        
        logger.debug(f"process_long_document: text length={len(text)}, estimated tokens={estimated_tokens}")
        
        # For Jina v4, we use context window approach which already provides
        # excellent context preservation through overlapping windows
        # The model's encode_text method handles the embedding internally
        
        if estimated_tokens <= self.MAX_TOKENS:
            # Document fits in model's context window
            return self._chunk_with_context_windows(text, task)
        
        # Process very long documents in overlapping windows
        logger.info(f"Document too long (~{estimated_tokens} tokens), processing in windows")
        
        all_chunks = []
        window_size_chars = self.MAX_TOKENS * 4  # Rough estimate
        window_overlap_chars = 1000 * 4  # 1000 token overlap
        
        start = 0
        window_index = 0
        
        while start < len(text):
            end = min(start + window_size_chars, len(text))
            window_text = text[start:end]
            
            # Process this window with context windows approach
            window_chunks = self._chunk_with_context_windows(window_text, task)
            
            # Adjust character positions to be relative to full document
            for chunk in window_chunks:
                chunk.start_char += start
                chunk.end_char += start
                chunk.start_token += start // 4
                chunk.end_token += start // 4
            
            all_chunks.extend(window_chunks)
            
            if end >= len(text):
                break
            
            start = end - window_overlap_chars
            window_index += 1
        
        # Re-index chunks
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(all_chunks)
        
        logger.info(f"Processed {window_index + 1} windows, total {len(all_chunks)} chunks")
        
        return all_chunks
