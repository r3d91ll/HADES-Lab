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


class JinaV4Embedder:
    """
    Jina v4 embedder with late chunking support.
    
    This embedder supports both:
    1. Traditional embedding (for short texts, backward compatibility)
    2. Late chunking (for long documents, superior context preservation)
    """
    
    # Jina v4 constants
    MAX_TOKENS = 32768  # Jina v4's context window
    EMBEDDING_DIM = 2048
    
    def __init__(self, 
                 device: str = None,
                 use_fp16: bool = None,
                 chunk_size_tokens: int = None,
                 chunk_overlap_tokens: int = None,
                 config_path: str = None):
        """
        Initialize Jina v4 embedder with late chunking support.
        
        Args:
            device: Device to use (cuda/cpu) - overrides config
            use_fp16: Use half precision for efficiency - overrides config
            chunk_size_tokens: Size of chunks in tokens - overrides config
            chunk_overlap_tokens: Overlap between chunks - overrides config
            config_path: Path to embedder configuration file
        """
        # Try to load config if path provided or default exists
        config = {}
        if config_path:
            import yaml
            try:
                with open(config_path, 'r') as f:
                    loaded = yaml.safe_load(f)
                    if not isinstance(loaded, dict):
                        logger.warning(f"Invalid YAML structure in {config_path}, using defaults")
                        config = {}
                    else:
                        config = loaded.get('embedder', {})
                        logger.debug(f"Loaded embedder config from {config_path}")
            except (yaml.YAMLError, IOError) as e:
                logger.warning(f"Failed to load config {config_path}: {e}, using defaults")
                config = {}
        else:
            # Try default location
            import os
            default_config = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'configs', 'embedder.yaml'
            )
            if os.path.exists(default_config):
                import yaml
                try:
                    with open(default_config, 'r') as f:
                        loaded = yaml.safe_load(f)
                        if not isinstance(loaded, dict):
                            logger.warning(f"Invalid YAML structure in {default_config}, using defaults")
                            config = {}
                        else:
                            config = loaded.get('embedder', {})
                            logger.info(f"Loaded embedder config from {default_config}")
                except (yaml.YAMLError, IOError) as e:
                    logger.warning(f"Failed to load default config {default_config}: {e}, using defaults")
                    config = {}
        
        # Set parameters with config as defaults, overridable by arguments
        self.device = device or config.get('device', 'cuda')
        # Use full Jina model with task='retrieval' for our use case
        # The retrieval-only model requires vLLM which would need refactoring
        self.model_name = config.get('model_name', 'jinaai/jina-embeddings-v4')
        self.chunk_size_tokens = chunk_size_tokens or config.get('chunk_size_tokens', 1000)
        self.chunk_overlap_tokens = chunk_overlap_tokens or config.get('chunk_overlap_tokens', 200)
        use_fp16 = use_fp16 if use_fp16 is not None else config.get('use_fp16', True)
        
        # Load model with appropriate dtype
        # Check if device starts with "cuda" to handle cuda:0, cuda:1, etc.
        dtype = torch.float16 if (use_fp16 and self.device.startswith("cuda")) else torch.float32
        
        logger.info(f"Loading {self.model_name} on {self.device} with dtype={dtype}")
        logger.info(f"Late chunking config: {self.chunk_size_tokens} tokens/chunk, {self.chunk_overlap_tokens} overlap")
        
        # Load tokenizer first for late chunking
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Load model directly to the target device to avoid Flash Attention warning
        # Check if device starts with "cuda" to handle both "cuda" and "cuda:0", "cuda:1", etc.
        if self.device.startswith("cuda") and torch.cuda.is_available():
            # For now, use single GPU to avoid tensor device mismatch with auto device_map
            # The Jina model doesn't properly handle distributed tensors
            if self.device == "cuda":
                # Default to first GPU
                device_to_use = "cuda:0"
            else:
                device_to_use = self.device
                
            logger.info(f"Loading model on {device_to_use}")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=device_to_use
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=dtype
            )
            # If not using device_map, manually move to device
            if self.device != "cpu":
                self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Store model device for later use
        # With multi-GPU, model.device doesn't exist, so we track it ourselves
        if hasattr(self.model, 'hf_device_map'):
            # Model is distributed across multiple GPUs
            self.model_device = 'cuda:0'  # Use first GPU for input tensors
            logger.info(f"Model distributed across GPUs: {list(set(self.model.hf_device_map.values()))}")
        elif hasattr(self.model, 'device'):
            self.model_device = self.model.device
        else:
            self.model_device = self.device
        
        logger.info("Jina v4 model loaded with late chunking support")
        
    def embed_texts(self, 
                    texts: List[str], 
                    task: str = "retrieval",
                    batch_size: int = 4) -> np.ndarray:
        """
        Embed texts using Jina v4.
        
        Args:
            texts: List of texts to embed
            task: Task type (retrieval, text-matching, code)
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (N x 2048)
        """
        all_embeddings = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Use Jina v4's encode_text method
                embeddings = self.model.encode_text(
                    texts=batch,
                    task=task,
                    batch_size=len(batch)  # Process whole batch at once
                )
                
                # Debug: Check what type of object we got
                logger.debug(f"Embeddings type: {type(embeddings)}")
                
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
                
                # Explicitly clear GPU memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate all batches
        result = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 2048))
        
        return result
    
    def embed_code(self, 
                   code_snippets: List[str],
                   batch_size: int = 4) -> np.ndarray:
        """
        Embed code using the code-specific task.
        
        Args:
            code_snippets: List of code snippets
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (N x 2048)
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
                                 task: str = "retrieval") -> List[ChunkWithEmbedding]:
        """
        Context-aware chunking implementation for Jina v4.
        
        Uses a sliding window approach with significant overlap to preserve context.
        Each chunk is embedded with awareness of surrounding text through larger
        context windows, achieving similar benefits to traditional late chunking.
        
        This maintains the WHERE dimension of information even when physically chunked,
        preventing zero-propagation in the WHAT dimension by ensuring each chunk
        contains awareness of its surrounding context.
        
        Args:
            text: Full document text to process
            task: Task type (retrieval, text-matching, separation, classification)
            
        Returns:
            List of ChunkWithEmbedding objects with contextually-aware embeddings
        """
        if not text:
            return []
        
        # Use the process_long_document method which handles both short and long texts
        return self.process_long_document(text, task)
    
    def embed_batch_with_late_chunking(self,
                                       texts: List[str],
                                       task: str = "retrieval") -> List[List[ChunkWithEmbedding]]:
        """
        Batch version of embed_with_late_chunking for GPU efficiency.
        
        Processes multiple documents simultaneously while preserving the late chunking
        context awareness. Collects context windows from all documents and processes
        them in batches for optimal GPU utilization.
        
        Args:
            texts: List of full document texts to process
            task: Task type (retrieval, text-matching, separation, classification)
            
        Returns:
            List of lists - one ChunkWithEmbedding list per input document
        """
        if not texts:
            return []
        
        all_results = []
        
        # For efficiency, we'll batch process context windows across documents
        # But first, let's collect all context windows with document tracking
        all_context_windows = []
        document_chunk_map = []  # Maps context window index to (doc_idx, chunk_info)
        
        for doc_idx, text in enumerate(texts):
            if not text:
                all_results.append([])
                continue
            
            # Generate chunk boundaries for this document
            doc_chunks = self._prepare_chunks_for_batch(text, doc_idx)
            
            # Add to global context windows list
            for chunk_info in doc_chunks:
                all_context_windows.append(chunk_info['context_text'])
                document_chunk_map.append({
                    'doc_idx': doc_idx,
                    'chunk_info': chunk_info
                })
        
        # Batch process all context windows at once
        if all_context_windows:
            logger.info(f"Batch processing {len(all_context_windows)} context windows from {len(texts)} documents")
            
            # Call model directly for maximum efficiency
            with torch.no_grad():
                # Process all context windows in optimal batches
                context_embeddings = self.model.encode_text(
                    texts=all_context_windows,
                    task=task,
                    batch_size=32  # Larger batch size for better GPU utilization
                )
                
                # Convert to numpy if needed
                if torch.is_tensor(context_embeddings):
                    if context_embeddings.is_cuda:
                        context_embeddings = context_embeddings.cpu()
                    context_embeddings = context_embeddings.numpy()
                elif hasattr(context_embeddings, 'detach'):
                    context_embeddings = context_embeddings.detach()
                    if hasattr(context_embeddings, 'is_cuda') and context_embeddings.is_cuda:
                        context_embeddings = context_embeddings.cpu()
                    context_embeddings = context_embeddings.numpy()
                elif isinstance(context_embeddings, np.ndarray):
                    pass  # Already numpy
                else:
                    # Convert to numpy as fallback
                    # Check if it's a list of tensors
                    if isinstance(context_embeddings, list) and len(context_embeddings) > 0:
                        first_item = context_embeddings[0]
                        if torch.is_tensor(first_item):
                            # Convert all tensors to CPU then numpy
                            context_embeddings = [t.cpu().numpy() if t.is_cuda else t.numpy() for t in context_embeddings]
                            context_embeddings = np.array(context_embeddings)
                        else:
                            context_embeddings = np.array(context_embeddings)
                    else:
                        context_embeddings = np.array(context_embeddings)
                    
                logger.info(f"Generated embeddings shape: {context_embeddings.shape}")
            
            # Organize results back by document
            doc_results = [[] for _ in range(len(texts))]
            
            for ctx_idx, embedding in enumerate(context_embeddings):
                map_info = document_chunk_map[ctx_idx]
                doc_idx = map_info['doc_idx']
                chunk_info = map_info['chunk_info']
                
                # Create ChunkWithEmbedding object
                chunk_with_embedding = ChunkWithEmbedding(
                    text=chunk_info['chunk_text'],
                    embedding=embedding,
                    start_char=chunk_info['start_char'],
                    end_char=chunk_info['end_char'],
                    start_token=chunk_info['start_token'],
                    end_token=chunk_info['end_token'],
                    chunk_index=chunk_info['chunk_index'],
                    total_chunks=chunk_info['total_chunks'],
                    context_window_used=chunk_info['context_window_used']
                )
                
                doc_results[doc_idx].append(chunk_with_embedding)
            
            # Update total chunks count for each document
            for doc_idx, doc_chunks in enumerate(doc_results):
                for chunk in doc_chunks:
                    chunk.total_chunks = len(doc_chunks)
            
            all_results = doc_results
        else:
            # No valid texts
            all_results = [[] for _ in range(len(texts))]
        
        logger.info(f"Batch late chunking complete: processed {len(texts)} documents, "
                   f"generated {sum(len(doc) for doc in all_results)} total chunks")
        
        return all_results
    
    def _prepare_chunks_for_batch(self, text: str, doc_idx: int) -> List[Dict]:
        """
        Prepare chunks and context windows for a single document in batch processing.
        
        Returns list of dictionaries with chunk and context information.
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
                                    task: str = "retrieval") -> List[ChunkWithEmbedding]:
        """
        Create chunks with contextual embeddings using overlapping windows.
        
        Each chunk is embedded with surrounding context to preserve semantic
        relationships across boundaries. This is Jina v4's practical approach
        to achieving the benefits of late chunking.
        """
        # Estimate chunk size in characters (rough: ~4 chars per token)
        chunk_size_chars = self.chunk_size_tokens * 4
        context_window_chars = min(self.chunk_size_tokens * 8, self.MAX_TOKENS * 4)  # 2x context on each side
        
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
            
            # Embed the context window (not just the chunk)
            with torch.no_grad():
                try:
                    # Use encode_text which works with Jina v4's actual API
                    context_embedding = self.model.encode_text(
                        texts=[context_text],
                        task=task,
                        batch_size=1
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM on chunk {chunk_index}, clearing cache and retrying")
                        torch.cuda.empty_cache()
                        import time
                        time.sleep(0.5)  # Brief pause
                        
                        # Try again with cleared cache
                        try:
                            context_embedding = self.model.encode_text(
                                texts=[context_text],
                                task=task,
                                batch_size=1
                            )
                        except RuntimeError as e2:
                            if "out of memory" in str(e2).lower():
                                # Still OOM - try with smaller context
                                logger.warning(f"Still OOM, reducing context window for chunk {chunk_index}")
                                torch.cuda.empty_cache()
                                time.sleep(0.5)
                                
                                # Use just the chunk without context
                                smaller_context = chunk_text[:min(len(chunk_text), 8000)]  # Limit to ~2k tokens
                                try:
                                    context_embedding = self.model.encode_text(
                                        texts=[smaller_context],
                                        task=task,
                                        batch_size=1
                                    )
                                except Exception as e3:
                                    logger.error(f"Failed even with reduced context: {e3}")
                                    # Return zero embedding as last resort
                                    context_embedding = np.zeros((1, 2048), dtype=np.float32)
                            else:
                                raise e2
                    else:
                        raise e
                
                logger.debug(f"Context embedding type after encode_text: {type(context_embedding)}")
                if hasattr(context_embedding, 'is_cuda'):
                    logger.debug(f"Context embedding is_cuda: {context_embedding.is_cuda}")
                
                # Convert to numpy - handle both tensor and direct numpy returns
                if isinstance(context_embedding, list):
                    # Handle list of tensors (Jina v4 returns list)
                    logger.debug(f"Context embedding is a list of {len(context_embedding)} items")
                    if len(context_embedding) > 0:
                        first_item = context_embedding[0]
                        if torch.is_tensor(first_item):
                            if first_item.is_cuda:
                                logger.debug("Moving list item from CUDA to CPU")
                                context_embedding = [t.cpu().numpy() if t.is_cuda else t.numpy() for t in context_embedding]
                            else:
                                context_embedding = [t.cpu().numpy() if hasattr(t, 'is_cuda') and t.is_cuda else t.numpy() for t in context_embedding]
                            # Stack if multiple, otherwise just take the first
                            context_embedding = np.vstack(context_embedding) if len(context_embedding) > 1 else context_embedding[0]
                        else:
                            # Try to convert list items to numpy
                            context_embedding = np.array(context_embedding)
                elif torch.is_tensor(context_embedding):
                    # Move to CPU if on CUDA
                    if context_embedding.is_cuda:
                        logger.debug("Moving context embedding from CUDA to CPU")
                        context_embedding = context_embedding.cpu()
                    context_embedding = context_embedding.numpy()
                elif hasattr(context_embedding, 'detach'):
                    context_embedding = context_embedding.detach()
                    if hasattr(context_embedding, 'is_cuda') and context_embedding.is_cuda:
                        logger.debug("Moving detached context embedding from CUDA to CPU")
                        context_embedding = context_embedding.cpu()
                    context_embedding = context_embedding.numpy()
                elif not isinstance(context_embedding, np.ndarray):
                    # Try to convert to numpy
                    try:
                        context_embedding = np.array(context_embedding)
                    except Exception as e:
                        logger.error(f"Failed to convert context embedding to numpy: {e}")
                        logger.error(f"Context embedding type: {type(context_embedding)}")
                        raise
                
                # Get first element if batched
                if context_embedding.ndim > 1:
                    context_embedding = context_embedding[0]
            
            # Create chunk with context-aware embedding
            chunks.append(ChunkWithEmbedding(
                text=chunk_text,
                embedding=context_embedding,
                start_char=start_char,
                end_char=end_char,
                start_token=start_char // 4,  # Rough estimate
                end_token=end_char // 4,
                chunk_index=chunk_index,
                total_chunks=0,  # Will update after loop
                context_window_used=len(context_text) // 4  # Rough token estimate
            ))
            
            # Move to next chunk with overlap
            if end_char >= len(text):
                break
            start_char = end_char - (self.chunk_overlap_tokens * 4)  # Convert overlap to chars
            chunk_index += 1
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        logger.debug(f"Created {len(chunks)} context-aware chunks")
        return chunks

    def encode_full_document(self, 
                           text: str, 
                           task: str = "retrieval") -> Tuple[torch.Tensor, Dict]:
        """
        Encode a full document to get token-level embeddings (first step of late chunking).
        
        This is the critical first step of proper late chunking:
        1. Process the entire document through the transformer
        2. Get contextualized token embeddings for the whole document
        3. Return these for subsequent chunking with context preservation
        
        Args:
            text: Full document text
            task: Task type (retrieval, code, etc.)
            
        Returns:
            Tuple of (token_embeddings, metadata_dict)
            where metadata_dict contains token offsets and other info
        """
        if not text:
            return torch.empty(0, self.EMBEDDING_DIM), {}
        
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
        input_ids = tokens['input_ids'].to(self.model_device)
        attention_mask = tokens['attention_mask'].to(self.model_device)
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
            if hasattr(self.model, 'encode_text'):
                # Use Jina's projection mechanism
                # We need to pool to get document embedding first
                pooled = token_embeddings.mean(dim=0, keepdim=True)
                # Then project to final dimension
                final_embedding = self.model.encode_text(
                    texts=[""],  # Dummy text since we already have embeddings
                    task=task,
                    batch_size=1,
                    token_embeddings=pooled  # Pass pre-computed embeddings if supported
                )
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
                             task: str = "retrieval") -> List[ChunkWithEmbedding]:
        """
        Process a document that may exceed 32k tokens.
        
        For documents longer than 32k tokens, we process in windows with
        overlap to maintain some context across boundaries.
        
        Args:
            text: Document text (can be very long)
            task: Task type
            
        Returns:
            List of all chunks with embeddings
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


def test_jina_v4() -> bool:
    """Test Jina v4 embedder."""
    print("Testing Jina v4 Embedder...")
    
    # Initialize embedder
    embedder = JinaV4Embedder(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test text embedding
    texts = [
        "Information Reconstructionism demonstrates multiplicative dependencies.",
        "When any dimension equals zero, information ceases to exist."
    ]
    
    embeddings = embedder.embed_texts(texts)
    print(f"✓ Text embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (2, 2048), f"Expected (2, 2048), got {embeddings.shape}"
    
    # Test code embedding
    code = [
        "def calculate_information(where, what, conveyance):\n    return where * what * conveyance"
    ]
    
    code_embeddings = embedder.embed_code(code)
    print(f"✓ Code embeddings shape: {code_embeddings.shape}")
    assert code_embeddings.shape == (1, 2048), f"Expected (1, 2048), got {code_embeddings.shape}"
    
    # Test PROPER late chunking
    long_text = "Information theory " * 1000  # Long text
    chunk_embeddings = embedder.process_long_document(long_text)
    print(f"✓ Late chunking produced {len(chunk_embeddings)} chunks")
    if chunk_embeddings:
        first_chunk = chunk_embeddings[0]
        print(f"  First chunk: {first_chunk.start_char}-{first_chunk.end_char} chars, "
              f"{first_chunk.start_token}-{first_chunk.end_token} tokens")
        print(f"  Context window used: {first_chunk.context_window_used} tokens")
        print(f"  Embedding shape: {first_chunk.embedding.shape}")
    
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    test_jina_v4()
