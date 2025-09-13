#!/usr/bin/env python3
"""
Base Extractor Interface

Defines the contract for all document extraction implementations.
Following the Conveyance Framework: extractors transform raw documents
into structured information, preserving the WHAT while enhancing accessibility.

Theory Connection:
Extractors are critical for the CONVEYANCE dimension - they transform
unstructured documents into actionable, structured data. High-quality
extraction directly increases the C value by making information more
readily usable.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of document extraction."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    equations: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    code_blocks: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class ExtractorConfig:
    """Configuration for extractors."""
    use_gpu: bool = True
    batch_size: int = 1
    timeout_seconds: int = 300
    extract_equations: bool = True
    extract_tables: bool = True
    extract_images: bool = True
    extract_code: bool = True
    extract_references: bool = True
    max_pages: Optional[int] = None
    ocr_enabled: bool = False


class ExtractorBase(ABC):
    """
    Abstract base class for all extractors.

    Defines the interface that all extraction implementations must follow
    to ensure consistency across different document types and approaches.
    """

    def __init__(self, config: Optional[ExtractorConfig] = None):
        """
        Initialize the extractor instance with the given configuration.
        
        If no config is provided, a default ExtractorConfig is created and assigned to self.config.
        
        Parameters:
            config (Optional[ExtractorConfig]): Extraction configuration to use; when None, a default is applied.
        """
        self.config = config or ExtractorConfig()

    @abstractmethod
    def extract(self,
               file_path: Union[str, Path],
               **kwargs) -> ExtractionResult:
        """
               Abstract method — extract structured content from a single document.
               
               Subclasses must override this to read and convert the document at `file_path` into an ExtractionResult.
               `file_path` is the path to the input document. `**kwargs` are extractor-specific options (for example: page ranges,
               timeout overrides, or OCR configuration) and may be ignored by some implementations.
               
               Returns:
                   ExtractionResult: standardized extraction output containing text, metadata, detected objects (tables, images,
                   equations, code blocks, references), optional error information, and processing time.
               """
        pass

    @abstractmethod
    def extract_batch(self,
                     file_paths: List[Union[str, Path]],
                     **kwargs) -> List[ExtractionResult]:
        """
                     Extract content from multiple documents and return a list of ExtractionResult objects in the same order.
                     
                     Each entry in the returned list corresponds to the path at the same index in `file_paths`. Additional keyword arguments are forwarded to the per-file `extract` implementation and may customize behavior (e.g., page range, OCR overrides) depending on the concrete extractor.
                     """
        pass

    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate that `file_path` refers to an existing, non-empty regular file.
        
        Performs three checks: the path exists, is a file (not a directory or other special entry), and has a non-zero size. On failure each check logs an error describing the reason.
        
        Parameters:
            file_path (str | Path): Path to the file to validate.
        
        Returns:
            bool: True if the path exists, is a regular file, and is not empty; otherwise False.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File does not exist: {path}")
            return False
        if not path.is_file():
            logger.error(f"Path is not a file: {path}")
            return False
        if path.stat().st_size == 0:
            logger.error(f"File is empty: {path}")
            return False
        return True

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """
        Return the list of file formats this extractor supports.
        
        Returns:
            List[str]: A list of supported file format identifiers (e.g., file extensions or MIME-type strings).
        """
        pass

    @property
    def supports_gpu(self) -> bool:
        """
        Return whether this extractor supports GPU acceleration.
        
        Subclasses that enable GPU processing should override this property to return True.
        
        Returns:
            bool: True if GPU acceleration is supported, otherwise False.
        """
        return False

    @property
    def supports_batch(self) -> bool:
        """Whether this extractor supports batch processing."""
        return True

    @property
    def supports_ocr(self) -> bool:
        """Whether this extractor supports OCR."""
        return False

    def get_extractor_info(self) -> Dict[str, Any]:
        """
        Return a dictionary summarizing this extractor's identity, capabilities, and core configuration.
        
        Returns:
            dict: Metadata containing:
                - "class" (str): extractor class name.
                - "supported_formats" (List[str]): formats the extractor supports.
                - "supports_gpu" (bool): whether GPU acceleration is supported.
                - "supports_batch" (bool): whether batch processing is supported.
                - "supports_ocr" (bool): whether OCR is supported.
                - "config" (dict): selected configuration values with keys
                  "use_gpu" (bool), "batch_size" (int), and "timeout_seconds" (int).
        """
        return {
            "class": self.__class__.__name__,
            "supported_formats": self.supported_formats,
            "supports_gpu": self.supports_gpu,
            "supports_batch": self.supports_batch,
            "supports_ocr": self.supports_ocr,
            "config": {
                "use_gpu": self.config.use_gpu,
                "batch_size": self.config.batch_size,
                "timeout_seconds": self.config.timeout_seconds
            }
        }