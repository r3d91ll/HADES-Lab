#!/usr/bin/env python3
"""
Extractor Factory

Factory pattern for creating extractor instances based on file type and configuration.
Supports automatic format detection and fallback strategies.

Theory Connection:
The factory enables adaptive extraction strategies based on document type,
maximizing the CONVEYANCE dimension by selecting the optimal extractor
for each document format.
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from .extractors_base import ExtractorBase, ExtractorConfig

logger = logging.getLogger(__name__)


class ExtractorFactory:
    """
    Factory for creating extractor instances.

    Manages the instantiation of different extractor types based on
    file format and configuration, with support for fallbacks.
    """

    # Registry of available extractors
    _extractors: Dict[str, type] = {}

    # Format to extractor mapping
    _format_mapping: Dict[str, str] = {
        '.pdf': 'docling',
        '.tex': 'latex',
        '.py': 'code',
        '.js': 'code',
        '.ts': 'code',
        '.java': 'code',
        '.cpp': 'code',
        '.c': 'code',
        '.rs': 'code',
        '.go': 'code',
    }

    @classmethod
    def register(cls, name: str, extractor_class: type):
        """
        Register an extractor class in the factory registry.
        
        Registers `extractor_class` under `name` so the factory can create instances
        of that extractor via `create`. This mutates the class-level extractor registry.
        
        Parameters:
            name (str): Key used to look up this extractor type.
            extractor_class (type): Extractor class to register (typically a subclass of ExtractorBase).
        """
        cls._extractors[name] = extractor_class
        logger.info(f"Registered extractor: {name}")

    @classmethod
    def create(cls,
              file_path: Optional[Union[str, Path]] = None,
              extractor_type: Optional[str] = None,
              config: Optional[ExtractorConfig] = None,
              **kwargs) -> ExtractorBase:
        """
              Create and return an extractor instance using the provided configuration, explicit type, or by auto-detecting type from a file path.
              
              If no config is provided, a new ExtractorConfig is built from kwargs. If extractor_type is not given but file_path is, the extractor type is determined from the file extension (with mappings for common formats, code files, and LaTeX-related extensions); when neither is provided the factory defaults to 'docling'. If the requested extractor type is not already registered the factory attempts on-demand registration; if it remains unavailable a ValueError is raised.
              
              Parameters:
                  file_path: Optional path used to auto-detect the extractor type when extractor_type is not provided.
                  extractor_type: Optional explicit extractor type name to use (overrides auto-detection).
                  config: Optional ExtractorConfig to pass to the extractor. If omitted, one is constructed from kwargs.
                  **kwargs: Extra parameters forwarded to ExtractorConfig when config is not supplied.
              
              Returns:
                  An instance of the resolved extractor class constructed with the provided config.
              
              Raises:
                  ValueError: If no extractor is registered (and cannot be auto-registered) for the resolved extractor type.
              """
        # Create config if not provided
        if config is None:
            config = ExtractorConfig(**kwargs)

        # Determine extractor type
        if extractor_type is None and file_path is not None:
            extractor_type = cls._determine_extractor_type(file_path)
        elif extractor_type is None:
            extractor_type = 'docling'  # Default

        if extractor_type not in cls._extractors:
            # Try to import and register on-demand
            cls._auto_register(extractor_type)

        if extractor_type not in cls._extractors:
            available = list(cls._extractors.keys())
            raise ValueError(
                f"No extractor registered for type '{extractor_type}'. "
                f"Available: {available}"
            )

        extractor_class = cls._extractors[extractor_type]
        logger.info(f"Creating {extractor_type} extractor")

        return extractor_class(config)

    @classmethod
    def _determine_extractor_type(cls, file_path: Union[str, Path]) -> str:
        """
        Determine the extractor type for a given file path based on its suffix.
        
        Uses the factory's internal extension-to-type mapping first, then falls back to:
        - 'latex' for common LaTeX-related extensions ('.tex', '.bib', '.cls', '.sty'),
        - 'code' for a curated set of programming-language file extensions,
        - 'docling' as the default for PDFs and any unknown/other extensions.
        
        Parameters:
            file_path (str | Path): Path to the file whose extractor type should be determined.
        
        Returns:
            str: One of the extractor type identifiers (e.g., 'docling', 'latex', 'code') as resolved for the given file.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        # Check format mapping
        if suffix in cls._format_mapping:
            return cls._format_mapping[suffix]

        # Check for LaTeX files
        if suffix in ['.tex', '.bib', '.cls', '.sty']:
            return 'latex'

        # Check for code files
        code_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
                          '.rs', '.go', '.rb', '.php', '.swift', '.kt']
        if suffix in code_extensions:
            return 'code'

        # Default to Docling for PDFs and unknown formats
        return 'docling'

    @classmethod
    def _auto_register(cls, extractor_type: str):
        """
        Attempt to import and register a built-in extractor implementation for the given extractor type.
        
        This performs on-demand registration by dynamically importing the extractor class corresponding to extractor_type
        and calling cls.register(name, extractor_class). Supported extractor_type values and their registrations:
        - "docling" -> DoclingExtractor
        - "latex" -> LatexExtractor
        - "code" -> CodeExtractor
        - "treesitter" -> TreeSitterExtractor
        - "robust" -> RobustExtractor
        
        If extractor_type is unrecognized, a warning is logged and no registration is attempted. Import failures are
        caught and logged; this function does not raise on import errors.
        """
        try:
            if extractor_type == 'docling':
                from .extractors_docling import DoclingExtractor
                cls.register('docling', DoclingExtractor)
            elif extractor_type == 'latex':
                from .extractors_latex import LatexExtractor
                cls.register('latex', LatexExtractor)
            elif extractor_type == 'code':
                from .extractors_code import CodeExtractor
                cls.register('code', CodeExtractor)
            elif extractor_type == 'treesitter':
                from .extractors_treesitter import TreeSitterExtractor
                cls.register('treesitter', TreeSitterExtractor)
            elif extractor_type == 'robust':
                from .extractors_robust import RobustExtractor
                cls.register('robust', RobustExtractor)
            else:
                logger.warning(f"Unknown extractor type: {extractor_type}")
        except ImportError as e:
            logger.error(f"Failed to import {extractor_type} extractor: {e}")

    @classmethod
    def create_for_file(cls,
                       file_path: Union[str, Path],
                       config: Optional[ExtractorConfig] = None,
                       **kwargs) -> ExtractorBase:
        """
                       Create and return an extractor tailored to the given file.
                       
                       This is a convenience wrapper around `create` that uses the file path to auto-detect the extractor type when one is not provided. If the file extension is unrecognized, the factory defaults to the 'docling' extractor. The provided `config`, if any, is passed through to the created extractor.
                       
                       Parameters:
                           file_path: Path to the input file used to determine the extractor type.
                           config: Optional extraction configuration to instantiate the extractor with.
                       
                       Returns:
                           An instance of ExtractorBase appropriate for the file.
                       """
        return cls.create(file_path=file_path, config=config, **kwargs)

    @classmethod
    def list_available(cls) -> Dict[str, Any]:
        """
        Return a mapping of registered extractor names to basic info.
        
        Each key is a registered extractor name; values are dicts containing:
        - "class": extractor class name
        - "module": extractor class module
        If retrieving info for an extractor fails, its value will be {"error": "<error message>"}.
        
        Returns:
            Dict[str, Any]: Mapping of extractor name to info or error.
        """
        available = {}
        for name, extractor_class in cls._extractors.items():
            try:
                available[name] = {
                    "class": extractor_class.__name__,
                    "module": extractor_class.__module__
                }
            except Exception as e:
                logger.warning(f"Failed to get info for {name}: {e}")
                available[name] = {"error": str(e)}

        return available

    @classmethod
    def get_format_mapping(cls) -> Dict[str, str]:
        """
        Return a shallow copy of the mapping from file extensions to extractor type names.
        
        The returned dictionary maps file suffixes (including the leading dot, e.g. '.py', '.tex') to
        the extractor type string used by the factory (e.g. 'code', 'latex', 'docling').
        
        Returns:
            Dict[str, str]: A shallow copy of the internal format-to-extractor mapping.
        """
        return cls._format_mapping.copy()