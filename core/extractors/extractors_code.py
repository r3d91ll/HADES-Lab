"""
Code File Extractor
===================

Extracts content from code files for embedding generation.
Integrates Tree-sitter for symbol table extraction to provide rich metadata
for Jina v4 coding LoRA embeddings.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from core.framework.extractors.tree_sitter_extractor import TreeSitterExtractor
logger = logging.getLogger(__name__)


class CodeExtractor:
    """
    Extract content from code files.
    
    This extractor represents the CONVEYANCE dimension for code,
    transforming source code into processable text while preserving
    semantic structure.
    """
    
    def __init__(self, use_tree_sitter: bool = True):
        """
        Create a CodeExtractor and prepare language comment heuristics.
        
        Initializes a mapping of common file extensions to their single-line comment marker (used by metadata heuristics). If `use_tree_sitter` is True, attempts to instantiate a TreeSitterExtractor for symbol/structure extraction; on failure Tree-sitter support is disabled and extraction will proceed without it.
        
        Parameters:
            use_tree_sitter (bool): If True, try to enable Tree-sitter-based extraction; if initialization fails, Tree-sitter will be disabled silently.
        """
        # Common comment patterns for different languages
        self.comment_patterns = {
            '.py': '#',
            '.js': '//',
            '.ts': '//',
            '.java': '//',
            '.c': '//',
            '.cpp': '//',
            '.go': '//',
            '.rs': '//',
            '.rb': '#',
            '.sh': '#',
            '.yaml': '#',
            '.yml': '#'
        }
        
        # Initialize Tree-sitter extractor
        self.use_tree_sitter = use_tree_sitter
        if use_tree_sitter:
            try:
                self.tree_sitter = TreeSitterExtractor()
                logger.info("Initialized CodeExtractor with Tree-sitter support")
            except Exception as e:
                logger.warning(f"Failed to initialize Tree-sitter: {e}")
                self.tree_sitter = None
                self.use_tree_sitter = False
        else:
            self.tree_sitter = None
            logger.info("Initialized CodeExtractor without Tree-sitter")
    
    def extract(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract the contents of a code file and return text plus lightweight, language-aware metadata.
        
        Reads the file at `file_path` as UTF-8 (errors ignored) and builds a result dict containing:
        - full_text, text: raw file content
        - markdown: fenced code block using the file extension as the language tag
        - num_lines: number of lines
        - file_size: file size in bytes
        - file_extension: suffix (e.g., ".py")
        - extractor: 'code_extractor_with_tree_sitter' if Tree-sitter provided enrichment, otherwise 'code_extractor'
        If a Tree-sitter extractor is available and succeeds, the result is enriched with:
        - symbols, code_metrics (from Tree-sitter 'metrics'), code_structure, language, and symbol_hash (when symbols exist)
        
        Parameters:
            file_path (str): Path to the code file to read.
        
        Returns:
            Optional[Dict[str, Any]]: Extraction result on success; None if the file does not exist or an error occurs.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract Tree-sitter symbols if available
            tree_sitter_data = {}
            if self.use_tree_sitter and self.tree_sitter:
                try:
                    tree_sitter_data = self.tree_sitter.extract_symbols(str(file_path), content)
                    logger.debug(f"Extracted Tree-sitter symbols for {file_path}")
                except Exception as e:
                    logger.warning(f"Tree-sitter extraction failed for {file_path}: {e}")
            
            result = {
                'full_text': content,
                'text': content,
                'markdown': f"```{file_path.suffix[1:]}\n{content}\n```",
                'num_lines': len(content.splitlines()),
                'file_size': file_path.stat().st_size,
                'file_extension': file_path.suffix,
                'extractor': 'code_extractor_with_tree_sitter' if tree_sitter_data else 'code_extractor'
            }
            
            # Add Tree-sitter data if available
            if tree_sitter_data:
                result['symbols'] = tree_sitter_data.get('symbols', {})
                result['code_metrics'] = tree_sitter_data.get('metrics', {})
                result['code_structure'] = tree_sitter_data.get('structure', {})
                result['language'] = tree_sitter_data.get('language')
                
                # Generate symbol hash for comparison
                if self.tree_sitter and tree_sitter_data.get('symbols'):
                    result['symbol_hash'] = self.tree_sitter.generate_symbol_hash(tree_sitter_data['symbols'])
            
            # Extract basic metadata (fallback for non-Tree-sitter)
            result['metadata'] = self._extract_metadata(content, file_path)
            
            # Merge Tree-sitter metrics into metadata if available
            if tree_sitter_data.get('metrics'):
                result['metadata'].update(tree_sitter_data['metrics'])
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract {file_path}: {e}")
            return None
    
    def _extract_metadata(self, content: str, file_path: Path) -> Dict[str, Any]:
        """
        Compute basic metadata for a source code file.
        
        Returns a dictionary of metrics computed from `content`. The returned keys are:
        - line_count: number of lines in the content.
        - char_count: total number of characters.
        - has_docstring: (Python) True if triple-quote docstrings are present.
        - import_count: approximate count of import/require statements.
        - function_count: approximate count of function definitions or arrow functions.
        - class_count: approximate count of class declarations.
        
        Language-specific behavior is inferred from file_path.suffix (lowercased):
        - .py: detects Python docstrings, lines starting with `import`/`from`, `def`, and `class`.
        - .js, .ts, .jsx, .tsx: counts occurrences of `import`/`require`, `function` and `=>`, and lines starting with `class`.
        - .java: counts lines starting with `import` and lines containing `class` with `{`.
        
        The counts are simple heuristics (line-based or substring matches) intended for lightweight metadata and may over- or under-count in complex code constructs.
        """
        lines = content.splitlines()
        
        metadata = {
            'line_count': len(lines),
            'char_count': len(content),
            'has_docstring': False,
            'import_count': 0,
            'function_count': 0,
            'class_count': 0
        }
        
        # Language-specific analysis
        ext = file_path.suffix.lower()
        
        if ext == '.py':
            # Python-specific analysis
            metadata['has_docstring'] = '"""' in content or "'''" in content
            metadata['import_count'] = sum(1 for line in lines if line.strip().startswith(('import ', 'from ')))
            metadata['function_count'] = sum(1 for line in lines if line.strip().startswith('def '))
            metadata['class_count'] = sum(1 for line in lines if line.strip().startswith('class '))
            
        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            # JavaScript/TypeScript analysis
            metadata['import_count'] = sum(1 for line in lines if 'import ' in line or 'require(' in line)
            metadata['function_count'] = content.count('function ') + content.count('=>')
            metadata['class_count'] = sum(1 for line in lines if line.strip().startswith('class '))
            
        elif ext == '.java':
            # Java analysis
            metadata['import_count'] = sum(1 for line in lines if line.strip().startswith('import '))
            metadata['class_count'] = sum(1 for line in lines if 'class ' in line and '{' in line)
            
        return metadata