"""
Code File Extractor
===================

Extracts content from code files for embedding generation.
Simple text extraction with optional syntax highlighting awareness.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CodeExtractor:
    """
    Extract content from code files.
    
    This extractor represents the CONVEYANCE dimension for code,
    transforming source code into processable text while preserving
    semantic structure.
    """
    
    def __init__(self):
        """
        Initialize the CodeExtractor instance.
        
        Sets up a mapping of common file extensions to their single-line comment marker (used by metadata heuristics) and performs any instance-level initialization.
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
        
        logger.info("Initialized CodeExtractor")
    
    def extract(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extracts text and basic metadata from a code file for embedding generation.
        
        Reads the file at `file_path` as UTF-8 (errors ignored) and returns a dictionary containing the raw content and simple derived fields:
        - full_text: the file's full raw content
        - text: same as full_text (kept for compatibility)
        - markdown: fenced code block using the file extension as the language tag
        - num_lines: number of lines in the file
        - file_size: file size in bytes
        - file_extension: file suffix (e.g., ".py")
        - extractor: constant string 'code_extractor'
        - metadata: language-aware summary (line_count, char_count, has_docstring, import_count, function_count, class_count)
        
        Parameters:
            file_path (str): Path to the code file to read.
        
        Returns:
            Optional[Dict[str, Any]]: The extraction result dictionary on success; None if the file does not exist or extraction fails.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic extraction - just return the content
            # In future, could add:
            # - Comment extraction
            # - Function/class detection
            # - Import analysis
            # - Tree-sitter parsing
            
            result = {
                'full_text': content,
                'text': content,
                'markdown': f"```{file_path.suffix[1:]}\n{content}\n```",
                'num_lines': len(content.splitlines()),
                'file_size': file_path.stat().st_size,
                'file_extension': file_path.suffix,
                'extractor': 'code_extractor'
            }
            
            # Extract basic metadata
            result['metadata'] = self._extract_metadata(content, file_path)
            
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