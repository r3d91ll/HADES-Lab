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
        """Initialize code extractor."""
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
        Extract content from a code file.
        
        Args:
            file_path: Path to code file
            
        Returns:
            Extracted content or None if extraction fails
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
        """Extract basic metadata from code content."""
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