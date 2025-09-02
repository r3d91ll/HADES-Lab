#!/usr/bin/env python3
"""
Simple Tree-sitter Test
=======================

Test Tree-sitter symbol extraction on a single Python file.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.framework.extractors.tree_sitter_extractor import TreeSitterExtractor
import logging

logging.basicConfig(level=logging.DEBUG)

def test_simple():
    """Test on this very file."""
    
    extractor = TreeSitterExtractor()
    
    # Test on this file
    result = extractor.extract_symbols(__file__)
    
    print(f"Language: {result.get('language')}")
    print(f"Symbols: {result.get('symbols', {})}")
    print(f"Metrics: {result.get('metrics', {})}")
    
    # Should find at least the test_simple function
    functions = result.get('symbols', {}).get('functions', [])
    print(f"\nFound {len(functions)} functions:")
    for func in functions:
        print(f"  - {func.get('name')} at line {func.get('line')}")


if __name__ == "__main__":
    test_simple()