"""
Tree-sitter Symbol Extractor
============================

Extracts symbol tables and structural information from code files using Tree-sitter.
This module is specifically designed for the GitHub pipeline to provide rich metadata
for the Jina v4 coding LoRA embeddings.

In Actor-Network Theory terms, this extractor serves as a translation device between
code syntax (tree structures) and semantic understanding (symbol tables). It transforms
the implicit knowledge embedded in code structure into explicit metadata that can
enhance embedding quality.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import hashlib

try:
    import tree_sitter
    from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("Tree-sitter not available - symbol extraction disabled")

logger = logging.getLogger(__name__)


class TreeSitterExtractor:
    """
    Extract symbol tables and code structure using Tree-sitter parsers.
    
    This extractor identifies and categorizes code symbols (functions, classes,
    variables, imports) to provide rich metadata for embedding generation.
    The symbol table acts as a 'boundary object' in ANT terms - maintaining
    coherence across different analytical contexts.
    """
    
    # Language mapping to Tree-sitter language names
    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'c_sharp',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.lua': 'lua',
        '.jl': 'julia',
        '.m': 'objc',
        '.mm': 'objc',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'bash',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sql': 'sql',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'ini',
        '.properties': 'ini',
        '.env': 'ini'
    }
    
    def __init__(self):
        """
        Initialize the TreeSitterExtractor.
        
        Sets up an internal parser cache (self.parsers). If Tree-sitter is not available, logs a warning and leaves the cache empty. If available, attempts to pre-load parsers for common languages (python, javascript, typescript, java, go, rust, cpp, c) and stores any successfully created parsers in the cache; failures to load individual language parsers are logged at debug level.
        """
        self.parsers = {}
        
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available - symbol extraction will be skipped")
            return
        
        # Pre-load common language parsers
        # get_parser(lang) returns a Parser instance already configured for that language
        for lang in ['python', 'javascript', 'typescript', 'java', 'go', 'rust', 'cpp', 'c']:
            try:
                parser = get_parser(lang)
                self.parsers[lang] = parser
                logger.debug(f"Loaded Tree-sitter parser for {lang}")
            except Exception as e:
                # Not all languages may be available
                logger.debug(f"Language {lang} not available: {e}")
    
    def extract_symbols(self, file_path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract symbols, metrics, and top-level structure from a source file using Tree-sitter.
        
        Parses the provided file (or supplied content) with a Tree-sitter parser detected from the file extension and returns a structured result containing language-specific symbol tables, approximate code metrics, and a high-level module structure. If Tree-sitter is unavailable, the language is unsupported, the file cannot be read, or parsing fails, an empty result structure is returned.
        
        Parameters:
            file_path (str): Path to the source file used to detect language and (if `content` is None) read content.
            content (Optional[str]): Optional file contents to parse; when provided, the file is not read from disk.
        
        Returns:
            Dict[str, Any]: A dictionary with keys:
                - symbols: dict of categorized symbols (functions, classes, imports, etc.) per language extractor.
                - metrics: approximate code metrics (lines_of_code, complexity, max_depth, node_count, avg_depth).
                - structure: high-level module structure (top-level functions/classes with line ranges).
                - language: detected Tree-sitter language name.
                - tree_sitter_version: version string of the tree_sitter package or 'unknown'.
        """
        if not TREE_SITTER_AVAILABLE:
            return self._empty_result()
        
        file_path = Path(file_path)
        
        # Detect language from file extension
        language = self._detect_language(file_path)
        if not language:
            logger.debug(f"No Tree-sitter parser for {file_path.suffix}")
            return self._empty_result()
        
        # Load content if not provided
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return self._empty_result()
        
        # Get or load parser
        parser = self._get_parser(language)
        if not parser:
            return self._empty_result()
        
        try:
            # Parse the code
            tree = parser.parse(bytes(content, 'utf-8'))
            
            # Extract symbols based on language
            symbols = self._extract_language_symbols(tree, content, language)
            
            # Calculate metrics
            metrics = self._calculate_metrics(tree, content)
            
            # Extract structure
            structure = self._extract_structure(tree, content, language)
            
            return {
                'symbols': symbols,
                'metrics': metrics,
                'structure': structure,
                'language': language,
                'tree_sitter_version': tree_sitter.__version__ if hasattr(tree_sitter, '__version__') else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path} with Tree-sitter: {e}")
            return self._empty_result()
    
    def _detect_language(self, file_path: Path) -> Optional[str]:
        """
        Return the detected Tree-sitter language name for a file based on its extension.
        
        Checks the file's suffix (case-insensitive) against the extractor's LANGUAGE_MAP and returns the corresponding language identifier (e.g., '.py' -> 'python'). Returns None if the extension is not recognized.
        """
        suffix = file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(suffix)
    
    def _get_parser(self, language: str):
        """Get or create parser for a language."""
        if language in self.parsers:
            return self.parsers[language]
        
        try:
            # get_parser takes the language name and returns a configured parser
            parser = get_parser(language)
            self.parsers[language] = parser
            return parser
        except Exception as e:
            logger.debug(f"Could not get parser for {language}: {e}")
            return None
    
    def _extract_language_symbols(self, tree, content: str, language: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Dispatch to a language-specific symbol extractor and return the collected symbol table.
        
        This calls the appropriate per-language extractor based on `language` and returns
        a dictionary of symbol categories (each a list of symbol dictionaries). Common
        keys produced include: `functions`, `classes`, `imports`, `variables`,
        `exports`, `interfaces`, `enums`, and `types`. For config formats (json/yaml/xml/toml)
        a minimal structural metadata extractor is used; unsupported languages fall back
        to a generic identifier-based extractor.
        
        Parameters:
            language (str): Tree-sitter language identifier (e.g., "python", "javascript",
                "java", "go", "rust", "c", "cpp", "json", "yaml", "xml", "toml").
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Mapping from category name to a list of
            symbol dictionaries produced by the language-specific extractor.
        """
        symbols = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'exports': [],
            'interfaces': [],
            'enums': [],
            'types': []
        }
        
        # Language-specific extractors
        if language == 'python':
            symbols = self._extract_python_symbols(tree.root_node, content)
        elif language in ['javascript', 'typescript']:
            symbols = self._extract_javascript_symbols(tree.root_node, content)
        elif language == 'java':
            symbols = self._extract_java_symbols(tree.root_node, content)
        elif language == 'go':
            symbols = self._extract_go_symbols(tree.root_node, content)
        elif language == 'rust':
            symbols = self._extract_rust_symbols(tree.root_node, content)
        elif language in ['c', 'cpp']:
            symbols = self._extract_c_symbols(tree.root_node, content)
        elif language in ['json', 'yaml', 'xml', 'toml']:
            # For config files, provide minimal structural metadata
            # Let Jina v4's coding LoRA handle semantic understanding
            symbols = self._extract_config_metadata(tree.root_node, content, language)
        else:
            # Generic extraction for other languages
            symbols = self._extract_generic_symbols(tree.root_node, content)
        
        return symbols
    
    def _extract_python_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract Python symbols from a Tree-sitter Python syntax node.
        
        Returns a dictionary of symbol categories discovered anywhere under the provided node:
        - functions: list of functions with keys `name`, `line` (1-based), `scope` (e.g., "module" or "module.Class"),
          `parameters` (list), `decorators` (list), and `docstring` (string or None).
        - classes: list of classes with keys `name`, `line` (1-based), `scope`, `bases` (list of base class names),
          `decorators` (list), and `docstring` (string or None).
        - imports: list of import statements with keys `statement` (source text), `line` (1-based), and `type`
          (`import` or `from_import`).
        - variables: list of module-level constants detected from uppercase identifiers with keys `name`, `line` (1-based),
          `type` (`constant`), and `scope`.
        - decorators: collected decorator nodes (kept for backward compatibility; may be empty).
        
        Parameters:
            node: Tree-sitter node representing a Python module or subtree to inspect.
            content (str): Full source text corresponding to the node (used to extract names, statements, and docstrings).
        
        Notes:
        - The function treats uppercase module-level assignments as constants.
        - Class member functions and nested definitions are recorded with an updated `scope` of the form
          "module.ClassName" when traversing class bodies.
        - Line numbers are 1-based.
        - Docstrings and decorators are obtained via helper methods on the extractor; if none are present the values will be None or empty lists as appropriate.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Symbol table grouped by category.
        """
        symbols = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'decorators': []
        }
        
        def traverse(node, scope='module'):
            """
            Recursively walk a Python Tree-sitter node to collect module-level symbols into the surrounding `symbols` buckets.
            
            This traversal recognizes:
            - function_definition: records name, 1-based line, scope, parameters, decorators, and docstring.
            - class_definition: records name, line, scope, base classes, decorators, and docstring; then walks the class body with an updated scope ("<parent>.<ClassName>") and avoids double-traversing the class children.
            - import_statement and import_from_statement: records the original statement text, line, and import type.
            - assignment at module scope: records identifiers that are all uppercase as probable constants.
            
            Side effects:
            - Mutates the outer-scope `symbols` dictionary by appending found entries to the keys 'functions', 'classes', 'imports', and 'variables'.
            
            Parameters:
            - node: Tree-sitter AST node to visit.
            - scope: dotted scope string (defaults to 'module'); used to qualify symbol scope when collected.
            """
            if node.type == 'function_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    func_info = {
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'scope': scope,
                        'parameters': self._extract_parameters(node, content),
                        'decorators': self._extract_decorators(node, content),
                        'docstring': self._extract_docstring(node, content)
                    }
                    symbols['functions'].append(func_info)
                    
            elif node.type == 'class_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = content[name_node.start_byte:name_node.end_byte]
                    class_info = {
                        'name': class_name,
                        'line': name_node.start_point[0] + 1,
                        'scope': scope,
                        'bases': self._extract_bases(node, content),
                        'decorators': self._extract_decorators(node, content),
                        'docstring': self._extract_docstring(node, content)
                    }
                    symbols['classes'].append(class_info)
                    # Traverse class body with updated scope
                    for child in node.children:
                        traverse(child, f"{scope}.{class_name}")
                    return  # Don't traverse children again
                    
            elif node.type in ['import_statement', 'import_from_statement']:
                import_info = {
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1,
                    'type': 'from_import' if node.type == 'import_from_statement' else 'import'
                }
                symbols['imports'].append(import_info)
                
            elif node.type == 'assignment' and scope == 'module':
                # Global variable assignments
                left = node.child_by_field_name('left')
                if left and left.type == 'identifier':
                    var_name = content[left.start_byte:left.end_byte]
                    if var_name.isupper():  # Likely a constant
                        symbols['variables'].append({
                            'name': var_name,
                            'line': left.start_point[0] + 1,
                            'type': 'constant',
                            'scope': scope
                        })
            
            # Traverse children
            for child in node.children:
                traverse(child, scope)
        
        traverse(node)
        return symbols
    
    def _extract_javascript_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract JavaScript/TypeScript top-level symbols from a Tree-sitter syntax node.
        
        Parses the provided Tree-sitter `node` (typically the root of a parsed JS/TS file) and collects language-level symbols used for indexing and lightweight metadata. Uses the raw file `content` to slice name and statement text and to compute 1-based line numbers.
        
        Parameters:
            node: Tree-sitter syntax node representing the AST subtree to scan.
            content (str): Source text corresponding to the parsed node.
        
        Returns:
            dict: A mapping of symbol categories to lists of symbol records. Categories and typical record fields:
              - 'functions': [{'name': str, 'line': int, 'async': bool, 'generator': bool}, ...]
              - 'classes': [{'name': str, 'line': int, 'extends': Optional[str]}, ...]
              - 'imports': [{'statement': str, 'line': int}, ...]
              - 'exports': [{'statement': str, 'line': int}, ...]
              - 'variables': [] (reserved for future use)
              - 'interfaces': [{'name': str, 'line': int}, ...]
              - 'types': [{'name': str, 'line': int}, ...]
        
        Notes:
            - Line numbers are 1-based.
            - Function records attempt to detect `async` and generator functions; class records include an `extends` entry when a superclass is present.
            - `content` is used for exact source slices (names and full import/export statements).
        """
        symbols = {
            'functions': [],
            'classes': [],
            'imports': [],
            'exports': [],
            'variables': [],
            'interfaces': [],
            'types': []
        }
        
        def traverse(node):
            """
            Recursively traverse a JavaScript/TypeScript Tree-sitter node and collect top-level symbols into the surrounding `symbols` dictionary.
            
            This visitor recognizes function declarations/expressions/arrow functions (records name, 1-based line, async, and generator), class declarations (name, line, extends), import and export statements (source text and line), interface declarations (name and line), and type alias declarations (name and line). It reads identifier text and full statements from the enclosing `content` string and appends entries to the outer-scope `symbols` lists. The function mutates `symbols` and has no return value.
            
            Parameters:
                node: A Tree-sitter AST node to visit.
            """
            if node.type in ['function_declaration', 'function_expression', 'arrow_function']:
                name_node = node.child_by_field_name('name')
                if name_node:
                    # Check for generator by looking for '*' token
                    # Rely on the node's generator attribute for JS/TS
                    is_generator = bool(getattr(node, 'generator', False))
                    
                    symbols['functions'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'async': self._has_child_type(node, 'async'),
                        'generator': is_generator
                    })
                    
            elif node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['classes'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'extends': self._get_extends(node, content)
                    })
                    
            elif node.type in ['import_statement', 'import_specifier']:
                symbols['imports'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })
                
            elif node.type in ['export_statement', 'export_specifier']:
                symbols['exports'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })
                
            elif node.type == 'interface_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['interfaces'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })
                    
            elif node.type == 'type_alias_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['types'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(node)
        return symbols
    
    def _extract_java_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract Java symbols from a Tree-sitter syntax node.
        
        Returns a dictionary of symbol categories discovered in the subtree rooted at `node`. Categories and the shape of each entry:
        - functions: list of {name: str, line: int, modifiers: List[str]} — Java method declarations.
        - classes: list of {name: str, line: int, modifiers: List[str]} — class declarations.
        - imports: list of {statement: str, line: int} — full import statements as source text.
        - interfaces: list of {name: str, line: int} — interface declarations.
        - enums: list of {name: str, line: int} — enum declarations.
        - annotations: list — reserved for annotation declarations (may be empty).
        
        Line numbers are 1-based. Names and statements are sliced from `content` using node byte ranges.
        """
        symbols = {
            'functions': [],
            'classes': [],
            'imports': [],
            'interfaces': [],
            'enums': [],
            'annotations': []
        }
        
        def traverse(node):
            """
            Recursively traverse a Tree-sitter AST node and collect Java symbol declarations into the enclosing `symbols` mapping.
            
            This helper walks the subtree rooted at `node` and appends discovered declarations to the surrounding `symbols` dict:
            - method_declaration -> functions: {'name', 'line', 'modifiers'}
            - class_declaration  -> classes:   {'name', 'line', 'modifiers'}
            - import_declaration -> imports:   {'statement', 'line'}
            - interface_declaration -> interfaces: {'name', 'line'}
            - enum_declaration   -> enums:     {'name', 'line'}
            
            Names and statement text are sliced from the `content` buffer using node byte ranges. Line numbers are 1-based. The function recurses into all child nodes and returns None.
            Parameters:
                node: A Tree-sitter AST node to inspect (recursively).
            """
            if node.type == 'method_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['functions'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'modifiers': self._extract_modifiers(node, content)
                    })
                    
            elif node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['classes'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'modifiers': self._extract_modifiers(node, content)
                    })
                    
            elif node.type == 'import_declaration':
                symbols['imports'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })
                
            elif node.type == 'interface_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['interfaces'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })
                    
            elif node.type == 'enum_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['enums'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(node)
        return symbols
    
    def _extract_go_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract a symbol table from a Tree-sitter Go syntax node.
        
        Parameters:
            node: Tree-sitter node representing a Go source file or subtree. The function walks this node to discover declarations.
            content (str): Original Go source text used to slice names/statements and compute line numbers.
        
        Returns:
            dict: A mapping with the following keys (each value is a list of dicts):
                - 'functions': entries with {'name': str, 'line': int, 'receiver': Optional[str]} for function/method declarations.
                - 'types': entries with {'name': str, 'line': int} for declared types (including structs, interfaces, type aliases).
                - 'imports': entries with {'statement': str, 'line': int} for import declarations (full import text trimmed).
                - 'interfaces': list reserved for interface-like declarations (may be populated via type declarations).
                - 'constants': entries with {'name': str, 'line': int} for constant declarations.
                - 'variables': list reserved for variable declarations (may be populated if detected).
        
        Notes:
            - Line numbers are 1-based.
            - Names and statements are taken by slicing the original `content` using node byte ranges.
        """
        symbols = {
            'functions': [],
            'types': [],
            'imports': [],
            'interfaces': [],
            'constants': [],
            'variables': []
        }
        
        def traverse(node):
            """
            Recursively traverse a Go syntax subtree and append discovered symbols (functions, types, imports, constants)
            to the enclosing `symbols` collection.
            
            This helper inspects node types relevant to Go declarations:
            - `function_declaration`: records function name, 1-based line number, and receiver (via self._extract_receiver).
            - `type_declaration` / `type_spec`: records type names and line numbers.
            - `import_declaration`: records the full import statement text and line number.
            - `const_declaration` / `const_spec`: records constant names and line numbers.
            
            Side effects:
            - Mutates the outer-scope `symbols` dict by appending entries.
            - Reads source text from the outer-scope `content` to extract names and statement text.
            
            Parameters:
                node: A Tree-sitter AST node representing the current subtree to inspect.
            """
            if node.type == 'function_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['functions'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'receiver': self._extract_receiver(node, content)
                    })
                    
            elif node.type == 'type_declaration':
                for spec in node.children:
                    if spec.type == 'type_spec':
                        name_node = spec.child_by_field_name('name')
                        if name_node:
                            symbols['types'].append({
                                'name': content[name_node.start_byte:name_node.end_byte],
                                'line': name_node.start_point[0] + 1
                            })
                            
            elif node.type == 'import_declaration':
                symbols['imports'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })
                
            elif node.type == 'const_declaration':
                for spec in node.children:
                    if spec.type == 'const_spec':
                        name_node = spec.child_by_field_name('name')
                        if name_node:
                            symbols['constants'].append({
                                'name': content[name_node.start_byte:name_node.end_byte],
                                'line': name_node.start_point[0] + 1
                            })
            
            for child in node.children:
                traverse(child)
        
        traverse(node)
        return symbols
    
    def _extract_rust_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract Rust symbols from a Tree-sitter syntax node.
        
        Traverses the given Tree-sitter node subtree and collects high-level Rust declarations into categorized lists:
        - functions: dicts with keys 'name', 'line', 'async'
        - structs: dicts with keys 'name', 'line'
        - enums: dicts with keys 'name', 'line'
        - traits: dicts with keys 'name', 'line'
        - imports: dicts with keys 'statement', 'line'
        - types: (reserved, may remain empty)
        - macros: (reserved, may remain empty)
        
        Parameters:
            node: Tree-sitter node serving as the root of the traversal (typically the parsed file or module node).
            content (str): Original source text used to extract exact name and statement substrings.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Mapping of symbol categories to lists of symbol metadata dictionaries.
        """
        symbols = {
            'functions': [],
            'structs': [],
            'enums': [],
            'traits': [],
            'imports': [],
            'types': [],
            'macros': []
        }
        
        def traverse(node):
            """
            Recursively traverse a Tree-sitter AST node and collect Rust-specific symbols into the enclosing `symbols` mapping.
            
            This visitor appends discovered entries to the surrounding `symbols` dictionary (functions, structs, enums, traits, imports). For each declaration it records the name (extracted from the node's `name` child when present) and the 1-based line number; import declarations record the full statement text and line. The function mutates `symbols` and relies on the outer-scope `content` string and `self._has_child_type` for async detection. No value is returned.
            """
            if node.type == 'function_item':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['functions'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1,
                        'async': self._has_child_type(node, 'async')
                    })
                    
            elif node.type == 'struct_item':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['structs'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })
                    
            elif node.type == 'enum_item':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['enums'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })
                    
            elif node.type == 'trait_item':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['traits'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })
                    
            elif node.type == 'use_declaration':
                symbols['imports'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })
            
            for child in node.children:
                traverse(child)
        
        traverse(node)
        return symbols
    
    def _extract_c_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract C/C++ symbols from a Tree-sitter syntax tree node.
        
        Traverses the provided Tree-sitter node and collects top-level and nested C/C++ declarations:
        - functions: name and 1-based line number (from function declarators)
        - structs: name and line number
        - classes: name and line number
        - includes: full include statement and line number
        - defines: macro name and line number
        - typedefs: (collected via typedef nodes when present)
        
        Parameters:
            node: Tree-sitter AST node to traverse (typically the root of a parsed C/C++ tree).
            content (str): Original source text; used to extract identifier text and statement slices.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: A mapping of symbol categories to lists of symbol records. Each record contains at least 'name' or 'statement' and 'line'.
        """
        symbols = {
            'functions': [],
            'structs': [],
            'classes': [],
            'includes': [],
            'defines': [],
            'typedefs': []
        }
        
        def traverse(node):
            """
            Recursively walk a C/C++ tree-sitter node subtree and collect top-level C/C++ symbols into the outer `symbols` mapping.
            
            This function visits nodes to find function definitions, struct and class declarations, preprocessor includes, and macro defines. For each discovered symbol it appends a dictionary with the symbol name (or full include/define statement) and its 1-based line number into the appropriate list in the surrounding `symbols` variable. It reads text slices from the surrounding `content` and uses the extractor's `_extract_function_name` helper to resolve function names.
            
            Parameters:
                node: A tree-sitter Node representing the current subtree root to traverse.
            
            Returns:
                None — results are recorded by mutating the enclosing `symbols` dictionary.
            """
            if node.type == 'function_definition':
                # Look for the declarator which contains the name
                declarator = node.child_by_field_name('declarator')
                if declarator:
                    name = self._extract_function_name(declarator, content)
                    if name:
                        symbols['functions'].append({
                            'name': name,
                            'line': node.start_point[0] + 1
                        })
                        
            elif node.type == 'struct_specifier':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['structs'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })
                    
            elif node.type == 'class_specifier':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['classes'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })
                    
            elif node.type == 'preproc_include':
                symbols['includes'].append({
                    'statement': content[node.start_byte:node.end_byte].strip(),
                    'line': node.start_point[0] + 1
                })
                
            elif node.type == 'preproc_def':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols['defines'].append({
                        'name': content[name_node.start_byte:name_node.end_byte],
                        'line': name_node.start_point[0] + 1
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(node)
        return symbols
    
    def _extract_config_metadata(self, node, content: str, language: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract minimal structural metadata from a parsed configuration file.
        
        This collects lightweight, format-specific counts and nesting information without
        attempting to interpret semantic meanings of keys or values. Supported formats
        are: "json", "yaml", "xml", and "toml". The function assumes `node` is a
        Tree-sitter root or subtree for the parsed content; reaching this code implies
        parsing succeeded.
        
        Returns:
            A dict with:
              - 'config_type' (str): same as the provided `language`.
              - 'structure_info' (list): a single-entry list containing a dict with:
                  - 'type': 'config_metadata'
                  - 'format': the config format (language)
                  - 'key_count' (int): number of key/property-like nodes found
                  - 'max_nesting_depth' (int): maximum traversal depth encountered
                  - 'is_valid_syntax' (bool): True when parsing reached this extractor
        """
        symbols = {
            'config_type': language,
            'structure_info': []
        }
        
        # Count basic structural elements without interpreting them
        key_count = 0
        max_depth = 0
        
        def traverse(node, depth=0):
            """
            Recursively traverse a Tree-sitter syntax node to count config keys/properties and track maximum nesting depth.
            
            This helper walks the syntax tree rooted at `node`, increments the nonlocal `key_count` when it encounters node types that represent keys/properties for the detected config `language` (json, yaml, xml, toml), and updates the nonlocal `max_depth` to reflect the deepest nesting seen.
            
            Parameters:
                node: A Tree-sitter AST node to traverse.
                depth (int): Current depth in the tree (root starts at 0).
            
            Side effects:
                Modifies the enclosing scope's `key_count` and `max_depth` nonlocal variables.
            """
            nonlocal key_count, max_depth
            max_depth = max(max_depth, depth)
            
            # Count keys/properties without interpreting their meaning
            if language == 'json':
                if node.type in ['pair', 'property']:
                    key_count += 1
            elif language == 'yaml':
                if node.type in ['block_mapping_pair', 'flow_pair']:
                    key_count += 1
            elif language == 'xml':
                if node.type in ['element', 'start_tag']:
                    key_count += 1
            elif language == 'toml':
                if node.type in ['pair', 'table']:
                    key_count += 1
            
            for child in node.children:
                traverse(child, depth + 1)
        
        traverse(node)
        
        # Provide minimal metadata
        symbols['structure_info'] = [{
            'type': 'config_metadata',
            'format': language,
            'key_count': key_count,
            'max_nesting_depth': max_depth,
            'is_valid_syntax': True  # If we got here, it parsed successfully
        }]
        
        return symbols
    
    def _extract_generic_symbols(self, node, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extracts a minimal symbol table from a syntax tree for languages without a dedicated extractor.
        
        This walks the given Tree-sitter `node` subtree and collects identifier occurrences whose text length is greater than two characters. For each identifier the function records its name and 1-based line number (derived from the node's start_point). Identifiers are deduplicated (first occurrence preserved) and the final list is truncated to at most 100 entries. The returned mapping contains two keys:
        - 'identifiers': list of { 'name': str, 'line': int }
        - 'literals': list (currently unused; returned for compatibility with other extractors)
        
        Parameters:
            node: Tree-sitter AST node to traverse (root of the subtree to scan).
            content (str): Source file content used to slice identifier text by node byte offsets.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Symbol categories with collected identifiers and literals.
        """
        symbols = {
            'identifiers': [],
            'literals': []
        }
        
        def traverse(node):
            """
            Recursively walk the given syntax subtree and collect identifier names into the surrounding `symbols['identifiers']` list.
            
            This function visits every node in the subtree rooted at `node`. When it encounters a node of type 'identifier', it extracts the source text using the outer-scope `content` bytes range and, if the identifier length is greater than two characters, appends a dict with keys 'name' and 'line' (1-based line number) to `symbols['identifiers']`. Short identifiers (length ≤ 2) are ignored. The function mutates the outer `symbols` structure and returns None.
            """
            if node.type == 'identifier':
                text = content[node.start_byte:node.end_byte]
                if len(text) > 2:  # Skip short identifiers
                    symbols['identifiers'].append({
                        'name': text,
                        'line': node.start_point[0] + 1
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(node)
        
        # Deduplicate identifiers
        seen = set()
        unique_identifiers = []
        for ident in symbols['identifiers']:
            if ident['name'] not in seen:
                seen.add(ident['name'])
                unique_identifiers.append(ident)
        symbols['identifiers'] = unique_identifiers[:100]  # Limit to top 100
        
        return symbols
    
    def _calculate_metrics(self, tree, content: str) -> Dict[str, Any]:
        """
        Compute simple code metrics from a Tree-sitter syntax tree and the source text.
        
        Returns a dictionary with:
        - lines_of_code (int): number of lines in the provided content.
        - complexity (int): a small cyclomatic-complexity estimate (starts at 1 and increments for control-flow constructs).
        - max_depth (int): maximum AST nesting depth visited (skips nodes inside string/comment nodes).
        - node_count (int): number of AST nodes counted (skips nodes inside string/comment nodes).
        - avg_depth (float): max_depth divided by node_count (guarded against division by zero).
        
        The function uses a language-specific set of control-flow node types based on tree.language (defaults to Python) when estimating complexity. It intentionally ignores nodes that represent strings or comments to avoid inflating metrics.
        """
        lines = content.count('\n') + 1
        
        # Count nodes and calculate depth and complexity
        node_count = 0
        max_depth = 0
        complexity = 1  # Start with 1 for the function/file itself
        
        # Define control flow node types per language
        control_flow_nodes = {
            'python': ['if_statement', 'elif_clause', 'else_clause', 'for_statement', 
                      'while_statement', 'try_statement', 'except_clause', 'with_statement'],
            'javascript': ['if_statement', 'else_clause', 'for_statement', 'for_in_statement',
                          'while_statement', 'do_statement', 'switch_statement', 'case_statement',
                          'catch_clause', 'ternary_expression'],
            'typescript': ['if_statement', 'else_clause', 'for_statement', 'for_in_statement',
                          'while_statement', 'do_statement', 'switch_statement', 'case_statement',
                          'catch_clause', 'ternary_expression'],
            'java': ['if_statement', 'else_clause', 'for_statement', 'enhanced_for_statement',
                    'while_statement', 'do_statement', 'switch_statement', 'case_statement',
                    'catch_clause', 'ternary_expression'],
            'c': ['if_statement', 'else_clause', 'for_statement', 'while_statement',
                 'do_statement', 'switch_statement', 'case_statement'],
            'cpp': ['if_statement', 'else_clause', 'for_statement', 'while_statement',
                   'do_statement', 'switch_statement', 'case_statement', 'catch_clause'],
            'go': ['if_statement', 'else_clause', 'for_statement', 'switch_statement',
                  'case_clause', 'type_switch_statement'],
            'rust': ['if_expression', 'else_clause', 'for_expression', 'while_expression',
                    'loop_expression', 'match_expression', 'match_arm']
        }
        
        # Get language-specific control flow nodes or use a default set
        language = getattr(tree, 'language', 'python')
        cf_nodes = control_flow_nodes.get(language, control_flow_nodes['python'])
        
        def traverse(node, depth=0, in_string_or_comment=False):
            """
            Recursively traverse AST nodes to update node count, maximum depth, and cyclomatic complexity.
            
            This helper visits each node in the Tree-sitter syntax tree, skipping nodes that are inside string or comment constructs. For every non-skipped node it increments the module-level node_count, updates max_depth, and increments complexity when the node's type is listed in `cf_nodes`. Traversal sets `in_string_or_comment` for any descendant nodes under string/comment nodes.
            
            Parameters:
                node: A Tree-sitter AST node to visit.
                depth (int): Current traversal depth (root call typically 0).
                in_string_or_comment (bool): Whether the current node is within a string or comment region.
            
            Side effects:
                Updates the enclosing scope's nonlocal variables `node_count`, `max_depth`, and `complexity`.
            """
            nonlocal node_count, max_depth, complexity
            
            # Skip nodes inside strings and comments
            if node.type in ['string', 'string_literal', 'comment', 'block_comment', 'line_comment']:
                in_string_or_comment = True
            
            if not in_string_or_comment:
                node_count += 1
                max_depth = max(max_depth, depth)
                
                # Increment complexity for control flow nodes
                if node.type in cf_nodes:
                    complexity += 1
            
            for child in node.children:
                traverse(child, depth + 1, in_string_or_comment)
        
        traverse(tree.root_node)
        
        return {
            'lines_of_code': lines,
            'complexity': complexity,
            'max_depth': max_depth,
            'node_count': node_count,
            'avg_depth': max_depth / max(node_count, 1)
        }
    
    def _extract_structure(self, tree, content: str, language: str) -> Dict[str, Any]:
        """
        Return a high-level module structure with top-level functions and classes.
        
        Builds a minimal hierarchical representation for the parsed file: a root node of type "module" with its detected language and a list of top-level children. Each child represents a top-level function or class and includes 1-based start and end line numbers. The extractor recognizes several common Tree-sitter node types for functions and classes across languages.
        
        Returns:
            dict: Structure with keys:
                - type (str): Always "module".
                - language (str): The detected language passed to the function.
                - children (list): List of dicts for top-level members. Each child has:
                    - type (str): "function" or "class".
                    - line (int): 1-based start line of the node.
                    - end_line (int): 1-based end line of the node.
        """
        structure = {
            'type': 'module',
            'language': language,
            'children': []
        }
        
        # Extract top-level structure
        for child in tree.root_node.children:
            if child.type in ['function_definition', 'function_declaration', 'function_item']:
                structure['children'].append({
                    'type': 'function',
                    'line': child.start_point[0] + 1,
                    'end_line': child.end_point[0] + 1
                })
            elif child.type in ['class_definition', 'class_declaration', 'class_specifier']:
                structure['children'].append({
                    'type': 'class',
                    'line': child.start_point[0] + 1,
                    'end_line': child.end_point[0] + 1
                })
        
        return structure
    
    # Helper methods
    def _extract_parameters(self, node, content: str) -> List[str]:
        """
        Return a list of parameter strings extracted from a function/method node.
        
        Searches the node's 'parameters' child and collects text for child nodes of types
        commonly used to represent parameters (e.g., 'identifier', 'typed_parameter',
        'simple_parameter'). Parameter text is trimmed and filtered to exclude bare
        punctuation like parentheses and commas.
        
        Parameters:
            node: Tree-sitter AST node representing a function or method declaration.
            content (str): Full source file content used to slice parameter text by byte offsets.
        
        Returns:
            List[str]: Ordered list of parameter source fragments as they appear in the signature.
        """
        params = []
        param_list = node.child_by_field_name('parameters')
        if param_list:
            for child in param_list.children:
                if child.type in ['identifier', 'typed_parameter', 'simple_parameter']:
                    param_text = content[child.start_byte:child.end_byte].strip()
                    if param_text and param_text not in ['(', ')', ',']:
                        params.append(param_text)
        return params
    
    def _extract_decorators(self, node, content: str) -> List[str]:
        """
        Return a list of Python decorator source strings for the given Tree-sitter node.
        
        Searches the node's direct children for nodes of type 'decorator' and returns the raw source slice for each decorator (including the leading '@' and any arguments). Does not recurse into deeper descendants and does not normalize or parse the decorator text.
        """
        decorators = []
        for child in node.children:
            if child.type == 'decorator':
                dec_text = content[child.start_byte:child.end_byte].strip()
                decorators.append(dec_text)
        return decorators
    
    def _extract_docstring(self, node, content: str) -> Optional[str]:
        """
        Extract the first-string literal docstring from a Tree-sitter node's body.
        
        Given a Tree-sitter node representing a module, class, or function, this returns
        the text of the leading string expression (the usual Python docstring) if present.
        The function inspects the node's `body` field, looks for an initial
        `expression_statement` whose child is a `string` node, slices the original
        source `content` using the string node's byte range, strips common Python
        string prefixes (r, f, fr, rf, etc.) and surrounding quotes (single, double,
        or triple), and returns the trimmed docstring text. Returns None when no
        leading string literal is found.
        
        Parameters:
            node: Tree-sitter AST node for a module/class/function (expected to have a `body` field).
            content (str): The full source text corresponding to the node, used to extract the literal slice.
        
        Returns:
            Optional[str]: The cleaned docstring text, or None if no docstring is present.
        """
        body = node.child_by_field_name('body')
        if body and body.children:
            first_stmt = body.children[0]
            if first_stmt.type == 'expression_statement':
                for child in first_stmt.children:
                    if child.type == 'string':
                        docstring = content[child.start_byte:child.end_byte]
                        
                        # Handle raw string prefixes (r, R, f, F, fr, rf, etc.)
                        prefix_len = 0
                        if docstring.lower().startswith(('r"""', "r'''", 'f"""', "f'''", 
                                                        'fr"""', "fr'''", 'rf"""', "rf'''")):
                            # Handle compound prefixes (fr, rf)
                            if docstring[:2].lower() in ('fr', 'rf'):
                                prefix_len = 2
                            else:
                                prefix_len = 1
                        elif docstring.lower().startswith(('r"', "r'", 'f"', "f'")):
                            prefix_len = 1
                        
                        # Strip prefix and quotes
                        if prefix_len > 0:
                            docstring = docstring[prefix_len:]
                        
                        # Clean up quotes
                        if docstring.startswith('"""') or docstring.startswith("'''"):
                            docstring = docstring[3:-3]
                        elif docstring.startswith('"') or docstring.startswith("'"):
                            docstring = docstring[1:-1]
                        
                        return docstring.strip()
        return None
    
    def _extract_bases(self, node, content: str) -> List[str]:
        """
        Return the list of base class names declared on a Python `class_definition` node.
        
        Scans the node's `superclasses` field and extracts identifier text slices from the original source `content`.
        Returns base class names in source order; returns an empty list if no superclasses are present.
        """
        bases = []
        superclasses = node.child_by_field_name('superclasses')
        if superclasses:
            for child in superclasses.children:
                if child.type == 'identifier':
                    bases.append(content[child.start_byte:child.end_byte])
        return bases
    
    def _has_child_type(self, node, child_type: str) -> bool:
        """
        Return True if the given Tree-sitter node has an immediate child with the specified node type.
        
        Parameters:
            node: A Tree-sitter AST node whose immediate children will be inspected.
            child_type (str): The node type string to match (e.g., "function_definition", "class_definition").
        
        Returns:
            bool: True if any direct child of `node` has type equal to `child_type`, otherwise False.
        
        Notes:
            - Only immediate/direct children are checked; descendants at deeper levels are not considered.
        """
        for child in node.children:
            if child.type == child_type:
                return True
        return False
    
    def _get_extends(self, node, content: str) -> Optional[str]:
        """
        Return the name of the first extended/base class from a JavaScript/TypeScript class heritage clause.
        
        Searches the node for a `heritage` field and, if it contains an `extends_clause`, extracts and returns the clause text with the leading `extends` keyword removed and surrounding whitespace trimmed. Returns None if no heritage or extends clause is present.
        
        Parameters:
            node: Tree-sitter node representing a class declaration or similar construct.
            content (str): Full source text used to slice the extends clause.
        
        Returns:
            Optional[str]: The base class name or expression as a string, or None if not found.
        """
        heritage = node.child_by_field_name('heritage')
        if heritage:
            for child in heritage.children:
                if child.type == 'extends_clause':
                    return content[child.start_byte:child.end_byte].replace('extends', '').strip()
        return None
    
    def _extract_modifiers(self, node, content: str) -> List[str]:
        """
        Extract Java modifiers from a Tree-sitter node.
        
        Given a node that may contain a `modifiers` field, returns the list of modifier tokens
        as they appear in the source (sliced from `content`). Typical modifiers include
        'public', 'private', 'protected', 'static', 'final', etc.
        
        Parameters:
            node: Tree-sitter node expected to have a `modifiers` child field.
            content (str): Full source text used to extract the modifier substrings.
        
        Returns:
            List[str]: Modifier tokens in source order. Empty list if no modifiers are present.
        """
        modifiers = []
        mod_node = node.child_by_field_name('modifiers')
        if mod_node:
            for child in mod_node.children:
                if child.type == 'modifier':
                    modifiers.append(content[child.start_byte:child.end_byte])
        return modifiers
    
    def _extract_receiver(self, node, content: str) -> Optional[str]:
        """
        Extract the Go method receiver declaration from a method node.
        
        Returns the source text for the first receiver parameter (e.g., "(r *MyType)")
        when present and parseable; otherwise returns None.
        
        Parameters:
            node: Tree-sitter node for a Go method or function declaration.
            content (str): File source used to slice the receiver text.
        
        Returns:
            Optional[str]: The raw receiver text or None if not found.
        """
        params = node.child_by_field_name('parameters')
        if params and params.children:
            first_param = params.children[0]
            if first_param.type == 'parameter_list':
                return content[first_param.start_byte:first_param.end_byte]
        return None
    
    def _extract_function_name(self, declarator, content: str) -> Optional[str]:
        """
        Extract the function name from a C/C++ declarator node.
        
        Searches the declarator's children for identifier-like nodes and returns the corresponding source text.
        Handles:
        - plain `identifier` and `field_identifier` nodes,
        - `function_declarator` nodes whose child contains the identifier,
        - `pointer_declarator` nodes by recursing into their children.
        
        Parameters:
            declarator: A Tree-sitter node representing a C/C++ declarator.
            content (str): The full source text from which node byte ranges are sliced.
        
        Returns:
            The function name as it appears in source, or None if no name can be determined.
        """
        if declarator.type == 'function_declarator':
            for child in declarator.children:
                if child.type == 'identifier':
                    return content[child.start_byte:child.end_byte]
                elif child.type == 'field_identifier':
                    return content[child.start_byte:child.end_byte]
        elif declarator.type == 'pointer_declarator':
            # Recursive call for pointer functions
            for child in declarator.children:
                name = self._extract_function_name(child, content)
                if name:
                    return name
        elif declarator.type == 'identifier':
            return content[declarator.start_byte:declarator.end_byte]
        return None
    
    def _empty_result(self) -> Dict[str, Any]:
        """
        Return a standardized empty extraction result.
        
        The result matches the shape produced by extract_symbols when no data can be extracted
        (e.g., unsupported language, parse failure, or missing tree-sitter). Fields:
        
        - symbols: categories for collected symbols (lists are empty).
        - metrics: numeric metrics initialized to zero.
        - structure: empty dict for hierarchical structure.
        - language: None to indicate no detected language.
        
        Returns:
            Dict[str, Any]: A dictionary with keys 'symbols', 'metrics', 'structure', and 'language'.
        """
        return {
            'symbols': {
                'functions': [],
                'classes': [],
                'imports': [],
                'variables': [],
                'exports': []
            },
            'metrics': {
                'lines_of_code': 0,
                'complexity': 0,
                'max_depth': 0,
                'node_count': 0
            },
            'structure': {},
            'language': None
        }
    
    def generate_symbol_hash(self, symbols: Dict[str, List]) -> str:
        """
        Return a stable SHA-256 hex digest representing the provided symbol table.
        
        This produces a deterministic fingerprint of the symbol table suitable for change detection:
        - Sorts categories and, within each category of dict items, sorts by each item's 'name' field.
        - Only the 'name' fields are used (items without a 'name' are ignored).
        - The resulting ordered string is hashed with SHA-256 and returned as a hex string.
        
        Parameters:
            symbols (Dict[str, List]): Mapping of symbol categories (e.g., "functions", "classes")
                to lists of symbol dictionaries. Each symbol dictionary is expected to contain a
                'name' key when applicable.
        
        Returns:
            str: Hex-encoded SHA-256 digest of the stable symbol representation.
        """
        # Create a stable string representation
        symbol_str = ""
        for category in sorted(symbols.keys()):
            symbol_str += f"{category}:"
            items = symbols[category]
            if items and isinstance(items[0], dict):
                names = sorted([item.get('name', '') for item in items if item.get('name')])
                symbol_str += ','.join(names)
            symbol_str += ';'
        
        return hashlib.sha256(symbol_str.encode()).hexdigest()