# GitHub Repository Processing Tool

## Overview

The GitHub tool processes repositories by cloning, extracting code/documentation, generating embeddings, and storing everything in a graph-based ArangoDB structure. It features Tree-sitter integration for symbol extraction and leverages Jina v4's coding LoRA for semantic understanding.

## Key Features

- **Graph-Based Storage**: Repository → File → Chunk → Embedding relationships
- **Tree-sitter Symbol Extraction**: AST-based extraction of functions, classes, imports
- **Jina v4 Coding LoRA**: Specialized embeddings for code understanding
- **Config File Support**: YAML, JSON, TOML, XML with minimal metadata extraction
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C/C++, Go, Rust

## Architecture

### Graph Structure

```
github_repositories (vertex)
    ↓ [github_repo_files edge]
github_papers (vertex) - file metadata
    ↓ [github_has_chunks edge]  
github_chunks (vertex) - text chunks
    ↓ [github_has_embeddings edge]
github_embeddings (vertex) - Jina v4 vectors
```

### Processing Pipeline

1. **Clone Repository**: Clone to temporary directory
2. **Extract Files**: Use CodeExtractor with Tree-sitter
3. **Generate Embeddings**: Jina v4 with coding LoRA
4. **Store in Graph**: Create vertices and edges
5. **Cleanup**: Remove temporary files

## Installation

```bash
# Install Tree-sitter parsers
pip install tree-sitter==0.20.4
pip install tree-sitter-languages

# Setup graph collections (first time only)
cd tools/github/
python setup_github_graph.py
```

## Usage

### Basic Repository Processing

```bash
# Process a single repository
python github_pipeline_manager.py --repo "owner/repo"

# Examples
python github_pipeline_manager.py --repo "dav/word2vec"
python github_pipeline_manager.py --repo "facebook/react"
```

### Configuration

Create a config file `configs/github_simple.yaml`:

```yaml
pipeline:
  name: "github_simple"
  version: "1.0.0"
  
processing:
  github:
    clone_dir: "/tmp/github_repos"
    cleanup_after_processing: true
    code_extensions: [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs"]
    doc_extensions: [".md", ".rst", ".txt"]
    config_extensions: [".yaml", ".yml", ".json", ".toml", ".xml"]
    skip_dirs: [".git", "node_modules", "__pycache__", "dist", "build"]
    max_file_size_mb: 10

phases:
  extraction:
    type: "code"
    workers: 8
    timeout_seconds: 10
    use_tree_sitter: true
    
  embedding:
    workers: 4
    gpu_devices: [0, 1]
    batch_size: 16
```

### Testing Tree-sitter Integration

```bash
# Test symbol extraction
python test_treesitter_simple.py

# Output shows extracted symbols:
# Functions: ['main', 'helper_function']
# Classes: ['MyClass', 'BaseClass']
# Imports: ['os', 'json', 'numpy']
```

## Querying the Graph

### ArangoDB Queries

```aql
// Find all files in a repository
FOR v, e, p IN 1..1 OUTBOUND 'github_repositories/owner_repo'
  GRAPH 'github_graph'
  FILTER IS_SAME_COLLECTION('github_papers', v)
  RETURN v.path

// Find all embeddings for a repository
FOR v, e, p IN 1..3 OUTBOUND 'github_repositories/owner_repo'
  GRAPH 'github_graph'
  FILTER IS_SAME_COLLECTION('github_embeddings', v)
  RETURN v

// Find similar code across repositories
FOR embedding IN github_embeddings
  FILTER embedding.similarity_score > 0.85
  LET repo_path = (
    FOR v, e, p IN 1..3 INBOUND embedding._id
      GRAPH 'github_graph'
      FILTER IS_SAME_COLLECTION('github_repositories', v)
      RETURN v.full_name
  )
  RETURN {
    file: embedding.source_file,
    repository: FIRST(repo_path),
    score: embedding.similarity_score
  }

// Theory-practice bridge detection
FOR paper IN arxiv_embeddings
  FOR code IN github_embeddings
    LET similarity = DOT(paper.embedding, code.embedding)
    FILTER similarity > 0.8
    RETURN {
      paper: paper.title,
      code: code.source_file,
      similarity: similarity
    }
```

## Tree-sitter Symbol Extraction

### Supported Languages

- **Python**: Functions, classes, imports, decorators
- **JavaScript/TypeScript**: Functions, classes, imports, exports
- **Java**: Classes, methods, imports, interfaces
- **C/C++**: Functions, structs, includes, classes
- **Go**: Functions, structs, imports, interfaces
- **Rust**: Functions, structs, imports, traits

### Config File Handling

For configuration files (YAML, JSON, TOML, XML), we extract minimal metadata:
- File format
- Key count
- Maximum nesting depth
- Syntax validity

**Important**: We don't interpret semantic meaning - that's Jina v4's job with its coding LoRA.

### Example Symbol Extraction

```python
# Input Python file
import numpy as np
from sklearn import metrics

class ModelEvaluator:
    def __init__(self):
        self.results = []
    
    def evaluate(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

# Extracted symbols:
{
  "imports": ["numpy", "sklearn.metrics"],
  "classes": [
    {
      "name": "ModelEvaluator",
      "line": 4,
      "methods": ["__init__", "evaluate"]
    }
  ],
  "functions": []
}
```

## Performance Metrics

- **Processing Rate**: ~100 files/minute
- **Tree-sitter Overhead**: ~0.1s per file
- **Embedding Generation**: ~8 files/second with GPU
- **Graph Storage**: <1ms per relationship
- **Memory Usage**: <4GB for typical repositories

## Troubleshooting

### Common Issues

1. **Tree-sitter version error**
   ```bash
   # Solution: Use specific version
   pip install tree-sitter==0.20.4
   ```

2. **Graph collections not found**
   ```bash
   # Solution: Setup collections first
   python setup_github_graph.py
   ```

3. **Repository too large**
   ```bash
   # Solution: Adjust max_file_size_mb in config
   # Or process in batches
   ```

4. **GPU memory issues**
   ```bash
   # Solution: Reduce batch_size in config
   # Or use fewer GPU workers
   ```

## Architecture Decision Records

### ADR-006: Let Jina Handle Config Semantics

**Decision**: Provide minimal metadata for config files, let Jina v4's coding LoRA handle semantic understanding.

**Rationale**: Jina v4 is specifically trained to understand config file semantics. Duplicating this capability would be redundant and less effective.

### ADR-007: Graph-Based Storage

**Decision**: Use graph structure with vertices and edges instead of flat collections.

**Rationale**: Enables both repository distinction and cross-repository analysis for theory-practice bridge detection.

## Future Enhancements

- [ ] Private repository support with authentication
- [ ] Incremental updates (only changed files)
- [ ] Commit history analysis
- [ ] Dependency graph extraction
- [ ] Code quality metrics
- [ ] License detection

## Related Documentation

- [GitHub Integration PRD](../../docs/prd/github_integration_prd.md)
- [Generic Document Processor](../../core/processors/README.md)
- [Tree-sitter Extractor](../../core/framework/extractors/tree_sitter_extractor.py)