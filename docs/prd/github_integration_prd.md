# Product Requirements Document: GitHub Repository Integration

## Document Information
- **Version**: 1.0
- **Date**: 2025-08-28
- **Status**: Draft
- **Owner**: HADES-Lab Team

## 1. Executive Summary

### 1.1 Purpose
Integrate GitHub repository processing capabilities into HADES-Lab to enable code analysis, embedding generation, and storage of source code alongside academic papers in ArangoDB.

### 1.2 Scope
This PRD covers the integration of GitHub repository cloning, code extraction, embedding generation, and database storage using the existing generic document processing architecture.

## 2. Problem Statement

### 2.1 Current State
- HADES-Lab successfully processes ArXiv papers (11.3 papers/minute)
- Generic document processor architecture supports multiple sources
- No current capability to process code repositories

### 2.2 Problems to Solve
1. Cannot analyze code implementations alongside papers
2. Missing source code in knowledge graph
3. No way to process GitHub repositories at scale
4. Cannot generate embeddings for code files

### 2.3 Opportunity
Enable processing of GitHub repositories to enrich the knowledge graph with practical implementations, supporting future research and analysis capabilities.

## 3. Goals and Success Metrics

### 3.1 Primary Goals
1. **Enable GitHub repository processing** through existing infrastructure
2. **Maintain system reliability** (100% success rate with fallbacks)
3. **Achieve efficient processing** (target: 100+ files/minute)
4. **Reuse existing architecture** (70%+ code reuse from ArXiv pipeline)

### 3.2 Success Metrics
- **Processing Rate**: ≥100 code files per minute
- **Success Rate**: 100% (with graceful degradation)
- **Code Reuse**: ≥70% from existing components
- **Storage Efficiency**: <10KB per file in ArangoDB
- **Memory Usage**: <8GB RAM for typical repositories

## 4. Functional Requirements

### 4.1 Repository Management
- **F1**: Clone GitHub repositories via HTTPS
- **F2**: Support public repositories (authentication optional)
- **F3**: Clean up cloned repositories after processing
- **F4**: Handle various repository sizes (up to 1GB)

### 4.2 Code Extraction
- **F5**: Extract text content from code files
- **F6**: Support multiple programming languages
- **F7**: Filter by file extensions
- **F8**: Skip binary and large files (>10MB)
- **F9**: Preserve file structure metadata

### 4.3 Embedding Generation
- **F10**: Use Jina v4 for code embeddings
- **F11**: Support late chunking for context preservation
- **F12**: Generate embeddings for all text files
- **F13**: Handle files up to 32k tokens

### 4.4 Database Storage
- **F14**: Store in ArangoDB collections with 'github' prefix
- **F15**: Save embeddings and metadata only (not raw code)
- **F16**: Use same schema as ArXiv documents
- **F17**: Support atomic transactions

### 4.5 Integration
- **F18**: Use GenericDocumentProcessor
- **F19**: Compatible with existing monitoring
- **F20**: Configuration via YAML files

## 5. Non-Functional Requirements

### 5.1 Performance
- **P1**: Process 100+ files per minute
- **P2**: Clone repositories in <60 seconds
- **P3**: Support parallel processing (8+ workers)
- **P4**: GPU acceleration for embeddings

### 5.2 Reliability
- **R1**: Handle network failures gracefully
- **R2**: Timeout protection for cloning (5 minutes)
- **R3**: Skip corrupted/unreadable files
- **R4**: Atomic database transactions

### 5.3 Scalability
- **S1**: Process repositories up to 10,000 files
- **S2**: Handle batch processing of multiple repos
- **S3**: Support incremental processing

### 5.4 Security
- **SC1**: No storage of sensitive code
- **SC2**: No execution of cloned code
- **SC3**: Sanitize file paths
- **SC4**: Respect .gitignore patterns

## 6. Technical Architecture

### 6.1 Components

```
GitHubDocumentManager
├── Repository cloning
├── File filtering
└── Task preparation

CodeExtractor
├── Text extraction
├── Metadata extraction
└── Language detection

GenericDocumentProcessor (existing)
├── Extraction phase
├── Embedding phase
└── Storage phase

ArangoDB Collections
├── github_papers (metadata)
├── github_chunks (text chunks)
├── github_embeddings (vectors)
└── github_structures (code structures)
```

### 6.2 Data Flow
1. Clone repository to temporary directory
2. Scan and filter files
3. Create DocumentTask for each file
4. Extract text content
5. Generate embeddings
6. Store in ArangoDB
7. Clean up temporary files

### 6.3 Configuration Schema
```yaml
phases:
  extraction:
    type: 'code'  # Use CodeExtractor
    workers: 8
    timeout_seconds: 10
  
  embedding:
    workers: 4
    gpu_devices: [0, 1]

processing:
  github:
    clone_dir: '/tmp/github_repos'
    cleanup_after_processing: true
    code_extensions: [.py, .js, ...]
    skip_dirs: [.git, node_modules, ...]
```

## 7. User Stories

### 7.1 As a Researcher
- I want to process GitHub repositories so I can analyze code implementations
- I want to search code by semantic similarity
- I want to see which files were successfully processed

### 7.2 As a System Operator
- I want to monitor processing progress
- I want to configure which file types to process
- I want to control resource usage

## 8. Constraints and Assumptions

### 8.1 Constraints
- Public repositories only (no authentication required initially)
- Maximum repository size: 1GB
- Maximum file size: 10MB
- Text files only (no binary processing)

### 8.2 Assumptions
- Git is installed on the system
- Network connectivity to GitHub
- Sufficient disk space for temporary clones
- ArangoDB is configured and accessible

## 9. Dependencies

### 9.1 External Dependencies
- GitHub.com availability
- Git command-line tool
- Network bandwidth for cloning

### 9.2 Internal Dependencies
- GenericDocumentProcessor
- ArangoDB instance
- Jina v4 embeddings
- GPU availability

## 10. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Large repository overwhelming system | High | Medium | Size limits, streaming processing |
| Network failures during clone | Medium | Medium | Retry logic, timeout protection |
| Malicious code in repositories | High | Low | No code execution, sandboxing |
| Storage exhaustion | High | Low | Cleanup after processing |

## 11. Implementation Phases

### Phase 1: Basic Integration (Week 1)
- Port existing GitHub tools
- Create GitHubDocumentManager
- Create CodeExtractor
- Basic testing with small repos

### Phase 2: Production Ready (Week 2)
- Add configuration management
- Implement error handling
- Add monitoring integration
- Performance optimization

### Phase 3: Enhancement (Future)
- Authentication for private repos
- Incremental processing
- Advanced code parsing (Tree-sitter)
- Language-specific extractors

## 12. Testing Requirements

### 12.1 Unit Tests
- Repository URL parsing
- File filtering logic
- Code extraction
- Error handling

### 12.2 Integration Tests
- End-to-end processing
- Database storage verification
- Configuration loading
- Resource cleanup

### 12.3 Performance Tests
- Processing rate measurement
- Memory usage monitoring
- GPU utilization
- Network bandwidth usage

## 13. Documentation Requirements

- User guide for GitHub processing
- Configuration reference
- API documentation
- Troubleshooting guide

## 14. Acceptance Criteria

The feature is complete when:
1. ✅ Can clone and process public GitHub repositories
2. ✅ Generates embeddings for code files
3. ✅ Stores data in ArangoDB successfully
4. ✅ Achieves 100+ files/minute processing rate
5. ✅ Handles errors gracefully
6. ✅ Cleans up resources properly
7. ✅ Integrates with existing monitoring
8. ✅ Documentation is complete

## 15. Future Enhancements

- Private repository support with authentication
- Incremental processing (only changed files)
- Advanced code parsing with Tree-sitter
- Language-specific embedding models
- Commit history analysis
- Dependency graph extraction
- License detection and compliance
- Code quality metrics

## Appendix A: Supported File Types

### Primary Languages
- Python (.py)
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)
- Java (.java)
- C/C++ (.c, .cpp, .h, .hpp)
- Go (.go)
- Rust (.rs)

### Documentation
- Markdown (.md)
- ReStructuredText (.rst)
- Plain text (.txt)

### Configuration
- YAML (.yaml, .yml)
- JSON (.json)
- TOML (.toml)
- XML (.xml)

## Appendix B: Example Usage

```bash
# Process single repository
python github_pipeline.py --repo "https://github.com/owner/repo"

# Process multiple repositories
python github_pipeline.py --repos "repo1" "repo2" "repo3"

# Process from file
python github_pipeline.py --file repositories.txt

# Custom configuration
python github_pipeline.py --config custom.yaml --repo "owner/repo"
```

## Sign-off

- [ ] Product Owner
- [ ] Technical Lead
- [ ] Development Team
- [ ] QA Team