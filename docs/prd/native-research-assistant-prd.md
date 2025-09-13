# Product Requirements Document: HADES Native Research Assistant

## Document Information
- **Version**: 1.0
- **Date**: 2025-01-09
- **Author**: HADES Development Team
- **Status**: Draft

## Executive Summary

Build a native, high-performance research assistant interface that eliminates network overhead by using Unix domain sockets and shared memory for all inter-process communication. The system will provide unified graph visualization and LLM chat interaction for a single researcher, optimizing for maximum performance on local hardware.

## Problem Statement

### Current Limitations
1. **Network Overhead**: Current ArangoDB connection uses TCP/IP stack even for localhost
2. **Fragmented Interface**: Separate tools for visualization, chat, and model interaction
3. **Performance Bottlenecks**: HTTP REST APIs add unnecessary latency for local operations
4. **Context Loss**: Graph exploration context not integrated with LLM interactions
5. **Development Friction**: Running experiments requires multiple terminals and VS Code

### User Need
Researchers need a unified, zero-latency interface for exploring knowledge graphs while maintaining conversational context with LLMs, enabling rapid hypothesis testing and exploration without network overhead.

## Solution Overview

A native GTK4/Vala application providing:
- Real-time graph visualization of millions of nodes
- Integrated chat interface with LLM models (Qwen/Claude)
- Direct Unix socket connections to all backend services
- Shared memory for graph data access
- Unified experiment execution environment

## Core Principles

1. **Zero Network Stack**: All IPC via Unix sockets or shared memory
2. **Single Machine Architecture**: Optimized for one user, one machine
3. **RAM-First Design**: Leverage 128GB+ RAM for in-memory operations
4. **Native Performance**: Compiled Vala/C code for UI responsiveness
5. **Unified Context**: Graph state informs chat context automatically

## Functional Requirements

### 1. Graph Visualization (P0)

**1.1 Rendering Performance**
- Render 1M+ nodes at 60 FPS using GPU acceleration
- Support zoom levels from full graph to individual nodes
- Hardware-accelerated canvas via GTK4/Cairo/Vulkan

**1.2 Graph Interaction**
- Pan, zoom, rotate with mouse/trackpad
- Node selection with details panel
- Edge filtering by type (citation, similarity, etc.)
- Community detection visualization
- Path finding between nodes

**1.3 Memory Architecture**
- Memory-mapped access to graph data
- Zero-copy rendering pipeline
- Direct access to ArangoDB's RocksDB files if possible

### 2. Chat Interface (P0)

**2.1 LLM Integration**
- Support for local Qwen models
- Support for Claude API (via Unix socket proxy)
- Model hot-swapping without restart
- Streaming responses with < 10ms first token

**2.2 Context Management**
- Selected graph nodes automatically added to context
- Conversation history with graph state snapshots
- Export conversations with graph references

**2.3 Code Execution**
- Execute Python/bash directly from chat
- Display results inline
- Access to HADES framework functions

### 3. Backend Communication (P0)

**3.1 Unix Socket Architecture**
- `/tmp/hades.sock` - Main control socket
- `/tmp/hades-arango.sock` - ArangoDB connection
- `/tmp/hades-model.sock` - Model inference
- `/tmp/hades-exec.sock` - Code execution

**3.2 Shared Memory Segments**
- `/dev/shm/hades_graph` - Graph structure (edges)
- `/dev/shm/hades_embeddings` - Node embeddings
- `/dev/shm/hades_metadata` - Node metadata

**3.3 Protocol**
- MessagePack for structured data
- Direct memory pointers for bulk data
- Async message passing with callbacks

### 4. Model Management (P1)

**4.1 Model Loading**
- Load/unload models on demand
- GPU memory management
- Model warm-up and caching

**4.2 Inference Pipeline**
- Batch processing for efficiency
- Priority queue for interactive requests
- Result caching with TTL

### 5. Experiment Runner (P1)

**5.1 Execution Environment**
- Isolated Python environments
- Resource monitoring (GPU, RAM, CPU)
- Live output streaming

**5.2 Results Visualization**
- Inline plots and charts
- Performance metrics dashboard
- Comparison across runs

## Non-Functional Requirements

### Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Graph render latency | < 16ms (60 FPS) | Frame time |
| Node selection response | < 1ms | Click to highlight |
| Chat first token | < 10ms | Message send to first char |
| IPC message latency | < 100μs | Socket round-trip |
| Memory usage | < 32GB for 10M nodes | RSS measurement |
| Startup time | < 2 seconds | Launch to interactive |

### System Requirements

**Minimum:**
- CPU: 8 cores
- RAM: 32GB
- GPU: RTX 3060 (12GB VRAM)
- OS: Ubuntu 22.04+
- Storage: NVMe SSD

**Recommended:**
- CPU: 16+ cores
- RAM: 128GB
- GPU: RTX 4090 (24GB VRAM)
- OS: Ubuntu 24.04
- Storage: NVMe SSD (2TB+)

## Technical Architecture

### Technology Stack

**Frontend:**
- Language: Vala 0.56+
- UI Framework: GTK4
- Graphics: Cairo/Vulkan
- Build: Meson

**Backend Integration:**
- IPC: Unix domain sockets
- Serialization: MessagePack
- Memory: POSIX shared memory
- Concurrency: GLib async

**Python Backend:**
- Framework: asyncio
- Models: PyTorch 2.0+
- Database: ArangoDB 3.11+
- Memory: multiprocessing.shared_memory

### Component Architecture

```
┌──────────────────────────────────────────────┐
│            Vala/GTK4 Application             │
├──────────────┬──────────────┬────────────────┤
│   GraphView  │  ChatView    │  ExperimentView│
├──────────────┴──────────────┴────────────────┤
│              IPC Layer (Unix Sockets)         │
├───────────────────────────────────────────────┤
│            Shared Memory Manager              │
└──────────────────────────────────────────────┘
                       ↕
┌──────────────────────────────────────────────┐
│            Python Backend Services           │
├──────────────┬──────────────┬────────────────┤
│   ArangoDB   │  Model Server│  Code Executor │
└──────────────┴──────────────┴────────────────┘
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Configure ArangoDB for Unix sockets
- [ ] Create basic Vala/GTK4 application shell
- [ ] Implement Unix socket communication
- [ ] Set up shared memory segments

### Phase 2: Graph Visualization (Week 3-4)
- [ ] Implement GPU-accelerated rendering
- [ ] Add pan/zoom/select interactions
- [ ] Memory-map graph data
- [ ] Optimize for 1M+ nodes

### Phase 3: Chat Integration (Week 5-6)
- [ ] Build chat UI component
- [ ] Connect to model backend
- [ ] Implement context management
- [ ] Add code execution

### Phase 4: Polish & Optimization (Week 7-8)
- [ ] Performance profiling
- [ ] Memory optimization
- [ ] UI polish
- [ ] Documentation

## Success Metrics

1. **Performance**
   - Achieve 60 FPS with 1M nodes rendered
   - Sub-millisecond graph operations
   - 10ms chat response latency

2. **Usability**
   - Single window for all operations
   - Keyboard shortcuts for common tasks
   - Preserve context across sessions

3. **Reliability**
   - Zero network timeouts (no network!)
   - Graceful degradation on resource limits
   - Automatic state recovery

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Vala learning curve | High | Start with simple prototype, iterate |
| ArangoDB Unix socket support | High | Fallback to TCP if needed initially |
| GPU driver compatibility | Medium | Support CPU fallback rendering |
| Memory fragmentation | Medium | Implement memory pooling |
| Model switching overhead | Low | Pre-load models in background |

## Open Questions

1. Should we support multiple graph layouts (force-directed, hierarchical, etc.)?
2. Do we need conversation branching/versioning?
3. Should we implement our own graph storage or keep ArangoDB?
4. How much graph editing capability is needed?
5. Should we support custom model fine-tuning from the UI?

## Dependencies

- ArangoDB must support Unix domain sockets
- GTK4 must be available on target system
- Sufficient GPU memory for models + visualization
- Python backend must implement async socket server

## Alternatives Considered

1. **Electron + Web Stack**: Rejected due to performance overhead
2. **Pure Python (PyQt/Tkinter)**: Rejected due to GIL and rendering limits
3. **Rust + egui**: Considered but Vala has better GTK integration
4. **C++ + Qt**: More complex than Vala for equivalent functionality

## Appendix

### A. Unix Socket Configuration for ArangoDB

```bash
# /etc/arangodb3/arangod.conf
[server]
endpoint = unix:///tmp/arangodb.sock
authentication-unix-sockets = false
```

### B. Shared Memory Layout

```c
struct GraphSharedMemory {
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t* edge_list;  // COO format
    float* node_positions; // x,y,z coordinates
    uint32_t* node_colors; // RGBA
    uint64_t version;      // For cache invalidation
};
```

### C. Message Protocol

```json
{
    "type": "graph_query",
    "id": "uuid",
    "payload": {
        "operation": "neighbors",
        "node_id": "arxiv_papers/2301.00001",
        "depth": 2
    }
}
```

## Review Checklist

- [ ] Performance targets achievable?
- [ ] Architecture supports future scaling?
- [ ] Dependencies clearly identified?
- [ ] Success metrics measurable?
- [ ] Implementation phases realistic?
- [ ] Risk mitigations adequate?

---

**Next Steps:**
1. Review and approve PRD
2. Set up development environment
3. Create GitHub issue from PRD
4. Begin Phase 1 implementation