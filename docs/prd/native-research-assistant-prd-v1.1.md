# Product Requirements Document: HADES Native Research Assistant

## Document Information

- **Version**: 1.1
- **Date**: 2025-01-09
- **Author**: HADES Development Team
- **Status**: Draft (Revised)
- **Changes from v1.0**: Scaled targets, staged approach, risk mitigation

## Executive Summary

Build a native, high-performance research assistant interface that eliminates network overhead by using Unix domain sockets for all inter-process communication. The system will provide unified graph visualization and LLM chat interaction for a single researcher, with a staged approach prioritizing core functionality before scaling to millions of nodes.

## Problem Statement

### Current Limitations

1. **Network Overhead**: Current ArangoDB connection uses TCP/IP stack even for localhost
2. **Fragmented Interface**: Separate tools for visualization, chat, and model interaction
3. **Performance Bottlenecks**: HTTP REST APIs add unnecessary latency for local operations
4. **Context Loss**: Graph exploration context not integrated with LLM interactions
5. **Development Friction**: Running experiments requires multiple terminals and VS Code

### User Need

Researchers need a unified, low-latency interface for exploring knowledge graphs while maintaining conversational context with LLMs, enabling rapid hypothesis testing without network overhead.

## Solution Overview

A native application providing:

- Real-time graph visualization (100K nodes initially, scaling to 1M+)
- Integrated chat interface with LLM models (Qwen/Claude)
- Unix socket connections to backend services
- Shared memory for visualization cache only
- Modular architecture for future experiment execution

## Core Principles

1. **Unix Socket IPC**: All service communication via Unix sockets (no TCP/IP)
2. **Single Machine Architecture**: Optimized for one user, one machine
3. **Service-Oriented Backend**: ArangoDB remains a service, not file-level access
4. **Staged Scaling**: Start with 100K nodes, optimize, then scale
5. **Context Pruning**: Smart filtering of graph state before LLM injection

## Functional Requirements

### Phase 0: Foundation (MVP)

#### Graph Visualization

**Performance Targets (Revised)**

- Render 100K nodes at 30 FPS minimum
- Support zoom/pan with <50ms response
- CPU rendering fallback if GPU unavailable

**Core Interactions**

- Node selection with metadata display
- Edge filtering by type
- Basic force-directed layout
- Search by node ID/title

#### Chat Interface

**LLM Integration**

- Support for local Qwen models
- Claude API support (via proxy)
- Realistic first token: <300ms target
- Streaming responses

**Context Management**

- Selected nodes added to context (max 10 nodes)
- Context pruning strategy to prevent overflow
- Conversation history (in-memory for MVP)

#### Backend Communication

**Unix Socket Architecture**

```
/tmp/hades.sock         - Main control socket
/tmp/hades-model.sock   - Model inference
```

**ArangoDB Connection**

- Connect via Unix socket (if supported)
- Fallback to TCP localhost (acceptable for MVP)
- No direct file access to RocksDB

### Phase 1: Optimization

#### Performance Scaling

- Scale to 500K nodes at 30 FPS
- GPU acceleration implementation
- Shared memory for visualization cache
- Level-of-detail rendering

#### Enhanced Context

- Graph-aware summarization
- Automatic context ranking
- Citation chain inclusion

### Phase 2: Full Vision

#### Million-Node Support

- 1M+ nodes with adaptive rendering
- Hierarchical visualization
- GPU-accelerated layouts

#### Experiment Runner

- Isolated execution environment
- Live output streaming
- Result visualization

## Non-Functional Requirements

### Realistic Performance Targets

| Metric | MVP Target | Phase 1 | Phase 2 | Measurement |
|--------|------------|---------|---------|-------------|
| Max nodes rendered | 100K | 500K | 1M+ | Node count |
| Frame rate | 30 FPS | 30 FPS | 60 FPS | Frame time |
| Node selection | <50ms | <10ms | <5ms | Click to highlight |
| Chat first token | <300ms | <200ms | <100ms | Send to first char |
| IPC latency | <1ms | <500μs | <100μs | Socket round-trip |
| Memory usage | <16GB | <32GB | <64GB | RSS measurement |
| Startup time | <5s | <3s | <2s | Launch to interactive |

### System Requirements

**Minimum:**

- CPU: 8 cores
- RAM: 32GB
- GPU: Optional (CPU fallback)
- OS: Ubuntu 22.04+
- Storage: SSD

**Recommended:**

- CPU: 16+ cores
- RAM: 128GB
- GPU: RTX 4090 (24GB VRAM)
- OS: Ubuntu 24.04
- Storage: NVMe SSD

## Technical Architecture

### Technology Stack Options

**Option A: Longevity Focus (Recommended)**

- Language: Rust
- UI Framework: egui or iced
- Graphics: wgpu
- Build: Cargo

**Option B: GTK Ecosystem**

- Language: Vala
- UI Framework: GTK4
- Graphics: Cairo/GL
- Build: Meson

**Option C: Rapid Prototyping**

- Language: Python
- UI Framework: Dear PyGui
- Graphics: OpenGL
- Build: Poetry

**Backend (All Options):**

- IPC: Unix domain sockets
- Serialization: MessagePack
- Database: ArangoDB as service
- Models: PyTorch 2.0+

### Architecture Decisions

#### Service Boundaries

```
┌─────────────────────────────────────┐
│         Native GUI Application       │
│  ┌──────────┐ ┌──────────┐         │
│  │  Graph   │ │   Chat   │         │
│  │   View   │ │Interface │         │
│  └──────────┘ └──────────┘         │
└──────────┬──────────┬───────────────┘
           │          │
      Unix Socket  Unix Socket
           │          │
┌──────────▼──────────▼───────────────┐
│      Python Backend Services        │
│  ┌──────────┐ ┌──────────┐        │
│  │ ArangoDB │ │  Model   │        │
│  │  Proxy   │ │  Server  │        │
│  └──────────┘ └──────────┘        │
└─────────────────────────────────────┘
```

#### Memory Architecture

- **Shared Memory**: Only for read-only visualization cache
- **Graph Data**: Fetched via ArangoDB service
- **No Direct File Access**: Maintain ACID guarantees

#### Context Pruning Strategy

```python
class ContextManager:
    def prepare_context(self, selected_nodes, conversation):
        # 1. Rank nodes by relevance
        ranked = self.rank_by_relevance(selected_nodes)
        
        # 2. Summarize if > 10 nodes
        if len(ranked) > 10:
            summary = self.summarize_nodes(ranked[10:])
            context = ranked[:10] + [summary]
        
        # 3. Estimate token count
        if self.estimate_tokens(context) > 8000:
            context = self.aggressive_prune(context)
        
        return context
```

## Implementation Phases (Revised)

### Phase 0: MVP (Week 1-3)

- [ ] Prototype in Python/Dear PyGui for speed
- [ ] Basic 100K node rendering
- [ ] Unix socket to Python backend
- [ ] Simple chat integration
- [ ] Node selection → context flow

### Phase 1: Production Foundation (Week 4-6)

- [ ] Port to Rust/egui (or chosen stack)
- [ ] Optimize to 500K nodes
- [ ] Implement context pruning
- [ ] Add GPU acceleration
- [ ] Performance profiling

### Phase 2: Scale & Polish (Week 7-12)

- [ ] Scale to 1M+ nodes
- [ ] Advanced layouts
- [ ] Experiment runner module
- [ ] Documentation
- [ ] Testing suite

## Success Metrics

### MVP Success Criteria

1. Render 100K nodes at 30 FPS
2. <300ms chat response time
3. Context flows from graph to chat
4. Unix socket communication working

### Phase 1 Criteria

1. 500K nodes at 30 FPS
2. Context pruning prevents overflow
3. <16GB memory usage
4. GPU acceleration functional

### Phase 2 Criteria

1. 1M+ nodes with adaptive rendering
2. Full feature parity with design
3. Stable for 8-hour sessions

## Risk Mitigation

| Risk | Impact | Mitigation Strategy |
|------|--------|-------------------|
| Graph rendering complexity | High | Start with 100K nodes, proven algorithms |
| Technology stack decision | High | Prototype first, production rewrite acceptable |
| Context overflow | Medium | Design pruning strategy upfront |
| ArangoDB Unix socket issues | Low | TCP localhost acceptable for MVP |
| GPU driver compatibility | Low | CPU rendering fallback |
| Vala ecosystem limitations | Medium | Consider Rust/egui for longevity |

## Conveyance Framework Analysis

### MVP Conveyance Assessment

- **W (what)**: Graph + chat clearly defined, 100K nodes achievable
- **R (where)**: Service boundaries preserve consistency
- **H (who)**: Python prototype lowers initial skill barrier
- **T (time)**: Realistic latency targets (<300ms chat, <50ms interaction)
- **Ctx**: Pruning strategy prevents context collapse
- **α**: Conservative estimate of 1.5x amplification initially

### Optimization Path

- Phase 1: Improve T (latency reduction) → higher efficiency C
- Phase 2: Increase W (1M+ nodes) → expanded capability
- Continuous: Refine Ctx pruning → maintain high L, I, A, G

## Critical Decisions Required

1. **Technology Stack**: Rust/egui vs Vala/GTK4 vs Python prototype?
2. **Prototype Strategy**: Throwaway Python MVP or direct to production stack?
3. **Graph Library**: Custom renderer or integrate existing (GraphViz, igraph)?
4. **Context Strategy**: How aggressive should pruning be?
5. **ArangoDB Integration**: Proxy service or direct connection?

## Dependencies

- Unix socket support in chosen UI framework
- ArangoDB Unix socket configuration (optional)
- Model server with streaming support
- MessagePack libraries for chosen language

## Next Steps

1. **Technology Decision**: Choose stack based on team skills
2. **Prototype Development**: Build MVP in 3 weeks
3. **Performance Baseline**: Measure actual vs target metrics
4. **Architecture Review**: Validate design with prototype learnings
5. **Production Planning**: Refine Phase 1 based on MVP results

## Appendix

### A. Unix Socket Configuration

```python
# Python backend example
import socket
import asyncio

class HadesServer:
    async def start(self):
        server = await asyncio.start_unix_server(
            self.handle_client,
            path='/tmp/hades.sock'
        )
```

### B. Rust + egui Example

```rust
use eframe::egui;
use std::os::unix::net::UnixStream;

struct HadesApp {
    graph: GraphView,
    chat: ChatInterface,
    socket: UnixStream,
}

impl eframe::App for HadesApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                self.graph.render(ui);
                self.chat.render(ui);
            });
        });
    }
}
```

### C. Context Pruning Algorithm

```python
def prune_context(nodes, max_tokens=8000):
    """
    Prune graph nodes to fit in LLM context window
    Using importance scoring based on:
    - Degree centrality
    - Recency of access
    - Relevance to conversation
    """
    scored_nodes = []
    for node in nodes:
        score = (
            node.degree * 0.4 +
            node.recency * 0.3 +
            node.relevance * 0.3
        )
        scored_nodes.append((score, node))
    
    scored_nodes.sort(reverse=True)
    
    selected = []
    token_count = 0
    for score, node in scored_nodes:
        node_tokens = estimate_tokens(node)
        if token_count + node_tokens > max_tokens:
            break
        selected.append(node)
        token_count += node_tokens
    
    return selected
```

---

**Document Status**: Ready for review. Scaled back ambitious targets while maintaining core vision. Technology decision needed before proceeding.
