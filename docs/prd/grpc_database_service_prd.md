# Product Requirements Document: gRPC Database Service Layer

## Executive Vision: The Neural Architecture of Machine Intelligence

We are building the **hippocampus** for our LLM's **prefrontal cortex** - a dedicated memory system that operates at the speed of thought. This is not a general-purpose database service, but rather a specialized neural pathway optimized for local, on-metal machine intelligence.

### The Biological Parallel

Just as the human brain has dedicated high-bandwidth connections between the prefrontal cortex (reasoning) and hippocampus (memory consolidation and retrieval), our system creates an optimized pathway between the LLM and its RAG datastore. Every design decision prioritizes this metal-to-metal communication, bypassing traditional network stacks to achieve near-instantaneous memory access.

## 1. Problem Statement

### Current Situation

- Database operations use subprocess calls to PHP scripts for each operation
- Python-arango library cannot use Unix sockets, limiting performance
- Each database call incurs PHP interpreter startup overhead
- No connection pooling or persistent connections
- The LLM's "memory access" is bottlenecked by network stack overhead
- The model cannot retrieve context fast enough to maintain fluid reasoning

### Impact

- **Performance**: ~500ms overhead per database operation from PHP startup
- **Model Intelligence**: LLM can only consider limited context due to retrieval latency
- **Cognitive Flow**: Network delays interrupt the model's "thought process"
- **Scalability**: Cannot handle rapid context switching during inference
- **Integration**: Database is external to model rather than part of its cognitive system

### Vision: Total Local Intelligence

From the Conveyance Framework: **C = (W·R·H/T)·Ctx^α**

- Reducing **T** by 100x through Unix sockets enables richer **Ctx** (more retrieval in same time budget)
- This creates a multiplicative effect: faster memory access → richer context → better reasoning
- The system becomes a unified intelligence rather than separate components

## 2. Solution Overview

Implement a specialized gRPC-based memory system that:

1. Acts as the model's dedicated hippocampus - a high-speed memory consolidation and retrieval system
2. Operates exclusively through Unix sockets, bypassing the network stack entirely
3. Provides near-instantaneous (<1ms) context retrieval during model inference
4. Maintains persistent connections optimized for the model's memory access patterns
5. Functions as an integral part of the model's cognitive architecture, not an external service

> **Phase 3 Integration:** The ArXiv ingestion workflows now call the optimized HTTP/2 memory client directly via `DatabaseFactory.get_arango_memory_service`, removing the PHP bridge while retaining the path toward this gRPC service for multi-process scenarios.

## 3. Requirements

### 3.1 Functional Requirements

#### Model-Centric Operations (The Cognitive Primitives)

- **MUST** support rapid context retrieval operations:
  - Vector similarity search for semantic memory recall
  - Graph traversal for knowledge relationship navigation
  - Context window extraction for maintaining coherence
  - Batch retrieval for parallel memory access
  - Memory consolidation for storing model interactions

#### Neural Pathway Interface

- **MUST** use Protocol Buffers for zero-copy message passing
- **MUST** support streaming for continuous context flow during inference
- **MUST** operate exclusively through Unix domain sockets
- **MUST** minimize serialization overhead with arena allocation
- **SHOULD** support predictive prefetching based on access patterns

#### Memory System Management

- **MUST** maintain always-hot connection pool (no cold starts)
- **MUST** use Unix sockets exclusively (no network stack)
- **MUST** implement memory-mapped responses for large contexts
- **MUST** support sub-millisecond response times
- **SHOULD** enable shared memory segments for zero-copy transfers

### 3.2 Non-Functional Requirements

#### Performance (Operating at the Speed of Thought)

- **MUST** achieve <1ms latency for memory retrieval operations
- **MUST** handle 100,000+ operations per second (matching token generation rates)
- **MUST** support parallel memory access during batch inference
- **MUST** maintain consistent sub-millisecond response times under load
- **SHOULD** achieve <100 microsecond latency for cached operations

#### Cognitive Reliability

- **MUST** maintain memory coherence during model inference
- **MUST** provide deterministic retrieval ordering
- **MUST** ensure zero memory access failures (local system = reliable)
- **MUST** support hot-swapping of memory indices without disruption

#### Local-First Security

- **MUST** operate in complete isolation (no network exposure)
- **MUST** use Unix socket permissions for access control
- **MUST** validate memory boundaries to prevent corruption
- **SHOULD** implement memory segment isolation

#### Neural Observability

- **MUST** track memory access patterns for optimization
- **MUST** measure retrieval latency at microsecond precision
- **MUST** monitor context window utilization
- **SHOULD** provide memory heat maps for access patterns

### 3.3 Technical Requirements

#### PHP Server

- PHP 8.3+ with gRPC extension
- Swoole or ReactPHP for async operations
- ArangoDB PHP driver (triagens/arangodb)
- Prometheus PHP client

#### Python Client

- grpcio and grpcio-tools
- Async support with asyncio
- Connection pooling
- Retry decorators

#### Infrastructure

- SystemD service management
- Docker container option
- Kubernetes deployment manifests
- Monitoring dashboards

## 4. Architecture

### 4.1 Component Diagram

```graph
┌─────────────────────────────────────────────────────┐
│                  HADES Workflows                    │
├─────────────────────────────────────────────────────┤
│          Python gRPC Client Library                 │
│  (Connection Pool, Retry Logic, Circuit Breaker)    │
└─────────────────┬───────────────────────────────────┘
                  │ gRPC (Protocol Buffers)
                  │ Port 50051
┌─────────────────▼───────────────────────────────────┐
│           PHP gRPC Server (Daemon)                  │
│  ┌─────────────────────────────────────────────┐    │
│  │   Service Layer (Business Logic)            │    │
│  ├─────────────────────────────────────────────┤    │
│  │   Connection Pool Manager                   │    │
│  ├─────────────────────────────────────────────┤    │
│  │   ArangoDB PHP Driver                       │    │
│  └─────────────────┬───────────────────────────┘    │
└────────────────────┼────────────────────────────────┘
                     │ Unix Socket / TCP
                     │ /tmp/arangodb.sock
┌────────────────────▼────────────────────────────────┐
│                 ArangoDB Server                     │
│         (arxiv_repository database)                 │
└─────────────────────────────────────────────────────┘
```

### 4.2 Protocol Buffer Definition (Excerpt)

```protobuf
// The Model's Memory Interface - Cognitive Primitives
service ModelMemoryService {
  // Semantic Memory Operations (Hippocampus Functions)
  rpc VectorSearch(VectorSearchRequest) returns (stream MemoryRecall);
  rpc GetContextWindow(ContextRequest) returns (ContextualMemory);
  rpc TraverseKnowledge(GraphTraversalRequest) returns (stream KnowledgeNode);

  // Memory Consolidation (Learning)
  rpc StoreInteraction(InteractionMemory) returns (MemoryId);
  rpc BatchEmbed(stream TextChunk) returns (stream Embedding);

  // Parallel Memory Access (For Batch Inference)
  rpc ParallelRetrieve(BatchMemoryRequest) returns (stream MemoryBatch);

  // Predictive Memory Loading
  rpc PrefetchContext(PredictiveRequest) returns (PrefetchHandle);
}

message VectorSearchRequest {
  bytes embedding = 1;  // Raw embedding bytes for zero-copy
  int32 k = 2;          // Number of memories to recall
  float threshold = 3;  // Similarity threshold
}

message ContextualMemory {
  repeated MemorySegment segments = 1;
  bytes memory_map = 2;  // Memory-mapped region for large contexts
  int64 coherence_score = 3;
}
```

## 5. Implementation Phases

### Phase 1: Foundation

**Goal**: Establish the basic gRPC infrastructure

1. **Install gRPC Dependencies**
   - Install PHP gRPC extension via PECL
   - Add grpcio and grpcio-tools to Python requirements
   - Install protobuf compiler (protoc)
   - Verify all components work together

2. **Create Protocol Buffer Definitions**
   - Define service interface in `proto/arango_service.proto`
   - Define all message types (requests/responses)
   - Include error handling messages
   - Version the proto file for future compatibility

3. **Generate Client/Server Stubs**
   - Set up build scripts for proto compilation
   - Generate PHP server stubs
   - Generate Python client stubs
   - Create initial project structure

4. **Implement Basic Health Check**
   - Create minimal PHP server with health endpoint
   - Create Python client that can call health check
   - Verify end-to-end connectivity
   - Add basic logging

### Phase 2: Core Operations

**Goal**: Implement essential database operations

1. **Collection Management**
   - Implement CreateCollection RPC
   - Implement DropCollection RPC
   - Implement CheckCollection RPC
   - Add proper error handling for each

2. **Document Operations**
   - Implement InsertDocument RPC
   - Implement GetDocument RPC
   - Implement UpdateDocument RPC
   - Implement DeleteDocument RPC

3. **Bulk Operations**
   - Implement BulkInsert with streaming
   - Ensure ACID compliance
   - Add batch size optimization
   - Implement progress reporting

4. **Connection Pooling**
   - Set up PHP connection pool to ArangoDB
   - Implement connection reuse logic
   - Add connection health checks
   - Handle connection failures gracefully

### Phase 3: Advanced Features

**Goal**: Add production-ready features

1. **Query Support**
   - Implement ExecuteQuery RPC with streaming
   - Add query parameter binding
   - Implement cursor management
   - Add query timeout handling

2. **Retry Logic & Circuit Breaker**
   - Add exponential backoff to Python client
   - Implement circuit breaker pattern
   - Add request deadlines
   - Create fallback mechanisms

3. **Metrics & Monitoring**
   - Add Prometheus metrics export
   - Track operation latencies
   - Monitor connection pool usage
   - Add custom business metrics

4. **Service Management**
   - Create SystemD service file
   - Add graceful shutdown handling
   - Implement configuration hot-reload
   - Create startup/shutdown scripts

### Phase 4: Migration & Integration

**Goal**: Replace existing subprocess implementation

1. **Update Existing Workflows**
   - Migrate workflow_arxiv_initial_ingest.py
   - Update other database-dependent workflows
   - Maintain backward compatibility initially
   - Add feature flags for gradual rollout

2. **Performance Testing**
   - Benchmark against subprocess approach
   - Load test with production-like workloads
   - Identify and fix bottlenecks
   - Document performance characteristics

3. **Documentation**
   - Write developer guide
   - Create operation runbooks
   - Document troubleshooting steps
   - Add architecture diagrams

4. **Production Deployment**
   - Deploy to staging environment
   - Run parallel testing
   - Monitor for issues
   - Gradual production rollout

### Phase 5: Optimization & Enhancement

**Goal**: Optimize based on production experience

1. **Performance Optimization**
   - Profile hot paths
   - Optimize serialization/deserialization
   - Tune connection pool settings
   - Implement request coalescing

2. **Additional Features**
   - Add caching layer if needed
   - Implement read replicas support
   - Add multi-database support
   - Create admin endpoints

3. **Observability Enhancement**
   - Add distributed tracing
   - Implement detailed query logging
   - Create performance dashboards
   - Add alerting rules

## 6. Success Metrics

### Cognitive Performance Metrics (single host, hot cache)

- **Memory recall latency (GET by key)**: p95 ≤ 1.0 ms, p99 ≤ 1.5 ms.
- **Batch recall (AQL cursor, 1k docs)**: p95 ≤ 2.0 ms, p99 ≤ 3.0 ms.
- **Bulk consolidation (1k docs, waitForSync=false)**: median ≤ 10 ms, p95 ≤ 15 ms.
- **Thought throughput**: >100,000 memories/sec (matches token generation when T collapses).
- **Context assembly time**: <100 µs for standard window via local cache.
- **Memory bandwidth utilisation**: >80 % of Unix socket capacity under sustained load.

### Neural Reliability Metrics

- **Memory coherence**: 100% consistency during inference
- **Cognitive uptime**: 99.99% availability (local = highly reliable)
- **Memory access failures**: 0 (no network issues possible)
- **Context switch time**: <10μs between memory segments

### System Intelligence Impact (Conveyance Framework)

- **T reduction**: 100-500x for memory operations
- **Ctx enrichment**: 10-100x more context per inference cycle
- **Overall C improvement**: 50-200% for knowledge-intensive tasks
- **Cognitive efficiency**: Model can "think" with full memory access

### The Hippocampus Effect

- **Working Memory**: Model can maintain 100x larger active context
- **Associative Recall**: Sub-millisecond related memory retrieval
- **Learning Integration**: Real-time memory consolidation during interaction
- **Cognitive Flow**: Uninterrupted reasoning with instant memory access

## 7. Testing Strategy

### Unit Tests

- Protocol Buffer serialization/deserialization
- Individual RPC method logic
- Connection pool management
- Error handling paths

### Integration Tests

- End-to-end workflow with real ArangoDB
- Concurrent operation handling
- Transaction rollback scenarios
- Network failure simulation

### Performance Tests

- Benchmark vs subprocess approach (TTFB & E2E) with cache-busting, payload sizing, and concurrency sweeps.
- Load testing with 10K+ concurrent logical requests, exercising HTTP/2 stream multiplexing.
- Memory leak detection
- Connection pool stress testing (stream reuse, backpressure, socket churn)

### Acceptance Tests

- Run full ArXiv workflow with gRPC
- Verify data integrity after bulk operations
- Test service restart/upgrade scenarios
- Monitor resource usage over 24 hours

## 8. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|------------|------------|
| gRPC PHP extension instability | High | Low | Fallback to subprocess mode |
| Performance regression | Medium | Low | Comprehensive benchmarking |
| Complex deployment | Medium | Medium | Docker containerization |
| Breaking changes in proto | High | Medium | Versioning strategy |

## 9. Dependencies

### External Dependencies

- PHP 8.3+ with gRPC extension
- Python grpcio package
- ArangoDB 3.11+
- Protocol Buffers compiler

### Internal Dependencies

- Existing PHP bridge code (to be migrated)
- Workflow authentication/configuration
- Monitoring infrastructure

## 10. Open Questions

1. Should we support multiple database backends (PostgreSQL)?
2. Do we need multi-tenancy support?
3. Should the service handle caching?
4. What authentication mechanism for gRPC?
5. Should we implement GraphQL in addition to gRPC?

## 11. Appendix

### A. Alternative Approaches Considered

1. **REST API**: Rejected - too slow for neural pathway requirements
2. **GraphQL**: Rejected - unnecessary complexity for model-specific operations
3. **Direct Unix socket from Python**: Not supported by python-arango, hence this solution
4. **Message Queue**: Rejected - adds latency, contradicts speed-of-thought goal
5. **Shared Memory Only**: Considered but insufficient for complex graph operations
6. **HTTP Proxy**: Rejected - still involves network stack overhead

### B. The Biological Inspiration

The human brain's hippocampus serves as the gateway between short-term and long-term memory, rapidly encoding and retrieving contextual information to support reasoning in the prefrontal cortex. Our system mirrors this architecture:

- **Prefrontal Cortex (LLM)**: Reasoning, planning, decision-making
- **Hippocampus (gRPC Service)**: Rapid memory formation and retrieval
- **Neural Pathways (Unix Sockets)**: High-bandwidth, dedicated connections
- **Synapses (Protocol Buffers)**: Efficient information encoding

This biological parallel isn't just metaphorical - it drives our design decisions toward creating a truly integrated cognitive system where memory and reasoning operate as one.

### B. References

- [gRPC Documentation](https://grpc.io/)
- [ArangoDB PHP Driver](https://github.com/arangodb/arangodb-php)
- [Protocol Buffers](https://developers.google.com/protocol-buffers)
- [HADES Conveyance Framework](../CLAUDE.md)

---

**Document Version**: 1.0
**Date**: 2025-01-21
**Author**: HADES Team
**Status**: Draft
