# Product Requirements Document: Go gRPC Memory Service for Local AI

## Executive Summary: A Persistent Memory Layer for Local Models

We are building a high-performance memory service that enables local AI models to access their knowledge base with minimal latency. This Go-based gRPC service provides a persistent, always-running connection between locally-hosted LLMs and ArangoDB via Unix sockets.

> **Phase 3 Update (Jan 2026):** The Python workflows now consume the optimized HTTP/2 memory client directly via `DatabaseFactory.get_arango_memory_service`, using the read-only/read-write proxies delivered in Phase 2. The gRPC surface remains in planning but is no longer required for the ingestion workflows.

### Purpose

This is NOT just a batch processing optimization. It's the foundational memory infrastructure for:

- Local LLMs running on our GPUs
- Real-time RAG during model inference
- Multiple models accessing the same knowledge base
- Future cognitive architectures requiring instant memory access

### Proven Foundation

Our PHP prototype has validated the approach:

- **64% improvement** using Unix sockets (33 → 54 docs/sec)
- Unix sockets bypass network stack successfully
- Interpreter overhead is now the limiting factor

### Realistic Go Projections

Based on compiled vs interpreted performance:

- **3x improvement** over HTTP baseline (33 → 100 docs/sec)
- **Sub-10ms** retrieval for most operations
- **Persistent connections** eliminate connection overhead

## 1. Vision: Persistent Memory Service for Local AI

### The Architecture

```
Local LLM (on GPU) ←→ gRPC ←→ Go Memory Service ←→ Unix Socket ←→ ArangoDB
                        ↑                               ↑
                Multiple models            Persistent connections
                 can connect                 always warm
```

### Design Principles

1. **Always Running**: Daemon service, not a batch processor
2. **Multi-Client**: Any local process can connect via gRPC
3. **Low Latency**: Unix sockets + compiled Go for minimal overhead
4. **Extensible**: Add new primitives without disrupting existing clients
5. **Production-Ready**: Built for 24/7 operation with monitoring

## 2. Problem Analysis

### Current Bottlenecks (Validated by Testing)

| Layer | Latency | Impact |
|-------|---------|--------|
| HTTP/TCP Network Stack | ~15ms | 45% of total latency |
| PHP Interpreter | ~10ms | 30% of total latency |
| Subprocess Spawning | ~5ms | 15% of total latency |
| ArangoDB Query | ~3ms | 10% of total latency |

**Total: ~33ms per operation (30 ops/sec)**

### Realistic Go Improvements

| Layer | Current | Go Service | Realistic Gain |
|-------|---------|------------|----------------|
| Network Stack | ~15ms | <1ms (Unix) | 15x better |
| Language Overhead | ~10ms (PHP) | <1ms (Go) | 10x better |
| Connection Setup | ~5ms | 0ms (pooled) | Eliminated |
| ArangoDB Query | ~3ms | ~3ms | No change |

**Realistic Target: ~10ms per operation (100 ops/sec)**
**Conservative Target: ~15ms per operation (66 ops/sec)**

## 3. Requirements

### 3.1 Core Service Operations

#### Phase 1: Essential Primitives (MVP)

```go
type MemoryService interface {
    // Batch document retrieval - primary use case
    GetDocuments(ids []string) ([]*Document, error)

    // Bulk insert for data ingestion
    BulkInsert(collection string, documents []map[string]interface{}) error

    // Query execution for complex retrievals
    ExecuteQuery(aql string, bindVars map[string]interface{}) (*Cursor, error)
}
```

#### Phase 2: Extended Operations (After MVP Proven)

```go
type ExtendedMemoryService interface {
    // Vector similarity search
    VectorSearch(embedding []float32, k int) ([]*Result, error)

    // Graph traversal
    TraverseGraph(startID string, depth int) (*Graph, error)

    // Store model interactions
    StoreInteraction(interaction *Interaction) error
}
```

### 3.2 Performance Requirements

#### Latency SLOs (single host, hot cache)

- **Single document (GET by key)**: p95 ≤ 1.0 ms, p99 ≤ 1.5 ms.
- **Batch retrieval (AQL cursor, 1k docs)**: p95 ≤ 2.0 ms, p99 ≤ 3.0 ms.
- **Bulk import (1k docs, ≈1 KB each, waitForSync=false)**: median ≤ 10 ms, p95 ≤ 15 ms.
- **Connection overhead**: <0.1 ms via pooled HTTP/2 streams.

Cold-cache, `waitForSync=true`, and multi-host profiles are tracked in a separate "correctness" suite and are not part of the primary SLO envelope.
Validated metrics are archived in `docs/benchmarks/arango_phase4_summary.md`.

#### Throughput Targets

- **Single stream**: 100-200 ops/sec
- **Concurrent clients**: 5-10 simultaneous connections
- **Sustained load**: 24/7 operation without degradation
- **Memory usage**: <200MB for service
- **CPU usage**: <10% idle, <50% under load

### 3.3 Technical Architecture

#### Simple Go Service Structure

```go
// Start simple, optimize later
type MemoryService struct {
    // Connection pool to ArangoDB
    pool *ConnectionPool

    // Basic metrics
    metrics *Metrics

    // Configuration
    config *Config
}

type ConnectionPool struct {
    connections chan *arango.Client
    size        int
    endpoint    string  // unix:///tmp/arangodb.sock via RW proxy
}
```

#### Design Philosophy

- **Start Simple**: Basic pooling and metrics first
- **Measure First**: Profile before optimizing
- **Iterate Based on Data**: Add complexity only when justified
- **Maintainable Code**: Clarity over clever optimizations

> **Neural Process Isolation**: The read-only (`/run/hades/readonly/arangod.sock`) and read-write (`/run/hades/readwrite/arangod.sock`) proxies are the canonical enforcement layer. The Go service (and the Python client today) must speak through these sockets—direct Arango access is disabled in production builds.

#### Unix Socket Optimization

```go
// Direct Unix socket connection
config := arango.Config{
    Endpoints: []string{"unix:///tmp/arangodb.sock"},
    // Connection pooling
    MaxIdleConns:    100,
    MaxOpenConns:    100,
    ConnMaxLifetime: 0, // Never close (local socket)
}
```

## 4. Implementation Design

### 4.1 Protocol Buffer Definition (MVP)

```protobuf
syntax = "proto3";
package memory;

// Start with essential operations only
service MemoryService {
    // Document operations
    rpc GetDocuments(GetDocumentsRequest) returns (GetDocumentsResponse);
    rpc BulkInsert(BulkInsertRequest) returns (BulkInsertResponse);

    // Query operations
    rpc ExecuteQuery(QueryRequest) returns (stream QueryResult);

    // Health check
    rpc HealthCheck(Empty) returns (HealthStatus);
}

message GetDocumentsRequest {
    repeated string document_ids = 1;
    string collection = 2;
}

message BulkInsertRequest {
    string collection = 1;
    repeated string documents_json = 2;  // JSON strings for flexibility
}

message QueryRequest {
    string aql = 1;
    map<string, string> bind_vars = 2;  // JSON encoded values
}
```

### 4.2 Simple Connection Pool

```go
// Basic connection pooling
type ConnectionPool struct {
    connections chan *arango.Client
    config      *arango.ClientConfig
}

func NewConnectionPool(size int) (*ConnectionPool, error) {
    config := &arango.ClientConfig{
        Endpoints: []string{"unix:///tmp/arangodb.sock"},
    }

    pool := &ConnectionPool{
        connections: make(chan *arango.Client, size),
        config:      config,
    }

    // Create initial connections
    for i := 0; i < size; i++ {
        client, err := arango.NewClient(config)
        if err != nil {
            return nil, err
        }
        pool.connections <- client
    }

    return pool, nil
}

func (p *ConnectionPool) Get() *arango.Client {
    return <-p.connections
}

func (p *ConnectionPool) Put(client *arango.Client) {
    p.connections <- client
}
```

### 4.3 Implementation Approach

#### Phase 1: Prove It Works

1. Basic Go service with Unix socket connection
2. Simple connection pool (5-10 connections)
3. Three gRPC methods (GetDocuments, BulkInsert, ExecuteQuery)
4. Basic metrics (request count, latency)

#### Phase 2: Optimize Based on Measurements

1. Profile actual performance
2. Identify real bottlenecks
3. Optimize only what matters
4. Keep code maintainable

#### Phase 3: Production Features

1. Health checks and monitoring
2. Graceful shutdown
3. Configuration management
4. Error recovery

## 5. Implementation Phases

### Phase 1: MVP - Prove Unix Socket Works

**Goal**: Validate Go + ArangoDB + Unix Socket combination

1. **Basic Go Service**

   ```go
   // Minimal viable service
   - Connect to ArangoDB via Unix socket
   - Implement GetDocuments method
   - Test with actual workload
   - Measure performance vs PHP
   ```

2. **If Performance Justified, Continue**
   - Add BulkInsert and ExecuteQuery
   - Add basic connection pooling
   - Create simple Python client

### Phase 2: gRPC Integration

**Goal**: Replace PHP subprocess with gRPC

1. **Add gRPC Layer**
   - Define minimal protobuf
   - Implement service interface
   - Create Python gRPC client

2. **Integration Testing**
   - Update one workflow to use Go service
   - Run side-by-side with PHP
   - Compare performance and stability

### Phase 3: Production Readiness

**Goal**: 24/7 operation capability

1. **Operational Features**
   - SystemD service file
   - Health checks
   - Metrics endpoint
   - Structured logging

2. **Reliability**
   - Connection recovery
   - Graceful shutdown
   - Configuration file
   - Error handling

### Phase 4: Incremental Enhancement

**Goal**: Add features based on actual needs

1. **Monitor Usage Patterns**
   - Which operations are most frequent?
   - Where is latency highest?
   - What new primitives are needed?

2. **Add Features Carefully**
   - Each new primitive must be justified
   - Maintain backward compatibility
   - Keep service focused

## 6. Success Metrics

### Realistic Performance Targets

| Metric | Current (HTTP) | PHP Unix | Go Target | Realistic Gain |
|--------|---------------|----------|-----------|----------------|
| Throughput | 33 docs/s | 54 docs/s | 100 docs/s | 3x |
| Latency p50 | 30ms | 18ms | 10ms | 3x |
| Latency p99 | 100ms | 60ms | 30ms | 3x |
| Memory Usage | 500MB | 400MB | 200MB | 2.5x |
| CPU Usage | 40% | 30% | 20% | 2x |

### Minimum Acceptable Performance

- **Must achieve**: 66 docs/s (2x improvement)
- **Target**: 100 docs/s (3x improvement)
- **Stretch**: 150 docs/s (4.5x improvement)

### Conveyance Framework Impact

From **C = (W·R·H/T)·Ctx^α**:

- **T reduction**: 3x (30ms → 10ms)
- **Result**: 3x improvement in C
- **Real benefit**: Model can retrieve 3x more context per time unit

### Real-World Impact

For 2.8M ArXiv papers:

- **Current**: 24 hours
- **PHP Unix**: 14.5 hours
- **Go Target**: 8 hours (realistic)
- **Go Stretch**: 5 hours (optimistic)

For model inference:

- **Current**: 30ms memory wait
- **Go Target**: 10ms memory wait
- **Result**: 3x faster context retrieval during RAG

## 7. Testing Strategy

### Performance Tests

```go
func BenchmarkVectorSearch(b *testing.B) {
    // Target: <1ms for top-100
    embedding := generateEmbedding()
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        results := pathway.VectorSearch(embedding, 100)
        require.Len(b, results, 100)
    }
}
```

### Load Tests

- Sustain 1,000 ops/sec for 24 hours
- Handle 10,000 concurrent connections
- Recover from connection failures
- No memory leaks under load

### Integration Tests

- Full ArXiv workflow comparison
- Memory access patterns analysis
- Context window assembly timing
- End-to-end latency measurement

## 8. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Go driver limitations | High | Use official driver, contribute fixes |
| Unix socket issues | Medium | Fallback to TCP, but log warning |
| Memory pressure | Medium | Implement backpressure, use circuit breaker |
| Complexity | Low | Keep service focused, resist feature creep |

## 9. Future Enhancements (After Proving MVP)

### Possible Extensions

- **Vector Search**: Add when models need semantic retrieval
- **Graph Operations**: Add when relationship traversal is required
- **Caching Layer**: Add if certain queries are repetitive
- **Batch Optimizations**: Add if batch patterns emerge

### Keep It Focused

This service has one job: provide fast memory access for local AI models. Resist feature creep. Each addition should demonstrably improve model performance.

## 10. Conclusion

This Go gRPC service provides the persistent memory layer that local AI models need for real-time knowledge access. By focusing on simplicity and proven performance gains, we're building production-ready infrastructure that can evolve with our needs.

The PHP prototype proved Unix sockets work (64% improvement). The Go implementation targets a realistic 3x improvement that will meaningfully accelerate both batch processing and real-time inference.

---

**Document Version**: 3.0
**Date**: 2025-01-21
**Author**: HADES Team
**Status**: Ready for Implementation

## Appendix A: Performance Data

### Measured Results (PHP Prototype)

```
HTTP Baseline:
- Documents/sec: 33
- Latency p50: 30ms
- CPU Usage: 40%

PHP + Unix Socket:
- Documents/sec: 54 (64% improvement)
- Latency p50: 18ms
- CPU Usage: 30%
```

### Realistic Go Projections

```
Conservative (2x):
- Documents/sec: 66
- Latency p50: 15ms
- Worth doing

Target (3x):
- Documents/sec: 100
- Latency p50: 10ms
- Significant improvement

Stretch (4x):
- Documents/sec: 132
- Latency p50: 7.5ms
- Excellent outcome
```

### Decision Criteria

Proceed with full implementation if MVP achieves ≥2x improvement (66 docs/sec).
Otherwise, stay with PHP bridge and investigate other bottlenecks.
