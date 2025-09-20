# ArangoDB Optimized Connection Architecture

## Summary

We evaluated and implemented a custom HTTP/2 client over Unix sockets to replace the PHP subprocess bridge. The new client reduces request latency by more than 10x (GET <1 ms, bulk insert of 1,000 documents ≈6 ms, query ≈0.7 ms) and supports proxy-based process isolation.

## Conveyance Scorecard (Efficiency stance)

| Dimension | Score | Evidence |
|-----------|-------|----------|
| **W** (What) | 0.95 | Full-response verification and payload integrity checks in the Phase 4 harness. |
| **R** (Where) | 0.92 | Dedicated RO/RW proxies on `/run/hades/...` with systemd-managed credentials. |
| **H** (Who) | 0.93 | Optimised Python client with HTTP/2 multiplexing; bounded connection pool. |
| **T** (Time) | 0.97 | Hot-cache p95 latency ≤1.0 ms (see [Phase 4 benchmarks](../benchmarks/arango_phase4_summary.md)). |
| **Ctx** (L/I/A/G) | 0.90 | L=0.90, I=0.90, A=0.90, G=0.90 via observability playbooks and isolation hardening. |

Using α = 1.7, Conveyance = (W·R·H / T) · Ctx^α ≈ **0.70**.

## Phase 1 – Prototype & Benchmark (Completed)
- HTTP/2 client (`core/database/arango/optimized_client.py`) with GET, insert, and query operations.
- Manual benchmark harness under `tests/benchmarks/arango_connection_test.py`.
- Benchmarks vs PHP bridge (GET ~100 ms, insert ~400+ ms, query ~200 ms): new client achieved ~0.4 ms / 6 ms / 0.7 ms respectively.
- Python CLI accepts Basic auth, configurable build, and Unix-socket path for flexibility.

## Phase 2 – Proxies & Isolation (Completed)
- Read-only proxy (`cmd/roproxy`): allows GET/HEAD/OPTIONS and read-only cursor usage.
- Read-write proxy (`cmd/rwproxy`): extends to document/collection writes and imports.
- Both proxies listen on configurable Unix sockets, enforce 0660 permissions, log requests, and leverage HTTP/2 over upstream sockets.
- Proxy benchmarks: GET ≈0.6 ms, insert ≈7 ms, query ≈0.8 ms.

## Phase 3 – Workflow Integration (Completed)
- Added `DatabaseFactory.get_arango_memory_service` with automatic proxy detection and environment overrides.
- Replaced PHP bridge calls in the size-sorted ingestion workflows with direct HTTP/2 operations.
- Updated monitoring and analysis utilities to consume the new memory client API.
- Added collection management helpers so dropping/creating collections no longer shells out to PHP.

## Phase 4 – Validation & Hardening (Completed)
- Expanded benchmark harness
  - Captures both **TTFB** and **E2E** latency, drains entire response bodies.
  - Supports cache-busting (multiple keys, bind variance), payload sizing, `waitForSync`, and concurrency sweeps.
  - Emits JSON reports to backfill cold vs hot curves.
- HTTP/2 enforcement: client now rejects non-H2 responses so regressions surface immediately.
- Proxy boundary formalised as the "neural process isolation" layer; documentation tracks systemd socket ownership and permissions plan.
- Updated success targets (single host, HTTP/2 over `/run/hades/...` proxies with warm cache; raw data in [Phase 4 benchmarks](../benchmarks/arango_phase4_summary.md)):
  - **GET by key**: p95 ≤ 1.0 ms, p99 ≤ 1.5 ms.
  - **AQL cursor (1k docs)**: p95 ≤ 2.0 ms, p99 ≤ 3.0 ms.
  - **Bulk insert 1k docs (~1 KB, waitForSync=false)**: median ≤ 10 ms, p95 ≤ 15 ms.
  - Benchmarks run under defined CPU budgets with concurrency sweeps.
- Observability checklist: client-side timing buckets, in-flight stream counters, queue depth, and server-side WAL/cache metrics tied together via `x-hades-trace`.
- Hot- and cold-cache benchmark results captured under `benchmarks/reports/` and summarised in `docs/benchmarks/arango_phase4_summary.md`.
- Systemd socket/service units manage the RO/RW proxies under `/run/hades/...`, ensuring they restart with ArangoDB (`docs/deploy/arango_proxy_systemd.md`).

### Verification Checklist
- Baseline (hot cache) runs recorded for GET, query, and insert with SLO comparisons.
- Cold-cache suite executed after Arango restart to capture correctness + sync-write latency.
- Concurrency sweep (1, 8, 32, 128) to surface head-of-line blocking.
- Payload sweep at 1 KB, 4 KB, 16 KB documents for bulk inserts.
- `waitForSync=true` benchmark reported alongside default asynchronous profile.
- JSON artefacts stored for regression diffing (see `benchmarks/reports/`).

## Beyond Phase 4
- Implement streaming cursor support and adaptive connection pooling once workload patterns justify it.
- Publish systemd/socket deployment runbooks for the RO/RW proxy pair and peer-credential enforcement.
- Investigate optional RAM caching for high-frequency hot sets if latency margins tighten further.

## Meta

- **Version:** 2025.09.20
- **Last updated:** 2025-09-20
