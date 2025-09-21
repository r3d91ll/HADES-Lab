# HADES-Lab — Heterogeneous Adaptive Dimensional Embedding System

HADES-Lab transforms long-form technical corpora into a context-preserving knowledge graph stored in ArangoDB. The stack is built around the Conveyance Framework, enforcing late chunking, Unix-socket HTTP/2 transports, and reproducible benchmarks for every workflow.

## Key Capabilities

- Late chunking extraction and embedding pipeline (Docling + Jina/SentenceTransformers) for PDFs, code, and hybrid corpora.
- Hardened ArangoDB HTTP/2 clients with dedicated RO/RW Unix-socket proxies at `/run/hades/readonly/arangod.sock` and `/run/hades/readwrite/arangod.sock`.
- Modular workflows in `core/workflows` with state management, resumable ingest, and Conveyance-aware orchestration.
- Observability and benchmarking via `core/monitoring`, `tests/benchmarks`, and structured reports in `benchmarks/reports/`.
- Legacy prototypes (PostgreSQL metadata, MCP servers) are archived under `Acheron/` and kept out of the production path.

## Repository Layout

```text
HADES-Lab/
├── core/                    # Production modules enforcing the Conveyance pipeline
│   ├── config/              # YAML-first configuration loader and defaults
│   ├── extractors/          # Docling/Tree-sitter powered content extraction (late chunking inputs)
│   ├── embedders/           # Jina & SentenceTransformers backends with chunk orchestration guarantees
│   ├── processors/          # High-level processors wiring extractors, embedders, and storage
│   ├── logging/             # Structured logging helpers emitting Conveyance metrics (writes to `core/logs/`)
│   ├── monitoring/          # Throughput, latency, and progress instrumentation utilities
│   ├── database/arango/     # HTTP/2 clients, Go proxies, PHP Unix bridge
│   ├── workflows/           # Late-chunking orchestrators and CLI entry points
│   └── tools/               # Domain utilities (ArXiv ingestion, RAG helpers, etc.)
├── tests/                   # pytest suites mirroring core packages and workflows
├── docs/                    # Conveyance framework, PRDs, benchmarks, deployment notes
├── setup/                   # Environment bootstrap & storage verification scripts (Postgres removed)
├── dev-utils/               # Operator utilities and ingest monitoring helpers
├── benchmarks/              # Captured benchmark artefacts (`reports/` JSON)
├── AGENTS.md                # Contributor guide and coding conventions
├── Acheron/                 # Archived experiments and legacy implementations (read-only)
└── …
```

## Core Modules Overview

- `core/extractors` isolates document parsing, OCR, and structural enrichment. Docling-backed extractors emit full-document payloads before late chunking, while Tree-sitter utilities surface code symbols.
- `core/embedders` implements the late chunking vector generators (Jina V4, SentenceTransformers variants) and the factory wiring batch sizes, devices, and fp16 modes.
- `core/processors` composes extractors, embedders, and database adapters into reusable document processors consumed by workflows.
- `core/logging` centralizes structured logging and Conveyance-specific log fields so telemetry lands consistently in `core/logs/` or downstream sinks.
- `core/monitoring` provides progress/throughput trackers, performance collectors, and metrics surfaces used by workflows, CLI monitors, and regression tests.
- `core/database/arango` ships the optimized HTTP/2 memory client and Unix-socket proxies; use it instead of legacy HTTP bridges.
- `core/workflows` exposes the CLI entry points (`workflow_pdf.py`, `workflow_arxiv_initial_ingest.py`, etc.) that orchestrate the subsystems end to end.

## Getting Started

1. Install dependencies: `poetry install`.
2. Configure environment: `cp .env.example .env` and set `ARANGO_PASSWORD`, `ARANGO_RO_SOCKET`, `ARANGO_RW_SOCKET`, GPU flags, and any pipeline overrides.
3. Build proxies: `go build ./core/database/arango/proxies/...` (binaries in `cmd/{roproxy,rwproxy}`); run with `LISTEN_SOCKET=/run/hades/readonly/arangod.sock UPSTREAM_SOCKET=/run/arangodb3/arangod.sock go run ./cmd/roproxy` (repeat for RW).
4. Verify tooling: `poetry run python setup/verify_environment.py` (MCP readiness output is legacy and can be ignored) and `poetry run python setup/verify_storage.py` once ArangoDB is reachable.
5. Optional automation: `bash setup/setup_local.sh` performs the checks above and validates GPU availability.

## Build, Test, and Quality Gates

- `poetry run python -m compileall core` — quick syntax sweep after editing Python modules.
- `poetry run ruff check` / `poetry run ruff format` — lint and auto-format to project standards.
- `poetry run pytest [-k pattern]` — execute unit and integration tests aligned with `tests/`.
- `go build ./core/database/arango/proxies/...` — ensure RO/RW proxy binaries remain buildable.
- `poetry run python tests/benchmarks/conveyance_logger.py --help` — emit Conveyance evidence for benchmark runs.

## Arango HTTP/2 Proxies & Clients

The optimized memory client (`core/database/arango/memory_client.py`) negotiates HTTP/2 over Unix sockets and prefers the hardened proxies:

- Read-only socket: `/run/hades/readonly/arangod.sock`
- Read-write socket: `/run/hades/readwrite/arangod.sock`

Override sockets with environment variables `ARANGO_RO_SOCKET`, `ARANGO_RW_SOCKET`, or bypass proxies via `ARANGO_SOCKET`. Proxy binaries accept `LISTEN_SOCKET` and `UPSTREAM_SOCKET` overrides—create directories ahead of time and keep permissions at 0640 (RO) / 0600 (RW). Regression tests and latency sampling live in `tests/benchmarks/arango_connection_test.py`.

## Workflows & Data Pipelines

Late chunking is mandatory across every workflow. Key entry points:

- `core/workflows/workflow_arxiv_initial_ingest.py` — CLI for large-scale ingest (`poetry run python core/workflows/workflow_arxiv_initial_ingest.py --help`) combining Docling extraction, Jina V4 embeddings, and HTTP/2 storage writes.
- `core/workflows/workflow_pdf_batch.py` / `workflow_pdf.py` — reusable batch/single PDF orchestrators with checkpointing support.
- `core/workflows/workflow_arxiv_single_pdf.py` — programmable single-paper Conveyance bundle generator for agent pipelines.

Supporting modules live in `core/extractors`, `core/processors`, `core/embedders`, and `core/database`. Compose new workflows via the factories in those packages while preserving the late chunking guarantee.

## Benchmarks & Monitoring

Benchmark evidence for Arango transports and workflow throughput resides in `docs/benchmarks/` with JSON artefacts tracked under `benchmarks/reports/`. Use `tests/benchmarks/arango_connection_test.py` for transport latency sampling and the scripts in `dev-utils/` (for example `monitor_workflow_logs.py`) for ingest telemetry. After deployments, run `poetry run python setup/verify_storage.py` to confirm collection/index health.

## Documentation & Contributor Resources

- `AGENTS.md` — contributor guidelines, naming conventions, and testing expectations.
- `docs/CONVEYANCE_FRAMEWORK.md` — theoretical baseline for Conveyance and late chunking.
- `docs/prd/` — product requirements and design decisions (completed PRDs live in `docs/prd/completed/`).
- `docs/deploy/` & `setup/` — runbooks for proxy/systemd configuration and environment validation.

Legacy PostgreSQL and MCP artifacts remain only for historical context (`Acheron/`, `CLAUDE.md`) and are not part of the supported architecture.

## License

Licensed under the Apache License 2.0 (`LICENSE`).
