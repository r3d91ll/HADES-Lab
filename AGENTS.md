# Repository Guidelines

## Project Structure & Module Organization

- `core/` holds Python source, including database clients (`core/database`), embedders (`core/embedders`), and workflows (`core/workflows`).
- `core/database/arango/proxies/` contains the Go RO/RW proxy sources (`cmd/roproxy`, `cmd/rwproxy`).
- `docs/` captures product requirements, benchmarks, and deployment runbooks; `setup/` contains automation scripts for local/bootstrap installs.
- Tests live under `tests/` and align with the package structure; use matching module paths when adding new suites.

## Build, Test, and Development Commands

- `poetry install` – create/update the virtualenv with project dependencies.
- `poetry run python -m compileall core` – quick syntax/bytecode sweep; run after touching Python modules.
- `poetry run pytest` – execute the Python test suite; add `-k` to target specific modules.
- `go build ./core/database/arango/proxies/...` – verify the RO/RW proxy binaries build cleanly.
- `poetry run ruff check` / `poetry run ruff format` – lint and auto-format Python code before committing.
- `poetry run python tests/benchmarks/conveyance_logger.py …` – translate benchmark JSON into Conveyance log entries (see docs/benchmarks/arango_phase4_summary.md for examples).

## Coding Style & Naming Conventions

- Python: 4-space indentation, type hints required for new public APIs, and docstrings for modules/classes/functions exposed outside a file.
- Follow PEP 8 naming (snake_case for functions/vars, PascalCase for classes). Keep module names lowercase.
- Go: leverage `gofmt` and conventional CamelCase identifiers; ensure proxy allowlists remain alphabetical.
- Avoid committing secrets; configuration belongs in `.env` or systemd drop-ins (e.g., `ARANGO_PASSWORD`, `ARANGO_RO_SOCKET`).

## Testing Guidelines

- Prefer `pytest`-style tests named `test_<feature>.py` with functions `test_<behavior>`.
- Mock external systems (ArangoDB, gRPC) and exercise both RO and RW paths of the memory client.
- Aim for ≥80 % docstring/type coverage (mirrors CI expectations) and capture evidence for Conveyance calculations when adding benchmarks.

## Commit & Pull Request Guidelines

- Commit messages follow the repository’s style: imperative mood with a concise summary (e.g., “Add HTTP/2 proxy rollback guard”), optionally referencing issues (`#51`).
- Each PR should include: linked issue, Conveyance Summary, W/R/H/T mapping, Performance Evidence, and Tests & Compatibility sections.
- Attach benchmark artefacts under `benchmarks/reports/` when changing performance-sensitive code, and note any required manual login steps (e.g., rebuilding systemd units).

## Conveyance Framework Expectations

- Frame design and review decisions using the efficiency view: `C = (W·R·H / T) · Ctx^α`, where `Ctx = wL·L + wI·I + wA·A + wG·G` and α is typically 1.7.
- Always report which factors improved (W, R, H, T, or Ctx) and include a zero-propagation check (if any base factor is 0 or T → ∞, declare C = 0).
- Benchmark notes must map measurements to W/R/H/T and cite context scores (L/I/A/G) so reviewers can recompute Conveyance.
- Late chunking is mandatory: encode documents once, then derive context-aware chunks. Choose embedders per workload (SentenceTransformers for high throughput, JinaV4 for fidelity) and document the trade-off in PRs.
- Further details on the conveyance framework can be found in docs/CONVEYANCE_FRAMEWORK.md

## Security & Configuration Tips

- Never hard-code credentials. Use environment variables (`ARANGO_RO_SOCKET`, `ARANGO_RW_SOCKET`, `ARANGO_HTTP_BASE_URL`) and keep `.env` out of version control.
- Run `setup/verify_storage.py` after deployments to confirm collection/index health before ingest workflows.

## CRITICAL: Late Chunking Principle

**MANDATORY**: All text chunking MUST use late chunking. Never use naive chunking.

### Why Late Chunking is Required

From the Conveyance Framework: **C = (W·R·H/T)·Ctx^α**

- **Naive chunking** breaks context awareness → Ctx approaches 0 → **C = 0** (zero-propagation)
- **Late chunking** preserves full document context → Ctx remains high → **C is maximized**
