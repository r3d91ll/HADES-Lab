# Repository Guidelines

Contributors keep this codebase consistent, measurable, and secure. Review these highlights before opening a pull request.

## Project Structure & Module Organization
- `core/` hosts Python source; subpackages such as `core/database` (Arango clients), `core/embedders`, and `core/workflows` drive runtime behavior.
- `core/database/arango/proxies/cmd/{roproxy,rwproxy}` contain Go binaries; keep RO/RW configs in sync.
- Place docs in `docs/` and setup automation in `setup/`; new tests belong under `tests/` mirroring package paths.

## Build, Test, and Development Commands
- `poetry install` — sync the virtualenv with declared dependencies.
- `poetry run python -m compileall core` — fast syntax/bytecode sweep after editing Python modules.
- `poetry run pytest` (optionally `-k pattern`) — run the Python test suite.
- `poetry run ruff check` / `poetry run ruff format` — lint and auto-format Python code.
- `go build ./core/database/arango/proxies/...` — ensure both proxy binaries still compile.

## Coding Style & Naming Conventions
- Python uses 4-space indents, type hints for new public APIs, and docstrings for exported symbols; keep module names lowercase and follow PEP 8 naming.
- Go code must remain gofmt-clean with CamelCase identifiers; proxy allowlists stay alphabetical.
- Avoid hard-coded credentials; rely on `.env` entries such as `ARANGO_RO_SOCKET`.

## Testing Guidelines
- Write `pytest` tests named `test_<feature>.py` with functions `test_<behavior>`.
- Mock ArangoDB/gRPC touchpoints and cover both RO and RW paths of the memory client.
- Target ≥80% docstring/type coverage; regenerate evidence for Conveyance benchmarks when metrics change.

## Commit & Pull Request Guidelines
- Commit summaries are imperative (e.g., `Add HTTP/2 proxy rollback guard`) with optional issue references (`#42`).
- PRs document linked issues, Conveyance Summary, W/R/H/T mapping, performance proof, and compatibility notes; attach benchmark artifacts under `benchmarks/reports/`.

## Security, Configuration, and Conveyance
- Keep secrets out of source; use environment variables and run `setup/verify_storage.py` after deployments.
- Late chunking is mandatory: encode full documents first, then derive context-aware chunks to preserve Conveyance.
- Always report how changes affect W, R, H, T, and context factors (L/I/A/G); if any base factor is zero, declare Conveyance `C = 0` per framework policy.
