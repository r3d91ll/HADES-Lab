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

## Machine Profile (HADES)
- **Host OS / Kernel:** Ubuntu 24.04 (Linux 6.14.0-29-generic `#29~24.04.1-Ubuntu` x86_64)
- **CPU:** AMD Ryzen Threadripper 7960X (24 cores / 48 threads, AVX512-capable)
- **Memory:** 256 GiB RAM (≈186 GiB available during profiling); 2 GiB swap
- **GPU:** 2× NVIDIA RTX A6000 (GA102GL) + 1× NVIDIA RTX 2000 Ada (AD107GL); `nvidia-smi` unavailable (NVML init fails), but CUDA 12.6 toolchain (`nvcc` 12.6.85) is installed
- **Storage:** Root on `/dev/nvme2n1p2` (1.8 TB, ~1.5 TB free)
- **Python Toolchain:** CPython 3.12.3 (Poetry-managed environment)
