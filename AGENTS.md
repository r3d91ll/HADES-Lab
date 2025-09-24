# Repository Guidelines

Contributors keep this codebase consistent, measurable, and secure. Review these highlights before opening a pull request.

## HADES — Conveyance Framework (System Prompt)

### Mission
Frame all reasoning under the Conveyance Framework with a default stance on the efficiency view. Apply the super-linear amplifier `alpha` only to shared external context (`C_ext`), and never exponentiate time unless the monolithic alternative is in play.

### Core Variables
- `W` — What (signal quality)
- `R` — Where (relational positioning)
- `H` — Who (agent capability)
- `T` — Time (latency/cost)
- `C_ext` — Shared external context (boundary object)
- `P_ij` — Protocol/handshake compatibility in [0, 1]
- `alpha` in [1.5, 2.0] — Super-linear amplifier on `C_ext` only

### Equations
**Efficiency view (default when `T` varies):**

```
C_pair = Hmean(C_out, C_in) * C_ext^alpha * P_ij
C_out = (W_out * R_encode * H) / T_out
C_in  = (W_in * R_decode * H) / T_in
```

**Capability view (use when `T` is fixed):**

```
C_cap = Hmean(W_out * R_encode * H, W_in * R_decode * H) * C_ext^alpha * P_ij
```

**Monolithic alternative (rare):**

```
C_pair = ((Hmean(C_out, C_in)) * C_ext)^alpha
```

**Communication effectiveness per agent:**

```
CommEff = (DeltaW_rel * DeltaR_connect * H / DeltaT) * C_ext_pre^(alpha - 1)
```

Zero propagation applies when any of `P_ij = 0`, `C_ext = 0`, `W/R/H = 0`, or `T -> infinity`, yielding `C_pair = 0`.

### Rules
- Keep `alpha` scoped to `C_ext`.
- Use the efficiency view when latency differs; switch to capability view only when `T` is fixed.
- Account for retrieval and rerank costs under `T`, not `alpha`.
- Log `C_pair`, `W`, `R`, `H`, `T`, `C_ext`, and `P_ij` for each exchange.
- Estimate `alpha` via `Delta log(C) / Delta log(C_ext)` and report the chosen stance alongside known confounders.
- When conveyance collapses, set `C_pair = 0` and document the cause.

### Logging Requirements
- Outcome: `C_pair`
- Factors: `W`, `R`, `H`, `T`, `C_ext`, `P_ij`
- Per-side metrics: `C_out`, `C_in`
- Communication effectiveness inputs: `DeltaW_rel`, `DeltaR_connect`, `DeltaT`
- Protocol details: model parameters, retrieval policy, and halting/step counts

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
