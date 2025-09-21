# vLLM Model Manager Service

## Summary

Provision a local vLLM-based model-serving layer that exposes an OpenAI-compatible, function-calling API over AF_UNIX sockets. The service loads Qwen/Qwen3-Coder-30B-A3B-Instruct from the default Hugging Face cache (`~/.cache/huggingface/modules/`) in development; production deployments pin artifacts under `/var/cache/huggingface/...` with `0700` permissions. The runtime enforces tool schemas defined by the virtual context PRD and delivers low-latency inference tightly coupled to the ArangoDB-backed memory system.

## Conveyance Scorecard (Efficiency stance)

| Dimension | Score | Evidence |
|-----------|-------|----------|
| **W** (What) | 0.84 | Function-calling runtime that emits tool_calls consumed by the memory executor enables agents to edit/retrieve context autonomously. |
| **R** (Where) | 0.86 | AF_UNIX-only API hosted on local metal, dedicated GPU, and explicit model vault path. |
| **H** (Who) | 0.88 | vLLM scheduler w/ dynamic batching + heartbeat backpressure integration; operator runbooks for finetuning. |
| **T** (Time) | 0.82 | P99 end-to-end latency ≤900 ms @ 2k input + 512 output tokens on dual RTX A6000 48GB GPUs (fp16, AF_UNIX transport). |
| **Ctx** (L/I/A/G) | 0.87 | L=0.90 (tool schema parity), I=0.86 (observability stack), A=0.88 (role-based AF_UNIX permissions), G=0.82 (GPU resource isolation). |

Using α = 1.7, Conveyance ≈ **0.61**. Zero-propagation: if vLLM loses AF_UNIX binding or function schema support, C = 0.

## Problem Statement

We need a deterministic, on-host LLM runtime that can execute the MemGPT-style tool schema against the Arango-backed memory service without incurring network overhead or external dependencies. The runtime must load Qwen3-Coder-30B-A3B-Instruct, expose OpenAI-compatible APIs for downstream clients (Cline, internal agents), and honour AF_UNIX-only access to align with our security model.

## Goals

- Serve Qwen/Qwen3-Coder-30B-A3B-Instruct via vLLM on local GPU(s) with AF_UNIX-only API endpoints.
- Provide an OpenAI-compatible REST API and WebSocket streaming interface restricted to Unix sockets for agents and tooling.
- Support function calling with schemas defined in `docs/prd/memgpt_virtual_context_arango_prd.md`; enforce tool argument validation and heartbeat flags.
- Manage model artifacts; development uses `~/.cache/huggingface/modules/`, production uses `/var/cache/huggingface/...` with `0700` permissions, and keep disk usage under control.
- Supply observability (latency, throughput, queue depth) and hook into Conveyance telemetry.

## Non-Goals

- Multi-host or horizontally scaled vLLM clusters (single-node only).
- UI/chat front-end (covered by separate PRD).
- Model fine-tuning pipelines (captured in future work).

## Stakeholders & Users

- Memory service developers integrating tool schemas.
- Agent runtime maintainers (Cline, internal orchestrators).
- Platform ops managing GPU capacity and OTA updates.

## Architecture Overview

1. **vLLM Runtime** loads Qwen3-Coder-30B-A3B-Instruct into configured GPU(s) using paged attention.
2. **API Layer (FastAPI/Uvicorn)** exposes OpenAI-compatible REST + streaming endpoints bound to AF_UNIX sockets (`http+unix://`).
3. **Controller** maps tool schemas, heartbeat parameters, and manages per-session state (temperature, stop sequences, etc.).
4. **Ingress Proxy** (optional) provides RBAC, auth, and request shaping for internal clients via AF_UNIX sockets.
5. **Telemetry Exporter** publishes Prometheus metrics and structured logs; surfaces Conveyance factors with request IDs tied to memory service interactions.

### Transport & Binding

- All API traffic uses AF_UNIX sockets under `/run/hades/vllm/`. No TCP/SSL listeners in production.
- Launch the vLLM OpenAI server via Uvicorn with `--uds /run/hades/vllm/api.sock`; directories `0700`, sockets `0660` (owner `hades`, group `hades-agents`). Example invocation: `uvicorn vllm.entrypoints.openai.api_server:app --uds /run/hades/vllm/api.sock`.
- Startup/CI guard fails if `ss -lntp` shows `python`/`uvicorn` bound to TCP, if `stat -c %a /run/hades/vllm` ≠ `700`, or if the socket lacks `660` permissions.

### Model Storage & Warmup

- Model weights pulled via Hugging Face into `~/.cache/huggingface/modules/transformers_modules/Qwen/Qwen3-Coder-30B-A3B-Instruct/`. In production, relocate artifacts to `/var/cache/huggingface/transformers_modules/...` with `0700` perms; the home-directory cache is for developer workflows only.
- Validate disk space ≥ model size + 20 % headroom.
- On startup, pre-warm attention cache and verify tokenizer/model revisions.
- Net namespace must deny outbound traffic; deployments preload or mirror Hugging Face artifacts so startup succeeds without WAN access.
- Version mismatch between cached model/tokenizer and expected revision blocks startup until reconciled (fail fast, no silent downgrade).

### Host Environment Baseline

- Motherboard: ASRock TRX50 WS; CPU: AMD Threadripper 7960X (24c/48t) with 256 GB ECC RDIMM.
- Storage tiers: ZFS pools (`bulk-store`, `dbpool`, `fastpool`) across NVMe RAID and HDD RAIDZ1; ensure `/var/cache/huggingface` resides on low-latency NVMe (`dbpool`).
- GPUs: 2× NVIDIA RTX A6000 48 GB (serving) + 1× RTX 2000 Ada 16 GB (display/light ML). PCIe lanes fully populated with NVMe drives.
- OS: Linux Mint 22.2 with ZFS; kernel tuned for pinned memory and GPU workloads.
- This baseline defines the primary production profile; alternate hardware (A100, L40S, etc.) follow the capacity plan for adjusted limits and SLOs.

## Data & Config

- `config/vllm.json`: model path, dtype (float16), tensor parallel degree, `max_context_tokens=8192`, `max_output_tokens=2048`, `tokens_in_flight_limit` (per capacity plan), and `trust_remote_code=false` (enable only if upstream model requires it).
- `policies/tools.yaml`: tool schemas (synced from memory PRD), heartbeat settings (`heartbeat_rps_limit=3`, `heartbeat_cooldown_ms=750`, default `max_chain_depth=8`, `max_chain_duration_ms=60000` mirrored from memory PRD). LLM Platform owns tuning; Platform Reliability owns dashboards and alerts.
- `auth/clients.yaml`: Unix user/group mapping to API keys, rate limits. Auth files are provisioned via systemd drop-ins or secret manager mounts and remain outside version control; CI enforces no secrets in the repository.

## API Contract

- **REST**: `POST /v1/chat/completions` with OpenAI schema, `model` fixed to `qwen3-coder-30b-a3b-instruct`.
- **Function calling**: `functions` + `function_call`. Responses contain `choices[].message.tool_calls` matching schema.
- vLLM validates schemas and emits tool_calls only; execution of tools (memory reads/writes, heartbeat chaining) occurs in the memory executor per `memgpt_virtual_context_arango_prd.md`.
- On startup the server loads tool schemas (`store_core`, `fetch_core`, `append_fifo`, `evict_fifo`, `write_recall`, `search_recall`, `search_archival`, `ingest_archival`, `record_heartbeat`) and rejects drift; mismatches fail fast with logged errors.
- **Streaming**: `POST /v1/chat/completions` with `stream=true`; server-sent events over AF_UNIX (Uvicorn supports `text/event-stream`).
- Streaming delivers token deltas; tool_call payloads are emitted once the JSON is complete (OpenAI-compatible behaviour).
- **Admin**: `GET /health`, `GET /metrics` (Prometheus), optional `POST /reload` (disabled by default).
- Clients pass `X-Request-ID` and `X-Conveyance-Session-ID`; server echoes them in responses and telemetry.
- Acceptance: absence of echoed IDs triggers alert and fails CI/startup checks.

### Constraints

- Total input+output tokens must be ≤8192; max output tokens per request default 2048. Requests exceeding limits return HTTP 400 with `TOKEN_BUDGET_EXCEEDED`.
- Acceptance: requests exceeding token limits must return HTTP 400 with `{ "error": "TOKEN_BUDGET_EXCEEDED" }` and echo `X-Request-ID` / `X-Conveyance-Session-ID` response headers.
- Max parallel requests defaults to 16 (dynamic batching) on dual RTX A6000 48GB GPUs; capacity plan documents overrides per hardware profile, but guard rails enforce the configured limit.
- Completion inference timeout: 45 s (tunable per hardware tier); tool execution timeouts remain the responsibility of the memory service.

## Performance Targets

- P95 latency ≤600 ms, P99 ≤900 ms for 2k input + 512 output tokens using continuous batching on dual RTX A6000 48GB GPUs (fp16).
- Report prefill vs. decode tokens/sec (Prometheus counters) and dynamic batching stats; track `vllm_batch_size` p50/p95/p99.
- Throughput ≥30 tokens/sec per request on RTX A6000 pair (fp16). L40S or fallback hardware publish revised SLOs in the capacity plan.
- Warm start time ≤120 s (model load + cache prime).
- GPU OOM or capacity saturation triggers 503 with `Retry-After`; optional feature flag enables failover to quantized 14B model with relaxed SLOs.

## Telemetry & Observability

- Metrics: `vllm_request_count`, `vllm_token_latency_ms`, `vllm_gpu_mem_bytes`, `vllm_gpu_memory_utilization`, `vllm_batch_size`, `vllm_queue_depth`, prefill vs. decode tokens/sec.
- Logs: request ID, `conveyance_session_id` (from header `X-Conveyance-Session-ID`), user, prompt tokens, completion tokens, tool calls, heartbeat loops.
- Traces: OTLP spans propagate `request_id` (`X-Request-ID`) and `conveyance_session_id` to align with memory service traces for end-to-end latency analysis.
- Acceptance: telemetry pipelines must record both IDs; missing IDs raise alerts.
- Alerts: queue depth, GPU OOM, heartbeat cooldown activation >5/min, AF_UNIX socket missing, and loss of telemetry correlation fields; thresholds sourced from the capacity plan for each hardware tier.

## Security & Access Control

- AF_UNIX sockets with filesystem ACLs; only `hades-agents` group can access completions; metrics socket may be group `hades-observability` with read-only ACLs.
- Authentication via static API keys stored in `auth/clients.yaml`; optional mutual SSH user gating.
- No outbound network; run in net namespace without default route unless explicitly allowed.
- Model cache directory restricted (`0700`). No prompts/completions written to disk; logs omit sensitive payloads.

## Deployment Plan

1. **Phase 0** – Design review & resource sizing (GPU, RAM, disk); align tool schema with memory PRD.
2. **Phase 1** – Prototype vLLM with AF_UNIX binding, load model locally, run baseline benchmarks.
3. **Phase 2** – Integrate function schema & telemetry; validate heartbeat interplay.
4. **Phase 3** – Harden security (auth, sandbox), add CI checks (no TCP listeners), document ops runbooks.
5. **Phase 4** – Staged rollout with synthetic load, chaos tests (GPU reset, cache eviction), sign-off with board.

## Test Plan & Acceptance Criteria

- **Unit/Dev**: tool schema validation; AF_UNIX binding tests; model load path correct.
- **Integration**: memory service tool call loop; heartbeat w/ `heartbeat_rps_limit` enforced; cold/warm latency measurements.
- Ensure streaming responses do not emit partial tool_call JSON; assert tool_call arrives only when complete payload is available.
- Verify token overage requests return 400 with `TOKEN_BUDGET_EXCEEDED` body and echoed correlation headers.
- **Load**: 16 concurrent sessions w/ 2k+2k tokens on dual RTX A6000 48 GB GPUs (fp16) with continuous batching enabled; capture prefill/decode throughput, queue depth, batch size, and p50/p95/p99 latency.
- **Security**: confirm no TCP listeners; UDS permissions; API key enforcement; net namespace isolation; outbound connectivity blocked (attempted egress fails) and model cache present locally.
- Auth secrets verified to reside outside repo; CI scan ensures no credentials committed.
- **Resilience**: GPU reset -> auto-reload; cache purge -> re-download; config reload failure -> safe fallback.

## Risks & Mitigations

- GPU scarcity → allow fallback to quantized model (e.g., Qwen3-14B) with adjusted SLOs.
- HF cache corruption → integrity checks, fallback path.
- AF_UNIX adoption in clients → provide wrapper library/SDK that swaps base URL transparently.
- Model upgrades → versioned config, staged rollout with A/B server.

## Open Questions

- Need for multi-GPU tensor parallelism or single GPU suffices initially?
- Do we expose streaming over AF_UNIX or rely on polled completions for first release?
- What finetuning cadence do we anticipate (affects downtime strategy)?

## References

- Memory PRD: `docs/prd/memgpt_virtual_context_arango_prd.md`
- vLLM docs (OpenAI API): <https://github.com/vllm-project/vllm>
- Qwen3-Coder-30B-A3B-Instruct: <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>
