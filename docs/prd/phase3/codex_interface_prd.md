# Codex Interface Integration

## Summary

Adopt the open-source `openai/codex` IDE-like interface as a temporary front-end for HADES. We will run Codex as a standalone web client that connects to our local vLLM OpenAI-compatible API via a thin gateway, inject the required Conveyance headers, surface tool-call telemetry, and avoid bespoke frontend work while we focus on backend hardening.

## Conveyance Scorecard (Efficiency stance)

| Dimension | Score | Evidence |
|-----------|-------|----------|
| **W** (What) | 0.80 | Enables agents/users to drive MemGPT+vLLM stack without new UI build; adds tool-call surfacing for debugging. |
| **R** (Where) | 0.85 | Hosted locally with AF_UNIX gateway; no external SaaS. |
| **H** (Who) | 0.83 | Minimal modifications to Codex; clear ownership between frontend tweaks and gateway team. |
| **T** (Time) | 0.84 | Latency dominated by vLLM; gateway adds <10 ms; quick adoption path. |
| **Ctx** (L/I/A/G) | 0.86 | L=0.88 (schema parity), I=0.84 (observability hooks), A=0.88 (socket ACLs + auth), G=0.83 (doc updates). |

Using α = 1.7, Conveyance ≈ **0.52**. Zero-propagation: if the gateway exposes TCP without auth or fails to emit Conveyance headers, declare C = 0.

## Problem Statement

We need a usable interface to exercise the HADES memory and model stack without committing resources to a custom frontend. The Codex project already implements an OpenAI-protocol UI; by adapting it to our security/telemetry requirements and bridging Codex’s HTTP expectations to our AF_UNIX-only vLLM API, we can test and demo the system quickly while deferring bespoke UI work.

## Goals

- Run Codex as a local web client (developer workstation or server-hosted) pointing at a proxy that speaks AF_UNIX to vLLM.
- Inject `X-Request-ID` / `X-Conveyance-Session-ID` headers on every request, echo them in responses, and surface tool_call/memory events in the UI log.
- Provide quick toggles for function-calling, streaming, temperature/max tokens, mirroring vLLM defaults.
- Keep Codex source modifications minimal, documented, and maintained via a fork or patch set pinned to a specific commit.
- Ensure gateway enforces auth, rate limits, and streaming pass-through, while exporting telemetry consistent with other PRDs.

## Non-Goals

- Building a new custom frontend or long-term UX (this is a stopgap).
- Exposing the gateway to public networks; access remains local/VPN.
- Replacing IDE integrations (Cline, etc.); they continue to use the same gateway.

## Stakeholders & Users

- Backend engineers validating MemGPT memory workflows.
- Applied research team generating prompts and analyzing tool-call traces.
- DevOps maintaining the proxy/gateway and deployment.

## Architecture Overview

1. **Codex Web Client** – Runs in browser; packaged via `npm`/`yarn`. Configured with API base URL, API key, streaming support, and new tool-call panel.
2. **HADES Proxy (codex-gateway)** – Nginx-based reverse proxy that:
   - Listens on HTTPS (localhost/loopback; TLS mandatory off-box) with upstream configured via AF_UNIX (`/run/hades/vllm/api.sock`).
   - Forwards requests to vLLM UDS using `proxy_pass http://unix:/run/hades/vllm/api.sock:/v1/...`, inserting auth, `X-Request-ID`, `X-Conveyance-Session-ID` (generate if missing), and validating payload size.
   - Streams responses in order, copies headers back to Codex, logs metrics. Startup/CI guard fails if gateway binds to non-loopback TCP without TLS or if `ss -lntp` shows unexpected listeners.
3. **vLLM Model Manager** – As defined in `docs/prd/model_manager_vllm_prd.md`.
4. **MemGPT Memory Service** – Remains unchanged; tool calls triggered via vLLM responses.
5. **Observability** – Proxy emits Prometheus metrics & logs correlated with backend IDs.

### Deployment Model

- Codex repo cloned separately under `tools/codex-ui/` with pinned commit.
- Build artifacts served via Node dev server or static build (e.g., `npm run build` + Nginx).
- Gateway runs as systemd service, reading vLLM UDS path from `/run/hades/vllm/api.sock`.
- Optional: host Codex and gateway on same machine as vLLM; remote users access via SSH port-forward or Tailscale (still using HTTPS endpoint).

## Modifications to Codex

- **Configuration**: Add `.env.local` or config file for API base URL, API key, Conveyance header names, default model (`qwen3-coder-30b-a3b-instruct`).
- **Header Injection**: Ensure every request includes user-provided `X-Request-ID` (generate client-side) and `X-Conveyance-Session-ID` (persist across chat session).
- **Tool-call Display**: Extend chat transcript to show tool_call JSON blocks, response payloads, and memory warnings from MemGPT; payloads redacted by default with a user-controlled reveal toggle for debugging.
- **Streaming Handling**: Respect SSE chunk order; display streaming tokens; flush helper.
- **Auth**: Provide UI for API key (if required) stored in browser local storage; avoid storing secrets server-side.
- **Telemetry Hooks**: Optionally emit browser console logs for debugging; final telemetry captured by gateway.

## Gateway Requirements

- Expose HTTPS (self-signed cert OK) on loopback or restricted interface; TLS is mandatory for any non-loopback access.
- Enforce rate limiting (default 5 requests/sec), payload max (e.g., 64 KB), timeout (60 s).
- Validate request JSON, ensure `model` matches allowed list, reject over-budget tokens (pre-check vs. configured limits).
- Forward streaming responses chunk-by-chunk; preserve SSE ordering; inject `X-Request-ID`/`X-Conveyance-Session-ID` if missing; propagate 4xx/5xx errors directly.
- Acceptance: 400 `TOKEN_BUDGET_EXCEEDED` responses pass through unmodified with echoed correlation headers; missing or altered headers trigger alerts/CI failures.
- Metrics: request count, latency, bytes, stream duration, error codes, queue depth.
- Logs: structured JSON with request/response IDs, user, path, status.
- Security: Basic auth or token guard; integrate with existing secret injection (systemd drop-in, Vault, etc.).

## Performance & Reliability Targets

- Added latency per request ≤10 ms P95 (gateway).
- Streaming throughput ≥90% of backend output; no buffering gaps >250 ms.
- Gateway uptime tied to backend (≥99% in staging); restart within 5 s.
- Codex build served locally; initial page load <3 s; subsequent prompt cycles rely on backend SLOs.

## Telemetry & Observability

- Gateway metrics exported via Prometheus (`hades_codex_requests_total`, `hades_codex_request_latency_ms`, `hades_codex_stream_bytes_total`).
- Logs include correlation IDs and tool-call summary counts; absence of IDs triggers alert/CI failure.
- Integrate with existing alerting: missing headers, auth failures, high error rates (>5% in 5 min), long response (>45 s).
- Optional browser logging disabled in production.

## Security & Access Control

- Gateway binds to localhost or restricted interface; rely on SSH/Tailscale for remote access and enforce TLS for any non-loopback usage.
- Auth tokens stored in systemd environment, not committed.
- TLS optional for localhost; mandatory when exposing beyond loopback.
- CORS limited to configured Codex host (`http://localhost:3000` etc.); CSRF protections enforced for any admin endpoints (keep admin disabled by default).
- No direct access to ArangoDB or MemGPT; all traffic flows through vLLM API. UI does not implement file uploads or ingestion bypass paths; any future ingestion must use approved pipelines.

## Deployment Plan

1. **Phase 0 – Eval**: Pin Codex commit, identify required patches, verify license compliance.
2. **Phase 1 – Gateway Prototype**: Implement AF_UNIX bridge with header injection; basic auth.
3. **Phase 2 – Codex Mods**: Add config, header injection, tool-call panel, streaming UI tweaks.
4. **Phase 3 – Telemetry & Security**: Add Prometheus metrics, logging, rate limiting, TLS.
5. **Phase 4 – Acceptance Testing**: Run functional, load, and failure scenarios; finalize docs.

## Test Plan & Acceptance Criteria

- **Unit**: header injection utilities; tool-call renderer; SSE handler; gateway rate limiter.
- **Integration**: full chat loop hitting vLLM via AF_UNIX; verify headers, token budget rejection passes through from backend; tool calls displayed.
- **Security**: ensure no public TCP exposure; auth required; secrets not in repo.
- **Performance**: measure added latency; streaming chunk timing.
- **Resilience**: restart gateway while streaming; ensure Codex reconnects gracefully.
- **User Acceptance**: backend engineers confirm workflow (prompt, inspect tool calls, respond) works end-to-end.

## Risks & Mitigations

- Codex upstream changes: pin commit; maintain patch notes.
- Browser caching of API keys: instruct users to clear or rotate; optional password manager integration.
- Gateway drift from backend headers: unit tests enforce parity with vLLM PRD.
- Long-term maintenance: schedule review when bespoke frontend effort begins.

## Open Questions

- Do we host Codex centrally or require engineers to run locally?
- Should gateway support multi-user auth/roles (read-only vs. edit)?
- Need for offline mode (bundled assets) vs. live dev server?
- How to persist chat transcripts (optional)?

## References

- Memory PRD: `docs/prd/memgpt_virtual_context_arango_prd.md`
- Model manager PRD: `docs/prd/model_manager_vllm_prd.md`
- Codex repo: https://github.com/openai/codex
