# Virtual Context Management via ArangoDB

## Summary

We will implement MemGPT‑inspired virtual context management that uses ArangoDB as the external memory tier. The feature introduces a bounded in‑context working memory, a FIFO conversation queue with recursive summarisation, and Arango‑backed recall/archival storage with token‑aware pagination and heartbeat chaining so agents can dynamically page information without exceeding model token limits.

## Conveyance Scorecard (Efficiency stance)

| Dimension         | Score | Evidence                                                                                                                                                                               |
| ----------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **W** (What)      | 0.87  | Working context + recall log retain critical state across sessions; paged archival search fulfils long‑horizon tasks with deterministic tool contracts.                                |
| **R** (Where)     | 0.89  | Arango collections (`core_memory`, `recall_log`, `archival_docs`) and `g_mem` graph expose deterministic RO/RW APIs over Unix sockets with revision control.                           |
| **H** (Who)       | 0.90  | Heartbeat‑driven function chaining keeps agents in control of multi‑step memory edits while logging termination reasons.                                                               |
| **T** (Time)      | 0.83  | Recall fetch P99 ≤250 ms (@ N≈1e6 messages) and archival search P99 ≤750 ms (@ N≈5e6 chunks, dim per chosen embedder) on production hardware.                                          |
| **Ctx** (L/I/A/G) | 0.88  | L=0.90 (late chunking compliance), I=0.86 (multi‑database observability), A=0.88 (automated pressure handling with summarisation drift alarms), G=0.88 (Unix socket security + audit). |

Using α = 1.7, Conveyance = (W·R·H / T) · Ctx^α ≈ **0.68**. Zero‑propagation: if any memory tier is unreachable or chunking reverts to naive mode, declare C = 0. ([MemGPT][1])

## Problem Statement

Modern LLM agents saturate their context windows long before conversations or document analyses finish. We need a first‑class virtual memory subsystem that lets agents persist, recall, and iterate on long‑lived state without giving up our ArangoDB‑based infrastructure or late chunking guarantees.

## Goals

* Deliver main‑memory vs external‑memory separation mirroring MemGPT: system instructions, working context, FIFO queue, and Arango recall/archival tiers. ([MemGPT][1])
* Expose function‑call APIs that allow the LLM to write, summarise, retrieve, paginate, and heartbeat autonomously while remaining idempotent and revision‑safe.
* Implement memory‑pressure heuristics (**warning at 70 %**, **flush when crossing 90 % back to 50 %**) with recursive summarisation of evicted turns. Intentional deviation from MemGPT’s example (**flush at 100 %**, evict \~50 %) provides headroom for tool outputs and tokenizer variance. ([MemGPT][1])
* Enforce late chunking during ingestion so archival documents keep workload‑aware context and comply with Conveyance zero‑propagation checks.
* Instrument Conveyance metrics (W/R/H/T/Ctx), heartbeat activity, and retrieval health for observability and review‑board evidence.

## Non‑Goals

* Training or fine‑tuning LLMs.
* Supporting non‑Arango backends or UI updates.
* Replacing existing ingestion automation or deployment scripts.

## Stakeholders & Primary Users

* Conversational agents requiring long‑term memory (product + CX).
* Document QA workflows handling multi‑document corpora (knowledge engineering).
* Platform reliability teams monitoring Arango latency, vector index capacity, and token budgets.

## Architecture Overview

1. **Event Dispatcher** normalises events to plain text and appends them to the FIFO queue. Event classes follow MemGPT: user, system (capacity warnings/alerts), user‑interaction (e.g., login notifications), and scheduled/timed ticks. ([MemGPT][1])
2. **LLM Processor** consumes main context (system instructions + working context + FIFO queue) and emits completion tokens interpreted as function calls.
3. **Function Executor** validates arguments, enforces token budgets, executes memory operations, captures runtime errors as system messages in‑context, and either honours `request_heartbeat=true` or yields to the next external event.
4. **Arango Memory Layer** persists working context snapshots, recall logs, archival documents, graph edges, and ArangoSearch views; vector search uses ArangoDB’s **experimental vector index (3.12.4+)** with `--experimental-vector-index`. Deployments must start with the flag; once enabled, it **cannot be disabled** and restores must also start with the flag. ([3.12 release notes][2]; [Vector functions][3])
5. **Telemetry & Conveyance Layer** records token utilisation, heartbeat chains, latency, summarisation drift, vector index health, and zero‑propagation checks for downstream analytics.

### Transport & Endpoints (AF\_UNIX only)

All database traffic transits **UNIX domain sockets** via the local Go proxy. ArangoDB binds **only** to a UDS endpoint (e.g., `unix:///run/arangodb3/arangod.sock`) using `--server.endpoint unix:///run/arangodb3/arangod.sock`. No `tcp://` or `ssl://` listeners are configured. Authentication remains enabled on UDS (`--server.authentication true` and `--server.authentication-unix-sockets true`, default `true`). Clients/drivers use UDS URLs (e.g., `http+unix:///run/arangodb3/arangod.sock`). Rollout guard fails if any TCP endpoint is detected. ([Server options][6])

> **Driver note (Node):** when using `arangojs`, UDS support requires the `undici` peer dependency. ([arangojs README][7])

## Data Model

* `core_memory` (document collection; 1 per agent):

  * `working_blocks: [ { id, role, content, token_estimate, pinned } ]`
  * `queue_head_summary: { content, token_estimate, revision }`
  * `token_budget: { warning, flush, target, max }`
  * `_rev`, `version`, `updated_at`
  * **Invariant:** system + working + FIFO + summary tokens ≤ `token_budget.max`. Violations log `TOKEN_BUDGET_EXCEEDED` to `memory_events`.

* `messages` (document collection):

  * `_key`, `agent_id`, `role`, `content`, `meta`, `created_at`, `token_estimate`
  * Index `(agent_id, created_at)` to support chronological traversals; participates as vertex targets of `recall_log` edges.

* `recall_log` (edge collection):

  * `_from` = `core_memory/{agent_id}` vertex, `_to` = message vertex (linear chain `prev_msg -> msg`)
  * `payload: { role, content, meta, token_estimate }`
  * `evicted_batch_id`, `summary_rev`, `created_at`
  * **Indexes:** `(_from, created_at)`; optional `_to` for reverse lookups.
  * ArangoSearch View provides BM25/phrase search over `payload.content` using the `text_en` analyzer. All PHRASE/BM25 queries **set analyzer context explicitly** (e.g., `ANALYZER(PHRASE(...), "text_en")`) to match the View definition. ([ArangoSearch functions & analyzer context][5])

* `archival_docs` (document collection, chunk‑level):

  * `doc_id`, `chunk_id`, `text`, `embedding`, `dimension`, `metric`, `embedding_version`, `model_id`, `provenance`, `chunk_links`, `freshness_ts`
  * Vector index via `APPROX_NEAR_COSINE` / `APPROX_NEAR_L2`; **no pre‑FILTER** is allowed between `FOR` and `LIMIT` when using these functions (apply metadata filters post‑query). Store index configuration (`defaultNProbe`, `metric`) alongside metadata; `dimension`/`metric` are immutable per index—changing embedder or dimensionality requires offline re‑index and backfill planning. ([Vector functions][3])

  **Vector top‑K AQL (copy‑ready):**

  ```aql
  FOR d IN archival_docs
    LET similarity = APPROX_NEAR_COSINE(d.embedding, @q, { nProbe: @n })
    SORT similarity DESC
    LIMIT @k
    RETURN MERGE({ _id: d._id, similarity }, d)
  ```

  *(Contract: **no `FILTER`** between `FOR` and `LIMIT` for pre‑filtering; apply any metadata filters in a post‑filter subquery.)* ([Vector functions][3])

* `memory_events` (document collection):

  * `type ∈ { warning, flush, hb_start, hb_end, summarize, error }`
  * `details`, `agent_id`, `ts`

* `g_mem` graph:

  * Vertices: `core_memory/{agent_id}` (one document per agent), `archival_docs`, `messages`
  * Edges: `recall_log`, `cites`, `derived_from`
  * Enables traversal‑based recall + hybrid retrieval.

## API / Function Schema (LLM‑facing)

All mutating calls require `_rev` (If‑Match) and `idempotency_key`; retries with the same key must be no‑ops. Responses return updated `_rev`, token estimates, and whether execution triggered a heartbeat opportunity.

* `store_core(block_id, content, policy, idempotency_key) -> { block_id, _rev, token_estimate }`
* `fetch_core(block_id) -> { block_id, content, _rev, token_estimate }`
* `append_fifo(message, role, idempotency_key) -> { message_id, token_estimate }`
* `evict_fifo(strategy='fifo', target_tokens, summarize=true, idempotency_key) -> { evicted_count, new_summary_tokens, after_occupancy }`
* `write_recall(entries[], idempotency_key) -> { inserted_ids, total_tokens }`
* `search_recall(query, limit, mode='text', heartbeat=false) -> { results, tokens_added }`
  *Phase 1: `mode='text'` (ArangoSearch View). Phase 2 (optional): extend schema with `payload.embedding` + vector index to support `vector`/`hybrid`.*
* `search_archival(query_vector, limit, page_token?, heartbeat=false, nProbe?) -> { results, next_page_token, tokens_added }`
* `ingest_archival(doc_id, chunks[], embeddings[], dimension, metric, embedding_version, model_id, idempotency_key) -> { inserted, vector_index_used }`
* `record_heartbeat(state) -> { chain_depth, duration_ms, terminated_reason? }`

**AQL generation rule for `search_archival`:** never insert `FILTER` between `FOR` and `LIMIT` with `APPROX_NEAR_*`. Apply metadata filters in a post‑filter subquery:

```aql
LET topk = (
  FOR d IN archival_docs
    SORT APPROX_NEAR_COSINE(d.embedding, @q, { nProbe: @n }) DESC
    LIMIT @k
    RETURN d
)
FOR d IN topk
  FILTER d.provenance.source == @src
  RETURN d
```

([Vector functions][3])

## **Executor behaviour**

* Reject operations that would exceed token budget and return `{ error: 'TOKEN_BUDGET_EXCEEDED', required_headroom }`.
* Enforce per‑page token accounting; stop paging when tokens remaining < page size.
* Runtime errors become system messages in the same turn for model introspection.
* If a tool response would exceed `tokens_remaining`, spool the full payload to `recall_log`, append a system notice referencing stored entry IDs, and provide a truncated in‑context preview.

## Memory Management Policies

* **Token Budgets (per LLM):** `warning = 0.70×context`, `flush = 0.90×context`, `target = 0.50×context`, `max = 1.00×context`. Crossing `warning` injects a memory‑pressure system message. Crossing `flush` triggers eviction until occupancy ≤ `target`. ([MemGPT][1])

* **Eviction:** default FIFO; `pinned` blocks cannot be evicted, and cumulative pinned tokens must remain ≤25 % of the window (`PIN_LIMIT_EXCEEDED` otherwise). Recursive summary budget ≤ 15 % of context. New summary = f(previous summary, evicted batch).
  **Atomicity:** `evict_fifo` performs three steps **atomically** in one transaction (stream in cluster / single‑request on single server): (1) append the evicted batch to `recall_log`, (2) recompute `queue_head_summary`, (3) remove FIFO messages. If any step fails, the transaction aborts and no state changes. All collections are declared at transaction start. Enforce payload caps to respect **128 MB max transaction size** and **idle timeout ≤ 120 s** for stream transactions; split batches if needed. ([Stream transactions][4])

* **Recall Rehydration:** retrieved messages append to FIFO tail; policy engine may propose `store_core` updates for critical facts (model decides).

* **Archival Paging:** page size ≤512 tokens; executor schedules heartbeat loops until yield, tokens depleted, or caps reached.

## Control Flow & Heartbeat

* Tool outputs MAY set `request_heartbeat=true` to re‑run inference immediately with updated context.
* Absence of the flag yields control until the next event.
* Guards: `max_chain_depth`, `max_chain_duration_ms`, and `tokens_remaining` gates. Defaults: `max_chain_depth=8`, `max_chain_duration_ms=60000`, `tokens_remaining` floor = 256 tokens. LLM Platform owns tuning via config service; terminations log `{ reason ∈ { depth, duration, tokens, explicit_yield } }` to `memory_events` and `record_heartbeat` payloads.
* Backpressure: enforce `heartbeat_rps_limit` gates (default 3 req/s per agent) and apply `heartbeat_cooldown_ms=750` after terminations caused by `depth`/`duration` to prevent thrash. Platform Reliability owns dashboarding and alert thresholds.

## Non‑Functional Requirements

* **Performance SLOs (P99 on production hardware):**

  * `recall_log` query (limit ≤20, N≈1e6 messages/agent) ≤250 ms.
  * `archival_docs` vector top‑k (k≤40, nProbe≤32, N≈5e6 chunks, embedding dimension per selected model) ≤750 ms.
  * `core_memory` write with optimistic locking ≤100 ms.

* **Performance validation:** Pre‑production load harness replays 1e6 recall messages and 5e6 archival chunks generated from staging exports; drivers run on c6id.8xlarge proxies with the Arango cluster tier mirrored from production. Capture P99 latencies from proxy histograms (Prometheus scrape) and Arango `_statisticsRaw` plus Conveyance logger annotations so the review board can reproduce the `(W·R·H/T)` evidence.

* **Capacity Planning:** document embedding dimension, metric, `embedding_version`, default `nProbe`, vector storage bytes/chunk, and expected corpus sizes documented pre‑rollout.

* **Vector index (experimental):** enable `--experimental-vector-index` (introduced 3.12.4). Once enabled it **cannot be disabled** (RocksDB column family). Restores/dumps must run with the flag. CI/startup guard enforces presence. ([3.12 release notes][2]; [Vector functions][3])

* **UDS guard:** deployment fails if `ss -lntp | grep -E ':8529|arangod'` returns any TCP listeners; success requires `lsof -U | grep arangod` to show the bound UDS path. ([Server options][6])

* **Reliability:** optimistic locking + idempotency guard against duplicate writes; exponential backoff on proxy failures; chaos tests covered in acceptance.

* **Compliance:** late chunking enforced at ingest; violations trigger zero‑propagation alarms and block ingestion.

## Telemetry & Observability

* Token gauges: `token.system`, `token.working`, `token.fifo`, `token.summary`, `token.free`.
* Heartbeats: `hb.depth`, `hb.duration_ms`, `hb.rate_limited`, `hb.terminated_reason`.
* Retrieval: `recall.hit_rate`, `recall.p99_ms`, `archival.p99_ms`, `archival.nProbe_distribution`, `archival.page_depth`, `archival.k_actual`, `archival.post_filter_drop_rate`.
* Summarisation drift: cosine similarity between recursive summary and evicted batch; auto re‑summarise below 0.85.
* Vector index health: cluster counts, nProbe overrides, flag presence.
* Audit trail: idempotency usage, actor IDs, diff footprint per write.
* Conveyance logger emitting W/R/H/T/Ctx per session + zero‑propagation flags.

## Security & Safety

* RO/RW Unix‑socket proxies with directory permissions `0700` and socket `0660`; credentials and endpoints via env vars (`ARANGO_RO_UDS`, `ARANGO_RW_UDS`). Drivers use `http+unix://%2Frun%2Farangodb3%2Farangod.sock` style URLs; production profiles ignore any HTTP base URL. (Driver UDS note: see [arangojs README][7])
* Enforce `--server.authentication true` and `--server.authentication-unix-sockets true` (default `true`). ([Server options][6])
* Enforce `_rev` + `idempotency_key` on all writes; log diffs to `memory_events` with actor metadata.
* Prompt‑injection / memory‑poisoning filter on retrieved text; tainted content requires explicit tool justification before promotion to `working_blocks`.
* Optional TTL policies for `recall_log` to meet compliance or PII retention requirements.

## Delivery Phases

1. **Phase 0 – Design Sign‑off:** Review data model, API contracts, heuristics, Arango flags, and capacity plan with architecture + platform teams.
2. **Phase 1 – Core Memory Service:** Implement collections, optimistic locking, idempotency, token accounting, and FIFO eviction; unit tests for thresholds and error propagation.
3. **Phase 2 – Retrieval & Heartbeat Loop:** Build recall/archival search functions, pagination, ArangoSearch View, heartbeat scheduler, and hybrid retrieval; integration tests with synthetic workloads.
4. **Phase 3 – Observability & Conveyance:** Instrument metrics, alerts, summarisation drift checks, Conveyance reporting, and security auditing.
5. **Phase 4 – Rollout & Hardening:** Load + chaos testing (vector scale, proxy restart), staged deployment, operational playbooks, and zero‑propagation validation.

## Risks & Mitigations

* **Token Estimate Drift:** tokenizer‑based counters vs actual prompt usage reconciliation; flag variance in telemetry.
* **Summarisation Loss:** regression tests + automated re‑summarisation when similarity <0.85.
* **Arango Contention / Vector Cost:** separate RO/RW proxies, connection pooling, sharded recall collection, monitor index RAM usage.
* **Heartbeat Loops:** enforce caps, termination logging, and watchdog alerts on excessive depth/duration.
* **Vector Index Stability:** track experimental flag, maintain blue/green upgrade plan to handle format changes.

## Test Plan & Acceptance Criteria

* **Unit:**

  * Token accounting for warning/flush/target thresholds across simulated context windows.
  * Idempotency: retry each mutating API call (same `idempotency_key`) → no duplicate effect.
  * Recursive summary: verify new summary footprint ≤15 % window.
  * Vector query generator: assert AQL emitted for `search_archival` never places `FILTER` before `LIMIT` when using `APPROX_NEAR_*`; subquery post‑filters behave as expected. ([Vector functions][3])
  * Analyzer regression: PHRASE/BM25 queries with explicit `ANALYZER()` return identical results to default analyzer contexts. ([ArangoSearch functions & analyzer context][5])

* **Integration:**

  * MemGPT DMR‑style recall: plant facts in `recall_log`, verify retrieval under token pressure with pagination.
  * Document QA: increasing `k` values ensure executor paginates without overflow; verify answers cite archival provenance (`_id`, `chunk_id`, offsets when available).
  * **Transactional eviction:** inject failure between `recall_log` append and FIFO removal to confirm transaction abort keeps queue + summary unchanged; verify batch size respects **128 MB** and **≤ 120 s** stream‑TX constraints. ([Stream transactions][4])
  * **Flush divergence scenario:** simulate large tool output to show `flush=90%` prevents overflow whereas `flush=100%` risks exceeding tokenizer cap; document rationale. ([MemGPT][1])
  * Heartbeat chain: 5 sequential tool calls with `request_heartbeat=true`; termination occurs at configured `max_chain_depth` with logged reason.

* **Load:**

  * Vector top‑k at N≈5e6 chunks / chosen dimension; measure P50/P95/P99 across `nProbe` sweeps.
  * Recall scans at N≈1e6 messages/agent with concurrent agents.

* **Chaos:**

  * Restart/**kill** proxy mid‑write; confirm retries respect idempotency and `_rev`.
  * Disable vector flag in staging → expect startup guard to fail deploy (preventing silent downgrade).
  * **Network isolation:** assert `ss -lntp | grep -E ':8529|arangod'` is empty, `lsof -U | grep arangod` returns UDS path, and drivers use `http+unix://` endpoints during tests. ([Server options][6]; [arangojs README][7])

## Open Questions

* Which embedding model/dimension and associated storage footprint do we standardise on for `archival_docs`?
* Preferred summarisation model (latency vs fidelity trade‑off) and batch size?
* Required concurrency targets for recall vs archival operations across peak traffic windows?
* Do we enable priority‑based eviction or hybrid policies (pinned categories) in Phase 1, or defer to later phases?

## Appendix – References & Principles

* **MemGPT** virtual memory architecture: system instructions, working context, FIFO queue, function calls, heartbeat chaining, recursive summarisation, pagination. ([MemGPT][1])
* **ArangoDB vector index**: release notes, experimental flag, startup/restoration constraints (introduced in 3.12.4). ([3.12 release notes][2])
* **AQL vector functions** (`APPROX_NEAR_*`), **no‑pre‑FILTER rule**, and paging guidance. ([Vector functions][3])
* **ArangoSearch analyzers** and `PHRASE`/`ANALYZER` usage for hybrid retrieval. ([ArangoSearch functions][5])
* **Server options**: `--server.endpoint unix:///…` (UDS), `--server.authentication-unix-sockets` (default `true`). ([Server options][6])
* **Driver UDS**: `arangojs` requires `undici` for UDS. ([arangojs README][7])

[1]: https://arxiv.org/pdf/2310.08560 "MemGPT: Towards LLMs as Operating Systems"
[2]: https://docs.arangodb.com/3.12/release-notes/version-3.12/whats-new-in-3-12/ "Features and Improvements in ArangoDB 3.12"
[3]: https://docs.arangodb.com/3.13/aql/functions/vector/ "Vector search functions in AQL | ArangoDB Documentation"
[4]: https://docs.arangodb.com/3.11/develop/transactions/stream-transactions/ "Stream Transactions | ArangoDB Documentation"
[5]: https://docs.arangodb.com/3.11/aql/functions/arangosearch/ "ArangoSearch functions in AQL"
[6]: https://docs.arangodb.com/3.12/components/arangodb-server/options/ "ArangoDB Server Options"
[7]: https://github.com/arangodb/arangojs "arangojs – official ArangoDB JS driver (UDS note)"

---

**What changed vs your last draft (at a glance):**

* Recomputed the Conveyance score (0.68) and deduplicated redundant citations so the scorecard math lines up with the appendix references.
* Added heartbeat guard defaults, ownership notes, and a production-mirroring performance validation plan to anchor the SLO commitments.
* Fixed the **AQL** example (uses `LET similarity` + `MERGE`), and reiterated the **no pre‑FILTER** rule. ([Vector functions][3])
* Corrected `recall_log` **indexing** to `(_from, created_at)` (edge collection semantics).
* **Merged** duplicate vector‑flag bullets; added the **cannot be disabled** clause. ([3.12 release notes][2])
* **AF\_UNIX** section now cites current **server options** and pins `--server.authentication-unix-sockets true` (default). ([Server options][6])
* Added **stream transaction** limits (128 MB / ≤ 120 s) and test guardrails. ([Stream transactions][4])
* Aligned `search_recall` to **text‑only in Phase 1**, with vector/hybrid as a Phase 2 schema extension.
* Added a brief **driver UDS** note for arangojs (`undici`). ([arangojs README][7])
