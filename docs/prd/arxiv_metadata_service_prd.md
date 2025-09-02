# PRD: ArXiv Metadata Service (Postgres)

## Summary

Build a Postgres‑backed metadata service for arXiv papers that ingests an initial offline snapshot and then maintains daily deltas via OAI‑PMH. The service tracks local artifacts (PDF and LaTeX), exposes fast list generation (IDs, stats, monthly splits) for downstream RAG pipelines, and keeps operations reproducible and observable. pgvector and embeddings are optional future extensions.

## Goals

- Single source of truth for arXiv metadata used by RAG pipelines.
- Fast, flexible list generation by time window, categories, and keywords.
- Offline‑first ingestion from the arXiv snapshot; daily incremental updates.
- Track local artifacts (PDF, LaTeX) and expose has_pdf/has_latex flags and paths.
- Deterministic exports compatible with existing tools/arxiv scripts.
- Operational transparency: idempotent runs, checkpoints, metrics, and logs.

## Non‑Goals (for initial delivery)

- Maintaining embeddings or pgvector indexes (prepare schema hooks, but do not compute indexes now).
- Running the full RAG pipeline; this service only feeds lists and metadata.
- Web UI; CLIs and SQL are sufficient initially.

## Users & Primary Use Cases

- Pipeline engineers: generate large ID lists and splits rapidly and reproducibly.
- Data ops: monitor nightly ingestion, PDF/LaTeX availability, and counts over time.
- Researchers: ad‑hoc queries (SQL) for trend analysis and monthly/yearly stats.
- Processing orchestrators: assess compute costs before committing resources to ingestion.

## Scope

- Initial load from snapshot JSONL: `/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json`.
- Daily deltas via OAI‑PMH ListRecords using from/until datestamps.
- Artifact scans for PDFs and LaTeX in local mirrors to set flags and paths.
- Export commands for ID lists, with/without artifacts, monthly splits, and stats, writing under `tools/arxiv/scripts/data/arxiv_collections/`.
- Optional API sweep only for recent months if needed; preferred path is OAI‑PMH.

Out of scope (initial): embedding pipeline, materialized semantic search, and MCP integration (can be added later).

## Key Requirements

### Functional

- Ingest snapshot (one‑time) and upsert per `arxiv_id`.
- Ingest deltas daily via OAI‑PMH; record ingestion runs and checkpoints.
- Parse publication timestamps robustly:
  - Prefer first `versions[0].created` if present (RFC‑822 or ISO‑like).
  - Fallback to `update_date` if needed.
  - Derive `year`, `month`, `yymm` fields from `published_at`.
- Maintain all categories for a paper; preserve `primary_category`.
- Track artifacts:
  - PDFs at `/bulk-store/arxiv-data/pdf/<YYMM>/<ID>.pdf` → `has_pdf`, `pdf_path`.
  - LaTeX sources at configurable base (e.g., `/bulk-store/arxiv-data/latex/**/<ID>/`) → `has_latex`, `latex_path`.
- Export lists:
  - Filter by year range, categories (e.g., `cs.LG`, `cs.CL`, `cs.CV`).
  - Optional keyword filters using Postgres FTS (english).
  - Optional caps: `--per-year-cap`, `--per-month-cap` for balanced selection.
  - Outputs: master ID list, with‑pdf, missing‑pdf, monthly lists, and stats JSON.
- Compute cost assessment:
  - Preview mode that returns statistics without generating files.
  - Document size distribution (small <1MB, medium 1-5MB, large 5-20MB, x-large >20MB).
  - Processing time estimates based on document count and complexity.
  - Resource requirements (storage, memory, GPU hours).
- Idempotent and resumable operations with clear logs and metrics.

### Non‑Functional

- Scale: handle up to ~2–3M papers (global) or subset efficiently; CS subset must be snappy.
- Performance targets:
  - Initial snapshot load: < 3 hours on modest hardware (batch COPY/UPSERT).
  - Daily deltas (24h window): < 10 minutes end‑to‑end.
  - Category+keyword list export over 1M rows: < 5 seconds with proper indexes.
- Reliability: recover from partial failures; ACID upserts; constraints prevent dupes.
- Observability: structured logs, per‑run metrics (inserted, updated, skipped, errors).
- Security: credentials via env vars; no secrets in repo; least‑privilege DB role.

## Data Model

### Tables

- papers (PK `arxiv_id`)
  - arxiv_id text PRIMARY KEY
  - title text NOT NULL
  - abstract text
  - primary_category text
  - published_at timestamptz
  - updated_at timestamptz
  - year int GENERATED (or managed by loader)
  - month int GENERATED (or managed by loader)
  - yymm text
  - doi text
  - license text
  - journal_ref text
  - has_pdf boolean DEFAULT false
  - pdf_path text
  - has_latex boolean DEFAULT false
  - latex_path text

- paper_categories
  - arxiv_id text REFERENCES papers(arxiv_id) ON DELETE CASCADE
  - category text NOT NULL
  - PRIMARY KEY (arxiv_id, category)

- versions (optional now)
  - arxiv_id text REFERENCES papers(arxiv_id) ON DELETE CASCADE
  - version int NOT NULL
  - created_at timestamptz
  - PRIMARY KEY (arxiv_id, version)

- ingest_runs
  - id bigserial PRIMARY KEY
  - source text CHECK (source IN ('snapshot','oai','api','fs-scan'))
  - started_at timestamptz NOT NULL DEFAULT now()
  - finished_at timestamptz
  - from_ts timestamptz
  - until_ts timestamptz
  - status text CHECK (status IN ('running','succeeded','failed'))
  - metrics jsonb DEFAULT '{}'::jsonb
  - last_cursor text NULL  -- for resuming OAI/API pagination

- embeddings (future)
  - arxiv_id text PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE
  - embedding vector(768)
  - created_at timestamptz DEFAULT now()

### Indexes

- papers: PRIMARY KEY (arxiv_id)
- B‑tree: `published_at`, `(year, month)`, `primary_category`, `yymm`
- B‑tree (optional): `has_pdf`, `has_latex` if these filters are frequent
- paper_categories: PRIMARY KEY (arxiv_id, category), B‑tree on `(category, arxiv_id)`
- Full‑text: GIN on `to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,''))`
- Vector (future): ivfflat/hnsw on `embeddings.embedding` with cosine ops

## Interfaces

### Configuration

- File: `tools/arxiv/configs/db.yaml` (YAML) with env overrides
  - postgres:
    - dsn: `ARXIV_PG_DSN` (e.g., `postgresql://user:pass@localhost:5432/arxiv`)
    - schema: default `public`
    - statement_timeout_ms: default 600000
    - pool_size, max_overflow
  - snapshot:
    - path: `/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json`
    - batch_size: 5000
  - oai:
    - base_url: `https://export.arxiv.org/oai2`
    - metadata_prefix: `arXiv`
    - rate_limit_rps: 2.0
    - retry/backoff: exp(100ms..5s), max_retries 5
  - artifacts:
    - pdf_glob: `/bulk-store/arxiv-data/pdf/*/*/*.pdf` (or structured by `<YYMM>/<ID>.pdf`)
    - latex_glob: `/bulk-store/arxiv-data/latex/**/<ID>/**.tex`

### Orchestration Pipeline Integration

The metadata service integrates with the ACID processing pipeline through a modular orchestration layer:

- **Search Module**: Accepts configuration (categories, keywords, time range) and queries PostgreSQL
- **Preview Module**: Generates compute cost assessment report without file generation
- **Refinement Module**: Allows iterative parameter adjustment based on preview results
- **Processing Module**: Feeds approved lists directly to ACID pipeline for ingestion

The orchestration flow follows a search → preview → refine → process pattern, with each module 
being interface-agnostic (CLI, MCP server, web UI). Initial implementation uses manual config 
editing with CLI confirmation; future phases may add interactive refinement.

### CLI Tools (new)

- `python tools/arxiv/db/load_snapshot_to_pg.py`
  - Args: `--config`, `--truncate-first`, `--max-rows`, `--workers`.
  - Behavior: COPY to temp table → UPSERT to `papers`/`paper_categories`/`versions`.

- `python tools/arxiv/db/harvest_oai_to_pg.py`
  - Args: `--config`, `--from`, `--until`, `--since-last-success`, `--checkpoint-interval`.
  - Behavior: harvest ListRecords; upsert; record `ingest_runs`; resume via `last_cursor`.

- `python tools/arxiv/db/scan_artifacts.py`
  - Args: `--config`, `--pdf`, `--latex`, `--reset-missing`.
  - Behavior: scan filesystem; update `has_pdf/pdf_path`, `has_latex/latex_path`.

- `python tools/arxiv/db/export_ids.py`
  - Args:
    - Time: `--start-year`, `--end-year`, `--months` (list), `--yymm-range`.
    - Filters: `--categories cs.LG cs.CL cs.CV`, `--keywords "attention|transformer|embedding|seq2seq|self-supervised|diffusion|VAE|RL|GNN"`, `--with-pdf`, `--missing-pdf`.
    - Caps: `--per-year-cap N`, `--per-month-cap N`.
    - Outputs: `--write-monthly-lists`, `--out-dir tools/arxiv/scripts/data/arxiv_collections/`.
  - Behavior: run SQL with indexes/FTS; write master IDs, with/missing PDF splits, monthly lists, and stats JSON (yearly/monthly counts).

### Example Queries

- Year/month histogram:

  ```sql
  SELECT date_trunc('month', published_at) AS ym, count(*)
  FROM papers
  WHERE published_at >= '2010-01-01' AND published_at < '2026-01-01'
  GROUP BY 1 ORDER BY 1;
  ```

- Categories + keywords:

  ```sql
  SELECT p.arxiv_id
  FROM papers p
  JOIN paper_categories c USING (arxiv_id)
  WHERE p.published_at >= '2010-01-01' AND p.published_at < '2025-01-01'
    AND c.category IN ('cs.LG','cs.CL','cs.CV')
    AND to_tsvector('english', coalesce(p.title,'') || ' ' || coalesce(p.abstract,''))
        @@ to_tsquery('attention | transformer | embedding | seq2seq | "self-supervised" | diffusion | VAE | "reinforcement & learning" | GNN');
  ```

- With/Missing PDFs:

  ```sql
  ... AND p.has_pdf = true; -- or false
  ```

## Data Processing Details

- Timestamp parsing precedence: first `versions[0].created` (RFC‑822 like: `"Mon, 9 Apr 2007 16:04:05 GMT"`), then ISO fallbacks; else `update_date`.
- Category handling: store all categories in `paper_categories`; designate `primary_category` from snapshot/API field.
- Deduplication: ON CONFLICT on `papers.arxiv_id` → UPDATE changed fields; re‑sync categories by upserting set difference.
- Derivations: compute `year`, `month`, `yymm` during load for fast grouping.
- Caps logic: when caps provided, stable order by `published_at, arxiv_id`; take top N per bucket.
- Stats: write JSON summary `{ total, by_year, by_month, by_category }` alongside IDs.

## Operations

- Deployment: docker‑compose for local; native Postgres for prod.
- Migrations: Alembic with `alembic.ini` and versions under `tools/arxiv/db/migrations/`.
- Scheduling: cron/systemd timer (daily, e.g., 03:00 UTC) to run `harvest_oai_to_pg.py` and `scan_artifacts.py`.
- Monitoring: logs to stdout + file; per‑run row counts; failure alerts (non‑zero exit, error summary file).
- Backups: rely on Postgres dumps/snapshots; no binary artifacts in DB.

## Acceptance Criteria

- Can load snapshot and produce:
  - Master list, with‑pdf list, missing‑pdf list, monthly lists, and stats for 2010–2024.
  - Execution matches current script behavior for category filters within ±1% (allowing parsing differences).
- Daily OAI run completes within target SLO and updates `ingest_runs` with metrics.
- Export queries over 1M rows complete within 5 seconds on a modest host.
- CLI tools have help, clear logs, and documented configuration.

## Milestones & Timeline

- M0: Schema + Alembic migration drafted, reviewed, and applied.
- M1: Snapshot loader (COPY/UPSERT) with timestamp parsing and categories.
- M2: Artifact scanner for PDFs and LaTeX; flags and paths populated.
- M3: Exporter CLI with caps and stats; parity with current outputs.
- M4: OAI‑PMH harvester with checkpoints and metrics; daily cron.
- M5: Docs, sample configs, and basic dashboards/logging.
- M6: Cutover: list generation switches from JSON builders to SQL exporter.

## Risks & Mitigations

- Timestamp format variance → multi‑format parser with tests; fallback to update_date.
- OAI throttling/outages → retries, backoff, resume via resumptionToken; widen window if needed.
- Filesystem drift (PDF/LaTeX moves) → periodic full rescan; configurable globs.
- Query performance regressions → indexes, `EXPLAIN ANALYZE`, materialized views for stats.
- Schema evolution → Alembic migrations; feature flags for new fields.

## Open Questions

- Exact LaTeX directory structure and glob pattern? Default vs per‑env overrides.
- Keep `versions` now or postpone to a later milestone?
- Enable FTS index immediately (cost ~seconds) or gate behind a flag?
- Continue API sweep for current month, or rely solely on OAI deltas once stable?
- Do we want an `artifacts` table for multi‑file tracking, or are flags on `papers` sufficient?

## Dependencies

- Postgres 14+ (for generated columns if used) and `pg_trgm`/`pgvector` extensions optional.
- Python 3.11; libraries: `psycopg` or SQLAlchemy for DB I/O; `lxml` or `sickle` for OAI.
- Existing local mirrors at `/bulk-store/arxiv-data/` for PDFs/LaTeX.

## Appendix: Example DDL (illustrative)

```sql
CREATE TABLE papers (
  arxiv_id text PRIMARY KEY,
  title text NOT NULL,
  abstract text,
  primary_category text,
  published_at timestamptz,
  updated_at timestamptz,
  year int,
  month int,
  yymm text,
  doi text,
  license text,
  journal_ref text,
  has_pdf boolean DEFAULT false,
  pdf_path text,
  has_latex boolean DEFAULT false,
  latex_path text
);

CREATE TABLE paper_categories (
  arxiv_id text REFERENCES papers(arxiv_id) ON DELETE CASCADE,
  category text NOT NULL,
  PRIMARY KEY (arxiv_id, category)
);

CREATE TABLE versions (
  arxiv_id text REFERENCES papers(arxiv_id) ON DELETE CASCADE,
  version int NOT NULL,
  created_at timestamptz,
  PRIMARY KEY (arxiv_id, version)
);

CREATE TABLE ingest_runs (
  id bigserial PRIMARY KEY,
  source text CHECK (source IN ('snapshot','oai','api','fs-scan')),
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  from_ts timestamptz,
  until_ts timestamptz,
  status text CHECK (status IN ('running','succeeded','failed')),
  metrics jsonb DEFAULT '{}'::jsonb,
  last_cursor text
);

-- Indexes
CREATE INDEX ON papers (published_at);
CREATE INDEX ON papers (year, month);
CREATE INDEX ON papers (primary_category);
CREATE INDEX ON papers (yymm);
CREATE INDEX ON paper_categories (category, arxiv_id);
CREATE INDEX papers_fts_idx ON papers USING gin (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,'')));
```

---
This PRD defines the initial scope and deliverables for the arXiv metadata service. Revisions will be tracked alongside Alembic migrations and CLI specs.
