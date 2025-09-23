# Sharding Toolkit Usage Guide

This guide explains how to use the Phase 2 sharding toolkit to accelerate large-scale ingestion, edge construction, and export jobs. It supplements the design spec (`1_design_sharding_toolkit.md`) with actionable instructions and examples for incorporating the toolkit into workflows.

## Overview

The toolkit decomposes high-volume jobs into shard-aligned partitions, executes them concurrently with leasing/backpressure, and emits telemetry for Conveyance dashboards.

Core modules:

- `core.tools.sharding.spec`: partition definitions and adapter protocol.
- `core.tools.sharding.planners`: helpers for hash/time partition plans.
- `core.tools.sharding.runner`: async runner with leases, retries, throttles.
- `core.tools.sharding.jobs`: job executors (Python/AQL/external).
- `core.tools.sharding.lease`: lease store interfaces (defaults to in-memory; stub for external coordinators if ever reintroduced).
- `core.tools.sharding.token_bucket`: shared throttling primitives.
- `core.tools.sharding.adapters`: dataset adapters (`ArxivPartitionAdapter` reference implementation).
- `core.tools.sharding.cli`: CLI entry point for ad-hoc runs.

## Requirements

- Python 3.12 environment via Poetry (`poetry shell`).
- Access to ArangoDB memory client when running AQL jobs.
- Workloads run in-process on a single host; in-memory leases are sufficient.

## Quick Start

1. **Select an adapter**
   ```python
   from core.tools.sharding.adapters import ArxivPartitionAdapter
   adapter = ArxivPartitionAdapter(total_buckets=64, shard_count=16)
   ```

2. **Define a job**
   ```python
   from core.tools.sharding.jobs import PythonShardJob

   def upsert_entities(spec, params):
       # call a builder function / execute AQL
       processed = ingest_partition(spec, params)
       return {"rows_read": processed, "rows_written": processed}

   job = PythonShardJob(upsert_entities)
   ```

3. **Run with the sharding runner**
   ```python
   import asyncio
   from core.tools.sharding import ShardRunner, InMemoryLeaseStore, NullTokenBucket

   async def main():
       runner = ShardRunner(
           adapter,
           job,
           lease_store=InMemoryLeaseStore(),
           concurrency=8,
           token_buckets={"default": NullTokenBucket()},
       )
       results = await runner.run()
       print(f"completed {len(results)} partitions")

   asyncio.run(main())
   ```

4. **CLI equivalent**
   ```bash
   poetry run python -m core.tools.sharding.cli \
     --adapter core.tools.sharding.adapters.arxiv:ArxivPartitionAdapter \
     --adapter-config '{"total_buckets": 64, "shard_count": 16}' \
     --job my.module:ArxivIngestJob \
     --job-config '{"foo": "bar"}' \
     --concurrency 16 \
     --rate-limit default=16
   ```

## Integrating into a Workflow

### 1. Implement a Dataset Adapter

Subclass `BasePartitionAdapter` and override:

- `describe_partitions()` – return `PartitionSpec` list using planners.
- `bind_params(spec)` – provide bind variables/context.
- `expected_invariants(spec, result)` – assert row counts, collisions, etc.
- Optional `pre_shard` / `post_shard` hooks.

```python
from core.tools.sharding.spec import BasePartitionAdapter, PartitionResult
from core.tools.sharding.planners import build_hash_partitions

class MyDatasetAdapter(BasePartitionAdapter):
    def describe_partitions(self):
        return build_hash_partitions(total_buckets=32, shard_count=8, prefix="my")

    def bind_params(self, spec):
        bounds = spec.bounds
        return {
            "hash_mod": bounds["hash_mod"],
            "hash_low": bounds["hash_low"],
            "hash_high": bounds["hash_high"],
        }

    def expected_invariants(self, spec, result: PartitionResult):
        if result.rows_written < result.rows_read * 0.9:
            raise ValueError("too many dropped rows")
```

### 2. Choose a Lease Store

- **Local/testing**: `InMemoryLeaseStore` (non-durable).
- **Future extension**: if distributed coordination returns, supply a durable `LeaseStore` implementation that handles `acquire`, `heartbeat`, `mark_succeeded`, and `mark_failed` across processes.

### 3. Configure Throttles

Token buckets coordinate concurrency across workers.

```python
from core.tools.sharding import FixedTokenBucket

buckets = {
    "default": FixedTokenBucket(capacity=32),  # limit simultaneous jobs
    "writes": FixedTokenBucket(capacity=8),    # optional extra channel
}
```
Pass `token_buckets=buckets` into `ShardRunner`.

### 4. Run in Workflow

Wrap the runner in your workflow script (Airflow, CLI, etc.). Example pseudo-code:

```python
async def run_ingest(adapter):
    job = PythonShardJob(ingest_partition)
    lease_store = InMemoryLeaseStore()
    token_buckets = {"default": FixedTokenBucket(32)}

    runner = ShardRunner(
        adapter,
        job,
        lease_store=lease_store,
        concurrency=32,
        max_retries=3,
        lease_ttl=180,
        token_buckets=token_buckets,
        metrics=emit_metrics,
    )
    results = await runner.run()
    publish_summary(results)
```

`emit_metrics` should push events to logging/Prometheus; `publish_summary` can update dashboards.

For HiRAG-specific use cases, import ready-made jobs such as `HiragEntityIngestJob`, `HiragRelationsJob`, `HiragHierarchyJob`, and `HiragSemanticEdgesJob` from `core.tools.sharding`.

### 5. Lease & Throttle Backends

- Default configuration uses in-memory leases and token buckets. In multi-process scenarios, provide custom factories that plug in a durable coordination backend.

## Monitoring & Telemetry

The runner emits metric events via an optional `metrics` callback. Recommended payloads:

- `shard.started` – include attempt count.
- `shard.succeeded` – include `rows_written`, duration.
- `shard.failed` – include error, attempt.
- `shard.skipped` – reasons (lease unavailable, invariant failure).

Collect aggregates (partitions processed, failure rate, p95 runtime) for Conveyance dashboards.

## Extending the Toolkit

- Implement alternate `LeaseStore`/`TokenBucket` backends if distributed coordination returns.
- Add adapters for other datasets (patents, Git repos).
- Plug `AqlShardJob` for direct AQL execution via memory client.
- Add resume functionality by seeding `ShardRunner.run(partitions=...)` with failed specs from manifest/lease store.

## Tips & Best Practices

- Align hash partitions with Arango `numberOfShards` to minimize cross-shard writes.
- Keep adapters idempotent—use deterministic `_key` in all UPSERTs.
- Define meaningful invariants; failing invariant raises `PartitionInvariantError` and stops the run.
- Start with lower concurrency; observe lock waits, adjust token buckets accordingly.
- Persist run results and failures for audit/retry.

## References

- Design Spec: `docs/prd/phase2/1_design_sharding_toolkit.md`
- Tests: `tests/core/tools/sharding/test_runner.py`
- CLI: `python -m core.tools.sharding.cli --help`

Contact the graph team for integration support or to register new adapters.
