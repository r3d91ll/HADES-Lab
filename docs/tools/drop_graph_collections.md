# Drop Graph Collections Tool

`core/tools/db/drop_graph_collections.py` provides a controlled way to clear the Phase 2 graph schema in ArangoDB without re-running Phase 1 ingest jobs. It talks to the configured memory client, tears down the named graph if requested, and removes the canonical collection set defined in `GraphCollections`.

## Default behaviour

Running the tool with no flags drops the `HADES_KG` named graph (without cascading collection removal) and then deletes the default collections:

- `entities`
- `clusters`
- `relations`
- `cluster_edges`
- `bridge_cache`
- `weight_config`
- `query_logs`
- `cluster_membership_journal`
- `papers_raw`

The client settings come from `ARANGO_*` environment variables via `DatabaseFactory`, matching normal Phase 2 jobs.

## Safety features

- **Dry runs**: `--dry-run` logs each action instead of executing it.
- **Confirmation prompt**: the command asks you to re-type the database name unless `--yes` is provided.
- **Subset control**: use `--collections` to name specific targets or `--skip` to omit entries from the default list. Names can be either the collection string (e.g. `relations`) or the corresponding `GraphCollections` attribute (e.g. `cluster_edges`).
- **Graph handling**: pass `--skip-graph` to leave the named graph intact, or `--graph-drop-collections` to ask Arango to cascade-delete collections owned by that graph definition.
- **Idempotency**: missing graphs/collections are ignored by default, so repeated runs are safe.

## Example commands

```bash
# Preview exactly what will be dropped without making changes
poetry run python core/tools/db/drop_graph_collections.py --dry-run --yes

# Drop everything except the bridge cache
poetry run python core/tools/db/drop_graph_collections.py --yes --skip bridge_cache

# Only drop relations and cluster edges, leaving the named graph in place
poetry run python core/tools/db/drop_graph_collections.py --skip-graph --collections relations cluster_edges --yes

# Remove the named graph and have Arango remove any collections it owns as part of that graph
poetry run python core/tools/db/drop_graph_collections.py --graph-drop-collections --yes
```

Use these commands after Phase 1 ingest has populated `_key`/`paper_key`; rerunning the Phase 2 builders will recreate the graph without reprocessing the raw source documents.
