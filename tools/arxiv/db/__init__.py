"""
ArXiv Metadata Service (Postgres)
---------------------------------

Package providing configuration, DB helpers, Alembic migrations, and CLI tools
to support the Postgres-backed arXiv metadata service described in
`docs/prd/arxiv_metadata_service_prd.md`.

Initial delivery focuses on scaffolding and CLI skeletons; functionality will
be built out incrementally across milestones (M0–M4).
"""

__all__ = [
    "config",
    "pg",
]

