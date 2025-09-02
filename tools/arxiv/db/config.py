from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PostgresConfig:
    dsn: str
    schema: str = "public"
    statement_timeout_ms: int | None = 600_000
    pool_size: int = 5
    max_overflow: int = 10


@dataclass(frozen=True)
class SnapshotConfig:
    path: str = "/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json"
    batch_size: int = 5000


@dataclass(frozen=True)
class OAIConfig:
    base_url: str = "https://export.arxiv.org/oai2"
    metadata_prefix: str = "arXiv"
    rate_limit_rps: float = 2.0
    initial_backoff_ms: int = 100
    max_backoff_ms: int = 5000
    max_retries: int = 5


@dataclass(frozen=True)
class ArtifactsConfig:
    # Support multiple globs to accommodate differing layouts
    pdf_globs: list[str] = field(
        default_factory=lambda: [
            "/bulk-store/arxiv-data/pdf/*/*.pdf",
            "/bulk-store/arxiv-data/latex/*/*.pdf",
        ]
    )
    latex_globs: list[str] = field(
        default_factory=lambda: [
            "/bulk-store/arxiv-data/latex/**/*.tex",
        ]
    )


@dataclass(frozen=True)
class ArxivDBConfig:
    postgres: PostgresConfig
    snapshot: SnapshotConfig = SnapshotConfig()
    oai: OAIConfig = OAIConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()


def _merge_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    # Environment override for DSN
    dsn = os.getenv("ARXIV_PG_DSN")
    if dsn:
        cfg.setdefault("postgres", {})["dsn"] = dsn
    else:
        # Fallback: build DSN from common POSTGRES_* env vars if provided
        pg_user = os.getenv("POSTGRES_USER") or os.getenv("PGUSER")
        pg_password = os.getenv("POSTGRES_PASSWORD") or os.getenv("PGPASSWORD")
        pg_host = os.getenv("POSTGRES_HOST") or os.getenv("PGHOST") or "localhost"
        pg_port = os.getenv("POSTGRES_PORT") or os.getenv("PGPORT") or "5432"
        pg_db = os.getenv("POSTGRES_DB") or os.getenv("PGDATABASE") or "arxiv"
        if pg_user and pg_password:
            cfg.setdefault("postgres", {})["dsn"] = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    # Optional: statement timeout override
    st = os.getenv("ARXIV_PG_STATEMENT_TIMEOUT_MS")
    if st:
        try:
            cfg.setdefault("postgres", {})["statement_timeout_ms"] = int(st)
        except ValueError:
            # Ignore invalid env override
            pass
    return cfg


def load_config(path: str | Path) -> ArxivDBConfig:
    """Load YAML config and apply environment overrides.

    Expected YAML structure:
      postgres: { dsn, schema, statement_timeout_ms, pool_size, max_overflow }
      snapshot: { path, batch_size }
      oai: { base_url, metadata_prefix, rate_limit_rps, ... }
      artifacts: { pdf_glob, latex_glob }
    """
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    else:
        loaded = {}

    cfg = _merge_env_overrides(loaded)

    pg = cfg.get("postgres", {})
    snapshot = cfg.get("snapshot", {})
    oai = cfg.get("oai", {})
    artifacts = cfg.get("artifacts", {})

    # Basic validation for DSN presence
    dsn = pg.get("dsn") or os.getenv("ARXIV_PG_DSN")
    if not dsn:
        # Keep object constructible; CLI will surface clearer error before DB use
        dsn = "postgresql://user:pass@localhost:5432/arxiv"

    return ArxivDBConfig(
        postgres=PostgresConfig(
            dsn=dsn,
            schema=pg.get("schema", "public"),
            statement_timeout_ms=pg.get("statement_timeout_ms", 600_000),
            pool_size=pg.get("pool_size", 5),
            max_overflow=pg.get("max_overflow", 10),
        ),
        snapshot=SnapshotConfig(
            path=snapshot.get("path", "/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json"),
            batch_size=int(snapshot.get("batch_size", 5000)),
        ),
        oai=OAIConfig(
            base_url=oai.get("base_url", "https://export.arxiv.org/oai2"),
            metadata_prefix=oai.get("metadata_prefix", "arXiv"),
            rate_limit_rps=float(oai.get("rate_limit_rps", 2.0)),
            initial_backoff_ms=int(oai.get("initial_backoff_ms", 100)),
            max_backoff_ms=int(oai.get("max_backoff_ms", 5000)),
            max_retries=int(oai.get("max_retries", 5)),
        ),
        artifacts=ArtifactsConfig(
            pdf_globs=(
                list(artifacts.get("pdf_globs", []))
                if "pdf_globs" in artifacts
                else ([artifacts["pdf_glob"]] if "pdf_glob" in artifacts else [
                    "/bulk-store/arxiv-data/pdf/*/*.pdf",
                    "/bulk-store/arxiv-data/latex/*/*.pdf",
                ])
            ),
            latex_globs=(
                list(artifacts.get("latex_globs", []))
                if "latex_globs" in artifacts
                else ([artifacts["latex_glob"]] if "latex_glob" in artifacts else [
                    "/bulk-store/arxiv-data/latex/**/*.tex",
                ])
            ),
        ),
    )
