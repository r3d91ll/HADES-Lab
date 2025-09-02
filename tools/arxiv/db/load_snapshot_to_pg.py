from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from core.framework.logging import LogManager

# Support running as script or module
try:  # when executed as a package module
    from .config import ArxivDBConfig, load_config
    from .utils import derive_parts, normalize_arxiv_id, parse_published_at, _parse_any_ts
    from .pg import get_connection
except Exception:  # when executed directly via path
    from tools.arxiv.db.config import ArxivDBConfig, load_config  # type: ignore
    from tools.arxiv.db.utils import (  # type: ignore
        derive_parts,
        normalize_arxiv_id,
        parse_published_at,
        _parse_any_ts,
    )
    from tools.arxiv.db.pg import get_connection  # type: ignore

logger = structlog.get_logger()


@dataclass
class PaperRow:
    arxiv_id: str
    title: str
    abstract: str | None
    primary_category: str | None
    published_at: str | None  # ISO string for COPY; DB will cast
    updated_at: str | None
    year: int | None
    month: int | None
    yymm: str | None
    doi: str | None
    license: str | None
    journal_ref: str | None


def iter_snapshot(path: Path, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_rows is not None and count >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                count += 1
                yield obj
            except json.JSONDecodeError:
                logger.warning("bad_json_line")


def to_paper_row(obj: dict[str, Any]) -> tuple[PaperRow, list[str]]:
    arxiv_id_raw: str = obj.get("id") or obj.get("arxiv_id") or ""
    arxiv_id = normalize_arxiv_id(arxiv_id_raw)

    pub, upd = parse_published_at(obj)
    year, month, yymm = derive_parts(pub)

    cats = obj.get("categories") or []
    if isinstance(cats, str):
        cats_list = cats.split()
    elif isinstance(cats, list):
        cats_list = [str(c) for c in cats]
    else:
        cats_list = []

    row = PaperRow(
        arxiv_id=arxiv_id,
        title=(obj.get("title") or "").strip(),
        abstract=(obj.get("abstract") or None),
        primary_category=obj.get("primary_category") or (cats_list[0] if cats_list else None),
        published_at=pub.isoformat() if pub else None,
        updated_at=upd.isoformat() if upd else None,
        year=year,
        month=month,
        yymm=yymm,
        doi=obj.get("doi"),
        license=obj.get("license"),
        journal_ref=obj.get("journal_ref"),
    )
    return row, cats_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Load arXiv snapshot JSONL into Postgres (COPY/UPSERT)")
    parser.add_argument("--config", type=str, default="tools/arxiv/configs/db.yaml")
    parser.add_argument("--truncate-first", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    LogManager.setup(log_level="INFO")
    cfg: ArxivDBConfig = load_config(args.config)
    logger.info("snapshot_loader_start", path=cfg.snapshot.path, truncate=args.truncate_first)

    snapshot_path = Path(cfg.snapshot.path)
    if not snapshot_path.exists():
        logger.error("snapshot_missing", path=str(snapshot_path))
        raise SystemExit(2)

    total_rows = 0
    total_categories = 0
    total_versions = 0
    batch_rows: list[PaperRow] = []
    batch_cats: list[tuple[str, str]] = []
    batch_vers: list[tuple[str, int, str | None]] = []

    did_truncate: bool = False

    def flush_batches() -> tuple[int, int, int]:
        nonlocal did_truncate
        if not batch_rows and not batch_cats and not batch_vers:
            return 0, 0, 0

        inserted_papers = 0
        inserted_cats = 0
        upserted_versions = 0

        with get_connection(cfg.postgres) as conn:
            conn.autocommit = False
            cur = conn.cursor()
            try:
                if args.truncate_first and not did_truncate:
                    # Truncate all related tables in a single statement to satisfy FK constraints
                    cur.execute(
                        "TRUNCATE TABLE paper_categories, versions, papers RESTART IDENTITY CASCADE;"
                    )
                    did_truncate = True

                # UPSERT papers in batch
                paper_params = [
                    (
                        r.arxiv_id,
                        r.title,
                        r.abstract,
                        r.primary_category,
                        r.published_at,
                        r.updated_at,
                        r.year,
                        r.month,
                        r.yymm,
                        r.doi,
                        r.license,
                        r.journal_ref,
                    )
                    for r in batch_rows
                ]
                if paper_params:
                    cur.executemany(
                        """
                        INSERT INTO papers (
                          arxiv_id, title, abstract, primary_category,
                          published_at, updated_at, year, month, yymm,
                          doi, license, journal_ref
                        ) VALUES (
                          %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                        )
                        ON CONFLICT (arxiv_id) DO UPDATE SET
                          title = EXCLUDED.title,
                          abstract = EXCLUDED.abstract,
                          primary_category = EXCLUDED.primary_category,
                          published_at = EXCLUDED.published_at,
                          updated_at = EXCLUDED.updated_at,
                          year = EXCLUDED.year,
                          month = EXCLUDED.month,
                          yymm = EXCLUDED.yymm,
                          doi = EXCLUDED.doi,
                          license = EXCLUDED.license,
                          journal_ref = EXCLUDED.journal_ref
                        ;
                        """,
                        paper_params,
                    )
                    inserted_papers = cur.rowcount

                # UPSERT categories
                if batch_cats:
                    cur.executemany(
                        """
                        INSERT INTO paper_categories (arxiv_id, category)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING;
                        """,
                        batch_cats,
                    )
                    inserted_cats = cur.rowcount

                # UPSERT versions
                if batch_vers:
                    cur.executemany(
                        """
                        INSERT INTO versions (arxiv_id, version, created_at)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (arxiv_id, version) DO UPDATE SET
                          created_at = EXCLUDED.created_at;
                        """,
                        batch_vers,
                    )
                    upserted_versions = cur.rowcount

                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error("snapshot_flush_failed", error=str(e))
                raise
            finally:
                try:
                    cur.close()
                except Exception:
                    pass

        batch_rows.clear()
        batch_cats.clear()
        batch_vers.clear()
        return inserted_papers, inserted_cats, upserted_versions

    batch_size = max(1, int(cfg.snapshot.batch_size))

    total_ip = total_ic = total_iv = 0

    for obj in iter_snapshot(snapshot_path, max_rows=args.max_rows):
        row, cats = to_paper_row(obj)
        batch_rows.append(row)
        for c in cats:
            batch_cats.append((row.arxiv_id, c))
        # versions: optional array of {version: 'vN', created: ts}
        versions = obj.get("versions") or []
        if isinstance(versions, list):
            for v in versions:
                if not isinstance(v, dict):
                    continue
                vraw = v.get("version")
                try:
                    vnum = int(str(vraw).lstrip("vV")) if vraw is not None else 1
                except Exception:
                    vnum = 1
                created_s = v.get("created")
                created_iso: str | None = None
                if isinstance(created_s, str):
                    dt = _parse_any_ts(created_s)
                    created_iso = dt.isoformat() if dt else None
                batch_vers.append((row.arxiv_id, vnum, created_iso))

        total_rows += 1
        total_categories += len(cats)
        total_versions += len(versions) if isinstance(versions, list) else 0

        if len(batch_rows) >= batch_size:
            ip, ic, iv = flush_batches()
            total_ip += ip
            total_ic += ic
            total_iv += iv
            logger.info("snapshot_batch_flushed", papers=ip, cats=ic, versions=iv)

        if total_rows % 100_000 == 0:
            logger.info("parsed_rows", rows=total_rows)

    # Flush remaining
    ip, ic, iv = flush_batches()
    total_ip += ip
    total_ic += ic
    total_iv += iv

    logger.info(
        "snapshot_loader_summary",
        rows=total_rows,
        categories=total_categories,
        versions=total_versions,
        inserted_papers=total_ip,
        inserted_categories=total_ic,
        upserted_versions=total_iv,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
