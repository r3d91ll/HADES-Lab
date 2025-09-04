from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from datetime import datetime
import os
import re

import structlog

from core.framework.logging import LogManager

# Support running as script or module
try:
    from .config import ArxivDBConfig, load_config
except Exception:
    from tools.arxiv.db.config import ArxivDBConfig, load_config  # type: ignore

logger = structlog.get_logger()


def build_where(
    start_year: int | None,
    end_year: int | None,
    months: list[int] | None,
    yymm_range: tuple[str, str] | None,
    categories: list[str] | None,
    keywords: str | None,
    with_pdf: bool,
    missing_pdf: bool,
    table_alias: str | None = None,
) -> tuple[str, list[Any]]:
    where: list[str] = []
    params: list[Any] = []
    prefix = f"{table_alias}." if table_alias else ""

    if start_year is not None:
        where.append(f"{prefix}year >= %s")
        params.append(start_year)
    if end_year is not None:
        where.append(f"{prefix}year <= %s")
        params.append(end_year)
    if months:
        placeholders = ",".join(["%s"] * len(months))
        where.append(f"{prefix}month IN ({placeholders})")
        params.extend(months)
    if yymm_range:
        where.append(f"{prefix}yymm >= %s AND {prefix}yymm <= %s")
        params.extend(list(yymm_range))
    if categories:
        placeholders = ",".join(["%s"] * len(categories))
        where.append(
            f"{prefix}arxiv_id IN (SELECT arxiv_id FROM paper_categories WHERE category IN ({placeholders}))"
        )
        params.extend(categories)
    if keywords:
        # Use websearch_to_tsquery for friendlier syntax (quotes, AND/OR, etc.)
        # Convert legacy '|' separators into OR for websearch syntax
        kw = re.sub(r"\|", " OR ", keywords)
        # Collapse whitespace
        kw = re.sub(r"\s+", " ", kw).strip()
        where.append(
            f"to_tsvector('english', coalesce({prefix}title,'') || ' ' || coalesce({prefix}abstract,'')) "
            f"@@ websearch_to_tsquery('english', %s)"
        )
        params.append(kw)
    if with_pdf and not missing_pdf:
        where.append(f"{prefix}has_pdf = true")
    if missing_pdf and not with_pdf:
        where.append(f"{prefix}has_pdf = false")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    return where_sql, params


def build_query(
    start_year: int | None,
    end_year: int | None,
    months: list[int] | None,
    yymm_range: tuple[str, str] | None,
    categories: list[str] | None,
    keywords: str | None,
    with_pdf: bool,
    missing_pdf: bool,
    per_year_cap: int | None,
    per_month_cap: int | None,
) -> tuple[str, list[Any], str]:
    where_sql, params = build_where(
        start_year,
        end_year,
        months,
        yymm_range,
        categories,
        keywords,
        with_pdf,
        missing_pdf,
        None,
    )

    base = f"SELECT arxiv_id, published_at, year, month, yymm FROM papers {where_sql} ORDER BY published_at, arxiv_id"

    # Apply caps by grouping
    monthly_sql = f"SELECT arxiv_id, year, month, yymm FROM ({base}) x"
    if per_month_cap is not None:
        q = (
            "SELECT arxiv_id FROM ("
            " SELECT arxiv_id, published_at, year, month, yymm,"
            "        ROW_NUMBER() OVER (PARTITION BY year, month ORDER BY published_at, arxiv_id) AS rn"
            f"   FROM ({base}) x"
            ") y WHERE rn <= %s"
        )
        params_q = list(params) + [per_month_cap]
        monthly_sql = (
            "SELECT arxiv_id, year, month, yymm FROM ("
            " SELECT arxiv_id, published_at, year, month, yymm,"
            "        ROW_NUMBER() OVER (PARTITION BY year, month ORDER BY published_at, arxiv_id) AS rn"
            f"   FROM ({base}) x"
            ") y WHERE rn <= %s"
        )
        return q, params_q, monthly_sql
    if per_year_cap is not None:
        q = (
            "SELECT arxiv_id FROM ("
            " SELECT arxiv_id, published_at, year,"
            "        ROW_NUMBER() OVER (PARTITION BY year ORDER BY published_at, arxiv_id) AS rn"
            f"   FROM ({base}) x"
            ") y WHERE rn <= %s"
        )
        params_q = list(params) + [per_year_cap]
        monthly_sql = (
            "SELECT arxiv_id, year, month, yymm FROM ("
            " SELECT arxiv_id, published_at, year, month, yymm,"
            "        ROW_NUMBER() OVER (PARTITION BY year ORDER BY published_at, arxiv_id) AS rn"
            f"   FROM ({base}) x"
            ") y WHERE rn <= %s"
        )
        return q, params_q, monthly_sql
    return f"SELECT arxiv_id FROM ({base}) x", params, monthly_sql


def _update_symlink(base: Path, target: Path) -> None:
    try:
        if base.exists() or base.is_symlink():
            base.unlink()
        # Create relative symlink for portability
        base.symlink_to(os.path.relpath(target, base.parent))
    except Exception:
        # Fallback: copy contents when symlink not allowed
        try:
            if target.is_file():
                base.write_bytes(target.read_bytes())
        except Exception:
            pass


def write_outputs(
    out_dir: Path,
    prefix: str,
    ids: Sequence[str],
    with_pdf: bool,
    missing_pdf: bool,
    stats: dict[str, Any],
    monthly: dict[tuple[int, int], list[str]] | None = None,
    suffix: str | None = None,
    update_symlinks: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir = out_dir / "monthly"
    monthly_dir.mkdir(exist_ok=True)

    stamp = suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
    master_ts = out_dir / f"{prefix}_sweep_{stamp}.txt"
    master_path = out_dir / f"{prefix}_sweep.txt"
    with master_ts.open("w", encoding="utf-8") as f:
        for i in ids:
            f.write(i + "\n")
    if update_symlinks:
        _update_symlink(master_path, master_ts)

    if with_pdf:
        with (out_dir / f"{prefix}_with_pdfs_{stamp}.txt").open("w", encoding="utf-8") as f:
            for i in ids:
                f.write(i + "\n")
        if update_symlinks:
            _update_symlink(out_dir / f"{prefix}_with_pdfs.txt", out_dir / f"{prefix}_with_pdfs_{stamp}.txt")
    if missing_pdf:
        with (out_dir / f"{prefix}_missing_pdfs_{stamp}.txt").open("w", encoding="utf-8") as f:
            for i in ids:
                f.write(i + "\n")
        if update_symlinks:
            _update_symlink(out_dir / f"{prefix}_missing_pdfs.txt", out_dir / f"{prefix}_missing_pdfs_{stamp}.txt")

    stats_ts = out_dir / f"{prefix}_sweep_stats_{stamp}.json"
    with stats_ts.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    if update_symlinks:
        _update_symlink(out_dir / f"{prefix}_stats.json", stats_ts)

    if monthly:
        for (year, month), mids in monthly.items():
            fname = f"arxiv_ids_{year}_{month:02d}_{stamp}.txt"
            fpath = monthly_dir / fname
            with fpath.open("w", encoding="utf-8") as f:
                for i in mids:
                    f.write(i + "\n")


def collect_ids(cfg: ArxivDBConfig, sql: str, params: list[Any]) -> list[str]:
    # Local import to avoid hard dependency on import-time
    try:
        from .pg import get_connection  # type: ignore
    except Exception:
        from tools.arxiv.db.pg import get_connection  # type: ignore

    ids: list[str] = []
    with get_connection(cfg.postgres) as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql, params)
            for row in cur.fetchall():
                ids.append(row[0])
        finally:
            try:
                cur.close()
            except Exception:
                pass
    return ids


def collect_stats(cfg: ArxivDBConfig, where_sql: str, params: list[Any]) -> dict[str, Any]:
    try:
        from .pg import get_connection  # type: ignore
    except Exception:
        from tools.arxiv.db.pg import get_connection  # type: ignore

    stats = {"total": 0, "by_year": {}, "by_month": {}, "by_category": {}}
    with get_connection(cfg.postgres) as conn:
        cur = conn.cursor()
        try:
            # Total
            cur.execute(f"SELECT count(*) FROM papers {where_sql}", params)
            stats["total"] = int(cur.fetchone()[0])

            # By year
            cur.execute(
                f"SELECT year, count(*) FROM papers {where_sql} GROUP BY year ORDER BY year",
                params,
            )
            for y, c in cur.fetchall():
                if y is not None:
                    stats["by_year"][str(int(y))] = int(c)

            # By month (YYYY-MM)
            cur.execute(
                f"SELECT yymm, count(*) FROM papers {where_sql} GROUP BY yymm ORDER BY yymm",
                params,
            )
            for ym, c in cur.fetchall():
                if ym:
                    # convert yymm (e.g., '2407') to '2024-07'
                    y = 2000 + int(str(ym)[:2])
                    m = int(str(ym)[2:])
                    key = f"{y}-{m:02d}"
                    stats["by_month"][key] = int(c)

            # By category: rebuild WHERE with table alias 'p'
            # Use the same filters as provided by original where_sql/params
            where_p_sql = where_sql.replace("year", "p.year").replace("month", "p.month").replace(
                "yymm", "p.yymm"
            ).replace("has_pdf", "p.has_pdf").replace("primary_category", "p.primary_category").replace(
                "title", "p.title"
            ).replace("abstract", "p.abstract")
            cur.execute(
                f"SELECT c.category, count(*) FROM paper_categories c JOIN papers p USING (arxiv_id) {where_p_sql} GROUP BY c.category ORDER BY c.category",
                params,
            )
            for cat, c in cur.fetchall():
                stats["by_category"][str(cat)] = int(c)
        finally:
            try:
                cur.close()
            except Exception:
                pass
    return stats


def main() -> None:
    """
    CLI entrypoint: parse arguments, query the database for arXiv IDs, gather stats, and write export files.
    
    Parses command-line options (filters, caps, PDF flags, output directory, and whether to write monthly lists), builds the SQL needed to select matching arXiv IDs, executes that query to collect IDs, computes summary statistics, optionally collects per-month ID lists, and writes the resulting ID lists and stats to disk. Errors while querying or computing statistics are logged and cause safe fallbacks (empty ID list or minimal stats); failures collecting monthly lists disable monthly output. Uses the loaded DB configuration and the module's helper functions for query construction, database access, and file output. Returns None.
    """
    parser = argparse.ArgumentParser(description="Export arXiv ID lists and stats from Postgres")
    parser.add_argument("--config", type=str, default="tools/arxiv/configs/db.yaml")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--months", type=int, nargs="*", default=None)
    parser.add_argument("--yymm-range", type=str, nargs=2, default=None, metavar=("START","END"))
    parser.add_argument("--categories", type=str, nargs="*", default=None)
    parser.add_argument("--keywords", type=str, default=None)
    parser.add_argument("--with-pdf", action="store_true")
    parser.add_argument("--missing-pdf", action="store_true")
    parser.add_argument("--per-year-cap", type=int, default=None)
    parser.add_argument("--per-month-cap", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/arxiv_collections/",
    )
    parser.add_argument("--write-monthly-lists", action="store_true")
    args = parser.parse_args()

    LogManager.setup(log_level="INFO")
    cfg = load_config(args.config)

    yymm_range = tuple(args.yymm_range) if args.yymm_range else None
    sql, params, monthly_sql = build_query(
        args.start_year,
        args.end_year,
        args.months,
        yymm_range,  # type: ignore
        args.categories,
        args.keywords,
        args.with_pdf,
        args.missing_pdf,
        args.per_year_cap,
        args.per_month_cap,
    )

    logger.info("export_query", sql=sql, params=params)
    try:
        ids = collect_ids(cfg, sql, params)
    except Exception as e:
        logger.error("export_query_failed", error=str(e))
        logger.info("hint", msg="Install DB driver and configure ARXIV_PG_DSN to run exports")
        ids = []

    # Stats via SQL
    where_sql, where_params = build_where(
        args.start_year,
        args.end_year,
        args.months,
        yymm_range,  # type: ignore
        args.categories,
        args.keywords,
        args.with_pdf,
        args.missing_pdf,
        None,
    )
    try:
        stats = collect_stats(cfg, where_sql, where_params)
    except Exception as e:
        logger.error("stats_failed", error=str(e))
        stats = {"total": len(ids), "by_year": {}, "by_month": {}, "by_category": {}}

    out_dir = Path(args.out_dir)
    prefix = "arxiv_ids"
    monthly: dict[tuple[int, int], list[str]] | None = None
    if args.write_monthly_lists:
        monthly = {}
        try:
            # collect arxiv_id, year, month respecting caps
            from .pg import get_connection

            with get_connection(cfg.postgres) as conn:
                cur = conn.cursor()
                try:
                    cur.execute(monthly_sql, params)
                    for aid, yr, mo, _yymm in cur.fetchall():
                        if yr is None or mo is None:
                            continue
                        monthly.setdefault((int(yr), int(mo)), []).append(aid)
                finally:
                    try:
                        cur.close()
                    except Exception:
                        pass
        except Exception as e:
            logger.error("monthly_collect_failed", error=str(e))
            monthly = None

    write_outputs(out_dir, prefix, ids, args.with_pdf, args.missing_pdf, stats, monthly)
    logger.info("export_done", count=len(ids), out_dir=str(out_dir))


if __name__ == "__main__":  # pragma: no cover
    main()
