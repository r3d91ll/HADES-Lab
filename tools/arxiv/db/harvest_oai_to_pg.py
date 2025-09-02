from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from core.framework.logging import LogManager

try:
    from .config import ArxivDBConfig, load_config
    from .pg import get_connection
    from .utils import normalize_arxiv_id, derive_parts, _parse_any_ts
except Exception:
    from tools.arxiv.db.config import ArxivDBConfig, load_config  # type: ignore
    from tools.arxiv.db.pg import get_connection  # type: ignore
    from tools.arxiv.db.utils import normalize_arxiv_id, derive_parts, _parse_any_ts  # type: ignore

logger = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest OAI-PMH deltas and upsert into Postgres")
    parser.add_argument("--config", type=str, default="tools/arxiv/configs/db.yaml")
    parser.add_argument("--from", dest="from_ts", type=str, default=None, help="ISO timestamp or YYYY-MM-DD")
    parser.add_argument("--until", dest="until_ts", type=str, default=None, help="ISO timestamp or YYYY-MM-DD")
    parser.add_argument("--since-last-success", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    args = parser.parse_args()

    LogManager.setup(log_level="INFO")
    cfg: ArxivDBConfig = load_config(args.config)

    # Resolve windows
    def _parse(s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except Exception:
            try:
                return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)
            except Exception:
                return None

    start = _parse(args.from_ts)
    end = _parse(args.until_ts)

    if args.since_last_success:
        with get_connection(cfg.postgres) as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT COALESCE(until_ts, finished_at)
                    FROM ingest_runs
                    WHERE source = 'oai' AND status = 'succeeded'
                    ORDER BY finished_at DESC NULLS LAST
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row and row[0] and start is None:
                    start = row[0]
            finally:
                try:
                    cur.close()
                except Exception:
                    pass
    if start is None:
        # Default to last 1 day
        start = datetime.now(tz=UTC) - timedelta(days=1)
    if end is None:
        end = datetime.now(tz=UTC)

    logger.info(
        "oai_harvest_start",
        base_url=cfg.oai.base_url,
        start=str(start) if start else None,
        end=str(end) if end else None,
        since_last_success=args.since_last_success,
    )

    # Start ingest_runs entry
    run_id: int | None = None
    with get_connection(cfg.postgres) as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO ingest_runs (source, from_ts, until_ts, status, metrics, last_cursor)
                VALUES ('oai', %s, %s, 'running', '{}'::jsonb, NULL)
                RETURNING id
                """,
                (start, end),
            )
            run_id = int(cur.fetchone()[0])
            conn.commit()
        finally:
            try:
                cur.close()
            except Exception:
                pass

    assert run_id is not None

    processed = 0
    upserted_papers = 0
    inserted_categories = 0
    errors = 0

    def update_metrics(last_cursor: str | None) -> None:
        try:
            with get_connection(cfg.postgres) as conn:
                cur2 = conn.cursor()
                cur2.execute(
                    """
                    UPDATE ingest_runs
                    SET metrics = metrics || %s::jsonb,
                        last_cursor = %s
                    WHERE id = %s
                    """,
                    (
                        json.dumps(
                            {
                                "processed": processed,
                                "upserted_papers": upserted_papers,
                                "inserted_categories": inserted_categories,
                                "errors": errors,
                            }
                        ),
                        last_cursor,
                        run_id,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning("metrics_update_failed", error=str(e))

    def http_get(url: str) -> bytes:
        backoff = cfg.oai.initial_backoff_ms / 1000.0
        for attempt in range(cfg.oai.max_retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    return resp.read()
            except Exception as e:
                if attempt >= cfg.oai.max_retries:
                    raise
                logger.warning("oai_http_retry", error=str(e), attempt=attempt + 1)
                time.sleep(backoff)
                backoff = min(backoff * 2, cfg.oai.max_backoff_ms / 1000.0)
        return b""

    base_params: dict[str, str] = {
        "verb": "ListRecords",
        "metadataPrefix": cfg.oai.metadata_prefix,
        "from": start.astimezone(UTC).strftime("%Y-%m-%d"),
        "until": end.astimezone(UTC).strftime("%Y-%m-%d"),
    }

    resume_token: str | None = None
    batch_rows: list[tuple[Any, ...]] = []
    batch_cats: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal upserted_papers, inserted_categories
        if not batch_rows and not batch_cats:
            return
        try:
            with get_connection(cfg.postgres) as conn:
                conn.autocommit = False
                cur = conn.cursor()
                try:
                    if batch_rows:
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
                            batch_rows,
                        )
                        upserted_papers += cur.rowcount
                    if batch_cats:
                        cur.executemany(
                            """
                            INSERT INTO paper_categories (arxiv_id, category)
                            VALUES (%s, %s)
                            ON CONFLICT DO NOTHING;
                            """,
                            batch_cats,
                        )
                        inserted_categories += cur.rowcount
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    try:
                        cur.close()
                    except Exception:
                        pass
        finally:
            batch_rows.clear()
            batch_cats.clear()

    while True:
        if resume_token:
            params = {"verb": "ListRecords", "resumptionToken": resume_token}
        else:
            params = dict(base_params)
        url = cfg.oai.base_url + "?" + urllib.parse.urlencode(params)
        xml = http_get(url)
        root = None
        try:
            root = urllib.parse.etree.fromstring(xml)  # type: ignore[attr-defined]
        except Exception:
            from xml.etree import ElementTree as ET

            root = ET.fromstring(xml)

        ns = {
            "oai": "http://www.openarchives.org/OAI/2.0/",
            "arXiv": "http://arxiv.org/OAI/arXiv/",
        }
        records = root.findall(".//oai:record", ns)
        for rec in records:
            try:
                md = rec.find("oai:metadata", ns)
                if md is None:
                    continue
                ax = md.find("arXiv:arXiv", ns)
                if ax is None:
                    continue
                arxiv_id = normalize_arxiv_id(ax.findtext("arXiv:id", default="", namespaces=ns))
                title = (ax.findtext("arXiv:title", default="", namespaces=ns) or "").strip()
                abstract = ax.findtext("arXiv:abstract", default=None, namespaces=ns)
                cats_str = ax.findtext("arXiv:categories", default="", namespaces=ns) or ""
                categories = [c for c in cats_str.split() if c]
                primary_category = categories[0] if categories else None
                created_s = ax.findtext("arXiv:created", default=None, namespaces=ns)
                updated_s = ax.findtext("arXiv:updated", default=None, namespaces=ns)
                pub_dt = _parse_any_ts(created_s) if created_s else None
                upd_dt = _parse_any_ts(updated_s) if updated_s else None
                year, month, yymm = derive_parts(pub_dt)
                doi = ax.findtext("arXiv:doi", default=None, namespaces=ns)
                license_ = ax.findtext("arXiv:license", default=None, namespaces=ns)
                journal_ref = ax.findtext("arXiv:journal-ref", default=None, namespaces=ns)

                batch_rows.append(
                    (
                        arxiv_id,
                        title,
                        abstract,
                        primary_category,
                        pub_dt.isoformat() if pub_dt else None,
                        upd_dt.isoformat() if upd_dt else None,
                        year,
                        month,
                        yymm,
                        doi,
                        license_,
                        journal_ref,
                    )
                )
                for c in categories:
                    batch_cats.append((arxiv_id, c))
                processed += 1

                if processed % args.checkpoint_interval == 0:
                    flush()
                    update_metrics(resume_token)

            except Exception as e:
                errors += 1
                logger.warning("oai_record_parse_error", error=str(e))

        # Resumption token
        rt_elem = root.find(".//oai:resumptionToken", ns)
        resume_token = rt_elem.text.strip() if rt_elem is not None and rt_elem.text else None

        # Rate limiting
        time.sleep(max(0.0, 1.0 / max(0.1, cfg.oai.rate_limit_rps)))

        if not resume_token:
            break

    # Final flush and mark run success
    flush()
    update_metrics(None)

    with get_connection(cfg.postgres) as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE ingest_runs SET status='succeeded', finished_at = now() WHERE id = %s",
                (run_id,),
            )
            conn.commit()
        finally:
            try:
                cur.close()
            except Exception:
                pass


if __name__ == "__main__":  # pragma: no cover
    main()
