from __future__ import annotations

import argparse
import glob
from pathlib import Path
import re

import structlog

from core.framework.logging import LogManager

# Support running as script or module
try:
    from .config import ArxivDBConfig, load_config
    from .pg import get_connection
    from .utils import normalize_arxiv_id
except Exception:
    from tools.arxiv.db.config import ArxivDBConfig, load_config  # type: ignore
    from tools.arxiv.db.pg import get_connection  # type: ignore
    from tools.arxiv.db.utils import normalize_arxiv_id  # type: ignore

logger = structlog.get_logger()

NEW_ID_RE = re.compile(r"^(?P<yymm>\d{4})\.(?P<num>\d{4,5})(?:v\d+)?$")
OLD_FLAT_RE = re.compile(r"^(?P<cat>[a-z\-]+)(?P<num>\d{6,7})(?:v\d+)?$", re.IGNORECASE)


def _candidates_from_stem(stem: str) -> list[str]:
    """Return possible arXiv ID candidates from a file stem.

    Handles new-style (YYMM.NNNN/NNNNN) and old-style (catYYMMNNN...) forms.
    """
    stem = stem.strip()
    out: list[str] = []
    m = NEW_ID_RE.match(stem)
    if m:
        # Normalize to the captured form without version
        out.append(f"{m.group('yymm')}.{m.group('num')}")
    m2 = OLD_FLAT_RE.match(stem)
    if m2:
        cat = m2.group('cat').lower()
        num = m2.group('num')
        out.append(f"{cat}/{num}")
    return out


def scan_pdfs(globs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pattern in globs:
        for fp in glob.iglob(pattern, recursive=True):
            p = Path(fp)
            if p.is_file() and p.suffix.lower() == ".pdf":
                cands = _candidates_from_stem(p.stem)
                for cid in cands:
                    mapping.setdefault(cid, str(p))
    return mapping


def scan_latex(globs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    # Broad glob: infer ID from parent dir or filename
    for pattern in globs:
        for fp in glob.iglob(pattern, recursive=True):
            p = Path(fp)
            if p.is_file() and p.suffix.lower() == ".tex":
                # Prefer parent directory name as ID, else filename stem
                parent_name = p.parent.name
                cands = _candidates_from_stem(parent_name)
                if not cands:
                    cands = _candidates_from_stem(p.stem)
                for cid in cands:
                    mapping.setdefault(cid, str(p.parent))
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan local filesystem for arXiv PDF/LaTeX artifacts and update DB flags")
    parser.add_argument("--config", type=str, default="tools/arxiv/configs/db.yaml")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--reset-missing", action="store_true")
    args = parser.parse_args()

    LogManager.setup(log_level="INFO")
    cfg: ArxivDBConfig = load_config(args.config)

    do_pdf = args.pdf or (not args.pdf and not args.latex)
    do_latex = args.latex or (not args.pdf and not args.latex)

    pdf_map: dict[str, str] = {}
    latex_map: dict[str, str] = {}

    if do_pdf:
        logger.info("scan_pdfs_start", globs=cfg.artifacts.pdf_globs)
        pdf_map = scan_pdfs(cfg.artifacts.pdf_globs)
        logger.info("scan_pdfs_done", count=len(pdf_map))

    if do_latex:
        logger.info("scan_latex_start", globs=cfg.artifacts.latex_globs)
        latex_map = scan_latex(cfg.artifacts.latex_globs)
        logger.info("scan_latex_done", count=len(latex_map))

    # Apply updates to DB
    updated_pdf = 0
    updated_latex = 0
    reset_pdf = 0
    reset_latex = 0

    with get_connection(cfg.postgres) as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            # Temp tables for efficient set-based updates
            if pdf_map:
                cur.execute("CREATE TEMP TABLE tmp_pdf (arxiv_id text PRIMARY KEY, pdf_path text) ON COMMIT DROP;")
                cur.executemany(
                    "INSERT INTO tmp_pdf (arxiv_id, pdf_path) VALUES (%s, %s)",
                    list(pdf_map.items()),
                )
                cur.execute(
                    """
                    UPDATE papers p
                    SET has_pdf = TRUE,
                        pdf_path = t.pdf_path
                    FROM tmp_pdf t
                    WHERE p.arxiv_id = t.arxiv_id
                    """
                )
                updated_pdf = cur.rowcount

            if latex_map:
                cur.execute(
                    "CREATE TEMP TABLE tmp_latex (arxiv_id text PRIMARY KEY, latex_path text) ON COMMIT DROP;"
                )
                cur.executemany(
                    "INSERT INTO tmp_latex (arxiv_id, latex_path) VALUES (%s, %s)",
                    list(latex_map.items()),
                )
                cur.execute(
                    """
                    UPDATE papers p
                    SET has_latex = TRUE,
                        latex_path = t.latex_path
                    FROM tmp_latex t
                    WHERE p.arxiv_id = t.arxiv_id
                    """
                )
                updated_latex = cur.rowcount

            if args.reset_missing and (do_pdf or do_latex):
                if do_pdf:
                    if pdf_map:
                        cur.execute(
                            """
                            UPDATE papers p
                            SET has_pdf = FALSE,
                                pdf_path = NULL
                            WHERE p.arxiv_id NOT IN (SELECT arxiv_id FROM tmp_pdf)
                              AND p.has_pdf = TRUE
                            """
                        )
                    else:
                        cur.execute(
                            "UPDATE papers SET has_pdf = FALSE, pdf_path = NULL WHERE has_pdf = TRUE"
                        )
                    reset_pdf = cur.rowcount
                if do_latex:
                    if latex_map:
                        cur.execute(
                            """
                            UPDATE papers p
                            SET has_latex = FALSE,
                                latex_path = NULL
                            WHERE p.arxiv_id NOT IN (SELECT arxiv_id FROM tmp_latex)
                              AND p.has_latex = TRUE
                            """
                        )
                    else:
                        cur.execute(
                            "UPDATE papers SET has_latex = FALSE, latex_path = NULL WHERE has_latex = TRUE"
                        )
                    reset_latex = cur.rowcount

            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("artifact_update_failed", error=str(e))
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass

    logger.info(
        "artifact_scan_summary",
        pdf_count=len(pdf_map),
        latex_count=len(latex_map),
        updated_pdf=updated_pdf,
        updated_latex=updated_latex,
        reset_pdf=reset_pdf,
        reset_latex=reset_latex,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
