from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20250901_000001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # DDL based on PRD Appendix, with safe IF NOT EXISTS where applicable
    op.execute(
        r"""
        CREATE TABLE IF NOT EXISTS papers (
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

        CREATE TABLE IF NOT EXISTS paper_categories (
          arxiv_id text REFERENCES papers(arxiv_id) ON DELETE CASCADE,
          category text NOT NULL,
          PRIMARY KEY (arxiv_id, category)
        );

        CREATE TABLE IF NOT EXISTS versions (
          arxiv_id text REFERENCES papers(arxiv_id) ON DELETE CASCADE,
          version int NOT NULL,
          created_at timestamptz,
          PRIMARY KEY (arxiv_id, version)
        );

        CREATE TABLE IF NOT EXISTS ingest_runs (
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
        CREATE INDEX IF NOT EXISTS idx_papers_published_at ON papers (published_at);
        CREATE INDEX IF NOT EXISTS idx_papers_year_month ON papers (year, month);
        CREATE INDEX IF NOT EXISTS idx_papers_primary_category ON papers (primary_category);
        CREATE INDEX IF NOT EXISTS idx_papers_yymm ON papers (yymm);
        CREATE INDEX IF NOT EXISTS idx_paper_categories_cat_id ON paper_categories (category, arxiv_id);
        """
    )

    # FTS index (optional, can be slow on large tables). Create but tolerate absence early on.
    op.execute(
        r"""
        DO $$
        BEGIN
          CREATE INDEX IF NOT EXISTS papers_fts_idx
          ON papers USING gin (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(abstract,'')));
        EXCEPTION WHEN OTHERS THEN
          -- ignore errors if extension not available; will be re-applied later
          NULL;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        r"""
        DROP INDEX IF EXISTS papers_fts_idx;
        DROP INDEX IF EXISTS idx_paper_categories_cat_id;
        DROP INDEX IF EXISTS idx_papers_yymm;
        DROP INDEX IF EXISTS idx_papers_primary_category;
        DROP INDEX IF EXISTS idx_papers_year_month;
        DROP INDEX IF EXISTS idx_papers_published_at;

        DROP TABLE IF EXISTS ingest_runs;
        DROP TABLE IF EXISTS versions;
        DROP TABLE IF EXISTS paper_categories;
        DROP TABLE IF EXISTS papers;
        """
    )
