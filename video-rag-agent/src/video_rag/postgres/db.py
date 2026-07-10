"""Postgres/pgvector connection and schema initialization."""

import psycopg
from psycopg.rows import dict_row

from video_rag.config import settings


def get_conn() -> psycopg.Connection:
    """Get a connection to the Postgres database with dict-row factory."""
    conn = psycopg.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        dbname=settings.pg_db,
        user=settings.pg_user,
        password=settings.pg_password,
        row_factory=dict_row,
    )
    conn.autocommit = True
    return conn


CREATE_VECTORS = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
""" .strip()

CREATE_VIDEOS = """
CREATE TABLE IF NOT EXISTS videos (
    id              UUID PRIMARY KEY,
    external_id     VARCHAR(32) UNIQUE NOT NULL,
    title           VARCHAR(255) NOT NULL,
    duration_seconds INTEGER NOT NULL,
    category        VARCHAR(50) NOT NULL,
    summary         TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
""" .strip()

CREATE_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS snapshots (
    id                  UUID PRIMARY KEY,
    video_id            UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    external_video_id   VARCHAR(32) NOT NULL,
    timestamp_seconds   INTEGER NOT NULL,
    start_seconds       INTEGER NOT NULL,
    end_seconds         INTEGER NOT NULL,
    duration_seconds    INTEGER NOT NULL,
    caption             TEXT NOT NULL,
    search_text         VARCHAR(512) NOT NULL,
    embedding           VECTOR(1536) NOT NULL,
    tsv                 TSVECTOR,
    created_at          TIMESTAMPTZ DEFAULT now()
);
""" .strip()

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_snapshots_hnsw
    ON snapshots USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_snapshots_tsv
    ON snapshots USING gin (tsv);

CREATE INDEX IF NOT EXISTS idx_snapshots_video
    ON snapshots (video_id);

CREATE INDEX IF NOT EXISTS idx_snapshots_timerange
    ON snapshots (start_seconds, end_seconds);

CREATE INDEX IF NOT EXISTS idx_videos_category
    ON videos (category);
""" .strip()

CREATE_TSV_TRIGGER = """
CREATE OR REPLACE FUNCTION snapshots_tsv_trigger()
RETURNS trigger AS $$
BEGIN
    NEW.tsv = to_tsvector('english', NEW.search_text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_snapshots_tsv ON snapshots;
CREATE TRIGGER trg_snapshots_tsv
    BEFORE INSERT OR UPDATE ON snapshots
    FOR EACH ROW EXECUTE FUNCTION snapshots_tsv_trigger();
""" .strip()


def init_db() -> None:
    """Create tables, indexes, and the tsv trigger if they do not exist."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(CREATE_VECTORS)
        cur.execute(CREATE_VIDEOS)
        cur.execute(CREATE_SNAPSHOTS)
        cur.execute(CREATE_INDEXES)
        cur.execute(CREATE_TSV_TRIGGER)
    conn.close()
    print("  ✅ Postgres schema initialized (videos + snapshots + indexes + tsv trigger)")


if __name__ == "__main__":
    print("=" * 60)
    print("POSTGRES DB INIT")
    print("=" * 60)
    try:
        init_db()
        print("  ✅ All tables created")
    except Exception as e:
        print(f"  ❌ Error: {e}")
