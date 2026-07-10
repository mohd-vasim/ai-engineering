"""Postgres/pgvector backend — hybrid search, ingestion, and query utilities."""

from video_rag.postgres.db import get_conn, init_db
from video_rag.postgres.ingest import count_points, ingest_industrial_data
from video_rag.postgres.search import (
    AGENT_TOOLS,
    build_filter,
    describe_table,
    generate_context,
    get_video_snapshots,
    hybrid_search,
    list_categories,
    list_tables,
    rerank,
    search_with_rerank,
)

__all__ = [
    "AGENT_TOOLS",
    "init_db",
    "get_conn",
    "ingest_industrial_data",
    "count_points",
    "hybrid_search",
    "search_with_rerank",
    "build_filter",
    "generate_context",
    "rerank",
    "get_video_snapshots",
    "list_categories",
    "list_tables",
    "describe_table",
]
