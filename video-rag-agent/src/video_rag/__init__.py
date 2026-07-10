"""Video RAG Agent — local package.

Two backends available:
  - video_rag.qdrant  (original Qdrant-based, legacy)
  - video_rag.postgres (Postgres/pgvector, default)

Default exports use the postgres backend.
"""

# Shared utilities (backend-agnostic)
from video_rag.embeddings import (
    derive_snapshot_ranges,
    embed_text,
    embed_texts,
    get_sparse_embedding,
    load_mock_data,
)

# Qdrant backend
from video_rag.qdrant import (
    AGENT_TOOLS_QDRANT,
    build_points,
    build_filter as build_filter_qdrant,
    count_points as count_points_qdrant,
    generate_context as generate_context_qdrant,
    hybrid_search as hybrid_search_qdrant,
    ingest_mock_data,
    recreate_collection,
    rerank as rerank_qdrant,
    search_by_video_id,
    search_with_rerank as search_with_rerank_qdrant,
)

# Postgres backend (default)
from video_rag.postgres import (
    build_filter,
    count_points,
    generate_context,
    get_video_snapshots,
    hybrid_search,
    ingest_industrial_data,
    init_db,
    list_categories,
    search_with_rerank,
)

__all__ = [
    # Shared
    "embed_text",
    "embed_texts",
    "get_sparse_embedding",
    "load_mock_data",
    "derive_snapshot_ranges",
    # Postgres (default)
    "init_db",
    "ingest_industrial_data",
    "hybrid_search",
    "search_with_rerank",
    "build_filter",
    "generate_context",
    "get_video_snapshots",
    "list_categories",
    "count_points",
    # Qdrant (explicit namespace)
    "AGENT_TOOLS_QDRANT",
    "recreate_collection",
    "ingest_mock_data",
    "build_points",
    "count_points_qdrant",
    "hybrid_search_qdrant",
    "search_with_rerank_qdrant",
    "search_by_video_id",
    "build_filter_qdrant",
    "generate_context_qdrant",
    "rerank_qdrant",
]
