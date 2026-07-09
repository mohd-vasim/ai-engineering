"""Video RAG Agent — local package.

Public API (AI-agent friendly):
  generate_context()   — one-call context retrieval (hybrid search + rerank + filters)
  hybrid_search()      — raw hybrid search (dense + sparse RRF)
  search_with_rerank() — hybrid search → Cohere rerank
  search_by_video_id() — fetch all snapshots for a video
  build_filter()       — build metadata filter conditions
  recreate_collection()— drop & recreate Qdrant collection
  ingest_mock_data()   — load mock data and upsert into Qdrant
  count_points()       — check collection size
  load_mock_data()     — load the JSON dataset
  embed_text()         — single text → dense vector
  embed_texts()        — batch texts → dense vectors
  get_sparse_embedding()— text → sparse BM25 vector
  derive_snapshot_ranges() — add start_seconds / end_seconds to snapshots
"""

from video_rag.embeddings import (
    derive_snapshot_ranges,
    embed_text,
    embed_texts,
    get_sparse_embedding,
    load_mock_data,
)
from video_rag.ingest import count_points, ingest_mock_data, recreate_collection
from video_rag.search import (
    build_filter,
    generate_context,
    hybrid_search,
    search_by_video_id,
    search_with_rerank,
)
from video_rag.tools import AGENT_TOOLS

__all__ = [
    # AI agent entry point
    "generate_context",
    "AGENT_TOOLS",
    # Search
    "hybrid_search",
    "search_with_rerank",
    "search_by_video_id",
    "build_filter",
    # Ingestion
    "recreate_collection",
    "ingest_mock_data",
    "count_points",
    # Data
    "load_mock_data",
    "embed_text",
    "embed_texts",
    "get_sparse_embedding",
    "derive_snapshot_ranges",
]
