"""Application configuration."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent.parent

load_dotenv(PROJECT_DIR / ".env")

@dataclass
class Settings:
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    qdrant_url: str = "http://localhost:6333"

    # Embedding model
    embedding_model: str = "text-embedding-3-small"

    # Rerank model
    rerank_model: str = "cohere/rerank-4-fast"

    # Qdrant collection
    collection_name: str = "video_captions"

    # Snapshot interval from the mock data
    snapshot_interval_seconds: int = 10

    # Dense vector size for text-embedding-3-small
    dense_vector_size: int = 1536

    # Postgres / pgvector settings
    pg_host: str = field(
        default_factory=lambda: os.environ.get("PG_HOST", "localhost")
    )
    pg_port: int = field(
        default_factory=lambda: int(os.environ.get("PG_PORT", "5432"))
    )
    pg_db: str = field(
        default_factory=lambda: os.environ.get("PG_DB", "video_analytics")
    )
    pg_user: str = field(
        default_factory=lambda: os.environ.get("PG_USER", "admin")
    )
    pg_password: str = field(
        default_factory=lambda: os.environ.get("PG_PASSWORD", "")
    )

    # Default data file
    data_file: str = "data/mock_industrial_videos.json"

    def validate(self):
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")


settings = Settings()
