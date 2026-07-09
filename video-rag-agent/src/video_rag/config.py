"""Application configuration."""

import os
from dataclasses import dataclass, field


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

    def validate(self):
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")


settings = Settings()
