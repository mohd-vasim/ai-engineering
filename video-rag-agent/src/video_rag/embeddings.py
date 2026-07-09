"""Embeddings module — create dense and sparse (latent) embeddings."""

import json
from pathlib import Path

from openai import OpenAI

from video_rag.config import settings


def get_openai_client() -> OpenAI:
    """Get an OpenAI-compatible client pointing to OpenRouter."""
    settings.validate()
    return OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )


# ---------------------------------------------------------------------------
# Dense embeddings
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Create dense embeddings for a list of texts using OpenRouter.

    Args:
        texts: List of text strings to embed.
        model: Model name (default: from settings).

    Returns:
        List of embedding vectors (list[float] each).
    """
    client = get_openai_client()
    model = model or settings.embedding_model

    response = client.embeddings.create(
        extra_headers={
            "HTTP-Referer": "https://github.com/mohd-vasim/ai-engineering",
            "X-OpenRouter-Title": "Video-RAG-Agent",
        },
        model=model,
        input=texts,
        encoding_format="float",
    )
    # OpenRouter may return errors embedded in the response (e.g. 429 overloaded)
    err = getattr(response, "error", None)
    if err is not None:
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise RuntimeError(f"OpenRouter embedding API error: {msg}")
    if response.data is None:
        raise RuntimeError(
            f"OpenRouter embedding API returned None for input of {len(texts)} text(s). "
            f"Response: {response.model_dump_json()}"
        )
    # Sort by index to preserve input order
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def embed_text(text: str, model: str | None = None) -> list[float]:
    """Create a dense embedding for a single text string."""
    return embed_texts([text], model=model)[0]


# ---------------------------------------------------------------------------
# Sparse (latent) embeddings via FastEmbed
# ---------------------------------------------------------------------------


def get_sparse_embedding(text: str) -> tuple[list[int], list[float]]:
    """Create a sparse (BM25-style) embedding using FastEmbed.

    Returns:
        (indices, values) tuple for use with qdrant_client.models.SparseVector.
    """
    try:
        from fastembed import SparseTextEmbedding
    except ImportError:
        raise ImportError(
            "fastembed is required for sparse embeddings; "
            "install with: uv pip install 'qdrant-client[fastembed]'"
        )

    model = SparseTextEmbedding("Qdrant/bm25")
    emb = list(model.embed(text))[0]
    return emb.indices.tolist(), emb.values.tolist()


# ---------------------------------------------------------------------------
# Load mock data
# ---------------------------------------------------------------------------


def load_mock_data(path: str | Path = "data/mock_video_captions.json") -> dict:
    """Load the mock video caption dataset from JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Derive start_seconds from snapshot list
# ---------------------------------------------------------------------------


def derive_snapshot_ranges(snapshots: list[dict]) -> list[dict]:
    """Add start_seconds to each snapshot based on the previous timestamp.

    The mock data only has timestamp_seconds per snapshot.
    This derives start_seconds logically:
      - First snapshot: start = 0
      - Subsequent snapshots: start = previous snapshot's timestamp_seconds
      - end = current timestamp_seconds

    Also includes duration_seconds = end - start.
    """
    enriched = []
    for i, snap in enumerate(snapshots):
        ts = snap["timestamp_seconds"]
        start_seconds = 0 if i == 0 else snapshots[i - 1]["timestamp_seconds"]
        enriched.append(
            {
                "start_seconds": start_seconds,
                "end_seconds": ts,
                "duration_seconds": ts - start_seconds,
                "caption": snap["caption"],
            }
        )
    return enriched


if __name__ == "__main__":
    import os

    # --- Test 1: Dense embedding for a single text ---
    print("=" * 60)
    print("TEST 1: Single text dense embedding")
    print("=" * 60)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("SKIP: OPENROUTER_API_KEY not set")
    else:
        try:
            emb = embed_text("A cat sitting on a mat", model="qwen/qwen3-embedding-4b")
            print(f"  Embedding dimension: {len(emb)}")
            print(f"  First 5 values: {emb[:5]}")
            print("  ✅ Single embedding OK")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # --- Test 2: Batch dense embeddings ---
    print()
    print("=" * 60)
    print("TEST 2: Batch dense embeddings (3 texts)")
    print("=" * 60)
    if api_key:
        try:
            texts = [
                "A cat sitting on a mat",
                "A dog running in the park",
                "Sunset over the ocean waves",
            ]
            embs = embed_texts(texts, model="qwen/qwen3-embedding-4b")
            print(f"  Number of embeddings: {len(embs)}")
            print(f"  Each dimension: {len(embs[0])}")
            print("  ✅ Batch embedding OK")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # --- Test 3: Sparse embedding ---
    print()
    print("=" * 60)
    print("TEST 3: Sparse (latent) embedding via FastEmbed")
    print("=" * 60)
    try:
        indices, values = get_sparse_embedding("A cat sitting on a mat")
        print(f"  Number of non-zero indices: {len(indices)}")
        print(f"  First 5 indices: {indices[:5]}")
        print(f"  First 5 values: {values[:5]}")
        print("  ✅ Sparse embedding OK")
    except Exception as e:
        print(f"  ❌ Sparse embedding error: {e}")

    # --- Test 4: Derive snapshot ranges ---
    print()
    print("=" * 60)
    print("TEST 4: Derive snapshot start/end ranges from mock data")
    print("=" * 60)
    data = load_mock_data()
    first_video = data["videos"][0]
    enriched = derive_snapshot_ranges(first_video["snapshots"])
    for s in enriched[:3]:
        print(
            f"  {s['start_seconds']}s -> {s['end_seconds']}s "
            f"(dur={s['duration_seconds']}s): {s['caption'][:60]}..."
        )
    print(f"  Total snapshots enriched: {len(enriched)}")
    print("  ✅ Snapshot range derivation OK")

    print()
    print("All embedding tests completed.")
