"""Qdrant ingestion — create collection, upsert points with dense + sparse vectors."""

import uuid

from qdrant_client import QdrantClient, models

from video_rag.config import settings
from video_rag.embeddings import (
    derive_snapshot_ranges,
    embed_texts,
    get_sparse_embedding,
    load_mock_data,
)


def get_client() -> QdrantClient:
    """Get a connected Qdrant client."""
    return QdrantClient(url=settings.qdrant_url)


def recreate_collection(client: QdrantClient | None = None) -> None:
    """Drop and recreate the collection with dense + sparse vector config.

    - Dense vector: float32, dimension = settings.dense_vector_size
    - Sparse vector: for hybrid (keyword + semantic) search
    """
    close = client is None
    client = client or get_client()

    # Delete if exists
    try:
        client.delete_collection(settings.collection_name)
        print(f"  Deleted existing collection '{settings.collection_name}'")
    except Exception:
        pass  # Collection didn't exist

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=settings.dense_vector_size,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
            ),
        },
    )

    # Create payload index for ordering by start_seconds
    client.create_payload_index(
        collection_name=settings.collection_name,
        field_name="start_seconds",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

    print(
        f"  Created collection '{settings.collection_name}' "
        f"(dense={settings.dense_vector_size}d, sparse=on)"
    )

    if close:
        client.close()


def build_points(data: dict) -> list[models.PointStruct]:
    """Build Qdrant PointStruct list from the mock dataset.

    Each snapshot becomes one point with:
      - Dense vector (from caption)
      - Sparse vector (from caption)
      - Payload with full metadata
    """
    all_points: list[models.PointStruct] = []
    texts_for_embedding: list[str] = []

    # Collect all captions first for batch embedding
    caption_index = []  # (video, enriched_snapshot) pairs
    for video in data["videos"]:
        enriched = derive_snapshot_ranges(video["snapshots"])
        for snap in enriched:
            caption_index.append((video, snap))
            texts_for_embedding.append(snap["caption"])

    print(f"  Generating {len(texts_for_embedding)} dense embeddings...")
    dense_vectors = embed_texts(texts_for_embedding)

    print(f"  Generating {len(texts_for_embedding)} sparse embeddings...")
    sparse_vectors = []
    for text in texts_for_embedding:
        indices, values = get_sparse_embedding(text)
        sparse_vectors.append((indices, values))

    for idx, ((video, snap), dense_vec, (sp_indices, sp_values)) in enumerate(
        zip(caption_index, dense_vectors, sparse_vectors, strict=True)
    ):
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{video['video_id']}_{snap['start_seconds']}_{snap['end_seconds']}",
            )
        )
        all_points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": models.SparseVector(
                        indices=sp_indices,
                        values=sp_values,
                    ),
                },
                payload={
                    # Video info
                    "video_id": video["video_id"],
                    "video_title": video["title"],
                    "video_duration": video["duration_seconds"],
                    "category": video["category"],
                    # Snapshot info
                    "start_seconds": snap["start_seconds"],
                    "end_seconds": snap["end_seconds"],
                    "duration_seconds": snap["duration_seconds"],
                    "caption": snap["caption"],
                    # Search context (combines title + caption for keyword matching)
                    "search_text": f"{video['title']}: {snap['caption']}",
                },
            )
        )

    return all_points


def ingest_mock_data(client: QdrantClient | None = None) -> int:
    """Load mock data, embed, and upsert into Qdrant.

    Returns:
        Number of points ingested.
    """
    close = client is None
    client = client or get_client()

    data = load_mock_data()
    points = build_points(data)

    # Upsert in batches of 10
    batch_size = 10
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=settings.collection_name,
            points=batch,
            wait=True,
        )
        total += len(batch)
        print(f"  Upserted {total}/{len(points)} points...")

    if close:
        client.close()

    return total


def count_points(client: QdrantClient | None = None) -> int:
    """Count points in the collection."""
    close = client is None
    client = client or get_client()
    result = client.count(collection_name=settings.collection_name)
    if close:
        client.close()
    return result.count


if __name__ == "__main__":
    import os
    import sys

    print("=" * 60)
    print("QDRANT INGESTION TESTS")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("SKIP: OPENROUTER_API_KEY not set — can't embed")
        sys.exit(0)

    # --- Test 1: Recreate collection ---
    print("\nTEST 1: Recreate collection")
    try:
        recreate_collection()
        print("  ✅ Collection created/ready")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise

    # --- Test 2: Build points ---
    print("\nTEST 2: Build points from mock data")
    data = load_mock_data()
    points = build_points(data)
    print(f"  Total points built: {len(points)}")
    print(f"  Sample payload keys: {list(points[0].payload.keys())}")
    print(f"  Vector keys: {list(points[0].vector.keys())}")
    if len(points) > 0:
        print("  ✅ Points built OK")

    # --- Test 3: Ingest into Qdrant ---
    print("\nTEST 3: Ingest into Qdrant")
    try:
        total = ingest_mock_data()
        print(f"  ✅ Ingested {total} points")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise

    # --- Test 4: Verify count ---
    print("\nTEST 4: Verify point count")
    cnt = count_points()
    print(f"  Collection has {cnt} points")
    assert cnt == len(points), f"Count mismatch: {cnt} != {len(points)}"
    print(f"  ✅ Count matches ({cnt})")

    print("\nAll ingestion tests passed!")
