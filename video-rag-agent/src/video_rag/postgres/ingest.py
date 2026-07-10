"""Postgres/pgvector ingestion — load industrial dataset, embed, and INSERT."""

import uuid

from video_rag.config import settings
from video_rag.embeddings import derive_snapshot_ranges, embed_texts, load_mock_data
from video_rag.postgres.db import get_conn


def ingest_industrial_data(json_path: str | None = None) -> int:
    """Load the industrial video dataset, embed captions, and insert into Postgres.

    Args:
        json_path: Path to the JSON file (default: from settings.data_file).

    Returns:
        Number of snapshots ingested.
    """
    path = json_path or settings.data_file
    print(f"  Loading data from {path}...")
    data = load_mock_data(path)
    videos = data["videos"]
    total_snapshots = 0

    conn = get_conn()

    # --- Insert videos first ---
    for v in videos:
        video_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"video_{v['video_id']}")
        )
        conn.execute(
            """
            INSERT INTO videos (id, external_id, title, duration_seconds, category, summary)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (external_id) DO UPDATE
                SET title = EXCLUDED.title,
                    duration_seconds = EXCLUDED.duration_seconds,
                    category = EXCLUDED.category,
                    summary = EXCLUDED.summary
            """,
            (
                video_id,
                v["video_id"],
                v["title"],
                v["duration_seconds"],
                v["category"],
                v.get("summary", ""),
            ),
        )

    print(f"  Upserted {len(videos)} videos")

    # --- Collect captions for batch embedding ---
    caption_index: list[tuple[dict, dict, str]] = []  # (video, enriched_snapshot, video_uuid)
    texts_for_embedding: list[str] = []

    for v in videos:
        v_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"video_{v['video_id']}"))
        enriched = derive_snapshot_ranges(v["snapshots"])
        for snap in enriched:
            caption_index.append((v, snap, v_uuid))
            texts_for_embedding.append(snap["caption"])

    print(f"  Generating {len(texts_for_embedding)} dense embeddings...")
    dense_vectors = embed_texts(texts_for_embedding)

    # --- Insert snapshots ---
    inserted = 0
    for (v, snap, v_uuid), dense_vec in zip(caption_index, dense_vectors, strict=True):
        snap_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{v['video_id']}_{snap['start_seconds']}_{snap['end_seconds']}",
            )
        )
        search_text = f"{v['title']}: {snap['caption']}"

        conn.execute(
            """
            INSERT INTO snapshots (
                id, video_id, external_video_id,
                timestamp_seconds, start_seconds, end_seconds, duration_seconds,
                caption, search_text, embedding
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
                SET caption = EXCLUDED.caption,
                    search_text = EXCLUDED.search_text,
                    embedding = EXCLUDED.embedding
            """,
            (
                snap_id,
                v_uuid,
                v["video_id"],
                snap["end_seconds"],
                snap["start_seconds"],
                snap["end_seconds"],
                snap["duration_seconds"],
                snap["caption"],
                search_text,
                dense_vec,
            ),
        )
        inserted += 1
        if inserted % 50 == 0:
            print(f"    Inserted {inserted}/{len(texts_for_embedding)} snapshots...")
            conn.commit()

    conn.commit()
    conn.close()

    print(f"  ✅ Inserted {inserted} snapshots")
    return inserted


def count_points() -> int:
    """Count total snapshots in the database."""
    conn = get_conn()
    row = conn.execute("SELECT COUNT(*) AS cnt FROM snapshots").fetchone()
    conn.close()
    return row["cnt"]


if __name__ == "__main__":
    import os

    print("=" * 60)
    print("POSTGRES INGESTION")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("SKIP: OPENROUTER_API_KEY not set — can't embed")
        import sys
        sys.exit(0)

    from video_rag.postgres.db import init_db

    print("\nStep 1: Initialize schema...")
    init_db()

    print("\nStep 2: Ingest data...")
    total = ingest_industrial_data()
    print(f"\n  Total snapshots in DB: {count_points()}")
    print("  ✅ Ingestion complete")
