"""Hybrid search + reranking — search captions with dense, sparse, and rerank."""

import os
from typing import Any

import httpx
from qdrant_client import QdrantClient, models

from video_rag.config import settings
from video_rag.embeddings import embed_text, get_sparse_embedding


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


# ---------------------------------------------------------------------------
# Filter builder — build Qdrant filter from metadata conditions
# ---------------------------------------------------------------------------
# Available payload fields:
#   video_id, category, video_title, start_seconds, end_seconds,
#   duration_seconds, caption, search_text ("title: caption")


def build_filter(
    *,
    video_id: str | None = None,
    category: str | None = None,
    start_seconds_gte: float | None = None,
    start_seconds_lte: float | None = None,
    end_seconds_gte: float | None = None,
    end_seconds_lte: float | None = None,
    duration_seconds_gte: float | None = None,
    duration_seconds_lte: float | None = None,
) -> models.Filter | None:
    """Build a Qdrant Filter from keyword metadata conditions (AND'd together)."""
    conditions: list = []

    if video_id is not None:
        conditions.append(
            models.FieldCondition(
                key="video_id", match=models.MatchValue(value=video_id)
            )
        )
    if category is not None:
        conditions.append(
            models.FieldCondition(
                key="category", match=models.MatchValue(value=category)
            )
        )

    rng: dict[str, float] = {}
    if start_seconds_gte is not None:
        rng["gte"] = start_seconds_gte
    if start_seconds_lte is not None:
        rng["lte"] = start_seconds_lte
    if rng:
        conditions.append(
            models.FieldCondition(key="start_seconds", range=models.Range(**rng))
        )

    rng_end: dict[str, float] = {}
    if end_seconds_gte is not None:
        rng_end["gte"] = end_seconds_gte
    if end_seconds_lte is not None:
        rng_end["lte"] = end_seconds_lte
    if rng_end:
        conditions.append(
            models.FieldCondition(key="end_seconds", range=models.Range(**rng_end))
        )

    rng_dur: dict[str, float] = {}
    if duration_seconds_gte is not None:
        rng_dur["gte"] = duration_seconds_gte
    if duration_seconds_lte is not None:
        rng_dur["lte"] = duration_seconds_lte
    if rng_dur:
        conditions.append(
            models.FieldCondition(key="duration_seconds", range=models.Range(**rng_dur))
        )

    if not conditions:
        return None

    return models.Filter(must=conditions)


# ---------------------------------------------------------------------------
# Hybrid search (dense + sparse with RRF fusion)
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    limit: int = 10,
    prefetch_limit: int = 50,
    query_filter: models.Filter | None = None,
    client: QdrantClient | None = None,
) -> list[models.ScoredPoint]:
    """Hybrid search using dense + sparse vectors with RRF fusion.

    Does two prefetches (dense + sparse) and fuses them with
    Reciprocal Rank Fusion (RRF) — all server-side in Qdrant.

    Args:
        query: Natural language query string.
        limit: Number of final results to return.
        prefetch_limit: Candidates to fetch per prefetch branch.
        query_filter: Optional metadata filter (use build_filter()).
        client: Optional QdrantClient.

    Returns:
        List of ScoredPoint sorted by hybrid relevance.
    """
    close = client is None
    client = client or get_client()

    dense_vec = embed_text(query)
    sp_indices, sp_values = get_sparse_embedding(query)

    # Shared filter clauses for each prefetch branch
    common_kwargs = {}
    if query_filter is not None:
        common_kwargs["filter"] = query_filter

    # Hybrid search with RRF fusion
    result = client.query_points(
        collection_name=settings.collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=prefetch_limit,
                **common_kwargs,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sp_indices,
                    values=sp_values,
                ),
                using="sparse",
                limit=prefetch_limit,
                **common_kwargs,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        with_vectors=False,
        limit=limit,
    )

    if close:
        client.close()

    return result.points


# ---------------------------------------------------------------------------
# Reranking with Cohere (via OpenRouter)
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    documents: list[str],
    model: str | None = None,
    top_n: int | None = None,
) -> list[dict]:
    """Rerank documents using Cohere Rerank via OpenRouter.

    Args:
        query: The original search query.
        documents: List of document texts to rerank.
        model: Rerank model name (default: from settings).
        top_n: Number of top results to return (default: all).

    Returns:
        List of dicts with keys: index, relevance_score, document.
        Sorted by relevance_score descending.
    """
    model = model or settings.rerank_model

    if not documents:
        return []

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n or len(documents),
    }

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mohd-vasim/ai-engineering",
        "X-OpenRouter-Title": "Video-RAG-Agent",
    }

    with httpx.Client(timeout=60.0) as http_client:
        resp = http_client.post(
            f"{settings.openrouter_base_url}/rerank",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    return data.get("results", [])


# ---------------------------------------------------------------------------
# High-level: hybrid search + rerank pipeline
# ---------------------------------------------------------------------------


def search_with_rerank(
    query: str,
    limit: int = 10,
    prefetch_limit: int = 50,
    rerank_top_k: int = 5,
    query_filter: models.Filter | None = None,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    """Full pipeline: hybrid search → rerank → return results.

    Steps:
      1. Hybrid search (dense + sparse RRF) in Qdrant
      2. Rerank results with Cohere Rerank via OpenRouter
      3. Return enriched results with payload + relevance scores

    Args:
        query: Natural language query.
        limit: Number of results from hybrid search.
        prefetch_limit: Prefetch candidates per branch.
        rerank_top_k: Number of top reranked results to return.
        query_filter: Optional metadata filter (use build_filter()).
        client: Optional QdrantClient.

    Returns:
        List of dicts with keys: score, payload, and rerank_score.
    """
    # Step 1: Hybrid search
    results = hybrid_search(
        query=query,
        limit=limit,
        prefetch_limit=prefetch_limit,
        query_filter=query_filter,
        client=client,
    )

    if not results:
        return []

    # Step 2: Rerank
    docs = [
        r.payload.get("search_text", "") or r.payload.get("caption", "")
        for r in results
    ]
    reranked = rerank(query=query, documents=docs, top_n=rerank_top_k)

    # Step 3: Map back to payloads
    final = []
    for r in reranked:
        idx = r["index"]
        original = results[idx]
        final.append(
            {
                "score": original.score,
                "rerank_score": r["relevance_score"],
                "payload": original.payload,
            }
        )

    # Sort by rerank_score descending
    final.sort(key=lambda x: x["rerank_score"], reverse=True)
    return final


# ---------------------------------------------------------------------------
# Search by video_id (exact filter)
# ---------------------------------------------------------------------------


def search_by_video_id(
    video_id: str,
    client: QdrantClient | None = None,
) -> list[models.ScoredPoint]:
    """Retrieve all snapshots for a specific video using payload filter."""
    close = client is None
    client = client or get_client()

    result = client.query_points(
        collection_name=settings.collection_name,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id),
                )
            ],
        ),
        query=models.OrderByQuery(
            order_by=models.OrderBy(
                key="start_seconds",
                direction=models.Direction.ASC,
            )
        ),
        with_payload=True,
        with_vectors=False,
        limit=100,
    )

    if close:
        client.close()

    return result.points


# ---------------------------------------------------------------------------
# generate_context — retrieve context for an AI agent
# ---------------------------------------------------------------------------
# This is the main function an AI agent should call. It returns structured
# context with relevant video caption snapshots + metadata, ready to be
# passed to an LLM prompt.


def generate_context(
    query: str,
    limit: int = 10,
    prefetch_limit: int = 50,
    rerank_top_k: int = 5,
    *,
    video_id: str | None = None,
    category: str | None = None,
    start_seconds_gte: float | None = None,
    start_seconds_lte: float | None = None,
    end_seconds_gte: float | None = None,
    end_seconds_lte: float | None = None,
    duration_seconds_gte: float | None = None,
    duration_seconds_lte: float | None = None,
    client: QdrantClient | None = None,
) -> dict[str, Any]:
    """Retrieve context for an AI agent via hybrid search + rerank + filters.

    Builds a metadata filter, runs hybrid search, reranks with Cohere,
    and returns structured context ready to inject into an LLM prompt.

    Args:
        query: Natural language query.
        limit: Number of results from hybrid search.
        prefetch_limit: Candidates per prefetch branch.
        rerank_top_k: Top N results after reranking.
        video_id: Filter to a specific video.
        category: Filter by category (nature, food, technology, sports, …).
        start_seconds_gte: Snapshot start >= N seconds.
        start_seconds_lte: Snapshot start <= N seconds.
        end_seconds_gte: Snapshot end >= N seconds.
        end_seconds_lte: Snapshot end <= N seconds.
        duration_seconds_gte: Snapshot duration >= N seconds.
        duration_seconds_lte: Snapshot duration <= N seconds.
        client: Optional QdrantClient.

    Returns:
        Dict with keys:
          - query: the original query
          - context: list of dicts (video_id, title, category, timestamp_range,
                     duration, caption)
          - count: number of context items
    """
    qfilter = build_filter(
        video_id=video_id,
        category=category,
        start_seconds_gte=start_seconds_gte,
        start_seconds_lte=start_seconds_lte,
        end_seconds_gte=end_seconds_gte,
        end_seconds_lte=end_seconds_lte,
        duration_seconds_gte=duration_seconds_gte,
        duration_seconds_lte=duration_seconds_lte,
    )

    results = search_with_rerank(
        query=query,
        limit=limit,
        prefetch_limit=prefetch_limit,
        rerank_top_k=rerank_top_k,
        query_filter=qfilter,
        client=client,
    )

    context = []
    for r in results:
        p = r["payload"]
        context.append(
            {
                "video_id": p.get("video_id"),
                "video_title": p.get("video_title"),
                "category": p.get("category"),
                "timestamp_range": f"{p.get('start_seconds')}s - {p.get('end_seconds')}s",
                "duration_seconds": p.get("duration_seconds"),
                "caption": p.get("caption"),
            }
        )

    return {
        "query": query,
        "context": context,
        "count": len(context),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID SEARCH + RERANK TESTS")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("SKIP: OPENROUTER_API_KEY not set")
        import sys

        sys.exit(0)

    # --- Test 1: Hybrid search ---
    print("\nTEST 1: Hybrid search (dense + sparse RRF)")
    try:
        results = hybrid_search("sunrise over mountains", limit=5)
        print(f"  Found {len(results)} results")
        for r in results:
            cap = r.payload.get("caption", "")[:60]
            vid = r.payload.get("video_id", "")
            print(f"    [{r.score:.4f}] {vid}: {cap}...")
        if results:
            print("  ✅ Hybrid search OK")
        else:
            print("  ⚠️  No results (collection may be empty)")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # --- Test 2: Rerank with Cohere ---
    print("\nTEST 2: Rerank with Cohere via OpenRouter")
    try:
        docs = [
            "A dark mountain silhouette against a pre-dawn sky with faint orange glow",
            "A bustling night market with colorful string lights hanging overhead",
            "The first rays of sunlight begin to illuminate the mountain peaks",
            "A street basketball court in an urban park with two players",
        ]
        reranked = rerank("sunrise over mountains at dawn", documents=docs, top_n=2)
        print(f"  Reranked {len(reranked)} results")
        for r in reranked:
            print(
                f"    [{r['relevance_score']:.4f}] doc {r['index']}: {docs[r['index']][:60]}..."
            )
        if reranked:
            print("  ✅ Rerank OK")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # --- Test 3: Full pipeline (search + rerank) ---
    print("\nTEST 3: Full hybrid search + rerank pipeline")
    try:
        final = search_with_rerank("cooking food in a pan", limit=8, rerank_top_k=3)
        print(f"  Final results: {len(final)}")
        for f in final:
            cap = f["payload"].get("caption", "")[:60]
            vid = f["payload"].get("video_id", "")
            print(
                f"    [dense={f['score']:.4f}, rerank={f['rerank_score']:.4f}] {vid}: {cap}..."
            )
        if final:
            print("  ✅ Full pipeline OK")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # --- Test 4: Search by video_id ---
    print("\nTEST 4: Search by video_id")
    try:
        snaps = search_by_video_id("video_001")
        print(f"  Found {len(snaps)} snapshots for video_001")
        for s in snaps[:3]:
            start = s.payload.get("start_seconds", "?")
            end = s.payload.get("end_seconds", "?")
            cap = s.payload.get("caption", "")[:50]
            print(f"    {start}s-{end}s: {cap}...")
        if snaps:
            print("  ✅ Video ID search OK")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # --- Test 5: build_filter ---
    print("\nTEST 5: build_filter function")
    f1 = build_filter(category="nature")
    print(f"  filter(category='nature'): {'created' if f1 else 'none'}")
    f2 = build_filter(video_id="video_005", start_seconds_gte=20)
    print(f"  filter(video_id + start>=20): {'created' if f2 else 'none'}")
    f3 = build_filter(start_seconds_gte=0, end_seconds_lte=30)
    print(f"  filter(start>=0, end<=30): {'created' if f3 else 'none'}")
    f4 = build_filter()
    print(f"  filter(empty): {f4}")
    if f1 and f2 and f3 and f4 is None:
        print("  ✅ build_filter OK")
    else:
        print("  ⚠️  build_filter unexpected")

    # --- Test 6: Hybrid search with metadata filter ---
    print("\nTEST 6: Hybrid search with category filter")
    try:
        qf = build_filter(category="food")
        results = hybrid_search("cooking", limit=5, query_filter=qf)
        print(f"  Found {len(results)} food results")
        for r in results:
            cat = r.payload.get("category", "?")
            cap = r.payload.get("caption", "")[:50]
            print(f"    [{r.score:.4f}] ({cat}) {cap}...")
        if results:
            all_food = all(r.payload.get("category") == "food" for r in results)
            print(f"  All results are 'food': {all_food}")
            if all_food:
                print("  ✅ Filtered hybrid search OK")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # --- Test 7: generate_context ---
    print("\nTEST 7: generate_context (AI agent entry point)")
    try:
        ctx = generate_context("sunrise mountain dawn", limit=5, rerank_top_k=3)
        print(f"  query: {ctx['query']}")
        print(f"  count: {ctx['count']}")
        for item in ctx["context"]:
            print(
                f"    {item['video_id']} | {item['timestamp_range']} | {item['caption'][:50]}..."
            )
        if ctx["count"] > 0:
            print("  ✅ generate_context OK")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # --- Test 8: generate_context with filters ---
    print("\nTEST 8: generate_context with timestamp + category filters")
    try:
        ctx = generate_context(
            "cooking",
            limit=5,
            rerank_top_k=3,
            category="food",
            duration_seconds_gte=5,
        )
        print(f"  query: {ctx['query']}")
        print(f"  count: {ctx['count']}")
        for item in ctx["context"]:
            print(
                f"    {item['video_id']} | {item['category']} | dur={item['duration_seconds']}s | {item['caption'][:40]}..."
            )
        if ctx["count"] > 0:
            print("  ✅ generate_context with filters OK")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    print("\nAll search tests completed!")
