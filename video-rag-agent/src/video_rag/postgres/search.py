"""Postgres/pgvector hybrid search — dense (pgvector) + sparse (tsvector) with rerank.

Tool-decorated functions live here (no separate tools.py) — LangChain accepts
plain callables, and @tool makes them ready for agent registration.
"""

import uuid
from typing import Any

import httpx

from video_rag.config import settings
from video_rag.embeddings import embed_text, load_mock_data
from video_rag.postgres.db import get_conn


# ---------------------------------------------------------------------------
# Filter builder — build SQL WHERE clause from metadata conditions
# ---------------------------------------------------------------------------

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
) -> tuple[str, list]:
    """Build a SQL WHERE clause + params from metadata conditions (AND'd).

    Returns:
        (where_clause, params_list). where_clause is empty string if no filters.
    """
    conditions: list[str] = []
    params: list = []

    if video_id is not None:
        conditions.append("s.external_video_id = %s")
        params.append(video_id)
    if category is not None:
        conditions.append("v.category = %s")
        params.append(category)
    if start_seconds_gte is not None:
        conditions.append("s.start_seconds >= %s")
        params.append(start_seconds_gte)
    if start_seconds_lte is not None:
        conditions.append("s.start_seconds <= %s")
        params.append(start_seconds_lte)
    if end_seconds_gte is not None:
        conditions.append("s.end_seconds >= %s")
        params.append(end_seconds_gte)
    if end_seconds_lte is not None:
        conditions.append("s.end_seconds <= %s")
        params.append(end_seconds_lte)
    if duration_seconds_gte is not None:
        conditions.append("s.duration_seconds >= %s")
        params.append(duration_seconds_gte)
    if duration_seconds_lte is not None:
        conditions.append("s.duration_seconds <= %s")
        params.append(duration_seconds_lte)

    if not conditions:
        return "", []

    return " AND ".join(conditions), params


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
# Hybrid search (dense pgvector + sparse tsvector, weighted combination)
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    limit: int = 10,
    prefetch_limit: int = 50,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Hybrid search using pgvector (dense) + tsvector (sparse) with weighted sum.

    Single SQL round-trip. Dense prefetch drives recall; combined score sorts.

    Args:
        query: Natural language query string.
        limit: Number of final results to return.
        prefetch_limit: Candidates for the dense prefetch branch.
        **filters: Keyword args for build_filter().

    Returns:
        List of dicts with keys: video_id, video_title, category,
        timestamp_range, duration_seconds, caption, dense_score, sparse_score.
    """
    dense_vec = embed_text(query)

    where_clause, params = build_filter(**filters)
    where_sql = where_clause

    sql = f"""
    WITH dense_prefetch AS (
        SELECT
            s.id, s.external_video_id, s.start_seconds, s.end_seconds,
            s.duration_seconds, s.caption, s.search_text,
            v.title AS video_title, v.category,
            1 - (s.embedding <=> %s::vector) AS dense_score,
            COALESCE(ts_rank(s.tsv, plainto_tsquery('english', %s)), 0) AS sparse_score
        FROM snapshots s
        JOIN videos v ON v.id = s.video_id
        {('WHERE ' + where_sql) if where_sql else ''}
        ORDER BY s.embedding <=> %s::vector
        LIMIT %s
    )
    SELECT *, (0.5 * dense_score + 0.5 * sparse_score) AS combined_score
    FROM dense_prefetch
    ORDER BY combined_score DESC
    LIMIT %s
    """

    conn = get_conn()
    sql_params = [dense_vec, query] + params + [dense_vec, prefetch_limit, limit]
    rows = conn.execute(sql, sql_params).fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append(
            {
                "id": str(row["id"]),
                "video_id": row["external_video_id"],
                "video_title": row["video_title"],
                "category": row["category"],
                "timestamp_range": f"{row['start_seconds']}s - {row['end_seconds']}s",
                "duration_seconds": row["duration_seconds"],
                "caption": row["caption"],
                "search_text": row["search_text"],
                "dense_score": float(row["dense_score"]),
                "sparse_score": float(row["sparse_score"]),
                "combined_score": float(row["combined_score"]),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Full pipeline: hybrid search + Cohere rerank
# ---------------------------------------------------------------------------


def search_with_rerank(
    query: str,
    limit: int = 10,
    prefetch_limit: int = 50,
    rerank_top_k: int = 5,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Full pipeline: hybrid search -> Cohere rerank -> return results.

    Args:
        query: Natural language query.
        limit: Number of results from hybrid search.
        prefetch_limit: Candidates per prefetch branch.
        rerank_top_k: Top N results after reranking.
        **filters: Keyword args for build_filter().

    Returns:
        List of dicts with payload + scores.
    """
    results = hybrid_search(
        query=query,
        limit=limit,
        prefetch_limit=prefetch_limit,
        **filters,
    )

    if not results:
        return []

    docs = [r["search_text"] for r in results]
    reranked = rerank(query=query, documents=docs, top_n=rerank_top_k)

    final = []
    for r in reranked:
        idx = r["index"]
        original = results[idx]
        final.append(
            {
                "dense_score": original["dense_score"],
                "sparse_score": original["sparse_score"],
                "rerank_score": r["relevance_score"],
                "payload": original,
            }
        )

    final.sort(key=lambda x: x["rerank_score"], reverse=True)
    return final


# ---------------------------------------------------------------------------
# generate_context — retrieve context for an AI agent
# ---------------------------------------------------------------------------


def generate_context(
    query: str,
    limit: int = 10,
    prefetch_limit: int = 50,
    rerank_top_k: int = 5,
    **filters: Any,
) -> dict[str, Any]:
    """Retrieve context via hybrid search + rerank + filters.

    Args:
        query: Natural language query.
        limit: Number of results from hybrid search.
        prefetch_limit: Candidates per prefetch branch.
        rerank_top_k: Top N results after reranking.
        **filters: Keyword args for build_filter().

    Returns:
        Dict with keys: query, context (list), count.
    """
    results = search_with_rerank(
        query=query,
        limit=limit,
        prefetch_limit=prefetch_limit,
        rerank_top_k=rerank_top_k,
        **filters,
    )

    context = []
    for r in results:
        p = r["payload"]
        context.append(
            {
                "video_id": p["video_id"],
                "video_title": p["video_title"],
                "category": p["category"],
                "timestamp_range": p["timestamp_range"],
                "duration_seconds": p["duration_seconds"],
                "caption": p["caption"],
            }
        )

    return {"query": query, "context": context, "count": len(context)}


# ---------------------------------------------------------------------------
# Utility functions (can be used directly as agent tools)
# ---------------------------------------------------------------------------


def get_video_snapshots(video_id: str) -> str:
    """Get all caption snapshots for a specific video by its video_id
    (e.g. 'factory_001', 'retail_005'). Returns the full timeline of captions with timestamps."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT s.start_seconds, s.end_seconds, s.duration_seconds,
               s.caption, v.title
        FROM snapshots s
        JOIN videos v ON v.id = s.video_id
        WHERE s.external_video_id = %s
        ORDER BY s.start_seconds ASC
        """,
        [video_id],
    ).fetchall()
    conn.close()

    if not rows:
        return f"No video found with id '{video_id}'."

    lines = [f"Timeline for {rows[0]['title']} ({video_id}):\n"]
    for r in rows:
        lines.append(
            f"  {r['start_seconds']}s - {r['end_seconds']}s "
            f"({r['duration_seconds']}s): {r['caption']}"
        )
    return "\n".join(lines)


def list_categories() -> str:
    """List all available video categories in the dataset.
    Use this to discover what kinds of videos are stored."""
    data = load_mock_data(settings.data_file)
    categories = sorted({v["category"] for v in data["videos"]})
    count = len(categories)
    lines = [f"Available categories ({count} total):\n"]
    for c in categories:
        videos = [v["video_id"] for v in data["videos"] if v["category"] == c]
        lines.append(f"  \u2022 {c} \u2014 {len(videos)} videos: {', '.join(videos)}")
    lines.append(
        f"\nTotal videos: {len(data['videos'])}, "
        f"total snapshots: {sum(len(v['snapshots']) for v in data['videos'])}"
    )
    return "\n".join(lines)


AGENT_TOOLS = [
    build_filter,
    rerank,
    hybrid_search,
    search_with_rerank,
    generate_context,
    get_video_snapshots,
    list_categories,
]

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("POSTGRES HYBRID SEARCH TESTS")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("SKIP: OPENROUTER_API_KEY not set")
        import sys
        sys.exit(0)

    # --- Test 1: build_filter ---
    print("\nTEST 1: build_filter")
    where, params = build_filter(category="safety")
    print(f"  filter(category='safety'): WHERE {where} params={params}")
    where2, params2 = build_filter(video_id="factory_001", start_seconds_gte=20)
    print(f"  filter(video_id + start>=20): WHERE {where2} params={params2}")
    where3, params3 = build_filter()
    print(f"  filter(empty): '{where3}' {params3}")
    assert where and where2 and where3 == "" and params3 == []
    print("  ✅ build_filter OK")

    # --- Test 2: Hybrid search ---
    print("\nTEST 2: Hybrid search")
    try:
        results = hybrid_search("workers without helmet", limit=5)
        print(f"  Found {len(results)} results")
        for r in results:
            print(f"    [{r['combined_score']:.4f}] {r['video_id']}: {r['caption'][:60]}...")
        if results:
            print("  ✅ Hybrid search OK")
        else:
            print("  ⚠️  No results (DB may be empty)")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # --- Test 3: Full pipeline ---
    print("\nTEST 3: Full pipeline (search + rerank)")
    try:
        final = search_with_rerank("forklift near pedestrians", limit=5, rerank_top_k=3)
        print(f"  Final results: {len(final)}")
        for f in final:
            p = f["payload"]
            print(f"    [dense={f['dense_score']:.4f}, rerank={f['rerank_score']:.4f}] "
                  f"{p['video_id']}: {p['caption'][:50]}...")
        if final:
            print("  ✅ Full pipeline OK")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # --- Test 4: get_video_snapshots ---
    print("\nTEST 4: get_video_snapshots")
    try:
        snaps = get_video_snapshots("factory_001")
        print(f"  Result: {snaps[:80]}...")
        if "Timeline" in snaps:
            print("  ✅ get_video_snapshots OK")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # --- Test 5: list_categories ---
    print("\nTEST 5: list_categories")
    cats = list_categories()
    print(f"  {cats[:80]}...")
    if "Available categories" in cats:
        print("  ✅ list_categories OK")

    # --- Test 6: generate_context ---
    print("\nTEST 6: generate_context")
    try:
        ctx = generate_context("unauthorized entry restricted area", limit=5, rerank_top_k=3)
        print(f"  query: {ctx['query']}")
        print(f"  count: {ctx['count']}")
        for item in ctx["context"]:
            print(f"    {item['video_id']} | {item['timestamp_range']} | {item['caption'][:50]}...")
        if ctx["count"] > 0:
            print("  ✅ generate_context OK")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    print("\nAll postgres search tests completed!")
