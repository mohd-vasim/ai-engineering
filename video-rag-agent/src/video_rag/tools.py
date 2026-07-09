"""LangChain tools for the Video RAG AI agent.

Each tool wraps a video_rag function and is decorated with @tool so it
can be registered with a LangChain agent directly.
"""

from typing import Optional

from langchain_core.tools import tool

from video_rag import (
    build_filter,
    generate_context,
    hybrid_search,
    search_by_video_id,
    search_with_rerank,
    count_points,
    load_mock_data,
)


@tool
def search_video_captions(
    query: str,
    limit: int = 10,
) -> str:
    """Search across all video caption snapshots using hybrid semantic + keyword search.
    Returns relevant video clips with their captions, timestamps, and video metadata.
    Use this when the user asks a general question about video content."""
    results = search_with_rerank(query=query, limit=limit, rerank_top_k=limit)
    if not results:
        return "No matching video clips found."

    lines = [f"Found {len(results)} relevant clips:\n"]
    for r in results:
        p = r["payload"]
        lines.append(
            f"• [{p.get('video_id')}] {p.get('video_title')} "
            f"| {p.get('start_seconds')}s-{p.get('end_seconds')}s "
            f"| {p.get('category')}"
            f"\n  Caption: {p.get('caption')}"
            f"\n  (score: {r['score']:.3f}, rerank: {r['rerank_score']:.3f})"
        )
    return "\n".join(lines)


@tool
def search_video_captions_filtered(
    query: str,
    category: Optional[str] = None,
    video_id: Optional[str] = None,
    start_seconds_gte: Optional[float] = None,
    start_seconds_lte: Optional[float] = None,
    end_seconds_gte: Optional[float] = None,
    end_seconds_lte: Optional[float] = None,
    duration_seconds_gte: Optional[float] = None,
    duration_seconds_lte: Optional[float] = None,
    limit: int = 10,
) -> str:
    """Search video captions with metadata filters.
    Use this when the user wants to narrow results by category, video, or time range.
    Example categories: nature, food, technology, sports, music, animals, fitness, urban, animation, craft, automotive.
    """
    qfilter = build_filter(
        category=category,
        video_id=video_id,
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
        rerank_top_k=limit,
        query_filter=qfilter,
    )
    if not results:
        return "No matching video clips found with those filters."

    lines = [f"Found {len(results)} matching clips:\n"]
    for r in results:
        p = r["payload"]
        lines.append(
            f"• [{p.get('video_id')}] {p.get('video_title')} "
            f"| {p.get('start_seconds')}s-{p.get('end_seconds')}s "
            f"| {p.get('category')}"
            f"\n  Caption: {p.get('caption')}"
        )
    return "\n".join(lines)


@tool
def get_video_snapshots(video_id: str) -> str:
    """Get all caption snapshots for a specific video by its video_id (e.g. 'video_001').
    Returns the full timeline of captions with timestamps."""
    results = search_by_video_id(video_id=video_id)
    if not results:
        return f"No video found with id '{video_id}'."

    lines = [f"Timeline for {results[0].payload.get('video_title')} ({video_id}):\n"]
    for r in results:
        p = r.payload
        lines.append(
            f"  {p.get('start_seconds')}s - {p.get('end_seconds')}s "
            f"({p.get('duration_seconds')}s): {p.get('caption')}"
        )
    return "\n".join(lines)


@tool
def list_video_categories() -> str:
    """List all available video categories in the dataset.
    Use this to discover what kinds of videos are stored."""
    data = load_mock_data()
    categories = sorted({v["category"] for v in data["videos"]})
    count = len(categories)
    lines = [f"Available categories ({count} total):\n"]
    for c in categories:
        videos = [v["video_id"] for v in data["videos"] if v["category"] == c]
        lines.append(f"  • {c} — {len(videos)} videos: {', '.join(videos)}")
    lines.append(
        f"\nTotal videos: {len(data['videos'])}, total snapshots: {sum(len(v['snapshots']) for v in data['videos'])}"
    )
    return "\n".join(lines)


@tool
def collection_stats() -> str:
    """Get statistics about the Qdrant collection: total points stored and status.
    Use this to verify data has been loaded."""
    try:
        cnt = count_points()
        data = load_mock_data()
        total_videos = len(data["videos"])
        return (
            f"Collection '{'video_captions'}' has {cnt} points.\n"
            f"Dataset contains {total_videos} videos with 100 snapshots total."
        )
    except Exception as e:
        return f"Could not reach Qdrant. Is it running? Error: {e}"


# ---- Export as a list for easy registration ----

AGENT_TOOLS = [
    search_video_captions,
    search_video_captions_filtered,
    get_video_snapshots,
    list_video_categories,
    collection_stats,
]
