---
description: Instructions for AI coding agents working on the video-rag-agent codebase.
---

# Video RAG Agent — Agent Instructions

A Python project that builds a **Video Analytics Agent** on top of a Qdrant
vector database. Caption snapshots from short videos are embedded (dense +
sparse), stored in Qdrant, and queried via hybrid search + Cohere rerank.
A LangChain tool layer exposes the search to an LLM agent.

> This file is for AI coding agents. For human-facing docs see [README.md](README.md)
> and [TEST_CASES.md](TEST_CASES.md). Do not duplicate their content here — link
> to them.

## Tech Stack & Versions

- **Python** `>=3.12` (see [pyproject.toml](pyproject.toml))
- **Package manager**: `uv` (uses `uv.lock`; do not introduce `pip` workflows)
- **Layout**: `src/`-layout package `video_rag` (no top-level `__init__.py` outside `src/`)
- **Core libs**: `openai`, `qdrant-client[fastembed]`, `httpx`, `langchain`, `langchain-openai`, `langchain-openrouter`
- **Embedding model**: `text-embedding-3-small` (1536-dim, set in [config.py](src/video_rag/config.py))
- **Rerank model**: `cohere/rerank-4-fast` (called via OpenRouter HTTP)
- **Sparse model**: `Qdrant/bm25` (FastEmbed)

## Project Structure

```
src/video_rag/
├── config.py      # Settings dataclass + env loading
├── embeddings.py  # Dense (OpenAI/OpenRouter) + sparse (FastEmbed BM25)
├── ingest.py      # Qdrant collection create + bulk upsert
├── search.py      # Filter builder, hybrid search (RRF), rerank, generate_context
├── tools.py       # LangChain @tool wrappers exposed as AGENT_TOOLS
├── prompt.py      # VIDEO_RAG_SYSTEM_PROMPT
└── qdrant/        # (reserved for future qdrant-specific helpers — currently empty)
notebooks/
├── main.ipynb              # End-to-end agent demo
└── qdrant_ingestion.ipynb  # Manual ingestion walkthrough
data/
└── mock_video_captions.json  # 17 videos, 100 snapshots, 11 categories
```

## Setup

Requires two external services:

1. **Qdrant** (Docker) — see the `docker run` block in [README.md](README.md).
   Container must be named `qdrant-vdb` (the `--name` flag must precede the image).
2. **OpenRouter API key** — set `OPENROUTER_API_KEY` in the environment.
   `config.Settings.validate()` raises if it is missing.

Install with:

```bash
uv sync
```

## Build / Test / Run

There is no `pytest` suite. Each module ends in a runnable `if __name__ == "__main__":`
block that prints test output. To exercise a module:

```bash
uv run python -m video_rag.embeddings
uv run python -m video_rag.ingest
uv run python -m video_rag.search
```

Manual end-to-end demos live in the notebooks under [notebooks/](notebooks/).
Open `main.ipynb` to see the full agent loop.

## Public API (entry points)

Re-exported from [src/video_rag/__init__.py](src/video_rag/__init__.py):

| Function | Purpose |
|---|---|
| `generate_context` | One-call context retrieval (hybrid + rerank + filters) — preferred agent entry point |
| `AGENT_TOOLS` | List of `@tool`-decorated LangChain tools |
| `hybrid_search` | Raw dense + sparse RRF search |
| `search_with_rerank` | Hybrid search → Cohere rerank pipeline |
| `search_by_video_id` | Fetch all snapshots for one video |
| `build_filter` | Construct a Qdrant `models.Filter` from kwargs |
| `recreate_collection` | Drop + recreate collection (destructive) |
| `ingest_mock_data` | Load JSON, embed, upsert |
| `count_points` | Collection size |
| `load_mock_data` | Read `data/mock_video_captions.json` |
| `embed_text` / `embed_texts` | Dense embeddings |
| `get_sparse_embedding` | BM25 sparse vector |
| `derive_snapshot_ranges` | Add `start_seconds`/`end_seconds` to raw snapshots |

## Conventions

- **Type hints**: use `X | None` (Python 3.12 union syntax), not `Optional[X]`.
- **Docstrings**: every public function has a one-line summary + `Args`/`Returns`
  where non-obvious. Match the existing style in the module you are editing.
- **Module-level test blocks**: each file has a `__main__` block that prints
  `  ✅ ...` / `  ❌ ...` lines. When adding new logic, mirror that pattern so
  the file can be smoke-tested with `python -m`.
- **Skip gracefully when env is missing**: tests that need the OpenRouter key
  check `os.environ.get("OPENROUTER_API_KEY", "")` and print `SKIP: ...` if
  absent. Follow the same pattern in any new test block.
- **OpenRouter headers**: every HTTP call to OpenRouter must send
  `HTTP-Referer: https://github.com/mohd-vasim/ai-engineering` and
  `X-OpenRouter-Title: Video-RAG-Agent`. See [embeddings.py](src/video_rag/embeddings.py) and
  [search.py](src/video_rag/search.py) for the exact strings.
- **Qdrant vectors**: every point has both a `dense` named vector and a
  `sparse` named vector — keep the names in sync across `recreate_collection`,
  `hybrid_search`, and `build_points`.
- **Deterministic point IDs**: snapshot IDs are derived via
  `uuid.uuid5(NAMESPACE_DNS, f"{video_id}_{start}_{end}")`. Do not switch to
  `uuid4` — re-ingestion must be idempotent.
- **Filter conventions**: snapshots have `start_seconds`, `end_seconds`, and
  `duration_seconds` as separate payload fields. For "first N seconds" queries
  use `end_seconds_lte=N` (or `start_seconds_lte=N` for the start). Do not
  approximate with Python-side filtering.
- **No silent fallbacks on Qdrant errors**: `recreate_collection` swallows the
  `delete_collection` error (collection may not exist yet) but raises on
  `create_collection`. Keep that asymmetry.
- **Tool wrappers**: every function in [tools.py](src/video_rag/tools.py) is
  decorated with `@tool` from `langchain_core.tools` and registered in
  `AGENT_TOOLS`. New tools must follow the same shape: rich docstring (this
  is what the LLM sees) + return a human-readable string, not a dict.

## Dataset

`data/mock_video_captions.json` contains 17 videos, 100 caption snapshots
(10s interval), 11 categories: `nature`, `food`, `technology`, `sports`,
`music`, `animals`, `fitness`, `urban`, `animation`, `craft`, `automotive`.
The full inventory and 23 sample queries are in [TEST_CASES.md](TEST_CASES.md) —
use them to validate any agent or search change.

## Common Pitfalls

- **OpenRouter `data=None`**: when OpenRouter is overloaded it returns
  `data=None` instead of raising. `embed_texts` already checks for this — do
  not remove that branch.
- **Docker flag ordering**: `--name qdrant-vdb` must come before the image
  name (`qdrant/qdrant`). Anything after the image name is treated as the
  container's command.
- **Rerank response shape**: OpenRouter's `/rerank` endpoint returns
  `{"results": [{"index", "relevance_score", "document"}, ...]}` — `index` is
  a position into the input list, not a Qdrant point id.
- **Sparse vector namespacing**: FastEmbed returns numpy arrays — convert
  with `.tolist()` before constructing `models.SparseVector`.
- **`search.py` is long (~450 lines)**: it owns the filter builder, hybrid
  search, rerank, video-id lookup, and `generate_context`. If you add a new
  search mode, add a section in the same file rather than splitting.

## When You Are Unsure

- Read the matching module first — each one is the source of truth for its
  area (e.g. payload schema is defined in [ingest.py:build_points](src/video_rag/ingest.py)).
- Check [TEST_CASES.md](TEST_CASES.md) for the expected user-facing behavior.
- Run the module's `__main__` block before opening a PR to catch obvious
  regressions in the search/embedding path.
