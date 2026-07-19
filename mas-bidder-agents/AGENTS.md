# mas-bidder-agents — Agent Instructions

Contract-Net Marketplace pattern for dynamic multi-agent task allocation.
A **Solicitor** auctions tasks to **BidderAgents** who bid on capability, cost,
and confidence. The design spec is defined in `docs/FDS.md` and `docs/FDS_v2.md`.

> **Current state:** scaffold only — `pyproject.toml` with zero deps, no code.
> The FDS docs are the source of truth for what to build.

## Key Facts

- **Python** `>=3.12`, **package manager**: `uv` (`uv sync`, `uv run`).
- **Layout**: use `src/`-layout (`src/mas_bidder_agents/...`) with `[tool.setuptools.packages.find] where = ["src"]` in `pyproject.toml`.
- **No tests / lint / CI exist yet** — add them alongside implementation. Use `pytest` and `ruff` (sibling project convention).

## Design (from FDS Docs)

Core auction flow:

1. **Announcement** — Solicitor broadcasts task with constraints + strict deadline.
2. **Bidding** — Each BidderAgent evaluates `can_handle()`, `estimate_compute_cost()`, `assess_capability()`, submits a `Bid`.
3. **Evaluation** — Solicitor scores bids via `Utility = α(Confidence) - β(Cost) - γ(ETA)` with configurable weights.
4. **Award** — Highest score wins; Solicitor calls `execute_contract()` on winner.
5. **Feedback** — Result updates the Reputation Registry (penalizes failures, discourages overconfident bidding).

Guardrails: enforce auction deadline, `NoBidsException` fallback, pluggable utility function weights. See `docs/FDS.md` and `docs/FDS_v2.md` for full details.

## Style

- Type hints: `X | None` (3.12 union syntax), not `Optional[X]`.
- Docstrings on every public function.
- Module-level `if __name__ == "__main__":` blocks for smoke-testing.

## Monorepo Context

This is one of several MAS sub-projects under `ai-engineering/`. Siblings
`auto-self-healing-agents/` and `mas-supply-chain-disruption/` implement
related patterns. No shared code currently — keep projects independent.
