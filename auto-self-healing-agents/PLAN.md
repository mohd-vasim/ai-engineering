# Implementation Plan — Auto-Healing Agent Resuscitation

> **Status:** 🟡 Planning complete · implementation in progress
> **Scope:** Auto-Healing Agent Resuscitation pattern only (Incremental Checkpointing explicitly out of scope)

---

## 1. Goal

Build a Python implementation of the **Auto-Healing Agent Resuscitation** pattern in which:

1. An external **Supervisor** continuously monitors a pool of long-running **Worker Agents** via a heartbeat mechanism.
2. When the supervisor detects a missed heartbeat (i.e., the agent's process has crashed), it **automatically restarts** the agent and re-initializes it to a clean state.
3. To avoid resource exhaustion, the supervisor applies a **crash-loop backoff** strategy — progressively longer waits between restart attempts for repeatedly failing agents.

The end deliverable is a runnable demo showing the full 5-step recovery cycle, plus a test suite covering the supervisor's decision logic.

---

## 2. Non-Goals (Explicitly Excluded)

- ❌ **Incremental Checkpointing** — state persistence across restarts. The reference doc covers this as a *separate* pattern; the user has scoped this work to auto-healing only.
- ❌ Distributed consensus / multi-supervisor leader election.
- ❌ Production-grade container orchestrator backends (Kubernetes) — only a pluggable interface and a stub in-process backend.
- ❌ Deep semantic health checks (e.g., "is the agent producing correct output") — heartbeats confirm liveness only.

---

## 3. Design Overview

```
                ┌────────────────────────────────────────────┐
                │              Supervisor                    │
                │                                            │
                │   ┌──────────────┐    ┌──────────────┐     │
                │   │ HealthCheck  │    │   Backoff    │     │
                │   │  (per-agent  │    │  (per-agent  │     │
                │   │  state)      │    │  delay calc) │     │
                │   └──────┬───────┘    └──────┬───────┘     │
                │          │                   │             │
                │          └─────┐   ┌─────────┘             │
                │                ▼   ▼                       │
                │         ┌──────────────────┐               │
                │         │  Restart Decision│               │
                │         └────────┬─────────┘               │
                │                  │                         │
                │                  ▼                         │
                │         ┌──────────────────┐               │
                │         │ RestartBackend   │               │
                │         │  (in-process /   │               │
                │         │   subprocess)    │               │
                │         └────────┬─────────┘               │
                └──────────────────┼─────────────────────────┘
                                   ▼
                ┌────────────────────────────────────────────┐
                │          Worker Agents (pool)              │
                │   DataProcessor-1   DataProcessor-2   ...  │
                │   [heartbeat]       [heartbeat]           │
                └────────────────────────────────────────────┘
```

### Component Responsibilities

| Component         | Responsibility                                                                 |
|-------------------|--------------------------------------------------------------------------------|
| `Supervisor`      | Owns the monitoring loop; queries `HealthCheck`; invokes `Backoff`; calls `RestartBackend`. |
| `HealthCheck`     | Per-agent state: last-seen heartbeat timestamp, consecutive-failure count.     |
| `Backoff`         | Computes delay before next restart (linear or exponential).                    |
| `RestartBackend`  | Pluggable strategy — `InProcessBackend` (re-instantiate) or `SubprocessBackend` (spawn). |
| `WorkerAgent`     | Abstract interface: `run()`, `heartbeat()`, `stop()`.                          |
| `DataProcessingAgent` | Reference implementation; simulates a long-running data stream consumer.   |

---

## 4. Phased Plan

### Phase 0 — Project Scaffolding ✅ (in progress)

- [x] Confirm `pyproject.toml` (Python 3.12, `langchain>=1.3.11`).
- [x] Create directory layout: `src/auto_self_healing_agents/...`
- [x] Create `README.md` and `PLAN.md` (this file).
- [ ] Add dev dependencies: `pytest`, `pytest-cov`, `pytest-mock`, `ruff`.

### Phase 1 — Core Abstractions

**Goal:** Define the contracts that everything else depends on.

| Task | File | Notes |
|------|------|-------|
| Define `WorkerAgent` ABC | `src/auto_self_healing_agents/agents/base.py` | Methods: `start()`, `stop()`, `is_alive()`, `heartbeat()`. |
| Define `Heartbeat` dataclass | `src/auto_self_healing_agents/supervisor/health_check.py` | `agent_id`, `timestamp`, `consecutive_failures`. |
| Define `RestartBackend` ABC | `src/auto_self_healing_agents/supervisor/restart.py` | Method: `restart(agent_id) -> WorkerAgent`. |
| Define `Backoff` strategy ABC | `src/auto_self_healing_agents/supervisor/backoff.py` | Method: `wait_seconds(consecutive_failures) -> float`. |

**Acceptance criteria:**
- All ABCs have explicit type hints and docstrings.
- ABCs raise `NotImplementedError` for abstract methods.

### Phase 2 — Reference Implementations

**Goal:** Build the concrete classes used by the demo.

| Task | File | Notes |
|------|------|-------|
| `DataProcessingAgent` | `src/auto_self_healing_agents/agents/data_processing_agent.py` | Simulates a worker that processes items; emits heartbeats; can be configured to crash randomly for demo. |
| `ExponentialBackoff` | `src/auto_self_healing_agents/supervisor/backoff.py` | `base * factor^n` clamped to `max_backoff`. |
| `InProcessRestartBackend` | `src/auto_self_healing_agents/supervisor/restart.py` | Re-instantiates the agent class and starts it. (Sufficient for the in-process demo.) |

**Acceptance criteria:**
- `DataProcessingAgent` runs in a background thread, emits heartbeats every N seconds.
- `ExponentialBackoff` returns sane values for n=0,1,2,5,10 (incl. clamping).
- `InProcessRestartBackend` returns a fresh, started agent instance.

### Phase 3 — Supervisor Logic

**Goal:** Implement the monitoring loop and decision logic.

| Task | File | Notes |
|------|------|-------|
| `Supervisor` class | `src/auto_self_healing_agents/supervisor/supervisor.py` | Holds `agent_ids`, `check_interval`, `heartbeat_timeout`. |
| Monitoring loop | same | Calls `is_agent_healthy(agent_id)`; on unhealthy, consults `Backoff` and `RestartBackend`. |
| Lifecycle: `start()`, `stop()`, `__enter__/__exit__` | same | Run loop in a daemon thread; clean shutdown. |
| Structured logging | `src/auto_self_healing_agents/observability/logger.py` | Use `logging` with a consistent format. |

**Acceptance criteria:**
- The loop runs every `check_interval` seconds.
- An agent with no heartbeat for >`heartbeat_timeout` is marked unhealthy.
- An unhealthy agent is restarted only after the backoff window has elapsed.
- Logs include: agent_id, action, timestamp, consecutive_failures.

### Phase 4 — End-to-End Demo

**Goal:** A runnable script that shows the full 5-step recovery cycle.

| Task | File | Notes |
|------|------|-------|
| `demo.py` | `src/auto_self_healing_agents/demo.py` | Creates 2 `DataProcessingAgent`s, a `Supervisor`, and runs the monitor for ~60s. |
| Inject crash | same | Programmatically kill an agent (e.g., raise an unhandled exception in its loop) to demonstrate auto-healing. |
| CLI entrypoint | same | `python -m auto_self_healing_agents.demo` |

**Acceptance criteria:**
- Running the demo produces a clear log trace covering: normal operation → crash → health check failure → resuscitation → recovery.
- The killed agent resumes heartbeating within a few cycles.

### Phase 5 — Tests

**Goal:** Validate the supervisor's decision logic in isolation.

| Test file | Coverage |
|-----------|----------|
| `tests/test_health_check.py` | Healthy / stale / consecutive-failure transitions. |
| `tests/test_backoff.py` | Exponential growth, clamping at `max_backoff`, `n=0` returns 0. |
| `tests/test_supervisor.py` | End-to-end: crash detection, restart trigger, backoff respected, multiple agents. |
| `tests/test_data_processing_agent.py` | Heartbeat emission, crash injection, lifecycle. |

**Acceptance criteria:**
- All tests pass.
- ≥ 80% line coverage on `supervisor/`.

### Phase 6 — Polish

- [ ] Add `ruff` config + run linter.
- [ ] Add type-check (`mypy --strict` or `pyright`).
- [ ] Add CI workflow (GitHub Actions): `pytest` on Python 3.12.
- [ ] Add a "Limitations & Trade-offs" section to README (✅ already done).
- [ ] Final review against the original docs/screenshots.

---

## 5. File-by-File Plan

```
src/auto_self_healing_agents/
├── __init__.py                              # public API exports
├── agents/
│   ├── __init__.py
│   ├── base.py                              # WorkerAgent ABC
│   └── data_processing_agent.py             # reference impl + crash simulator
├── supervisor/
│   ├── __init__.py
│   ├── supervisor.py                        # main loop, lifecycle
│   ├── health_check.py                      # per-agent heartbeat tracker
│   ├── backoff.py                           # exponential/linear strategies
│   └── restart.py                           # pluggable restart backends
├── observability/
│   ├── __init__.py
│   └── logger.py                            # structured logging config
└── demo.py                                  # end-to-end demonstration

tests/
├── __init__.py
├── test_health_check.py
├── test_backoff.py
├── test_supervisor.py
└── test_data_processing_agent.py
```

---

## 6. Key Interfaces (Sketch)

```python
# agents/base.py
class WorkerAgent(ABC):
    agent_id: str
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_alive(self) -> bool: ...
    def heartbeat(self) -> bool: ...  # True if healthy

# supervisor/health_check.py
@dataclass
class HealthState:
    agent_id: str
    last_heartbeat: float
    consecutive_failures: int = 0

    def record_heartbeat(self) -> None: ...
    def record_miss(self) -> None: ...
    def is_healthy(self, now: float, timeout: float) -> bool: ...

# supervisor/backoff.py
class Backoff(ABC):
    def wait_seconds(self, consecutive_failures: int) -> float: ...

class ExponentialBackoff(Backoff):
    def __init__(self, base: float, factor: float, max_wait: float): ...

# supervisor/restart.py
class RestartBackend(ABC):
    def restart(self, agent_id: str) -> WorkerAgent: ...

class InProcessRestartBackend(RestartBackend):
    def __init__(self, agent_factory: Callable[[str], WorkerAgent]): ...

# supervisor/supervisor.py
class Supervisor:
    def __init__(
        self,
        agent_ids: list[str],
        agent_factory: Callable[[str], WorkerAgent],
        check_interval: float = 10.0,
        heartbeat_timeout: float = 30.0,
        backoff: Backoff | None = None,
        restart_backend: RestartBackend | None = None,
    ): ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def monitor_forever(self) -> None: ...
    def is_agent_healthy(self, agent_id: str) -> bool: ...
    def resuscitate(self, agent_id: str) -> None: ...
```

---

## 7. Acceptance Criteria (End-to-End)

- [ ] `uv run python -m auto_self_healing_agents.demo` runs successfully.
- [ ] The demo log shows the 5-step auto-healing workflow within ~60 seconds.
- [ ] `uv run pytest` passes with ≥ 80% coverage on `supervisor/`.
- [ ] A `DataProcessingAgent` killed mid-run resumes heartbeating without any human action.
- [ ] Backoff delays are observed in the log when a crash is repeated in quick succession.
- [ ] README + PLAN accurately describe the final implementation.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Demo runs too long / never terminates | Add a `max_runtime` parameter that auto-stops the demo. |
| Tests are flaky due to timing (`time.sleep`) | Use `freezegun` or inject a clock; use generous tolerances. |
| In-process restart loses the original process id | Document this clearly; mention real backends (subprocess / k8s) as future work. |
| Heartbeat thread leaks if `stop()` is never called | Use `daemon=True` threads; ensure `__exit__` calls `stop()`. |

---

## 9. Progress Tracker

| Phase | Description                          | Status |
|-------|--------------------------------------|--------|
| 0     | Project scaffolding                  | 🟡 In progress |
| 1     | Core abstractions (ABCs + dataclass) | ⬜ Not started |
| 2     | Reference implementations            | ⬜ Not started |
| 3     | Supervisor logic & monitoring loop   | ⬜ Not started |
| 4     | End-to-end demo                      | ⬜ Not started |
| 5     | Test suite                           | ⬜ Not started |
| 6     | Polish (lint, types, CI)             | ⬜ Not started |

---

## 10. References

- Source material: [docs/](docs/) (book screenshots, pages 224–227).
- *Auto-Healing Agent Resuscitation* — chapter 7, section 7.6.
- *Agentic Architectural Patterns for Building Multi-Agent Systems*.
