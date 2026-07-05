# Auto-Self-Healing Agents

A Python implementation of the **Auto-Healing Agent Resuscitation** pattern for building highly available, multi-agent systems. An external supervisor continuously monitors worker agents via a heartbeat mechanism and automatically restarts any agent whose process has crashed — without human intervention.

> Based on the architectural pattern documented in *"Agentic Architectural Patterns for Building Multi-Agent Systems"*.

---

## Table of Contents

- [Why Auto-Healing?](#why-auto-healing)
- [The Pattern](#the-pattern)
- [How It Works](#how-it-works)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Limitations & Trade-offs](#limitations--trade-offs)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [References](#references)

---

## Why Auto-Healing?

In long-running, stateful multi-agent systems, agents are deployed as **persistent processes** (microservices, daemons, long-lived workers). A bug, corrupted dependency, or unhandled exception can cause an agent's process to crash completely — taking it offline and halting any workflow that depends on it.

Without an automated recovery mechanism:

- The agent remains offline until an operations team manually intervenes.
- Critical workflows stall, leading to extended downtime and SLA violations.
- On-call engineers are paged for issues that could be resolved automatically.

**Auto-Healing Agent Resuscitation** solves this by introducing an external supervisor that detects failures (via heartbeats) and automatically restarts the failed agent — restoring availability with no human in the loop.

---

## The Pattern

An **external supervisor** (or orchestrator) continuously monitors the health of its **worker agents**, typically via a *heartbeat* signal. If an agent becomes unresponsive or terminates unexpectedly, the supervisor:

1. Detects the missed heartbeats.
2. Logs the failure.
3. Triggers a **resuscitation protocol** — restart the agent's process and re-initialize it to a clean state.

```
   ┌──────────────────┐                  ┌──────────────────┐
   │    Supervisor    │   heartbeat      │   Worker Agent   │
   │  (health check)  │◄──────────────── │  (DataProcessor) │
   └──────────────────┘                  └──────────────────┘
            │                                    │
            │  detects missed heartbeats         │
            ▼                                    │
   ┌──────────────────┐                          │
   │  Resuscitate     │  ── restart process ──►  │
   │  Agent           │                          │
   └──────────────────┘                          │
            │                                    │
            └──────────── new heartbeat ─────────┘
```

---

## How It Works

The auto-healing workflow has five steps:

| # | Step                | Description                                                                                          |
|---|---------------------|------------------------------------------------------------------------------------------------------|
| 1 | **Normal operation** | The worker agent is running and periodically sends a heartbeat to the supervisor.                  |
| 2 | **Process crash**    | The agent encounters a critical bug (e.g., a memory error) and its process terminates unexpectedly. |
| 3 | **Health check failure** | The supervisor's monitoring loop doesn't receive a heartbeat within the expected interval. It marks the agent as unhealthy. |
| 4 | **Resuscitation**    | The supervisor logs the failure and triggers the restart protocol (e.g., process manager, container orchestrator). |
| 5 | **Recovery**         | The agent is restarted from its original definition, re-initializes, and resumes sending heartbeats. The system recovers automatically — no human intervention. |

### The Monitoring Loop

```
   Start Monitoring Loop
            │
            ▼
   ┌──────────────────┐
   │ Is Agent Healthy?│
   └──────────────────┘
       /          \
     Yes           No
      │             │
      ▼             ▼
  Wait Next      Log Failure
  Interval           │
      ▲             ▼
      │      ┌──────────────────┐
      │      │ Resuscitate Agent│
      │      └──────────────────┘
      └───────────────┘
```

---

## Features

- ✅ **Heartbeat-based health monitoring** — supervisor polls worker agents on a configurable interval.
- ✅ **Automatic restart / resuscitation** — crashed agents are restarted and re-initialized without manual intervention.
- ✅ **Crash loop backoff** — supervisor waits progressively longer between restart attempts for repeatedly failing agents (prevents resource exhaustion).
- ✅ **Pluggable restart backend** — abstracts how an agent's process is restarted (in-process, subprocess, container orchestrator).
- ✅ **Configurable monitoring interval & timeouts** — tune sensitivity and recovery time.
- ✅ **Failure logging & observability** — structured logs for every health check, crash detection, and resuscitation.
- ✅ **End-to-end demo** — simulates a crashing `DataProcessingAgent` and shows the full recovery cycle.

---

## Project Structure

```
auto-self-healing-agents/
├── docs/                              # Source documentation (book screenshots)
│   └── *.png
├── src/
│   └── auto_self_healing_agents/
│       ├── __init__.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py                # Abstract WorkerAgent interface
│       │   └── data_processing_agent.py  # Reference DataProcessingAgent implementation
│       ├── supervisor/
│       │   ├── __init__.py
│       │   ├── supervisor.py          # Main Supervisor class
│       │   ├── health_check.py        # Heartbeat / health-check logic
│       │   ├── backoff.py             # Crash loop backoff strategy
│       │   └── restart.py             # Pluggable restart backends
│       ├── observability/
│       │   ├── __init__.py
│       │   └── logger.py              # Structured logging setup
│       └── demo.py                    # End-to-end demo script
├── tests/
│   ├── __init__.py
│   ├── test_supervisor.py
│   ├── test_health_check.py
│   ├── test_backoff.py
│   └── test_data_processing_agent.py
├── pyproject.toml
├── uv.lock
├── PLAN.md                            # Implementation plan & progress tracker
└── README.md                          # This file
```

---

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd auto-self-healing-agents

# Install dependencies (creates .venv automatically)
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

**Requirements:**

- Python ≥ 3.12
- `uv` package manager (or `pip` if you prefer)
- Dependencies: `langchain>=1.3.11`

---

## Quick Start

Run the end-to-end demo to see auto-healing in action:

```bash
uv run python -m auto_self_healing_agents.demo
```

You'll see output similar to:

```
SUPERVISOR: Starting process for Agent DataProcessor-1...
SUPERVISOR: Starting process for Agent DataProcessor-2...
SUPERVISOR: Monitoring loop started.
SUPERVISOR: Agent DataProcessor-1 is healthy. ✓
SUPERVISOR: Agent DataProcessor-2 is healthy. ✓
... (10s wait)
SUPERVISOR: Agent DataProcessor-1 is unhealthy. ✗
SUPERVISOR: Agent DataProcessor-1 is unhealthy. ✗
SUPERVISOR: Logging failure for Agent DataProcessor-1.
SUPERVISOR: Attempting resuscitation of Agent DataProcessor-1.
SUPERVISOR: Restarting Agent DataProcessor-1...
SUPERVISOR: Starting process for Agent DataProcessor-1...
SUPERVISOR: Agent DataProcessor-1 has been resuscitated. ✓
SUPERVISOR: Agent DataProcessor-1 is healthy. ✓
```

---

## Usage

### 1. Define a Worker Agent

Subclass `WorkerAgent` and implement `run()` and `heartbeat()`:

```python
from auto_self_healing_agents.agents.base import WorkerAgent

class MyAgent(WorkerAgent):
    def run(self) -> None:
        # Long-running work loop
        while not self.should_stop:
            self.process_next_item()

    def heartbeat(self) -> bool:
        # Return True if this agent is healthy
        return self.is_alive() and not self.has_crashed()
```

### 2. Create a Supervisor

```python
from auto_self_healing_agents.supervisor.supervisor import Supervisor

supervisor = Supervisor(
    agent_ids=["DataProcessor-1", "DataProcessor-2"],
    check_interval=10,           # seconds between health checks
    heartbeat_timeout=30,        # seconds before declaring unhealthy
    max_backoff=300,             # max backoff between restart attempts
)
```

### 3. Start Monitoring

```python
supervisor.start()   # runs forever; use supervisor.stop() in another thread to halt
```

Or in a managed context:

```python
with Supervisor(agent_ids=["DataProcessor-1"]) as supervisor:
    supervisor.monitor_forever()
```

---

## Configuration

| Parameter          | Default | Description                                                        |
|--------------------|---------|--------------------------------------------------------------------|
| `check_interval`   | `10`    | Seconds between health-check cycles.                               |
| `heartbeat_timeout`| `30`    | Seconds without a heartbeat before an agent is marked unhealthy.   |
| `max_backoff`      | `300`   | Maximum backoff delay (seconds) between restart attempts.          |
| `backoff_factor`   | `2.0`   | Multiplier applied to backoff delay after each consecutive failure. |
| `max_restarts`     | `None`  | Optional cap on restart attempts before giving up (per agent).     |

---

## Architecture

### Components

| Component         | Responsibility                                                                |
|-------------------|-------------------------------------------------------------------------------|
| **Supervisor**    | Owns the monitoring loop, calls `is_agent_healthy` for each worker, decides when to restart. |
| **HealthCheck**   | Tracks per-agent heartbeat state (last-seen timestamp, failure count).        |
| **Backoff**       | Computes the wait time before the next restart attempt (linear or exponential). |
| **RestartBackend**| Pluggable strategy for actually restarting a process (in-process / subprocess / Kubernetes). |
| **WorkerAgent**   | The long-running agent; emits heartbeats and exposes a liveness check.       |

### Failure Detection

- Each worker emits a heartbeat (timestamp or health-flag update) on a fixed cadence.
- The supervisor records the last-seen timestamp per agent.
- If `now - last_seen > heartbeat_timeout`, the agent is considered unhealthy.
- Unhealthy agents trigger the resuscitation protocol.

### Crash Loop Backoff

To avoid resource exhaustion when an agent has a persistent bug, the supervisor applies a **backoff strategy**:

- 1st failure → restart immediately.
- 2nd consecutive failure → wait `backoff_factor^1 * base` seconds.
- 3rd consecutive failure → wait `backoff_factor^2 * base` seconds.
- … up to `max_backoff`.

This prevents tight restart loops while still allowing recovery from transient failures.

---

## Limitations & Trade-offs

> These are inherent to the Auto-Healing pattern and not specific to this implementation.

1. **Masking bugs** — A persistent crash-causing bug can lead to a "crash loop" where the agent is constantly restarted. Backoff mitigates resource waste but does not eliminate the underlying problem. Pair with alerting (e.g., PagerDuty) on repeated restarts.
2. **State loss** — This implementation handles **process-level** recovery only. State persistence across restarts requires a separate pattern (e.g., *Incremental Checkpointing*) — explicitly out of scope for this project.
3. **Single-region supervision** — The supervisor is itself a single point of failure. For production-grade HA, run multiple supervisors with leader election (e.g., etcd, ZooKeeper).
4. **Health-check granularity** — A heartbeat confirms *liveness* but not *correctness*. An agent may be alive but stuck or producing garbage output. Consider deeper health probes for production use.

---

## Testing

```bash
# Run the full test suite
uv run pytest

# Run with coverage
uv run pytest --cov=auto_self_healing_agents --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_supervisor.py -v
```

Tests cover:

- ✅ Heartbeat detection (healthy, missed, stale)
- ✅ Restart / resuscitation triggering
- ✅ Backoff calculation under repeated failures
- ✅ Supervisor recovery from simulated agent crashes
- ✅ End-to-end demo flow

---

## Roadmap

This project implements the **Auto-Healing Agent Resuscitation** pattern only. Potential extensions:

- [ ] **Incremental Checkpointing** — persist agent state to recover from long-running workflow failures.
- [ ] **Distributed supervision** — leader election for multiple supervisors.
- [ ] **Metrics & dashboards** — Prometheus / OpenTelemetry integration.
- [ ] **Container orchestrator backend** — Kubernetes liveness probes + restart policy.
- [ ] **Alerting** — PagerDuty / Slack notifications on repeated restarts.

---

## References

- *"Agentic Architectural Patterns for Building Multi-Agent Systems"* (chapter 7, pages 224–227) — source material.
- [LangChain documentation](https://python.langchain.com/) — used for agent abstractions.
- Related patterns: *Incremental Checkpointing*, *Supervisor-Worker*, *Circuit Breaker*.

---

## License

MIT (or your preferred license)
