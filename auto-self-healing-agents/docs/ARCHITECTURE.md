# Architecture & Visualizations

> Mermaid diagrams for the **Auto-Healing Agent Resuscitation** implementation.
> See [README.md](../README.md) for prose and [PLAN.md](../PLAN.md) for the build plan.

---

## Table of Contents

1. [System Context](#1-system-context)
2. [The 5-Step Auto-Healing Workflow](#2-the-5-step-auto-healing-workflow)
3. [Supervisor Monitoring Loop](#3-supervisor-monitoring-loop)
4. [Component Diagram](#4-component-diagram)
5. [Heartbeat & Restart Sequence](#5-heartbeat--restart-sequence)
6. [Crash-Loop Backoff Timeline](#6-crash-loop-backoff-timeline)
7. [Redis Key Layout](#7-redis-key-layout)
8. [POC Architecture (Notebook)](#8-poc-architecture-notebook)
9. [Failure & Recovery States](#9-failure--recovery-states)

---

## 1. System Context

High-level view: the supervisor sits *outside* the worker pool and observes it indirectly through **Redis** (shared state) and **MLflow** (observability). It does not communicate with workers directly.

```mermaid
flowchart LR
    User([Operator / Workflow])

    subgraph Supervisor[Supervisor Process]
        SL[Monitoring Loop]
        SD[Decision Logic]
        BR[Backoff Strategy]
        RB[Restart Backend]
    end

    subgraph Workers[Worker Pool]
        W1[DataProcessor-1]
        W2[DataProcessor-2]
        W3[DataProcessor-N]
    end

    Redis[(Upstash Redis<br/>heartbeats + state)]
    MLflow[(MLflow on Databricks<br/>traces + metrics)]

    User -->|configures| Supervisor
    SL -->|read hb| Redis
    SD -->|decision| BR
    BR -->|when ready| RB
    RB -->|restart| Workers
    Workers -->|write hb| Redis
    SL -.->|trace spans| MLflow
    SD -.->|log metrics| MLflow
```

---

## 2. The 5-Step Auto-Healing Workflow

Sequence diagram showing the full recovery cycle from the docs.

```mermaid
sequenceDiagram
    autonumber
    participant W as Worker Agent
    participant R as Redis
    participant S as Supervisor
    participant M as MLflow
    participant P as Process Mgr<br/>(Restart Backend)

    Note over W,P: Step 1 — Normal operation
    loop every heartbeat_interval
        W->>R: SET ash:hb:{id} Heartbeat{ts, alive=true}
    end

    Note over W,P: Step 2 — Process crash (unhandled exception)
    W--xW: process terminates (no more heartbeats)

    Note over W,P: Step 3 — Health check failure
    S->>R: GET ash:hb:{id}
    R-->>S: {ts: stale}
    S->>S: now - ts > heartbeat_timeout?<br/>→ UNHEALTHY
    S->>R: write HealthState{consecutive_failures++}
    S->>M: @mlflow.trace check_and_resuscitate

    Note over W,P: Step 4 — Resuscitation
    S->>S: backoff.wait_seconds(consecutive_failures)
    S->>P: restart(agent_id)
    P->>W: spawn fresh process
    W->>R: SET ash:hb:{id} Heartbeat{ts, alive=true}

    Note over W,P: Step 5 — Recovery
    S->>R: GET ash:hb:{id}
    R-->>S: {ts: fresh}
    S->>S: reset consecutive_failures = 0
    S->>R: log SupervisorAction{action=restarted}
    S->>M: log_metric total_restarts++
```

---

## 3. Supervisor Monitoring Loop

The exact flowchart from the docs, re-rendered as a Mermaid `stateDiagram-v2`.

```mermaid
stateDiagram-v2
    [*] --> Monitoring

    Monitoring: Start Monitoring Loop
    CheckHealthy: Is Agent Healthy?<br/>(heartbeat fresh?)

    Monitoring --> CheckHealthy: tick

    CheckHealthy --> Wait: Yes
    CheckHealthy --> LogFail: No

    Wait: Wait Next Interval
    LogFail: Log Failure<br/>(consecutive_failures++)

    Wait --> CheckHealthy: next tick

    LogFail --> InBackoff: backoff active
    LogFail --> Resuscitate: backoff elapsed

    InBackoff: Backoff<br/>(log backoff_skipped)
    InBackoff --> Wait: wait window

    Resuscitate: Resuscitate Agent<br/>(spawn fresh process)
    Resuscitate --> ResetCount
    ResetCount: reset failure counter
    ResetCount --> Wait

    Wait --> [*]: stop()
```

---

## 4. Component Diagram

Class structure of the POC + planned package layout. ABCs are in `<<abstract>>` style; concrete classes extend them.

```mermaid
classDiagram
    class WorkerAgent {
        <<abstract>>
        +agent_id: str
        +start() void
        +stop() void
        +crash() void
        +is_alive() bool
    }

    class DataProcessingAgent {
        +heartbeat_interval: float
        +crash_after_n_heartbeats: int?
        +crash_probability: float
        -_thread: Thread
        -_stop_event: Event
        -_run() void
    }

    class HeartbeatStore {
        +write_heartbeat(hb) void
        +read_heartbeat(agent_id) Heartbeat?
        +read_health(agent_id) HealthState
        +write_health(state) void
        +log_action(action) void
        +recent_actions(n) list
    }

    class Backoff {
        +base: float
        +factor: float
        +max_wait: float
        +wait_seconds(n) float
    }

    class Supervisor {
        +agent_specs: list
        +check_interval: float
        +heartbeat_timeout: float
        -_agents: dict
        -_monitor() void
        +start() void
        +stop() void
        +force_crash(agent_id) void
        +state() dict
    }

    class Heartbeat {
        <<Pydantic>>
        +agent_id: str
        +timestamp: float
        +alive: bool
        +sequence: int
    }

    class HealthState {
        <<Pydantic>>
        +agent_id: str
        +last_seen: float
        +consecutive_failures: int
        +total_restarts: int
        +last_action: str
    }

    class SupervisorAction {
        <<Pydantic>>
        +agent_id: str
        +action: Literal
        +details: str
        +consecutive_failures: int
        +backoff_seconds: float
    }

    WorkerAgent <|-- DataProcessingAgent
    DataProcessingAgent --> HeartbeatStore : writes heartbeats
    Supervisor --> WorkerAgent : manages pool
    Supervisor --> Backoff : consults
    Supervisor --> HeartbeatStore : reads/writes
    HeartbeatStore --> Heartbeat
    HeartbeatStore --> HealthState
    HeartbeatStore --> SupervisorAction
```

---

## 5. Heartbeat & Restart Sequence

A closer look at one full supervisor cycle.

```mermaid
sequenceDiagram
    participant Tick as Monitor Tick
    participant Sup as check_and_resuscitate()
    participant Store as HeartbeatStore
    participant Bk as Backoff
    participant Fac as agent_factory()
    participant New as New Agent

    Tick->>Sup: invoke(agent_id)
    Sup->>Store: read_heartbeat(agent_id)
    Store-->>Sup: Heartbeat | None

    alt heartbeat fresh
        Sup->>Store: read_health(agent_id)
        Store-->>Sup: HealthState
        Sup->>Store: write_health(consecutive_failures=0)
        Sup->>Store: log_action(action=healthy)
    else heartbeat stale or missing
        Sup->>Store: read_health(agent_id)
        Store-->>Sup: HealthState{consecutive_failures=N}
        Sup->>Bk: wait_seconds(N)
        Bk-->>Sup: wait_seconds

        alt inside backoff window
            Sup->>Store: log_action(action=backoff_skipped)
        else outside backoff window
            Sup->>Fac: agent_factory()
            Fac-->>Sup: new_agent
            Sup->>New: start()
            Sup->>Store: write_health(total_restarts++)
            Sup->>Store: log_action(action=restarted)
        end
    end
```

---

## 6. Crash-Loop Backoff Timeline

How backoff delays grow across consecutive failures. The cap (`max_wait`) prevents unbounded growth.

```mermaid
gantt
    title Crash-Loop Backoff Schedule (base=2s, factor=2, max=20s)
    dateFormat  X
    axisFormat %s

    section Failure 1
    Crash → Detect          :a1, 0, 1s
    Backoff (2s)            :crit, b1, after a1, 2s
    Restart                 :active, c1, after b1, 1s

    section Failure 2
    Crash → Detect          :a2, after c1, 1s
    Backoff (4s)            :crit, b2, after a2, 4s
    Restart                 :active, c2, after b2, 1s

    section Failure 3
    Crash → Detect          :a3, after c2, 1s
    Backoff (8s)            :crit, b3, after a3, 8s
    Restart                 :active, c3, after b3, 1s

    section Failure 4
    Crash → Detect          :a4, after c3, 1s
    Backoff (16s)           :crit, b4, after a4, 16s
    Restart                 :active, c4, after b4, 1s

    section Failure 5+
    Crash → Detect          :a5, after c4, 1s
    Backoff (capped 20s)    :crit, b5, after a5, 20s
    Restart                 :active, c5, after b5, 1s
```

---

## 7. Redis Key Layout

Storage organization under the `ash:` (auto-self-healing) prefix. Heartbeats and state are per-agent; the action log is shared.

```mermaid
graph TD
    Root["ash:* (Upstash Redis)"]

    HB["ash:hb:{agent_id}<br/>SET → Heartbeat JSON<br/>TTL: none (overwritten)"]
    ST["ash:state:{agent_id}<br/>SET → HealthState JSON"]
    ACT["ash:actions<br/>LIST, capped at 100<br/>LPUSH + LTRIM"]

    W1HB["ash:hb:DataProcessor-1"]
    W1ST["ash:state:DataProcessor-1"]
    W2HB["ash:hb:DataProcessor-2"]
    W2ST["ash:state:DataProcessor-2"]
    WNHB["ash:hb:CrashyProcessor"]
    WNST["ash:state:CrashyProcessor"]

    Root --> HB
    Root --> ST
    Root --> ACT

    HB --> W1HB
    HB --> W2HB
    HB --> WNHB

    ST --> W1ST
    ST --> W2ST
    ST --> WNST

    classDef hb fill:#fef3c7,stroke:#b45309
    classDef st fill:#dbeafe,stroke:#1d4ed8
    classDef act fill:#dcfce7,stroke:#15803d
    class W1HB,W2HB,WNHB hb
    class W1ST,W2ST,WNST st
    class ACT act
```

### Key access pattern

| Key | Written by | Read by | Operation |
|---|---|---|---|
| `ash:hb:{id}` | Worker | Supervisor | `SET` (overwrite) / `GET` |
| `ash:state:{id}` | Supervisor | Supervisor | `SET` / `GET` |
| `ash:actions` | Supervisor | Anyone (debug/UI) | `LPUSH` + `LTRIM` + `LRANGE` |

---

## 8. POC Architecture (Notebook)

What runs where in [notebooks/poc.ipynb](../notebooks/poc.ipynb). The notebook is the only "client" — workers and supervisor live as in-process threads.

```mermaid
flowchart TB
    subgraph NB[Jupyter Notebook Kernel]
        T1[Thread: agent-DataProcessor-1]
        T2[Thread: agent-DataProcessor-2]
        TS[Thread: supervisor]
        HF[HeartbeatStore<br/>in-memory wrapper]
    end

    subgraph UR[Upstash Redis]
        HB["ash:hb:*"]
        ST["ash:state:*"]
        AC["ash:actions"]
    end

    subgraph DB[Databricks MLflow]
        SP[Trace Spans<br/>is_agent_healthy<br/>check_and_resuscitate<br/>supervisor.monitor_cycle]
        MT[Metrics<br/>*_consecutive_failures<br/>*_total_restarts]
    end

    T1 -->|write_heartbeat| HF
    T2 -->|write_heartbeat| HF
    HF -->|SET/GET| UR

    TS -->|read_health /<br/>read_heartbeat| HF
    HF -->|SET/GET| UR
    TS -->|force_crash| T1
    TS -->|new DataProcessingAgent.start| T1

    TS -.->|@mlflow.trace| SP
    TS -.->|mlflow.log_metric| MT
```

---

## 9. Failure & Recovery States

A `stateDiagram-v2` of a single worker agent's lifecycle, as seen by the supervisor. The supervisor's view of an agent transitions through these states based on heartbeat freshness and restart events.

```mermaid
stateDiagram-v2
    [*] --> Starting

    Starting: Starting<br/>(process spawn, init)

    Healthy: Healthy<br/>(heartbeat fresh)
    Stale: Stale<br/>(1 missed heartbeat)
    Unhealthy: Unhealthy<br/>(≥ 2 consecutive failures)
    Restarting: Restarting<br/>(backoff elapsed, restart in progress)
    BackingOff: Backing Off<br/>(restart attempted, in cooldown)
    Dead: Permanently Dead<br/>(max_restarts exceeded — not in POC)

    Starting --> Healthy: first heartbeat seen
    Healthy --> Stale: heartbeat age > timeout
    Stale --> Healthy: heartbeat recovered
    Stale --> Unhealthy: another missed tick
    Unhealthy --> BackingOff: restart attempted
    BackingOff --> Healthy: new heartbeat within window
    BackingOff --> Unhealthy: still no heartbeat
    Unhealthy --> Dead: max_restarts exceeded
    Dead --> [*]
```

---

## Reading Order

If you're new to the codebase, read the diagrams in this order:

1. **[§1 System Context](#1-system-context)** — see the big picture
2. **[§2 5-Step Workflow](#2-the-5-step-auto-healing-workflow)** — understand the core loop
3. **[§3 Monitoring Loop](#3-supervisor-monitoring-loop)** — the supervisor's decision state machine
4. **[§4 Component Diagram](#4-component-diagram)** — the code structure
5. **[§5 Heartbeat & Restart Sequence](#5-heartbeat--restart-sequence)** — drill into one cycle
6. **[§7 Redis Key Layout](#7-redis-key-layout)** — understand persistence
7. **[§8 POC Architecture](#8-poc-architecture-notebook)** — see how the notebook ties it together
8. **[§9 Failure & Recovery States](#9-failure--recovery-states)** — worker lifecycle from supervisor's view

---

## Mermaid Rendering

These diagrams render in:

- ✅ **GitHub** — Mermaid is supported in markdown files in any repo
- ✅ **VS Code** — with the *Markdown Preview Mermaid Support* extension (or built-in if you have GitHub Copilot)
- ✅ **Databricks notebooks** — Mermaid rendering in markdown cells (since DB Runtime 11+)
- ✅ **MkDocs / Docusaurus** — with the `mermaid2` plugin
- ✅ **mkdocs-material** — with `pymdownx.superfences` + custom fence `mermaid`

For a quick local preview:

```bash
# Option 1: npx (no install)
npx -y @mermaid-js/mermaid-cli -i docs/ARCHITECTURE.md -o /tmp/preview

# Option 2: VS Code — open this file and use the Mermaid preview extension
```
