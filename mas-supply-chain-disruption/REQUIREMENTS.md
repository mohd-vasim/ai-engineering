# Functional Design Specification: Shared Epistemic Memory

**Document Type:** Functional Design Specification (FDS)  
**Version:** 1.0  
**Date:** 2026-06-27  
**Status:** Draft  

---

## 1. Purpose

This document specifies the design and implementation requirements for the **Shared Epistemic Memory (SEM)** pattern in a multi-agent system. It is intended as a reference for a coding agent to implement a production-ready SEM module from scratch.

---

## 2. Problem Statement

In a multi-agent system, each agent operates within its own context window — a private, ephemeral snapshot of the world. When one agent learns a new fact (e.g., a shipment is delayed, a server is down), that knowledge does not automatically propagate to other agents. This causes:

- **Fragmented knowledge:** Agents act on stale or incomplete information
- **Lossy communication:** Passing state through message chains loses nuance over hops
- **No ground truth:** No authoritative source to resolve conflicting agent beliefs

The result is **semantic drift** — agents in the same system end up with inconsistent world models and produce incoherent collective behavior.

---

## 3. Solution Overview

The SEM pattern introduces a **single, persistent, centralized memory store** that exists outside any individual agent's context. All agents in a workflow can:

- **READ** the current world state at any point
- **WRITE** structured updates when they observe new facts
- **TRUST** the store as the authoritative source of truth

This store acts as a global scratchpad — the canonical state of the task.

---

## 4. System Architecture

```
┌────────────────────────────────────────────────────────┐
│                   Agent Collective                     │
│                                                        │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────┐  │
│  │ Monitoring   │   │  Logistics   │   │ Customer  │  │
│  │   Agent      │   │    Agent     │   │  Notif.   │  │
│  └──────┬───────┘   └──────┬───────┘   └─────┬─────┘  │
│         │ WRITE            │ READ/WRITE       │ READ   │
│         └──────────────────┼──────────────────┘        │
│                            ▼                           │
│              ┌─────────────────────────┐               │
│              │   Shared Epistemic      │               │
│              │   Memory (Redis/        │               │
│              │   Memcached)            │               │
│              │   - Source of Truth     │               │
│              │   - Typed Schema        │               │
│              │   - TTL per entry       │               │
│              └─────────────────────────┘               │
└────────────────────────────────────────────────────────┘
```

---

## 5. Functional Requirements

### 5.1 Core Operations

| Operation | Description |
|-----------|-------------|
| `read(key)` | Retrieve current value for a key. Returns `None` if key does not exist. |
| `write(key, value)` | Write a typed value to a key. Must be atomic. |
| `update(key, partial)` | Merge a partial update into an existing record. |
| `delete(key)` | Remove a key from the store. |
| `list_keys(prefix)` | List all keys matching a given prefix. |
| `get_metadata(key)` | Return the entry's `source_agent_id`, `timestamp`, and `ttl`. |

### 5.2 Schema Enforcement

Every memory entry **must** be a typed, structured object — not free-form text. The store must reject writes that do not conform to a registered schema.

```python
# Example: shipment schema
class ShipmentStatus(BaseModel):
    shipment_id: str
    status: Literal["On Time", "Delayed", "Cancelled"]
    reason: Optional[str] = None
    source_agent_id: str
    timestamp: datetime
    ttl_seconds: int = 300
```

### 5.3 TTL and Staleness Tracking

Every entry must carry:

- `timestamp` — ISO 8601 UTC datetime of when the fact was written
- `ttl_seconds` — how long this fact is considered fresh
- `source_agent_id` — which agent wrote this fact

Downstream agents must check staleness before acting:

```python
def is_fresh(entry: MemoryEntry) -> bool:
    age = (datetime.utcnow() - entry.timestamp).total_seconds()
    return age < entry.ttl_seconds
```

If an entry is stale, the consuming agent should log a warning and optionally re-verify before acting.

### 5.4 Concurrency and Atomicity

Multiple agents may write simultaneously. The implementation must:

- Use **atomic operations** (Redis `SET NX`, Lua scripts, or transactions)
- Implement **optimistic locking** via versioning on critical keys
- Never allow partial writes to be visible

```python
# Redis atomic write with version check
def atomic_update(redis_client, key, new_value, expected_version):
    with redis_client.pipeline() as pipe:
        while True:
            try:
                pipe.watch(key)
                current = pipe.get(key)
                current_data = json.loads(current) if current else {}
                if current_data.get("version", 0) != expected_version:
                    raise VersionConflictError(key)
                pipe.multi()
                new_value["version"] = expected_version + 1
                pipe.set(key, json.dumps(new_value))
                pipe.execute()
                break
            except redis.WatchError:
                continue
```

---

## 6. Non-Functional Requirements

| Requirement | Specification |
|-------------|---------------|
| **Latency** | Read/write operations must complete in < 10ms (p95) |
| **Persistence** | Store must survive process restarts (Redis AOF or RDB persistence enabled) |
| **Availability** | Use Redis Sentinel or Cluster for HA; store must not be a single point of failure |
| **Throughput** | Must handle concurrent reads/writes from N agents without degradation |
| **Observability** | Every read and write must emit a structured log entry with key, agent_id, timestamp |

---

## 7. Implementation Specification

### 7.1 Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Backing store | Redis 7+ | Atomic ops, persistence, pub/sub, low latency |
| Schema validation | Pydantic v2 | Type safety, serialization, clear error messages |
| Serialization | JSON | Human-readable, debuggable |
| Python client | `redis-py` | Standard, async support |

### 7.2 Module Structure

```
shared_epistemic_memory/
├── __init__.py
├── store.py            # Core SEM class — read/write/delete/list
├── schemas.py          # Pydantic models for all memory entry types
├── exceptions.py       # VersionConflictError, SchemaValidationError, StaleEntryWarning
├── middleware.py       # TTL checker, staleness logger
├── tools.py            # Typed tool wrappers exposed to agents
└── tests/
    ├── test_store.py
    ├── test_schemas.py
    └── test_concurrency.py
```

### 7.3 Core `SharedEpistemicMemory` Class

```python
import json
import redis
from datetime import datetime
from typing import Any, Optional, Type
from pydantic import BaseModel

class SharedEpistemicMemory:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._schema_registry: dict[str, Type[BaseModel]] = {}

    def register_schema(self, key_prefix: str, schema: Type[BaseModel]):
        self._schema_registry[key_prefix] = schema

    def _resolve_schema(self, key: str) -> Optional[Type[BaseModel]]:
        for prefix, schema in self._schema_registry.items():
            if key.startswith(prefix):
                return schema
        return None

    def write(self, key: str, value: dict, source_agent_id: str, ttl_seconds: int = 300):
        schema = self._resolve_schema(key)
        if schema:
            validated = schema(**value, source_agent_id=source_agent_id,
                               timestamp=datetime.utcnow(), ttl_seconds=ttl_seconds)
            payload = validated.model_dump_json()
        else:
            raise ValueError(f"No schema registered for key: {key}")

        self.client.setex(key, ttl_seconds, payload)
        self._log("WRITE", key, source_agent_id)

    def read(self, key: str) -> Optional[dict]:
        raw = self.client.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        self._log("READ", key, agent_id="consumer")
        return data

    def delete(self, key: str):
        self.client.delete(key)

    def list_keys(self, prefix: str) -> list[str]:
        return self.client.keys(f"{prefix}*")

    def _log(self, operation: str, key: str, agent_id: str):
        print(json.dumps({
            "operation": operation,
            "key": key,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }))
```

### 7.4 Typed Agent Tools

Agents must never call `write(key, raw_text)`. Instead, expose domain-specific typed tools:

```python
def update_shipment_status(
    memory: SharedEpistemicMemory,
    shipment_id: str,
    status: str,
    reason: Optional[str],
    agent_id: str
):
    memory.write(
        key=f"shipments:{shipment_id}",
        value={"shipment_id": shipment_id, "status": status, "reason": reason},
        source_agent_id=agent_id,
        ttl_seconds=600
    )

def log_event(
    memory: SharedEpistemicMemory,
    event: str,
    agent_id: str
):
    existing = memory.read("events:log") or {"events": []}
    existing["events"].append({"event": event, "timestamp": datetime.utcnow().isoformat()})
    memory.write("events:log", existing, source_agent_id=agent_id, ttl_seconds=3600)
```

### 7.5 Agent Integration Pattern

```python
class LogisticsAgent:
    def run(self, memory: SharedEpistemicMemory):
        events = memory.read("events:log")
        if not events:
            return

        for event in events["events"]:
            if "Storm" in event["event"]:
                update_shipment_status(
                    memory,
                    shipment_id="shipment_B2",
                    status="Delayed",
                    reason=event["event"],
                    agent_id="LogisticsAgent"
                )

class CustomerNotificationAgent:
    def run(self, memory: SharedEpistemicMemory):
        shipment = memory.read("shipments:shipment_B2")
        if shipment and shipment["status"] == "Delayed":
            self._notify_customer(shipment["shipment_id"], shipment["reason"])

    def _notify_customer(self, shipment_id: str, reason: str):
        print(f"[NOTIFY] Shipment {shipment_id} delayed: {reason}")
```

---

## 8. Error Handling

| Error | Cause | Handling |
|-------|-------|----------|
| `SchemaValidationError` | Write payload does not match registered schema | Reject write, log error, agent retries with corrected payload |
| `VersionConflictError` | Concurrent write collision on same key | Retry with exponential backoff (max 3 attempts) |
| `StaleEntryWarning` | Entry TTL exceeded at read time | Log warning, agent must re-verify or skip action |
| `ConnectionError` | Redis unavailable | Raise immediately; orchestrator handles failover |
| `KeyNotFoundError` | Read on non-existent key | Return `None`; agent treats as "unknown state" |

---

## 9. Testing Requirements

### 9.1 Unit Tests

- Schema validation rejects malformed payloads
- TTL expiry correctly marks entries stale
- `list_keys(prefix)` returns only matching keys
- `delete` removes key and subsequent reads return `None`

### 9.2 Concurrency Tests

- Two agents writing to the same key simultaneously → only one write wins, no corruption
- Optimistic locking raises `VersionConflictError` on stale version

### 9.3 Integration Tests

- Full supply chain workflow: `MonitoringAgent` → write event → `LogisticsAgent` → read/write shipment → `CustomerNotificationAgent` → read and notify
- Verify notification is triggered exactly once per event

---

## 10. Observability Checklist

Every deployment must have:

- [ ] Structured JSON logs for every read/write/delete operation
- [ ] `source_agent_id` present on every log entry
- [ ] Redis key count and memory usage monitored via metrics
- [ ] Alert on TTL expiry rate exceeding threshold (signals facts are "rotting" too fast)
- [ ] Alert on `VersionConflictError` rate (signals contention between agents)

---

## 11. Open Questions / Decisions Needed

| # | Question | Impact |
|---|----------|--------|
| 1 | Should the store support pub/sub for reactive agents (agent gets notified on write)? | High — changes agent loop design |
| 2 | Should TTL be per-schema or per-entry? | Medium — per-entry is more flexible but harder to enforce |
| 3 | Should stale entries be readable or blocked? | Medium — blocking is safer but breaks agents that need historical data |
| 4 | Is Redis Cluster needed or is Sentinel sufficient for HA? | High — depends on write throughput requirements |

---

## 12. References

- *Agentic Architectural Patterns for Building Multi-Agent Systems*, Chapter 6, pp. 183–186
- Wegner, D. (1987). Transactive Memory: A Contemporary Analysis of the Group Mind
- Redis documentation: Atomic operations, Lua scripting, AOF persistence