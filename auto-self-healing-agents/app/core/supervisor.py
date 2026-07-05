"""Supervisor and agent classes for the Streamlit app."""

import random
import threading
import time
from typing import Callable, Optional

from app.core.redis_store import HeartbeatStore, SupervisorAction, store


class DataProcessingAgent:
    """A long-running worker that emits heartbeats and can be crashed for demo."""
    
    def __init__(
        self,
        agent_id: str,
        heartbeat_interval: float = 2.0,
        crash_after_n_heartbeats: Optional[int] = None,
        crash_probability: float = 0.0,
    ):
        self.agent_id = agent_id
        self._interval = heartbeat_interval
        self._crash_after_n = crash_after_n_heartbeats
        self._crash_probability = crash_probability
        self._sequence = 0
        self._stop_event = threading.Event()
        self._crashed_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._crashed_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"agent-{self.agent_id}", daemon=True
        )
        self._thread.start()
    
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def crash(self) -> None:
        """Simulate a hard process crash."""
        self._crashed_event.set()
        self._stop_event.set()
    
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._crashed_event.is_set()
    
    @property
    def is_crashed(self) -> bool:
        return self._crashed_event.is_set()
    
    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sequence += 1
            
            # Random crash
            if self._crash_probability > 0 and random.random() < self._crash_probability:
                self._crashed_event.set()
                return
            
            time.sleep(self._interval)
            
            # Emit heartbeat
            hb = Heartbeat(agent_id=self.agent_id, sequence=self._sequence, alive=True)
            store.write_heartbeat(hb)
            
            # Deterministic crash
            if self._crash_after_n is not None and self._sequence >= self._crash_after_n:
                self._crashed_event.set()
                return


class Backoff:
    """Exponential backoff with a cap."""
    
    def __init__(self, base: float = 2.0, factor: float = 2.0, max_wait: float = 60.0):
        self.base = base
        self.factor = factor
        self.max_wait = max_wait
    
    def wait_seconds(self, consecutive_failures: int) -> float:
        if consecutive_failures <= 0:
            return 0.0
        wait = self.base * (self.factor ** (consecutive_failures - 1))
        return min(wait, self.max_wait)


def is_agent_healthy(agent_id: str, heartbeat_timeout: float) -> tuple[bool, Optional[Heartbeat], float]:
    """Check if an agent's last heartbeat is within the timeout window."""
    hb = store.read_heartbeat(agent_id)
    if hb is None:
        return False, None, float("inf")
    age = time.time() - hb.timestamp
    return (age <= heartbeat_timeout), hb, age


def check_and_resuscitate(
    agent_id: str,
    agent_factory: Callable[[], DataProcessingAgent],
    current_agent: Optional[DataProcessingAgent],
    heartbeat_timeout: float,
    backoff: Backoff,
) -> tuple[DataProcessingAgent, SupervisorAction]:
    """One monitoring cycle for a single agent."""
    healthy, hb, age = is_agent_healthy(agent_id, heartbeat_timeout)
    state = store.read_health(agent_id)
    
    if healthy:
        state.consecutive_failures = 0
        state.last_action = "monitored"
        state.last_action_at = time.time()
        state.last_seen = hb.timestamp if hb else state.last_seen
        store.write_health(state)
        action = SupervisorAction(
            agent_id=agent_id,
            action="healthy",
            details=f"heartbeat age={age:.1f}s",
            consecutive_failures=0,
        )
        store.log_action(action)
        return current_agent, action
    
    # Unhealthy
    state.consecutive_failures += 1
    wait = backoff.wait_seconds(state.consecutive_failures)
    time_since_last_action = time.time() - state.last_action_at if state.last_action_at else float("inf")
    
    # Respect backoff window
    if state.last_action == "restarted" and time_since_last_action < wait:
        state.last_action = "backoff"
        state.last_action_at = time.time()
        store.write_health(state)
        action = SupervisorAction(
            agent_id=agent_id,
            action="backoff_skipped",
            details=f"in backoff window ({time_since_last_action:.1f}s < {wait:.1f}s)",
            consecutive_failures=state.consecutive_failures,
            backoff_seconds=wait,
        )
        store.log_action(action)
        return current_agent, action
    
    # Resuscitate
    new_agent = agent_factory()
    new_agent.start()
    state.total_restarts += 1
    state.last_action = "restarted"
    state.last_action_at = time.time()
    state.consecutive_failures = 0
    store.write_health(state)
    action = SupervisorAction(
        agent_id=agent_id,
        action="restarted",
        details=f"agent resuscitated (was {age:.1f}s stale, attempt #{state.total_restarts})",
        consecutive_failures=state.consecutive_failures,
        backoff_seconds=wait,
    )
    store.log_action(action)
    return new_agent, action


class Supervisor:
    """External supervisor that monitors a pool of worker agents."""
    
    def __init__(
        self,
        agent_specs: list[dict],
        check_interval: float = 5.0,
        heartbeat_timeout: float = 8.0,
        backoff: Optional[Backoff] = None,
    ):
        self._specs = agent_specs
        self._check_interval = check_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._backoff = backoff or Backoff()
        self._agents: dict[str, DataProcessingAgent] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def _factory(self, spec: dict) -> DataProcessingAgent:
        return DataProcessingAgent(
            agent_id=spec["agent_id"],
            heartbeat_interval=spec.get("heartbeat_interval", 2.0),
            crash_after_n_heartbeats=spec.get("crash_after_n_heartbeats"),
            crash_probability=spec.get("crash_probability", 0.0),
        )
    
    def start(self) -> None:
        for spec in self._specs:
            agent = self._factory(spec)
            agent.start()
            self._agents[spec["agent_id"]] = agent
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, name="supervisor", daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        self._stop_event.set()
        for agent in self._agents.values():
            agent.stop()
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            for spec in self._specs:
                aid = spec["agent_id"]
                current = self._agents.get(aid)
                if current is None:
                    continue
                
                new_agent, action = check_and_resuscitate(
                    agent_id=aid,
                    agent_factory=lambda s=spec: self._factory(s),
                    current_agent=current,
                    heartbeat_timeout=self._heartbeat_timeout,
                    backoff=self._backoff,
                )
                self._agents[aid] = new_agent
            
            time.sleep(self._check_interval)
    
    def force_crash(self, agent_id: str) -> None:
        """Demo helper: programmatically crash an agent."""
        if agent_id in self._agents:
            self._agents[agent_id].crash()
    
    def state(self) -> dict:
        return {
            aid: {
                "is_alive": agent.is_alive(),
                "is_crashed": agent.is_crashed,
                "health": store.read_health(aid).model_dump(),
            }
            for aid, agent in self._agents.items()
        }