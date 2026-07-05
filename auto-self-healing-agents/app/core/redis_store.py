"""Redis-backed heartbeat store for the Streamlit app."""

import time
import uuid
from typing import Literal, Optional

import streamlit as st
from pydantic import BaseModel, Field


class Heartbeat(BaseModel):
    """A single heartbeat emitted by a worker agent."""
    agent_id: str
    timestamp: float = Field(default_factory=time.time)
    alive: bool = True
    sequence: int = 0


class HealthState(BaseModel):
    """The supervisor's tracked state for a single agent."""
    agent_id: str
    last_seen: float = 0.0
    consecutive_failures: int = 0
    total_restarts: int = 0
    last_action: Literal["none", "monitored", "restarted", "backoff"] = "none"
    last_action_at: float = 0.0


class SupervisorAction(BaseModel):
    """A record of one decision the supervisor made."""
    action_id: str = Field(default_factory=lambda: f"act-{uuid.uuid4().hex[:8]}")
    timestamp: float = Field(default_factory=time.time)
    agent_id: str
    action: Literal["healthy", "missed_heartbeat", "restarted", "backoff_skipped", "backoff_respected"]
    details: str = ""
    consecutive_failures: int = 0
    backoff_seconds: float = 0.0


class HeartbeatStore:
    """In-memory heartbeat store for Streamlit (no Redis dependency for demo)."""
    
    def __init__(self):
        self._heartbeats: dict[str, Heartbeat] = {}
        self._health_states: dict[str, HealthState] = {}
        self._actions: list[SupervisorAction] = []
        self._sequences: dict[str, int] = {}
    
    def clear_all(self):
        """Clear all stored data."""
        self._heartbeats.clear()
        self._health_states.clear()
        self._actions.clear()
        self._sequences.clear()
    
    def write_heartbeat(self, hb: Heartbeat) -> None:
        self._heartbeats[hb.agent_id] = hb
        self._sequences[hb.agent_id] = hb.sequence
    
    def read_heartbeat(self, agent_id: str) -> Optional[Heartbeat]:
        return self._heartbeats.get(agent_id)
    
    def read_health(self, agent_id: str) -> HealthState:
        if agent_id not in self._health_states:
            self._health_states[agent_id] = HealthState(agent_id=agent_id)
        return self._health_states[agent_id]
    
    def write_health(self, state: HealthState) -> None:
        self._health_states[state.agent_id] = state
    
    def log_action(self, action: SupervisorAction) -> None:
        self._actions.insert(0, action)
        # Keep only last 100 actions
        self._actions = self._actions[:100]
        # Also add to session state for Streamlit
        if "action_log" in st.session_state:
            st.session_state.action_log.insert(0, action)
            st.session_state.action_log = st.session_state.action_log[:100]
    
    def recent_actions(self, n: int = 20) -> list[SupervisorAction]:
        return self._actions[:n]
    
    def get_all_agents(self) -> list[str]:
        return list(self._heartbeats.keys())


# Global store instance
store = HeartbeatStore()