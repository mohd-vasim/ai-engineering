"""Session state management for Streamlit app."""

import streamlit as st
from typing import Optional


def init_session_state():
    """Initialize all session state variables."""
    
    # Supervisor state
    if "supervisor" not in st.session_state:
        st.session_state.supervisor = None
    
    if "supervisor_running" not in st.session_state:
        st.session_state.supervisor_running = False
    
    # Agent configs
    if "agent_configs" not in st.session_state:
        st.session_state.agent_configs = [
            {"agent_id": "DataProcessor-1", "heartbeat_interval": 2.0, "crash_after_n_heartbeats": None, "crash_probability": 0.0},
            {"agent_id": "DataProcessor-2", "heartbeat_interval": 2.0, "crash_after_n_heartbeats": None, "crash_probability": 0.0},
        ]
    
    # Supervisor settings
    if "check_interval" not in st.session_state:
        st.session_state.check_interval = 5.0
    
    if "heartbeat_timeout" not in st.session_state:
        st.session_state.heartbeat_timeout = 8.0
    
    if "backoff_base" not in st.session_state:
        st.session_state.backoff_base = 2.0
    
    if "backoff_factor" not in st.session_state:
        st.session_state.backoff_factor = 2.0
    
    if "backoff_max" not in st.session_state:
        st.session_state.backoff_max = 60.0
    
    # Metrics history for charts
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = {
            "timestamps": [],
            "agents": {},
        }
    
    # Action log
    if "action_log" not in st.session_state:
        st.session_state.action_log = []


def reset_session():
    """Reset session state for a fresh start."""
    st.session_state.supervisor = None
    st.session_state.supervisor_running = False
    st.session_state.metrics_history = {
        "timestamps": [],
        "agents": {},
    }
    st.session_state.action_log = []