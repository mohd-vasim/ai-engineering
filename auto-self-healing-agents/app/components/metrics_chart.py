"""Metrics chart component."""

import streamlit as st
import pandas as pd


def render_metrics_chart(metrics_history: dict):
    """Render metrics over time as line charts."""
    
    if not metrics_history.get("timestamps"):
        st.info("Collecting metrics... Start the supervisor to see charts.")
        return
    
    timestamps = metrics_history["timestamps"]
    agents = metrics_history.get("agents", {})
    
    if not agents:
        return
    
    # Create DataFrame for restarts
    restart_data = {"Timestamp": timestamps}
    failure_data = {"Timestamp": timestamps}
    
    for agent_id, data in agents.items():
        restart_data[agent_id] = data.get("restarts", [])
        failure_data[agent_id] = data.get("failures", [])
    
    # Restarts chart
    st.subheader("📊 Total Restarts Over Time")
    df_restarts = pd.DataFrame(restart_data)
    st.line_chart(df_restarts.set_index("Timestamp"), use_container_width=True)
    
    # Failures chart
    st.subheader("⚠️ Consecutive Failures Over Time")
    df_failures = pd.DataFrame(failure_data)
    st.line_chart(df_failures.set_index("Timestamp"), use_container_width=True)


def update_metrics_history(supervisor_state: dict):
    """Update the metrics history with current supervisor state."""
    import time
    
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = {
            "timestamps": [],
            "agents": {},
        }
    
    # Add current timestamp
    st.session_state.metrics_history["timestamps"].append(time.strftime("%H:%M:%S"))
    
    # Keep only last 50 data points
    if len(st.session_state.metrics_history["timestamps"]) > 50:
        st.session_state.metrics_history["timestamps"] = st.session_state.metrics_history["timestamps"][-50:]
    
    # Update each agent's metrics
    for agent_id, state in supervisor_state.items():
        if agent_id not in st.session_state.metrics_history["agents"]:
            st.session_state.metrics_history["agents"][agent_id] = {
                "restarts": [],
                "failures": [],
            }
        
        health = state.get("health", {})
        st.session_state.metrics_history["agents"][agent_id]["restarts"].append(health.get("total_restarts", 0))
        st.session_state.metrics_history["agents"][agent_id]["failures"].append(health.get("consecutive_failures", 0))
        
        # Trim to last 50
        st.session_state.metrics_history["agents"][agent_id]["restarts"] = st.session_state.metrics_history["agents"][agent_id]["restarts"][-50:]
        st.session_state.metrics_history["agents"][agent_id]["failures"] = st.session_state.metrics_history["agents"][agent_id]["failures"][-50:]