"""Auto-Self-Healing Agents Streamlit App.

A visual demonstration of the Agent Resuscitation pattern where an external supervisor
monitors worker agents via heartbeats, detects crashes, and automatically restarts them
with exponential backoff.
"""

import time

import streamlit as st

from app.core.session import init_session_state, reset_session
from app.core.redis_store import store
from app.core.supervisor import Backoff, Supervisor
from app.components.agent_card import render_agent_card
from app.components.action_table import render_action_table
from app.components.metrics_chart import render_metrics_chart, update_metrics_history


# Page config
st.set_page_config(
    page_title="Auto-Self-Healing Agents",
    page_icon="🔄",
    layout="wide",
)

# Initialize session state
init_session_state()


def start_supervisor():
    """Start the supervisor with current settings."""
    if st.session_state.supervisor_running:
        return
    
    # Clear previous state
    store.clear_all()
    reset_session()
    
    # Create backoff
    backoff = Backoff(
        base=st.session_state.backoff_base,
        factor=st.session_state.backoff_factor,
        max_wait=st.session_state.backoff_max,
    )
    
    # Create and start supervisor
    supervisor = Supervisor(
        agent_specs=st.session_state.agent_configs,
        check_interval=st.session_state.check_interval,
        heartbeat_timeout=st.session_state.heartbeat_timeout,
        backoff=backoff,
    )
    supervisor.start()
    
    st.session_state.supervisor = supervisor
    st.session_state.supervisor_running = True


def stop_supervisor():
    """Stop the supervisor."""
    if st.session_state.supervisor:
        st.session_state.supervisor.stop()
    st.session_state.supervisor_running = False


def crash_agent(agent_id: str):
    """Force crash an agent."""
    if st.session_state.supervisor:
        st.session_state.supervisor.force_crash(agent_id)


# Sidebar
with st.sidebar:
    st.title("🔧 Configuration")
    
    # Agent configs
    st.subheader("Agent Configuration")
    
    for i, config in enumerate(st.session_state.agent_configs):
        with st.expander(f"Agent: {config['agent_id']}", expanded=True):
            new_id = st.text_input(
                "Agent ID",
                value=config["agent_id"],
                key=f"agent_id_{i}",
            )
            interval = st.number_input(
                "Heartbeat Interval (s)",
                min_value=0.5,
                max_value=10.0,
                value=config["heartbeat_interval"],
                step=0.5,
                key=f"interval_{i}",
            )
            crash_after = st.number_input(
                "Crash After N Heartbeats (0=disabled)",
                min_value=0,
                max_value=100,
                value=config.get("crash_after_n_heartbeats") or 0,
                step=1,
                key=f"crash_after_{i}",
            )
            crash_prob = st.slider(
                "Random Crash Probability",
                min_value=0.0,
                max_value=1.0,
                value=config["crash_probability"],
                step=0.05,
                key=f"crash_prob_{i}",
            )
            
            # Update config
            st.session_state.agent_configs[i]["agent_id"] = new_id
            st.session_state.agent_configs[i]["heartbeat_interval"] = interval
            st.session_state.agent_configs[i]["crash_after_n_heartbeats"] = crash_after if crash_after > 0 else None
            st.session_state.agent_configs[i]["crash_probability"] = crash_prob
    
    st.divider()
    
    # Supervisor settings
    st.subheader("Supervisor Settings")
    
    st.session_state.check_interval = st.slider(
        "Check Interval (s)",
        min_value=1.0,
        max_value=30.0,
        value=st.session_state.check_interval,
        step=1.0,
    )
    
    st.session_state.heartbeat_timeout = st.slider(
        "Heartbeat Timeout (s)",
        min_value=2.0,
        max_value=60.0,
        value=st.session_state.heartbeat_timeout,
        step=2.0,
    )
    
    st.divider()
    
    # Backoff settings
    st.subheader("Backoff Settings")
    
    st.session_state.backoff_base = st.number_input(
        "Base Wait (s)",
        min_value=1.0,
        max_value=10.0,
        value=st.session_state.backoff_base,
        step=1.0,
    )
    
    st.session_state.backoff_factor = st.number_input(
        "Backoff Factor",
        min_value=1.0,
        max_value=5.0,
        value=st.session_state.backoff_factor,
        step=0.5,
    )
    
    st.session_state.backoff_max = st.number_input(
        "Max Wait (s)",
        min_value=10.0,
        max_value=300.0,
        value=st.session_state.backoff_max,
        step=10.0,
    )
    
    st.divider()
    
    # Control buttons
    st.subheader("Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", type="primary", use_container_width=True, disabled=st.session_state.supervisor_running):
            start_supervisor()
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop", type="secondary", use_container_width=True, disabled=not st.session_state.supervisor_running):
            stop_supervisor()
            st.rerun()
    
    if st.button("🔄 Reset", use_container_width=True):
        stop_supervisor()
        reset_session()
        store.clear_all()
        st.rerun()
    
    # Backoff calculator
    st.divider()
    st.subheader("Backoff Calculator")
    
    failures = st.slider("Consecutive Failures", 1, 10, 3)
    backoff = Backoff(
        base=st.session_state.backoff_base,
        factor=st.session_state.backoff_factor,
        max_wait=st.session_state.backoff_max,
    )
    wait = backoff.wait_seconds(failures)
    st.info(f"Wait time: **{wait:.1f}s**")


# Main content
st.title("🔄 Auto-Self-Healing Agents")
st.markdown("""
This demo shows the **Agent Resuscitation Pattern** where an external supervisor monitors 
worker agents via heartbeats, detects crashes when heartbeats are missed, and automatically 
restarts agents with exponential backoff to prevent thrashing.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📋 Action Log", "📈 Backoff Visualizer"])

with tab1:
    st.subheader("Agent Status")
    
    if not st.session_state.supervisor_running:
        st.warning("👆 Start the supervisor from the sidebar to begin monitoring.")
    else:
        # Get current state
        supervisor_state = st.session_state.supervisor.state() if st.session_state.supervisor else {}
        
        # Update metrics
        update_metrics_history(supervisor_state)
        
        # Render agent cards
        cols = st.columns(len(supervisor_state) if supervisor_state else 2)
        for i, (agent_id, state) in enumerate(supervisor_state.items()):
            with cols[i % len(cols)]:
                render_agent_card(agent_id, state, on_crash_callback=crash_agent)
        
        # Render metrics chart
        st.divider()
        render_metrics_chart(st.session_state.metrics_history)

with tab2:
    st.subheader("Supervisor Action Log")
    
    filter_action = st.selectbox(
        "Filter by action",
        ["All", "healthy", "restarted", "backoff_skipped"],
    )
    
    actions = st.session_state.get("action_log", [])
    render_action_table(actions, filter_action)

with tab3:
    st.subheader("Exponential Backoff Visualizer")
    
    # Show backoff curve
    import pandas as pd
    
    max_failures = 10
    backoff = Backoff(
        base=st.session_state.backoff_base,
        factor=st.session_state.backoff_factor,
        max_wait=st.session_state.backoff_max,
    )
    
    data = {
        "Failures": list(range(1, max_failures + 1)),
        "Wait Time (s)": [backoff.wait_seconds(i) for i in range(1, max_failures + 1)],
    }
    df = pd.DataFrame(data)
    
    st.line_chart(df.set_index("Failures"), use_container_width=True)
    
    st.markdown("""
    **How it works:**
    - After each missed heartbeat, the supervisor waits `base * factor^(failures-1)` seconds
    - This prevents rapid restart cycles when an agent keeps failing
    - The wait time is capped at `max_wait` to avoid indefinite delays
    """)

# Auto-refresh when supervisor is running
if st.session_state.supervisor_running:
    time.sleep(0.5)
    st.rerun()