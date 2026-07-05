"""Agent status card component."""

import streamlit as st


def render_agent_card(agent_id: str, state: dict, on_crash_callback=None):
    """Render a single agent status card."""
    
    is_alive = state.get("is_alive", False)
    is_crashed = state.get("is_crashed", False)
    health = state.get("health", {})
    
    # Determine status
    if is_crashed:
        status = "crashed"
        status_color = "🔴"
    elif is_alive:
        status = "healthy"
        status_color = "🟢"
    else:
        status = "unknown"
        status_color = "🟡"
    
    # Calculate heartbeat age
    hb_age = ""
    if health.get("last_seen", 0) > 0:
        import time
        age = time.time() - health["last_seen"]
        hb_age = f"{age:.1f}s ago"
    
    # Build card content
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {status_color} {agent_id}")
            st.markdown(f"**Status:** {status}")
            st.markdown(f"**Last heartbeat:** {hb_age}")
            st.markdown(f"**Total restarts:** {health.get('total_restarts', 0)}")
            st.markdown(f"**Consecutive failures:** {health.get('consecutive_failures', 0)}")
            st.markdown(f"**Last action:** {health.get('last_action', 'none')}")
        
        with col2:
            if on_crash_callback and st.button("💥 Crash", key=f"crash_{agent_id}"):
                on_crash_callback(agent_id)
        
        st.divider()


def render_agent_metrics(state: dict) -> dict:
    """Extract metrics from agent state for charting."""
    health = state.get("health", {})
    return {
        "restarts": health.get("total_restarts", 0),
        "failures": health.get("consecutive_failures", 0),
        "is_alive": state.get("is_alive", False),
    }