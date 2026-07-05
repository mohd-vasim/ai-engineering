"""Action log table component."""

import streamlit as st
import pandas as pd


def render_action_table(actions: list, filter_action: str = "All"):
    """Render the action log as a filterable table."""
    
    if not actions:
        st.info("No actions recorded yet. Start the supervisor to see activity.")
        return
    
    # Convert to DataFrame
    data = []
    for action in actions:
        import time
        timestamp = time.strftime("%H:%M:%S", time.localtime(action.timestamp))
        data.append({
            "Time": timestamp,
            "Agent": action.agent_id,
            "Action": action.action,
            "Failures": action.consecutive_failures,
            "Backoff": f"{action.backoff_seconds:.1f}s" if action.backoff_seconds > 0 else "-",
            "Details": action.details,
        })
    
    df = pd.DataFrame(data)
    
    # Filter by action type
    if filter_action != "All":
        df = df[df["Action"] == filter_action.lower()]
    
    # Style the Action column
    def color_action(action):
        if action == "healthy":
            return "🟢 healthy"
        elif action == "restarted":
            return "🔄 restarted"
        elif action == "backoff_skipped":
            return "⏳ backoff_skipped"
        else:
            return action
    
    df["Action"] = df["Action"].apply(color_action)
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )