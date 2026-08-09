"""
Main entry point for Multi-Agent Formation Control Streamlit Application.
Sets up navigation, theme defaults, and global session state.
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env", override=True)

# Configure page metadata
st.set_page_config(
    page_title="Multi-Agent Swarm Formation Control",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize global shared session state
if "latest_mission" not in st.session_state:
    from core.graph import build_and_run_full_mission
    # Run an initial default simulation on first launch so the dashboard is pre-populated
    st.session_state.latest_mission = build_and_run_full_mission(
        mission_brief="Conduct precision survey of 100m x 80m agricultural field with tree hazard at (20, 40)."
    )

if "selected_sim_step" not in st.session_state:
    st.session_state.selected_sim_step = 0

# Define multi-page navigation using modern st.navigation
pages = {
    "Flight Operations": [
        st.Page("app_pages/mission_control.py", title="Mission control & simulation", icon=":material/flight_takeoff:"),
        st.Page("app_pages/fds_compliance.py", title="FDS verification matrix", icon=":material/verified_user:"),
    ],
    "Analytics & Records": [
        st.Page("app_pages/telemetry_history.py", title="Mission telemetry history", icon=":material/database:"),
        st.Page("app_pages/mlflow_view.py", title="MLflow observability", icon=":material/analytics:"),
    ],
    "System Architecture": [
        st.Page("app_pages/architecture.py", title="Control laws & architecture", icon=":material/account_tree:"),
    ],
}

selected_page = st.navigation(pages, position="sidebar")
selected_page.run()
