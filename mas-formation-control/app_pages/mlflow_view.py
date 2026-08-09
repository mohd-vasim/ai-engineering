"""
Databricks MLflow Observability & Trace Telemetry Inspector.
Displays experiment tracking parameters, autologging configuration, and payload optimization details.
"""
import os
import streamlit as st

st.header("📊 Databricks MLflow observability & traces", divider="blue")

# --- Environment & Connection Status ---
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "databricks")
experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID", "3192447675404693")
databricks_host = os.environ.get("DATABRICKS_HOST", "")
has_token = bool(os.environ.get("DATABRICKS_TOKEN", ""))

with st.container(border=True):
    st.subheader("MLflow experiment connection")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.metric("Tracking URI", tracking_uri, border=True)
    with col_e2:
        st.metric("Experiment ID", experiment_id, border=True)
    with col_e3:
        st.metric("Authentication", "Databricks Token Configured" if has_token else "Local Fallback", border=True)

    if databricks_host:
        st.caption(f"Host: `{databricks_host}`")

# --- MLflow Autologging Architecture ---
with st.container(border=True):
    st.subheader("⚡ LangChain & LangGraph Autologging")
    st.markdown("""
    All mission invocations, LLM reasoning spans, structured outputs, and tool calls are automatically traced via:
    ```python
    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id="3192447675404693")
    mlflow.langchain.autolog()
    ```
    """)

    st.markdown("""
    **What is captured in each MLflow run:**
    1. **LLM Spans:** Gemini 3.5 Flash prompt, temperature, token usage, latency.
    2. **Structured Outputs:** Pydantic validation for `SwarmDispatchPlan` and `FDSVerificationVerdict`.
    3. **Tool Invocations:** `run_swarm_simulation_tool` and `analyze_telemetry_metrics_tool`.
    4. **Artifacts & Plots:** Generated spatial trajectory plots saved to `docs/swarm_trajectory_telemetry.png`.
    """)

# --- Telemetry Payload Optimization (O(N^2) vs O(N)) ---
with st.container(border=True):
    st.subheader("🚀 Telemetry Payload Optimization")
    st.markdown("""
    To eliminate verbose JSON blob logs in the MLflow UI, we redesigned the step logging payload:
    """)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.error("❌ **Previous Scheme: $O(N^2)$ Growing Blobs**")
        st.code("""
# At step 159, logged 160 previous points
# for every drone:
{
  "step": 159,
  "trajectories": {
    "Drone_A": [[...160 points...]],
    "Drone_B": [[...160 points...]]
  }
}
# Result: ~1.6MB nested string in MLflow
        """, language="json")

    with col_p2:
        st.success("✅ **Optimized Scheme: $O(N)$ Compact Snapshots**")
        st.code("""
# Stores ONLY current instantaneous position:
{
  "step": 159,
  "mean_error": 0.041,
  "min_clearance": 9.982,
  "positions": {
    "Drone_A": [20.0, 50.0],
    "Drone_B": [10.0, 40.0]
  }
}
# Result: Lightweight, readable MLflow traces
        """, language="json")

# --- Databricks Navigation Guide ---
with st.container(border=True):
    st.subheader("🔍 Inspecting Traces in Databricks Workspace")
    st.markdown("""
    1. Open your Databricks Workspace URL.
    2. Navigate to **Experiments** in the left navigation sidebar.
    3. Select Experiment ID **`3192447675404693`**.
    4. Open the **Traces** tab to view the live execution tree, LangGraph node execution order, and latency breakdowns.
    """)
