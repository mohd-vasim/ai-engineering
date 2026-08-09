"""
Mission Telemetry History & SQLite Database Records Viewer.
Queries data/telemetry.db and provides comparative analytics across historical swarm runs.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from core.db import get_all_mission_runs, get_mission_stats

st.header("🗄️ Mission telemetry history & persistent database", divider="violet")

# Fetch all runs
df = get_all_mission_runs()
stats = get_mission_stats()

if df.empty:
    st.info("No mission records found in `data/telemetry.db`. Launch missions from the Mission Control page to record telemetry.")
    st.stop()

# --- Top Stats KPI Row ---
with st.container(horizontal=True):
    st.metric("Total recorded missions", f"{stats['total_runs']}", border=True)
    st.metric("Overall success rate", f"{stats['success_rate']:.1f}%", border=True)
    st.metric("Avg max deviation", f"{stats['avg_max_dev']:.3f} m", border=True)
    st.metric("Avg final error", f"{stats['avg_final_err']:.3f} m", border=True)
    st.metric("Avg min clearance", f"{stats['avg_min_clear']:.3f} m", border=True)

# --- Filters ---
with st.container(border=True):
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        formations = ["All"] + list(df["formation"].dropna().unique())
        selected_formation = st.selectbox("Filter by formation", formations, index=0)
    with col_f2:
        status_options = ["All", "Passed Only", "Failed Only"]
        selected_status = st.selectbox("Filter by compliance status", status_options, index=0)
    with col_f3:
        min_drones, max_drones = int(df["num_drones"].min()), int(df["num_drones"].max())
        drone_range = st.slider("Filter by drone count", min_value=3, max_value=8, value=(min_drones, max_drones))

# Apply filters
filtered_df = df.copy()
if selected_formation != "All":
    filtered_df = filtered_df[filtered_df["formation"] == selected_formation]
if selected_status == "Passed Only":
    filtered_df = filtered_df[filtered_df["mission_ok"] == 1]
elif selected_status == "Failed Only":
    filtered_df = filtered_df[filtered_df["mission_ok"] == 0]

filtered_df = filtered_df[
    (filtered_df["num_drones"] >= drone_range[0]) & 
    (filtered_df["num_drones"] <= drone_range[1])
]

# Display Table
st.subheader(f"Mission Records ({len(filtered_df)} runs matching filter)")

display_df = filtered_df.copy()
display_df["status_badge"] = display_df["mission_ok"].apply(lambda x: "✅ PASS" if x == 1 else "❌ FAIL")
display_df = display_df[[
    "status_badge", "run_id", "timestamp", "num_drones", "formation",
    "spacing_m", "max_dev_m", "final_err_m", "min_clear_m", "summary"
]].rename(columns={
    "status_badge": "Verdict",
    "run_id": "Run ID",
    "timestamp": "Timestamp (UTC)",
    "num_drones": "Drones",
    "formation": "Formation",
    "spacing_m": "Spacing (m)",
    "max_dev_m": "Max Dev (m)",
    "final_err_m": "Final Err (m)",
    "min_clear_m": "Min Clear (m)",
    "summary": "Executive Summary",
})

st.dataframe(
    display_df,
    hide_index=True,
    column_config={
        "Verdict": st.column_config.TextColumn(width="small"),
        "Run ID": st.column_config.TextColumn(width="small"),
        "Timestamp (UTC)": st.column_config.TextColumn(width="medium"),
        "Max Dev (m)": st.column_config.NumberColumn(format="%.3f m"),
        "Final Err (m)": st.column_config.NumberColumn(format="%.3f m"),
        "Min Clear (m)": st.column_config.NumberColumn(format="%.3f m"),
        "Spacing (m)": st.column_config.NumberColumn(format="%.1f m"),
    }
)

# Download CSV
csv_data = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Export filtered mission records to CSV",
    data=csv_data,
    file_name="mas_swarm_mission_telemetry.csv",
    mime="text/csv",
)

# --- Comparative Analytics ---
st.subheader("Comparative telemetry analytics")
col_c1, col_c2 = st.columns(2, border=True)

with col_c1:
    st.markdown("**Max Deviation by Formation Type**")
    if not filtered_df.empty:
        fig_bar = px.box(
            filtered_df,
            x="formation",
            y="max_dev_m",
            color="formation",
            points="all",
            template="plotly_dark",
            labels={"formation": "Formation Type", "max_dev_m": "Max Deviation (m)"},
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            height=280,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, config={"displayModeBar": False})

with col_c2:
    st.markdown("**Inter-Agent Clearance vs Drone Count**")
    if not filtered_df.empty:
        fig_scatter = px.scatter(
            filtered_df,
            x="num_drones",
            y="min_clear_m",
            color="formation",
            size="spacing_m",
            hover_data=["run_id", "final_err_m"],
            template="plotly_dark",
            labels={"num_drones": "Number of Drones", "min_clear_m": "Min Clearance (m)"},
        )
        fig_scatter.add_hline(y=3.0, line_dash="dot", line_color="#34D399", annotation_text="Safety Limit 3.0m")
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            height=280,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_scatter, config={"displayModeBar": False})
