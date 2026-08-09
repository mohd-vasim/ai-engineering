"""
FDS Verification Matrix & Formal Specification Compliance Inspector.
Evaluates all 5 stages of the Formation Control pattern from the book specification.
"""
import streamlit as st
import plotly.graph_objects as go

st.header("📋 FDS verification & compliance matrix", divider="green")

mission = st.session_state.get("latest_mission")
if not mission:
    st.warning("No mission telemetry found. Please launch a mission from the Mission Control page first.")
    st.stop()

plan = mission["plan"]
metrics = mission["metrics"]
verdict = mission["verdict"]

# --- Overall Verdict Banner ---
is_overall_ok = verdict.get("overall_mission_success", False)

with st.container(border=True):
    col_verdict_icon, col_verdict_text = st.columns([1, 4])
    with col_verdict_icon:
        if is_overall_ok:
            st.markdown("### 🟢 PASS")
            st.caption("FDS Specification Verified")
        else:
            st.markdown("### 🔴 FAIL")
            st.caption("Compliance Violation Detected")
    with col_verdict_text:
        st.markdown(f"**Executive Summary:** {verdict.get('executive_summary', 'No summary generated.')}")
        st.caption(f"Run ID: `{mission.get('run_id', 'N/A')}` | Evaluated by: **Gemini 3.5 Flash (Structured Output)**")

st.subheader("Stage-by-stage verification breakdown")

stages = [
    ("formation_rule_verified", "Stage 1: Formation Rule", "Maintain designated inter-agent spacing and bearing from designated neighbor", ":material/straighten:"),
    ("coordinated_movement_verified", "Stage 2: Coordinated Movement", "All follower drones track lead drone velocity without lag or trajectory divergence", ":material/navigation:"),
    ("dynamic_adaptation_verified", "Stage 3: Dynamic Adaptation", "Autonomous obstacle detection, repulsive force generation, and avoidance maneuver", ":material/change_circle:"),
    ("self_organization_verified", "Stage 4: Self-Organization", "Neighbor drones sense avoidance deviation and yield to maintain minimum clearance >= 3.0m", ":material/hub:"),
    ("re_formation_verified", "Stage 5: Re-Formation", "Once obstacle is cleared, swarm re-accelerates and restores formation error <= 2.0m", ":material/restart_alt:"),
]

for stage_key, stage_title, stage_desc, icon_name in stages:
    stage_data = verdict.get(stage_key, {})
    passed = stage_data.get("passed", False)
    details = stage_data.get("details", "Evaluated per FDS standard.")

    with st.container(border=True):
        col_status, col_content = st.columns([1, 4])
        with col_status:
            if passed:
                st.markdown(f"**{icon_name} PASSED**")
                st.caption("Status: ✅ Compliant")
            else:
                st.markdown(f"**{icon_name} FAILED**")
                st.caption("Status: ❌ Non-Compliant")
        with col_content:
            st.markdown(f"#### {stage_title}")
            st.markdown(f"*{stage_desc}*")
            st.info(f"**Technical Evaluation:** {details}", icon=":material/info:")

# --- Quantitative Verification Scorecards ---
st.subheader("Quantitative compliance thresholds")

col_g1, col_g2, col_g3 = st.columns(3, border=True)

with col_g1:
    st.markdown("**Inter-Agent Clearance**")
    min_clear = metrics.get("minimum_clearance_meters", 0.0)
    fig_g1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min_clear,
        number={"suffix": " m", "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 15]},
            "bar": {"color": "#34D399" if min_clear >= 3.0 else "#F87171"},
            "threshold": {
                "line": {"color": "#F87171", "width": 3},
                "thickness": 0.75,
                "value": 3.0,
            },
            "steps": [
                {"range": [0, 3.0], "color": "rgba(248, 113, 113, 0.2)"},
                {"range": [3.0, 15], "color": "rgba(52, 211, 153, 0.2)"},
            ],
        },
    ))
    fig_g1.update_layout(height=180, margin=dict(l=15, r=15, t=25, b=15), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_g1, config={"displayModeBar": False})
    st.caption("Target: Clearance $\ge 3.0$m to prevent collision")

with col_g2:
    st.markdown("**Final Formation Error**")
    final_err = metrics.get("final_formation_error_meters", 0.0)
    fig_g2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_err,
        number={"suffix": " m", "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 5]},
            "bar": {"color": "#34D399" if final_err <= 2.0 else "#F87171"},
            "threshold": {
                "line": {"color": "#F87171", "width": 3},
                "thickness": 0.75,
                "value": 2.0,
            },
            "steps": [
                {"range": [0, 2.0], "color": "rgba(52, 211, 153, 0.2)"},
                {"range": [2.0, 5.0], "color": "rgba(248, 113, 113, 0.2)"},
            ],
        },
    ))
    fig_g2.update_layout(height=180, margin=dict(l=15, r=15, t=25, b=15), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_g2, config={"displayModeBar": False})
    st.caption("Target: Convergence error $\le 2.0$m after obstacle")

with col_g3:
    st.markdown("**Peak Avoidance Deviation**")
    max_dev = metrics.get("max_deviation_meters", 0.0)
    fig_g3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max_dev,
        number={"suffix": " m", "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 10]},
            "bar": {"color": "#60A5FA"},
            "steps": [
                {"range": [0, 10], "color": "rgba(96, 165, 250, 0.2)"},
            ],
        },
    ))
    fig_g3.update_layout(height=180, margin=dict(l=15, r=15, t=25, b=15), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_g3, config={"displayModeBar": False})
    st.caption("Measurement: Maximum lateral evasion deflection")

# --- Raw Verdict JSON Expander ---
with st.expander("Structured Pydantic Verdict Payload (JSON)", expanded=False):
    st.json(verdict)
