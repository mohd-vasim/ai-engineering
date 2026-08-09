"""
Interactive Mission Control & Swarm Simulation Studio.
Provides real-time 2D arena visualization, time scrubber, telemetry charts, and LLM dispatch controls.
"""
import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from core.physics import simulate_swarm, CircularObstacle
from core.graph import build_and_run_full_mission, get_gemini_llm

st.header("🛸 Swarm mission control & 2D flight simulator", divider="blue")

# --- Sidebar Controls ---
with st.sidebar:
    st.subheader("Mission configuration")
    
    scenario = st.selectbox(
        "Scenario preset",
        [
            "Standard Agricultural Survey (Single Tree)",
            "Narrow Orchard Passage",
            "Dense High-Coverage Survey",
            "Clear Field Baseline",
        ],
        index=0,
    )

    # Preset defaults
    if scenario == "Standard Agricultural Survey (Single Tree)":
        default_drones = 5
        default_formation = "grid"
        default_spacing = 10.0
        default_obs = [{"x": 20.0, "y": 40.0, "radius": 3.5, "label": "Tree Obstacle"}]
        default_brief = "Conduct a precision agricultural survey of a 100m x 80m wheat field. One tree obstacle detected at (20, 40) with 3.5m radius."
    elif scenario == "Narrow Orchard Passage":
        default_drones = 4
        default_formation = "line"
        default_spacing = 8.0
        default_obs = [{"x": 20.0, "y": 40.0, "radius": 4.5, "label": "Orchard Hazard"}]
        default_brief = "Survey a narrow corridor in an orchard. Tight obstacle boundary requires high clearance and agile formation."
    elif scenario == "Dense High-Coverage Survey":
        default_drones = 7
        default_formation = "v_shape"
        default_spacing = 9.0
        default_obs = [{"x": 20.0, "y": 45.0, "radius": 3.0, "label": "Irrigation Tower"}]
        default_brief = "Maximum field survey sweep across wide crop zone with central irrigation obstacle."
    else:
        default_drones = 5
        default_formation = "grid"
        default_spacing = 10.0
        default_obs = []
        default_brief = "Baseline nominal flight without obstacles across clear field."

    planner_mode = st.segmented_control(
        "Planner mode",
        options=["Gemini AI agent", "Manual operator"],
        default="Gemini AI agent",
    )

    brief_input = st.text_area(
        "Mission brief",
        value=default_brief,
        height=85,
        help="Context analyzed by Gemini LLM to decide optimal swarm size, layout, and spacing.",
    )

    with st.expander("Obstacle placement", expanded=False):
        has_obs = st.checkbox("Include obstacle", value=len(default_obs) > 0)
        if has_obs:
            obs_x = st.number_input("Obstacle X (m)", value=20.0, step=1.0)
            obs_y = st.number_input("Obstacle Y (m)", value=40.0, step=1.0)
            obs_r = st.number_input("Obstacle radius (m)", value=3.5, min_value=1.0, max_value=10.0, step=0.5)
            active_obstacles = [{"x": obs_x, "y": obs_y, "radius": obs_r, "label": "Field Hazard"}]
        else:
            active_obstacles = []

    if planner_mode == "Manual operator":
        num_drones_input = st.slider("Swarm size (drones)", min_value=3, max_value=8, value=default_drones)
        formation_input = st.segmented_control(
            "Formation shape",
            options=["grid", "v_shape", "line"],
            default=default_formation,
        )
        spacing_input = st.slider("Inter-drone spacing (m)", min_value=6.0, max_value=16.0, value=default_spacing, step=0.5)
    else:
        num_drones_input = None
        formation_input = None
        spacing_input = None

    with st.expander("Physics & control gains", expanded=False):
        kp_val = st.slider("Proportional gain (Kp)", 0.5, 3.0, 1.4, 0.1)
        tol_val = st.slider("Tolerance epsilon (m)", 0.05, 1.0, 0.2, 0.05)
        sim_speed = st.slider("Leader survey speed (m/s)", 1.0, 4.0, 2.5, 0.5)

    launch_clicked = st.button("🚀 Launch mission & simulate", type="primary")

# Handle mission execution
if launch_clicked:
    with st.spinner("Executing LangGraph StateGraph & decentralized physics simulation..."):
        if planner_mode == "Gemini AI agent":
            res = build_and_run_full_mission(
                mission_brief=brief_input,
                field_width=100.0,
                field_height=80.0,
                obstacles=active_obstacles,
            )
        else:
            manual_plan_dict = {
                "num_drones": num_drones_input,
                "formation_type": formation_input,
                "spacing_m": spacing_input,
                "justification": f"Manual operator dispatched {num_drones_input} drones in '{formation_input}' at {spacing_input}m spacing.",
            }
            res = build_and_run_full_mission(
                mission_brief=brief_input,
                field_width=100.0,
                field_height=80.0,
                obstacles=active_obstacles,
                manual_plan=manual_plan_dict,
            )
        st.session_state.latest_mission = res
        st.session_state.selected_sim_step = 0
        st.toast("Mission completed successfully!", icon="✅")

mission = st.session_state.latest_mission
plan = mission["plan"]
metrics = mission["metrics"]
verdict = mission["verdict"]
step_logs = mission["step_logs"]
trajectories = mission["trajectories"]
obstacles = mission["obstacles"]

# --- Top KPI Row ---
with st.container(horizontal=True):
    st.metric(
        "Deployed formation",
        f"{plan.get('formation_type', 'grid').upper()}",
        f"{plan.get('num_drones', 5)} Drones @ {plan.get('spacing_m', 10):.1f}m",
        border=True,
    )
    st.metric(
        "Max formation deviation",
        f"{metrics.get('max_deviation_meters', 0.0):.3f} m",
        "Obstacle peak deflection",
        border=True,
    )
    st.metric(
        "Final convergence error",
        f"{metrics.get('final_formation_error_meters', 0.0):.3f} m",
        "Target: < 2.0 m",
        border=True,
    )
    st.metric(
        "Min inter-drone clearance",
        f"{metrics.get('minimum_clearance_meters', 0.0):.3f} m",
        "Safety threshold: > 3.0 m",
        border=True,
    )
    is_ok = verdict.get("overall_mission_success", False)
    st.metric(
        "FDS compliance verdict",
        "VERIFIED PASS" if is_ok else "COMPLIANCE FAIL",
        "5/5 Stages Passed" if is_ok else "Review Requirements",
        border=True,
    )

# --- Main Layout: 2D Spatial Map + Time-Series HUD ---
col_map, col_hud = st.columns([3, 2], border=True)

with col_map:
    st.subheader("2D Flight arena & spatial trajectories")
    
    # Time scrubber slider
    sim_step = st.slider(
        "Timeline scrubber (step / time = 0.1s)",
        min_value=0,
        max_value=len(step_logs) - 1,
        value=st.session_state.get("selected_sim_step", 0),
        step=1,
        key="scrubber_slider",
    )

    current_log = step_logs[sim_step]
    current_positions = current_log["positions"]
    
    # Build 2D Plotly Arena
    fig_arena = go.Figure()

    # Add full trajectory paths
    colors = px.colors.qualitative.Plotly
    drone_ids = list(trajectories.keys())
    
    for i, aid in enumerate(drone_ids):
        pts = np.array(trajectories[aid])
        color = colors[i % len(colors)]
        
        # Path trail up to current step
        cur_pts = pts[:sim_step + 1]
        fig_arena.add_trace(go.Scatter(
            x=cur_pts[:, 0],
            y=cur_pts[:, 1],
            mode="lines",
            name=f"{aid} Trail",
            line=dict(color=color, width=2.5),
            hoverinfo="skip",
        ))

        # Initial launch marker
        fig_arena.add_trace(go.Scatter(
            x=[pts[0, 0]],
            y=[pts[0, 1]],
            mode="markers",
            marker=dict(size=8, color=color, symbol="circle-open", line=dict(width=2)),
            name=f"{aid} Origin",
            showlegend=False,
        ))

        # Current instantaneous position marker
        cur_pos = current_positions[aid]
        symbol = "star" if aid == "Drone_A" else "circle"
        marker_size = 14 if aid == "Drone_A" else 11
        
        fig_arena.add_trace(go.Scatter(
            x=[cur_pos[0]],
            y=[cur_pos[1]],
            mode="markers+text",
            text=[f" {aid}"],
            textposition="top right",
            textfont=dict(size=10, color="#F1F5F9"),
            marker=dict(size=marker_size, color=color, symbol=symbol, line=dict(color="#FFFFFF", width=1.5)),
            name=f"{aid} (t={sim_step*0.1:.1f}s)",
            hovertext=f"<b>{aid}</b><br>X: {cur_pos[0]:.2f}m<br>Y: {cur_pos[1]:.2f}m",
            hoverinfo="text",
        ))

    # Add Obstacle Circles
    for obs in obstacles:
        # Obstacle solid core
        theta = np.linspace(0, 2 * np.pi, 50)
        obs_x = obs["x"] + obs["radius"] * np.cos(theta)
        obs_y = obs["y"] + obs["radius"] * np.sin(theta)
        fig_arena.add_trace(go.Scatter(
            x=obs_x,
            y=obs_y,
            fill="toself",
            fillcolor="rgba(248, 113, 113, 0.4)",
            line=dict(color="#F87171", width=2),
            name=f"{obs['label']} (r={obs['radius']}m)",
            hovertext=f"<b>{obs['label']}</b><br>Radius: {obs['radius']}m",
            hoverinfo="text",
        ))

        # Safety avoidance margin (radius + 3.0m)
        margin_r = obs["radius"] + 3.0
        safe_x = obs["x"] + margin_r * np.cos(theta)
        safe_y = obs["y"] + margin_r * np.sin(theta)
        fig_arena.add_trace(go.Scatter(
            x=safe_x,
            y=safe_y,
            mode="lines",
            line=dict(color="#FBBF24", width=1.5, dash="dot"),
            name="Avoidance Threshold",
            hoverinfo="skip",
        ))

    fig_arena.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        xaxis=dict(
            title="Field X (meters)",
            range=[-10, 50],
            zeroline=False,
            gridcolor="#334155",
        ),
        yaxis=dict(
            title="Field Y (meters)",
            range=[-15, 65],
            zeroline=False,
            gridcolor="#334155",
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=480,
    )

    st.plotly_chart(fig_arena, config={"displayModeBar": True})

with col_hud:
    st.subheader("Telemetry & flight HUD")
    
    # Instantaneous status at current scrubber step
    with st.container(border=True):
        cur_err = current_log["mean_error"]
        cur_clear = current_log["min_clearance"]
        is_drone_c_avoiding = current_log.get("drone_c_avoiding", 0.0) > 0.5
        
        st.markdown(f"**Step {sim_step} / 159** (Flight Time: `{sim_step*0.1:.1f}s`)")
        st.write(f"• **Instantaneous error:** `{cur_err:.3f} m`")
        st.write(f"• **Instantaneous clearance:** `{cur_clear:.3f} m`")
        if is_drone_c_avoiding:
            st.error("🚨 **Obstacle evasion ACTIVE:** Drone C executing tangential avoidance maneuver", icon=":material/warning:")
        else:
            st.success("🟢 **Formation nominal:** Coordinated survey tracking", icon=":material/check_circle:")

    # Formation Error Chart
    steps = [l["step"] for l in step_logs]
    errors = [l["mean_error"] for l in step_logs]
    clearances = [l["min_clearance"] for l in step_logs]

    fig_telemetry = go.Figure()
    fig_telemetry.add_trace(go.Scatter(
        x=steps,
        y=errors,
        mode="lines",
        name="Formation Error (m)",
        line=dict(color="#F87171", width=2),
    ))
    fig_telemetry.add_trace(go.Scatter(
        x=steps,
        y=clearances,
        mode="lines",
        name="Min Clearance (m)",
        line=dict(color="#34D399", width=2),
    ))
    # Vertical line indicating scrubber position
    fig_telemetry.add_vline(x=sim_step, line_width=1.5, line_dash="dash", line_color="#60A5FA")
    # Threshold horizontal lines
    fig_telemetry.add_hline(y=0.2, line_dash="dot", line_color="#F87171", annotation_text="Tolerance (0.2m)", annotation_position="top left")
    fig_telemetry.add_hline(y=3.0, line_dash="dot", line_color="#34D399", annotation_text="Safety Limit (3.0m)", annotation_position="bottom left")

    fig_telemetry.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        title=dict(text="Formation error & peer clearance over time", font=dict(size=12)),
        xaxis=dict(title="Step", gridcolor="#334155"),
        yaxis=dict(title="Meters (m)", gridcolor="#334155"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.3, font=dict(size=9)),
        height=260,
    )
    st.plotly_chart(fig_telemetry, config={"displayModeBar": False})

# --- Gemini Swarm Planner Justification Card ---
with st.container(border=True):
    st.subheader("🧠 Gemini LLM mission dispatch reasoning")
    col_plan1, col_plan2 = st.columns([1, 3])
    with col_plan1:
        st.markdown(f"**Dispatched size:** `{plan.get('num_drones', 5)}` Drones")
        st.markdown(f"**Formation type:** `{plan.get('formation_type', 'grid')}`")
        st.markdown(f"**Inter-drone spacing:** `{plan.get('spacing_m', 10.0):.1f} m`")
    with col_plan2:
        st.info(f"**Planner justification:** {plan.get('justification', 'Nominal mission plan deployed.')}", icon=":material/lightbulb:")
