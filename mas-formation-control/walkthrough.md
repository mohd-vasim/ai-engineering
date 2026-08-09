# Multi-Agent Formation Control Streamlit Application Walkthrough

We designed and built a multi-page **Streamlit Application** for the **Multi-Agent Formation Control** system, adhering to the specifications from Chapter 5 of *Agentic Architectural Patterns for Building Multi-Agent Systems* and the LangGraph StateGraph architecture.

---

## 🖥️ Screen Architecture

The application is structured into **5 dedicated screens** using modern Streamlit multi-page navigation (`st.navigation`):

```
mas-formation-control/
├── .streamlit/config.toml        # Dark slate theme with Inter & JetBrains Mono fonts
├── core/
│   ├── physics.py                # Decentralized DroneAgent, Vector math, 2D simulation
│   ├── graph.py                  # LangGraph StateGraph, Gemini structured output, mission orchestrator
│   └── db.py                     # SQLite reader/writer for data/telemetry.db
├── streamlit_app.py              # Main entrypoint & shared session state
└── app_pages/
    ├── mission_control.py        # 1. 2D Flight Simulator & Interactive HUD
    ├── fds_compliance.py         # 2. 5-Stage FDS Formal Compliance Matrix
    ├── telemetry_history.py      # 3. SQLite Mission History & Comparative Analytics
    ├── architecture.py           # 4. Decentralized Control Laws & Math Explainer
    └── mlflow_view.py            # 5. Databricks MLflow Traces & Payload Optimization
```

---

## 🌟 Detailed Screen Overview

### 1. 🛸 Mission Control & 2D Flight Simulator (`mission_control.py`)
- **Interactive 2D Flight Arena (Plotly)**:
  - Visualizes drone trajectories, starting origins, and instantaneous position markers.
  - Renders circular obstacle hazard zones and yellow dashed avoidance margins ($r + 3.0\text{m}$).
- **Timeline Scrubber (0–159 steps)**:
  - Slide through time ($0.1\text{s}$ per step) to inspect exact swarm coordinates, velocity vectors, and avoidance states.
- **Telemetry HUD**:
  - Live formation error (m), inter-agent clearance (m), and obstacle avoidance indicator.
- **Dispatch Modes**:
  - **Gemini AI Agent Planner**: Analyzes mission briefs to automatically decide drone count ($3–8$), formation (`grid`, `v_shape`, `line`), and spacing.
  - **Manual Swarm Operator**: Full manual control of swarm size, geometry, and spacing.
- **Mission Presets**: Standard Agricultural Survey, Narrow Orchard Passage, High-Density Survey, Clear Field.

---

### 2. 📋 FDS Verification Matrix (`fds_compliance.py`)
Directly verifies the **5 FDS Stages** specified in Chapter 5 of the book:
1. **Stage 1: Formation Rule** — Inter-drone distance and neighbor offset adherence.
2. **Stage 2: Coordinated Movement** — Follower tracking of leader velocity along survey path.
3. **Stage 3: Dynamic Adaptation** — Autonomous obstacle detection and tangential evasion maneuver.
4. **Stage 4: Self-Organization** — Neighbor peer-yielding forces to guarantee safety clearance $\ge 3.0\text{m}$.
5. **Stage 5: Re-Formation** — Convergence back to nominal formation error $\le 2.0\text{m}$ after hazard.
- Includes **Quantitative Gauge Meters** and full **Gemini Structured Output JSON inspection**.

---

### 3. 🗄️ Mission Telemetry History (`telemetry_history.py`)
- Live interface to persistent SQLite database (`data/telemetry.db`).
- **KPI Metrics**: Total Missions, Overall Pass Rate (%), Average Max Deviation, Average Clearance.
- **Interactive Filters**: Filter by formation type, pass/fail status, and drone count range.
- **Data Table**: Searchable, sortable table with formatted metrics and pass/fail badges.
- **Export**: One-click **CSV Download**.
- **Comparative Analytics**: Box plot of max deviation by formation and scatter plot of clearance vs drone count.

---

### 4. 📐 Architecture & Control Laws (`architecture.py`)
- **Why Drones Use Pure Math vs LLM for Mission Planning**:
  - Highlights $100\text{Hz}$ execution, zero API token cost, and sub-millisecond safety guarantees for drone agents vs strategic planning for Gemini.
- **Figure 5.14 Control Loop**: Interactive Mermaid diagram of the decentralized sense-evaluate-act loop.
- **Mathematical Formulations**: Formatted LaTeX equations for tracking error, proportional control, obstacle repulsion, and peer yielding.
- **LangGraph Workflow**: Mermaid diagram of the 6-node StateGraph pipeline.

---

### 5. 📊 MLflow Observability (`mlflow_view.py`)
- **Databricks Connection Status**: Tracks Experiment ID `3192447675404693` on Databricks.
- **Payload Optimization**: Explains the fix from $O(N^2)$ growing nested history blobs to $O(N)$ lightweight position snapshots.
- **Databricks Navigation Guide**: Step-by-step instructions for inspecting traces and latency in the Databricks UI.

---

## 🚀 Running the Application

The application is running and accessible at:
```
Local URL: http://localhost:8501
```

To start or restart the app manually:
```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```
