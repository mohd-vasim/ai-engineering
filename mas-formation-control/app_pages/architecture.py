"""
Multi-Agent Formation Control Architecture & Control Laws.
Explains the theoretical foundations from Chapter 5, the decentralized control loop, and LLM roles.
"""
import streamlit as st

st.header("📐 Architecture & decentralized control laws", divider="blue")

st.markdown("""
This application implements the **Formation Control Pattern** from *Agentic Architectural Patterns for Building Multi-Agent Systems* (Chapter 5, pp. 200–203).

Unlike centralized multi-agent systems where a single controller calculates global coordinates, Formation Control distributes the control logic across individual decentralized agents.
""")

# --- Why Pure Math for Drones vs LLM for Mission ---
with st.container(border=True):
    st.subheader("💡 Why drones use pure math while LLM orchestrates missions")
    st.markdown("""
    A fundamental architectural design decision in multi-agent robotics is the separation of **tactical physics control** from **strategic mission orchestration**:
    """)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("##### ⚡ Local Drone Agents (Physics & Control Laws)")
        st.write("• **Update Frequency:** Sub-millisecond execution ($100\\text{ Hz}$).")
        st.write("• **Zero Network/API Latency:** Operates directly on local sensor streams.")
        st.write("• **Deterministic Safety:** Math guarantees collision avoidance ($d \\ge 3.0\\text{m}$).")
        st.write("• **Extreme Scalability:** Scales to 100s of drones with zero LLM API cost.")
    
    with col_t2:
        st.markdown("##### 🧠 Strategic Mission Agent (Gemini 3.5 Flash)")
        st.write("• **Mission Brief Synthesis:** Interprets natural language mission context.")
        st.write("• **Swarm Topology Dispatch:** Decides drone count, formation geometry, and spacing.")
        st.write("• **FDS Verification Verdict:** Validates the 5-stage formal compliance.")
        st.write("• **Structured Output:** Emits validated Pydantic schemas for downstream systems.")

# --- Figure 5.14 Control Loop ---
with st.container(border=True):
    st.subheader("🔁 Figure 5.14 — Decentralized Drone Agent Control Loop")
    
    st.markdown("""
    Every drone in the swarm continuously executes the localized sense-evaluate-act loop:
    """)
    
    st.markdown("""
```mermaid
graph TD
    A[1. Sense Neighbor Position] --> B[2. Calculate Desired Position: Neighbor + Offset]
    B --> C[3. Compute Error: Desired - Current]
    C --> D{4. Error > Tolerance?}
    D -- Yes --> E[Calculate Adjustment Vector & Adjust Velocity]
    D -- No --> F[Maintain Current Heading & Velocity]
    E --> G[Apply Obstacle & Peer Repulsion Forces]
    F --> G
    G --> H[Update Position Integrator]
    H --> A
    
    style A fill:#1E293B,stroke:#60A5FA,stroke-width:2px,color:#F1F5F9
    style B fill:#1E293B,stroke:#60A5FA,stroke-width:2px,color:#F1F5F9
    style C fill:#1E293B,stroke:#60A5FA,stroke-width:2px,color:#F1F5F9
    style D fill:#334155,stroke:#FBBF24,stroke-width:2px,color:#F1F5F9
    style E fill:#1E293B,stroke:#34D399,stroke-width:2px,color:#F1F5F9
    style F fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#F1F5F9
    style G fill:#1E293B,stroke:#F87171,stroke-width:2px,color:#F1F5F9
    style H fill:#1E293B,stroke:#A78BFA,stroke-width:2px,color:#F1F5F9
```
""")

# --- Mathematical Formulations ---
st.subheader("Mathematical control laws")

col_m1, col_m2 = st.columns(2, border=True)

with col_m1:
    st.markdown("##### 1. Desired Position & Tracking Error")
    st.latex(r"p_{\text{desired}, i}(t) = p_{\text{neighbor}}(t) + \Delta_{\text{offset}, i}")
    st.latex(r"e_i(t) = p_{\text{desired}, i}(t) - p_i(t)")
    st.markdown("##### 2. Proportional Velocity Adjustment")
    st.latex(r"v_{\text{adj}, i} = K_p \cdot e_i(t) \quad \text{if } \|e_i(t)\| > \varepsilon_{\text{tolerance}}")

with col_m2:
    st.markdown("##### 3. Obstacle Repulsion & Tangent Bypass")
    st.latex(r"F_{\text{obs}} = \hat{u}_{\text{obs}} \cdot (\delta_{\text{safety}} - d_{\text{obs}}) \cdot K_{\text{rep}} + \hat{t} \cdot K_{\text{tan}}")
    st.markdown("##### 4. Peer-Yielding Self-Organization")
    st.latex(r"F_{\text{peer}} = \sum_{j \ne i} \hat{u}_{ij} \cdot \max(0, \delta_{\text{peer}} - d_{ij}) \cdot K_{\text{peer}}")

# --- LangGraph 6-Node Pipeline ---
with st.container(border=True):
    st.subheader("🔗 LangGraph 6-Node StateGraph Workflow")
    
    st.markdown("""
```mermaid
graph LR
    START --> assess[assess_mission]
    assess --> plan[plan_swarm<br/><b>Gemini LLM</b>]
    plan --> dispatch[dispatch_simulation<br/><b>Decentralized Physics</b>]
    dispatch --> analyze[analyze_telemetry<br/><b>Metrics Engine</b>]
    analyze --> persist[persist_telemetry<br/><b>SQLite DB</b>]
    persist --> verdict[generate_verdict<br/><b>Gemini FDS Verdict</b>]
    verdict --> END
    
    style assess fill:#1E293B,stroke:#94A3B8,color:#F1F5F9
    style plan fill:#1E293B,stroke:#60A5FA,stroke-width:3px,color:#F1F5F9
    style dispatch fill:#1E293B,stroke:#34D399,stroke-width:2px,color:#F1F5F9
    style analyze fill:#1E293B,stroke:#FBBF24,stroke-width:2px,color:#F1F5F9
    style persist fill:#1E293B,stroke:#A78BFA,stroke-width:2px,color:#F1F5F9
    style verdict fill:#1E293B,stroke:#60A5FA,stroke-width:3px,color:#F1F5F9
```
""")
