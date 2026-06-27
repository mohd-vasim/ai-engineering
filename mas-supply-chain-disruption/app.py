import os
import time
import uuid
import json
from datetime import datetime

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

load_dotenv()

import mlflow
from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

os.environ["DATABRICKS_TOKEN"] = os.environ.get("DATABRICKS_TOKEN", "")
os.environ["DATABRICKS_HOST"] = os.environ.get("DATABRICKS_HOST", "")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "databricks")
MLFLOW_REGISTRY_URI = os.environ.get("MLFLOW_REGISTRY_URI", "databricks-uc")
MLFLOW_EXPERIMENT_ID = os.environ.get("MLFLOW_EXPERIMENT_ID", "")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
if MLFLOW_EXPERIMENT_ID:
    mlflow.set_experiment(experiment_id=MLFLOW_EXPERIMENT_ID)
else:
    mlflow.set_experiment("/Users/supply-chain-disruption-poc")

mlflow.langchain.autolog()

try:
    from upstash_redis import Redis
    redis_client = Redis(
        url=os.environ.get("UPSTASH_REDIS_REST_URL", ""),
        token=os.environ.get("UPSTASH_REDIS_REST_TOKEN", ""),
    )
except Exception:
    redis_client = None


NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"
CHAT_MODEL_NAME = "openai/gpt-oss-120b"


class ShipmentStatus(BaseModel):
    shipment_id: str
    status: Literal["in_transit", "delayed", "rerouted", "delivered", "lost"]
    reason: str = ""
    source_agent_id: str
    timestamp: float = Field(default_factory=time.time)
    ttl_seconds: int = 3600
    version: int = 1


class EventLog(BaseModel):
    event_id: str
    event_type: Literal["disruption_detected", "reroute_planned", "customer_notified", "delivery_confirmed"]
    shipment_id: str
    details: str = ""
    source_agent_id: str
    timestamp: float = Field(default_factory=time.time)
    ttl_seconds: int = 86400
    version: int = 1


class SharedEpistemicMemory:
    def __init__(self, client):
        self._client = client

    def write(self, key: str, entry: BaseModel) -> None:
        if self._client is None:
            return
        self._client.set(key, entry.model_dump_json())

    def read(self, key: str) -> Optional[BaseModel]:
        if self._client is None:
            return None
        raw = self._client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        entry = self._deserialize(key, raw)
        if entry is None:
            return None
        if not self.is_fresh(entry):
            return None
        return entry

    def update(self, key: str, entry: BaseModel, expected_version: int) -> BaseModel:
        current = self.read(key)
        if current is None:
            raise KeyError(f"Key {key!r} not found")
        if current.version != expected_version:
            raise ValueError(f"Version conflict on {key!r}")
        updated = entry.model_copy(update={"version": expected_version + 1})
        self._client.set(key, updated.model_dump_json())
        return updated

    def delete(self, key: str) -> None:
        if self._client is None:
            return
        self._client.delete(key)

    def list_keys(self) -> list[str]:
        if self._client is None:
            return []
        keys = self._client.keys("*")
        return [k.decode("utf-8") if isinstance(k, bytes) else k for k in (keys or [])]

    def is_fresh(self, entry: BaseModel) -> bool:
        age = time.time() - entry.timestamp
        return age < entry.ttl_seconds

    def snapshot(self) -> dict:
        result = {}
        for key in self.list_keys():
            raw = self._client.get(key)
            if raw is None:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = self._deserialize(key, raw)
            if entry is not None:
                result[key] = entry.model_dump()
        return result

    def clear_all(self) -> None:
        if self._client is None:
            return
        for key in self.list_keys():
            self._client.delete(key)

    def _deserialize(self, key: str, raw: str) -> Optional[BaseModel]:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if key.startswith("shipment:"):
            return ShipmentStatus.model_validate(data)
        if key.startswith("evt-"):
            return EventLog.model_validate(data)
        for model_cls in (ShipmentStatus, EventLog):
            try:
                return model_cls.model_validate(data)
            except Exception:
                continue
        return None


@tool
def log_event(
    event_type: str,
    shipment_id: str,
    details: str,
    source_agent_id: str,
) -> str:
    """Log a discrete event to the shared memory."""
    event = EventLog(
        event_id=f"evt-{uuid.uuid4().hex[:8]}",
        event_type=event_type,
        shipment_id=shipment_id,
        details=details,
        source_agent_id=source_agent_id,
    )
    memory.write(event.event_id, event)
    return f"Logged event {event.event_id} ({event_type}) for shipment {shipment_id}"


@tool
def update_shipment_status(
    shipment_id: str,
    status: str,
    reason: str,
    source_agent_id: str,
) -> str:
    """Update a shipment's status in the shared memory."""
    key = f"shipment:{shipment_id}"
    existing = memory.read(key)

    if existing is None:
        entry = ShipmentStatus(
            shipment_id=shipment_id,
            status=status,
            reason=reason,
            source_agent_id=source_agent_id,
        )
        memory.write(key, entry)
        return f"Created shipment {shipment_id} with status={status} (v{entry.version})"

    updated = memory.update(
        key,
        existing.model_copy(update={"status": status, "reason": reason, "source_agent_id": source_agent_id}),
        expected_version=existing.version,
    )
    return f"Updated shipment {shipment_id} to status={status} (v{updated.version})"


@tool
def read_memory(key: str) -> str:
    """Read an entry from the shared memory by its key."""
    entry = memory.read(key)
    if entry is None:
        return "NOT_FOUND"
    return entry.model_dump_json()


memory = SharedEpistemicMemory(redis_client)


def init_agents():
    if "agents_initialized" in st.session_state:
        return True

    if not NVIDIA_API_KEY:
        st.session_state.llm_error = "NVIDIA_API_KEY not set"
        return False

    try:
        llm = init_chat_model(
            CHAT_MODEL_NAME,
            model_provider="nvidia",
            base_url=BASE_URL,
            api_key=NVIDIA_API_KEY,
        )
    except Exception as e:
        st.session_state.llm_error = str(e)
        return False

    st.session_state.monitoring_agent = create_agent(
        llm,
        tools=[log_event],
        system_prompt=(
            "You are the MonitoringAgent. Your job is to detect supply chain disruptions "
            "(storms, port closures, road blocks) and log them as events. "
            "Use the log_event tool with event_type='disruption_detected'. "
            "Always pass source_agent_id='monitoring-agent'. "
            "Do NOT update shipment statuses — that's the LogisticsAgent's job."
        ),
    )

    st.session_state.logistics_agent = create_agent(
        llm,
        tools=[read_memory, update_shipment_status],
        system_prompt=(
            "You are the LogisticsAgent. Your job is to read disruption events from "
            "shared memory and update affected shipment statuses. "
            "First call read_memory to discover recent events, then call "
            "update_shipment_status with status='delayed' or 'rerouted'. "
            "Always pass source_agent_id='logistics-agent'. "
            "Do NOT notify customers — that's the CustomerNotificationAgent's job."
        ),
    )

    st.session_state.customer_agent = create_agent(
        llm,
        tools=[read_memory, log_event],
        system_prompt=(
            "You are the CustomerNotificationAgent. Your job is to read shipment "
            "statuses from shared memory and log customer notification events for "
            "any shipment that is delayed, rerouted, or lost. "
            "First call read_memory to discover shipment statuses, then call "
            "log_event with event_type='customer_notified'. "
            "Always pass source_agent_id='customer-notification-agent'."
        ),
    )

    st.session_state.agents_initialized = True
    return True


def run_agent(agent, prompt: str, label: str):
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    final_msg = result["messages"][-1]
    return final_msg.content, result


def format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def get_memory_data():
    snapshot = memory.snapshot()
    shipments = []
    events = []

    for key, data in snapshot.items():
        if key.startswith("shipment:"):
            shipments.append(data)
        elif key.startswith("evt-"):
            events.append(data)

    shipments.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    events.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    return shipments, events


def create_event_timeline(events):
    if not events:
        return None

    df_events = []
    for evt in events:
        df_events.append({
            "event_id": evt.get("event_id", ""),
            "event_type": evt.get("event_type", ""),
            "shipment_id": evt.get("shipment_id", ""),
            "source_agent_id": evt.get("source_agent_id", ""),
            "timestamp": evt.get("timestamp", 0),
            "details": evt.get("details", ""),
        })

    fig = px.timeline(
        df_events,
        x_start="timestamp",
        x_end="timestamp",
        y="shipment_id",
        color="event_type",
        hover_data=["details", "source_agent_id"],
        title="Event Timeline",
    )
    fig.update_layout(
        height=300,
        showlegend=True,
        xaxis_title="Time",
        yaxis_title="Shipment",
    )
    return fig


def create_shipment_status_chart(shipments):
    if not shipments:
        return None

    status_counts = {}
    for sh in shipments:
        status = sh.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    fig = px.pie(
        names=list(status_counts.keys()),
        values=list(status_counts.values()),
        title="Shipment Status Distribution",
    )
    fig.update_layout(height=250, showlegend=True)
    return fig


st.set_page_config(
    page_title="Multi-Agent Supply Chain Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📦 Multi-Agent Supply Chain Disruption Dashboard")
st.markdown("**Shared Epistemic Memory (SEM)** — Interact with agents step by step")

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []
if "agent_responses" not in st.session_state:
    st.session_state.agent_responses = {}
if "agents_initialized" not in st.session_state:
    st.session_state.agents_initialized = False
if "llm_error" not in st.session_state:
    st.session_state.llm_error = None


with st.sidebar:
    st.header("Configuration")

    with st.expander("⚙️ API Settings", expanded=True):
        api_key_status = "✅ Set" if NVIDIA_API_KEY else "❌ Missing"
        st.text(f"NVIDIA API: {api_key_status}")

        redis_status = "✅ Connected" if redis_client else "⚠️ Using demo mode"
        st.text(f"Upstash Redis: {redis_status}")

        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "databricks")
        st.text(f"MLflow: {mlflow_uri}")

    st.divider()

    with st.expander("🎯 Disruption Input", expanded=True):
        st.selectbox(
            "Shipment ID",
            ["SHP-001", "SHP-002", "SHP-003", "SHP-004", "SHP-005"],
            index=0,
            key="shipment_id",
        )

        st.selectbox(
            "Event Type",
            ["disruption_detected", "reroute_planned", "customer_notified", "delivery_confirmed"],
            index=0,
            key="event_type",
        )

        st.text_area(
            "Details / Scenario",
            value="A severe storm has closed the I-80 corridor in Nebraska.",
            height=80,
            key="scenario",
        )

    st.divider()

    with st.expander("🚚 Logistics Options", expanded=True):
        st.selectbox(
            "New Status",
            ["delayed", "rerouted", "in_transit", "delivered", "lost"],
            index=0,
            key="new_status",
        )
        st.text_input("Reason", value="Weather conditions", key="status_reason")

    st.divider()

    if st.button("🗑️ Clear Memory Store", use_container_width=True):
        memory.clear_all()
        st.session_state.agent_logs = []
        st.session_state.agent_responses = {}
        st.rerun()

    st.divider()

    st.caption("**Workflow**")
    st.markdown("1. 🔍 Run MonitoringAgent")
    st.markdown("2. 🚚 Run LogisticsAgent")
    st.markdown("3. 📬 Run CustomerAgent")


col_main, col_memory = st.columns([3, 2], gap="medium")


with col_main:
    st.subheader("Interactive Agent Control")

    if st.session_state.llm_error:
        st.error(f"LLM Error: {st.session_state.llm_error}")

    if not st.session_state.agents_initialized and not NVIDIA_API_KEY:
        st.warning("Enter NVIDIA_API_KEY in .env to enable agents")
    else:
        if st.button("🔍 Initialize Agents", use_container_width=True):
            with st.spinner("Initializing agents..."):
                success = init_agents()
                if success:
                    st.success("Agents initialized!")
                    st.rerun()
                else:
                    st.error(f"Failed: {st.session_state.llm_error}")

        st.divider()

        st.markdown("### Step 1: 🔍 Monitoring Agent")
        st.markdown("*Log disruption events to shared memory*")

        col_m1, col_m2 = st.columns([3, 1])
        with col_m1:
            st.text_input("Shipment ID", value=st.session_state.get("shipment_id", "SHP-001"), key="mon_shipment_id")
            st.text_area("Event Details", value=st.session_state.get("scenario", ""), height=60, key="mon_details")
        with col_m2:
            st.selectbox("Event Type", ["disruption_detected", "reroute_planned", "customer_notified", "delivery_confirmed"], index=0, key="mon_event_type")

        if st.button("▶️ Run MonitoringAgent", use_container_width=True, key="btn_monitoring"):
            init_agents()
            if not st.session_state.get("agents_initialized"):
                st.error("Agents not initialized")
            else:
                with st.spinner("MonitoringAgent working..."):
                    try:
                        prompt = f"Log a {st.session_state.mon_event_type} event for shipment {st.session_state.mon_shipment_id}. Details: {st.session_state.mon_details}"
                        response, trace = run_agent(
                            st.session_state.monitoring_agent,
                            prompt,
                            "MonitoringAgent",
                        )
                        st.session_state.agent_responses["monitoring"] = {"response": response, "trace": trace}
                        st.session_state.agent_logs.append({
                            "agent": "MonitoringAgent",
                            "response": response,
                            "timestamp": time.time(),
                        })
                        st.success("MonitoringAgent completed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.divider()

        st.markdown("### Step 2: 🚚 Logistics Agent")
        st.markdown("*Update shipment status based on events*")

        if st.button("📋 Show Current Memory", use_container_width=True, key="btn_show_memory"):
            shipments, events = get_memory_data()
            if shipments or events:
                st.session_state.show_memory_preview = True
            else:
                st.info("Memory is empty")

        if st.session_state.get("show_memory_preview"):
            with st.expander("Current Memory State", expanded=True):
                snapshot = memory.snapshot()
                if snapshot:
                    st.json(snapshot)
                else:
                    st.info("No data in memory")

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.selectbox("New Status", ["delayed", "rerouted", "in_transit", "delivered", "lost"], index=0, key="log_status")
        with col_l2:
            st.text_input("Shipment ID", value=st.session_state.get("shipment_id", "SHP-001"), key="log_shipment_id")

        st.text_input("Reason", value="Weather delay", key="log_reason")

        if st.button("▶️ Run LogisticsAgent", use_container_width=True, key="btn_logistics"):
            init_agents()
            if not st.session_state.get("agents_initialized"):
                st.error("Agents not initialized")
            else:
                with st.spinner("LogisticsAgent working..."):
                    try:
                        prompt = f"""Check shared memory for recent disruption events. Update shipment {st.session_state.log_shipment_id} status to '{st.session_state.log_status}'. Reason: {st.session_state.log_reason}"""
                        response, trace = run_agent(
                            st.session_state.logistics_agent,
                            prompt,
                            "LogisticsAgent",
                        )
                        st.session_state.agent_responses["logistics"] = {"response": response, "trace": trace}
                        st.session_state.agent_logs.append({
                            "agent": "LogisticsAgent",
                            "response": response,
                            "timestamp": time.time(),
                        })
                        st.success("LogisticsAgent completed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.divider()

        st.markdown("### Step 3: 📬 Customer Notification Agent")
        st.markdown("*Notify customers about affected shipments*")

        if st.button("▶️ Run CustomerNotificationAgent", use_container_width=True, key="btn_customer"):
            init_agents()
            if not st.session_state.get("agents_initialized"):
                st.error("Agents not initialized")
            else:
                with st.spinner("CustomerNotificationAgent working..."):
                    try:
                        prompt = "Check shared memory for shipment statuses. For any shipment that is delayed, rerouted, or lost, log a customer_notified event."
                        response, trace = run_agent(
                            st.session_state.customer_agent,
                            prompt,
                            "CustomerNotificationAgent",
                        )
                        st.session_state.agent_responses["customer"] = {"response": response, "trace": trace}
                        st.session_state.agent_logs.append({
                            "agent": "CustomerNotificationAgent",
                            "response": response,
                            "timestamp": time.time(),
                        })
                        st.success("CustomerNotificationAgent completed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    st.divider()
    st.subheader("Execution Log")

    if st.session_state.agent_logs:
        for log in reversed(st.session_state.agent_logs):
            ts = format_timestamp(log["timestamp"])
            st.markdown(f"**[{ts}] {log['agent']}**: {log['response'][:150]}...")
    else:
        st.info("Run an agent to see logs here")

    st.divider()
    st.subheader("Agent Responses & Traces")

    if st.session_state.agent_responses:
        for agent_key, agent_name in [("monitoring", "🔍 MonitoringAgent"), ("logistics", "🚚 LogisticsAgent"), ("customer", "📬 CustomerNotificationAgent")]:
            if agent_key in st.session_state.agent_responses:
                data = st.session_state.agent_responses[agent_key]
                with st.expander(f"{agent_name}"):
                    st.text_area("Response", data["response"], height=100, disabled=True)
                    st.divider()
                    st.markdown("**Trace:**")
                    st.json(data["trace"])
    else:
        st.info("Run agents to see their responses and traces")


with col_memory:
    st.subheader("📊 Memory Store")

    tab1, tab2, tab3 = st.tabs(["📦 Shipments", "📋 Events", "📈 Analytics"])

    with tab1:
        shipments, events = get_memory_data()

        if shipments:
            df_shipments = []
            for sh in shipments:
                df_shipments.append({
                    "shipment_id": sh.get("shipment_id", ""),
                    "status": sh.get("status", ""),
                    "reason": sh.get("reason", "")[:50],
                    "agent": sh.get("source_agent_id", ""),
                    "version": sh.get("version", 1),
                    "timestamp": format_timestamp(sh.get("timestamp", 0)),
                })

            st.dataframe(df_shipments, hide_index=True)
        else:
            st.info("No shipments in memory")

        st.divider()

        with st.expander("🔍 Raw Memory Snapshot"):
            snapshot = memory.snapshot()
            if snapshot:
                st.json(snapshot)
            else:
                st.info("Empty")

    with tab2:
        if events:
            df_events = []
            for evt in events:
                df_events.append({
                    "event_id": evt.get("event_id", ""),
                    "event_type": evt.get("event_type", ""),
                    "shipment_id": evt.get("shipment_id", ""),
                    "agent": evt.get("source_agent_id", ""),
                    "details": evt.get("details", "")[:60],
                    "timestamp": format_timestamp(evt.get("timestamp", 0)),
                })

            st.dataframe(df_events, hide_index=True)
        else:
            st.info("No events in memory")

    with tab3:
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_pie = create_shipment_status_chart(shipments)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No data")

        with col_chart2:
            fig_timeline = create_event_timeline(events)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No events")


st.divider()

with st.expander("📚 Documentation", expanded=False):
    st.markdown("""
    **How this works:**

    1. **MonitoringAgent** detects disruptions (storms, port closures) and logs them as events
    2. **LogisticsAgent** reads disruption events and updates shipment statuses (delayed/rerouted)
    3. **CustomerNotificationAgent** reads shipment statuses and logs notification events for affected customers

    **Key concepts:**
    - **Shared Epistemic Memory (SEM)**: All agents coordinate through a common memory store, not direct messaging
    - **Optimistic Locking**: Each entry has a version number to prevent silent overwrites
    - **TTL/Staleness**: Entries expire after a set time, stale entries are treated as missing
    - **Pydantic Schemas**: All memory entries are typed and validated
    - **MLflow Tracing**: All LLM calls, tool calls, and agent steps are traced to Databricks MLflow
    """)