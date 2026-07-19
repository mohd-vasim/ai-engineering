import pandas as pd
import streamlit as st

from mas_bidder_agents.bidder import OllamaBidderAgent
from mas_bidder_agents.llm import get_llm
from mas_bidder_agents.models import Task, UtilityWeights
from mas_bidder_agents.reputation import ReputationRegistry
from mas_bidder_agents.solicitor import NoBidsException
from mas_bidder_agents.utility import utility_score

st.set_page_config(
    page_title="Contract-Net Auction",
    page_icon=":material/gavel:",
    layout="wide",
)

REPUTATION_FILE = "reputation.json"
DEFAULT_AGENTS = [
    ("coder", ["code", "debugging"]),
    ("writer", ["writing", "poetry"]),
    ("ml_eng", ["ml", "data_science"]),
]
DEFAULT_TASKS = [
    Task("t1", "Write a Python function to sort integers using quicksort", "code"),
    Task("t2", "Write a haiku about gradient descent", "poetry"),
    Task("t3", "Fix this JS typo: function greet(name {{ return 'Hello ' + name }}", "debugging"),
    Task("t4", "Analyze sales trend from: Q1=100, Q2=150, Q3=200, Q4=250", "data_science"),
    Task("t5", "Write a short story about a robot learning to paint", "writing"),
]


def init_session():
    if "agents" not in st.session_state:
        st.session_state.agents = list(DEFAULT_AGENTS)
    if "reputation" not in st.session_state:
        st.session_state.reputation = ReputationRegistry(REPUTATION_FILE)
    if "llm" not in st.session_state:
        st.session_state.llm = get_llm()
    if "auction_history" not in st.session_state:
        st.session_state.auction_history = []
    if "weights" not in st.session_state:
        st.session_state.weights = UtilityWeights(alpha=1.0, beta=1.0, gamma=1.0)


@st.dialog("Add agent")
def add_agent_dialog():
    name = st.text_input("Agent name", placeholder="e.g. analyst", label_visibility="collapsed")
    domains = st.text_input("Domains (comma separated)", placeholder="e.g. analysis, reporting", label_visibility="collapsed")
    if st.button("Add", type="primary", use_container_width=True):
        if name.strip() and domains.strip():
            st.session_state.agents.append((name.strip(), [d.strip() for d in domains.split(",")]))
            st.rerun()


def render_sidebar():
    with st.sidebar:
        st.header(":material/tune: Configuration")

        with st.expander(":material/robot_2: Agents", expanded=True):
            for i, (name, domains) in enumerate(st.session_state.agents):
                cols = st.columns([3, 8, 1])
                cols[0].markdown(":material/person:")
                cols[1].markdown(f"**{name}**  \n*{', '.join(domains)}*")
                if cols[2].button("✕", key=f"rm_{i}", help=f"Remove {name}"):
                    st.session_state.agents.pop(i)
                    st.rerun()
            if st.button(":material/add: Add agent", use_container_width=True):
                add_agent_dialog()

        with st.expander(":material/assignment: Task", expanded=True):
            task_options = {f"{t.task_type}: {t.description[:50]}...": t for t in DEFAULT_TASKS}
            task_options["Custom task..."] = None
            selected = st.selectbox(
                "Select a preset or custom",
                list(task_options.keys()),
                label_visibility="collapsed",
            )
            if task_options[selected] is not None:
                task = task_options[selected]
                st.text_input("Description", value=task.description, key="task_desc", label_visibility="collapsed")
                st.text_input("Type", value=task.task_type, key="task_type", label_visibility="collapsed")
            else:
                st.text_input("Description", placeholder="Describe the task...", key="task_desc", label_visibility="collapsed")
                st.text_input("Type", placeholder="e.g. code, writing, analysis", key="task_type", label_visibility="collapsed")

        with st.expander(":material/scale: Utility weights"):
            st.session_state.weights.alpha = st.slider("Confidence (α)", 0.0, 5.0, st.session_state.weights.alpha, 0.1)
            st.session_state.weights.beta = st.slider("Cost (β)", 0.0, 5.0, st.session_state.weights.beta, 0.1)
            st.session_state.weights.gamma = st.slider("ETA (γ)", 0.0, 5.0, st.session_state.weights.gamma, 0.1)

        run_btn = st.button(
            ":material/play_arrow: Run auction",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.agents or not st.session_state.get("task_desc", "").strip(),
        )

        st.divider()
        with st.expander(":material/history: Reputation history"):
            rep_data = st.session_state.reputation.scores
            if rep_data:
                for agent_id, s in rep_data.items():
                    total = s["successes"] + s["failures"]
                    rate = f"{s['successes']}/{total}" if total > 0 else "0/0"
                    st.markdown(f"**{agent_id}**: {rate} ({s['successes']} success, {s['failures']} failure)")
            else:
                st.caption("No reputation data yet.")

        return run_btn


def render_auction_phase(phase_num, label, icon):
    cols = st.columns([1, 20])
    cols[0].markdown(f"**:material/{icon}:**")
    cols[1].markdown(f"**Phase {phase_num}:** {label}")
    st.divider()


def run_auction_flow(task, agents, reputation, llm, weights):
    st.subheader(f":material/gavel: Auction for: *{task.description}*")
    st.caption(f"Task type: `{task.task_type}`")
    st.divider()

    # Phase 1: Announcement
    render_auction_phase(1, "Task announcement", "campaign")
    with st.container(border=True):
        constraints = task.constraints
        st.markdown(f"**Task:** {task.description}")
        st.markdown(f"**Type:** `{task.task_type}`")
        st.markdown(f"**Max cost:** ${constraints['max_cost']}  |  **Max ETA:** {constraints['max_eta_hours']}h")
        st.markdown(f"**Bidders:** {', '.join(a[0] for a in agents)}")

    # Phase 2: Bidding
    render_auction_phase(2, "Agent evaluation & bidding", "how_to_vote")
    bid_results = []
    for agent_id, domains in agents:
        agent = OllamaBidderAgent(agent_id, domains, llm)
        with st.status(f"Evaluating **{agent_id}**...", expanded=True) as status:
            st.markdown(f"*Domains:* {', '.join(domains)}")
            bid = agent.evaluate(task)
            if bid is None:
                status.update(label=f":material/block: **{agent_id}** — cannot handle", state="complete")
                bid_results.append((agent, None, None))
            else:
                adj_conf = reputation.adjust(agent_id, bid.confidence)
                status.update(
                    label=f":material/how_to_vote: **{agent_id}** — bids ${bid.cost:.0f} | {bid.confidence:.0%} conf (adj: {adj_conf:.0%}) | {bid.eta_minutes:.0f}min ETA",
                    state="complete",
                )
                st.markdown(f"> {bid.reasoning}")
                bid_results.append((agent, bid, adj_conf))

    # Phase 3: Evaluation
    render_auction_phase(3, "Bid evaluation & scoring", "leaderboard")
    scored = []
    for agent, bid, adj_conf in bid_results:
        if bid is None:
            continue
        u = utility_score(adj_conf, bid.cost, bid.eta_minutes, weights)
        scored.append((agent, bid, adj_conf, u))

    if not scored:
        st.error(":material/error: No agent can handle this task. Auction failed.")
        raise NoBidsException(task.description)

    df = pd.DataFrame(
        [
            {
                "Agent": s[0].agent_id,
                "Domains": ", ".join(s[0].domains),
                "Confidence": f"{s[1].confidence:.0%}",
                "Adjusted": f"{s[2]:.0%}",
                "Cost": f"${s[1].cost:.0f}",
                "ETA (min)": f"{s[1].eta_minutes:.0f}",
                "Utility": round(s[3], 2),
            }
            for s in scored
        ]
    )
    scored.sort(key=lambda x: x[3], reverse=True)
    df = df.sort_values("Utility", ascending=False)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Phase 4: Award
    render_auction_phase(4, "Contract award", "handshake")
    winner_agent, winner_bid, _, winner_utility = scored[0]
    with st.container(border=True):
        cols = st.columns([1, 5])
        cols[0].markdown(":material/trophy:")
        cols[1].markdown(f"**{winner_agent.agent_id}** wins with utility **{winner_utility:.2f}**")
        st.markdown(f"Bid: ${winner_bid.cost:.0f} | {winner_bid.confidence:.0%} confidence | {winner_bid.eta_minutes:.0f}min ETA")
        if scored[1:]:
            runner_up = scored[1]
            st.caption(f"Runner-up: **{runner_up[0].agent_id}** (utility {runner_up[3]:.2f})")

    # Phase 5: Execution
    render_auction_phase(5, "Task execution", "checklist")
    with st.status(f"Executing task on **{winner_agent.agent_id}**...", expanded=True) as status:
        result = winner_agent.execute(task)
        reputation.record(winner_agent.agent_id, result.success)
        emoji = ":material/check_circle:" if result.success else ":material/error:"
        status.update(
            label=f"{emoji} {'Success' if result.success else 'Failed'}",
            state="complete",
        )
        if result.output:
            st.code(result.output)
        if result.error:
            st.error(result.error)

    # Summary
    st.divider()
    st.success(f"Auction complete. **{winner_agent.agent_id}** executed the task {'successfully' if result.success else 'with failures'}.")

    return winner_agent, winner_bid, result


def main():
    init_session()
    run_auction = render_sidebar()

    st.title(":material/gavel: Contract-Net Marketplace")
    st.markdown(
        "A **Solicitor** auctions tasks to **BidderAgents** who bid on capability, cost, and confidence. "
        "The best bid wins and executes the task."
    )

    if st.session_state.auction_history:
        with st.expander(":material/history: Previous auctions", expanded=False):
            for i, desc in enumerate(st.session_state.auction_history):
                st.caption(f"#{i + 1}: {desc}")

    if run_auction:
        desc = st.session_state.task_desc.strip()
        ttype = st.session_state.task_type.strip()
        if not desc or not ttype:
            st.warning("Please enter both a task description and type.")
            st.stop()

        task = Task(f"custom_{len(st.session_state.auction_history)}", desc, ttype)
        agents = list(st.session_state.agents)
        reputation = st.session_state.reputation
        llm = st.session_state.llm
        weights = st.session_state.weights

        try:
            winner, bid, result = run_auction_flow(task, agents, reputation, llm, weights)
            st.session_state.auction_history.append(f"{winner.agent_id} → {desc[:60]}")
        except NoBidsException:
            st.session_state.auction_history.append(f"NO BIDS → {desc[:60]}")
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
