"""Streamlit app"""

import sys
from uuid import uuid4
import logging
import streamlit as st
from video_rag.agent import VideoRagAgent


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("main")
logger.info("Env with arg: %s", sys.argv)


try:
    ENV = sys.argv[1]
except IndexError:
    ENV = "prod"

# agent = AI Coworker()


def chat_ui():
    """Main app interface"""

    st.title("Welcome to Video Analytics Agent")

    st.markdown(
        """
    <style>
    /* Remove border & background from status (expander-based) */
    div[data-testid="stExpander"] > details {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    /* Remove border from summary header */
    div[data-testid="stExpander"] summary {
        border: none !important;
        background: transparent !important;
        padding: 0.25rem 0 !important;
    }

    /* Remove border from expanded content */
    div[data-testid="stExpanderDetails"] {
        border: none !important;
        background: transparent !important;
        padding-left: 0 !important;
    }
    </style>

    <script>
    (function () {
        const observer = new MutationObserver(() => {
            document
              .querySelectorAll('div[data-testid="stExpander"] summary p')
              .forEach(p => {
                  if (p.textContent.trim() === "Completed") {
                      // Remove the whole status block safely
                      const wrapper = p.closest('div[data-testid="stLayoutWrapper"]');
                      if (wrapper) {
                          wrapper.remove();
                      }
                  }
              });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    })();
    </script>
    """,
        unsafe_allow_html=True,
    )

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())

    if "agent" not in st.session_state:
        st.session_state.agent = VideoRagAgent()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:

        st.sidebar.header("Video Analytics Agent")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask about video footages"
    )

    if prompt:

        # User prompt augmentation
        user_prompt = prompt

        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # User's block
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # AI response block
        with st.chat_message("assistant"):
            response_box = st.empty()
            response = ""

            agent = st.session_state.agent

            with st.status("") as response_status:
                for chunk in agent.invoke_stream(
                    query=user_prompt, session_id=st.session_state.session_id
                ):
                    if not isinstance(chunk, dict):
                        continue

                    # Message chunk
                    response_status.update(label=chunk.get("content"), expanded=True)

                    # When agent start writes, last message will have keys type: text
                    if chunk.get("action") == "Generating":
                        response_status.update(label="", expanded=False)
                        response += chunk.get("content")
                        response_box.write(response)

                response_status.update(label="", expanded=False)

        st.session_state.messages.append({"role": "assistant", "content": response})
        logger.info("Current session id: %s", st.session_state.session_id)


if __name__ == "__main__":
    print("Starting streamlit chat interface")
    chat_ui()
    # NOTE: >>> uv run streamlit run coworker/streamlit_app.py --logger.level=debug -- dev
