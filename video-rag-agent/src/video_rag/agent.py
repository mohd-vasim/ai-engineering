"""LLM"""

from os import getenv
from typing import List, Generator, Dict
import logging
from pathlib import Path
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelFallbackMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from video_rag.postgres.search import tool_messages, AGENT_TOOLS
from video_rag.prompt import VIDEO_RAG_SYSTEM_PROMPT

load_dotenv()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("agent")

def setup_agent(system_prompt: str, tools: List):
    """Setup agent"""
    try:

        # Different models to avoid fallback
        gpt_oss_120b_openrouter = init_chat_model(
            model="openai:openai/gpt-oss-120b",
            api_key=getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

        return create_agent(
            model=gpt_oss_120b_openrouter,
            system_prompt=system_prompt,
            tools=tools,
            checkpointer=InMemorySaver(),
            middleware=[
                ToolCallLimitMiddleware(thread_limit=10, run_limit=10)
            ],
        )

    except Exception as e:
        print(f"Error: {e}")
        raise


class VideoRagAgent:
    """VideoRagAgent"""

    def __init__(self):
        self.memory: Dict[str, List] = {}
        self.agent = setup_agent(
            system_prompt=VIDEO_RAG_SYSTEM_PROMPT,
            tools=AGENT_TOOLS,
        )

        logger.info("Agent initialized, memory: %s", self.memory)

    def invoke(self, query: str, session_id: str) -> Dict:
        """Invoke agent"""
        try:
            response = self.agent.invoke(
                {"messages": [HumanMessage(query)]},
                stream_mode="values",
                config={"configurable": {"thread_id": session_id}},
            )
            return response

        except Exception as e:
            logger.error("Error: %s", e, exc_info=True)
            raise

    def invoke_stream(self, query: str, session_id: str) -> Generator[Dict, None, None]:
        """Invoke agent"""
        try:
            response = ""

            # logger.info(
            #     "Memory status: %s",
            #     self.memory,
            # )

            # if session_id not in self.memory:
            #     self.memory[session_id] = []

            # self.memory[session_id].append(HumanMessage(query))

            for chunk, _ in self.agent.stream(
                {"messages": [HumanMessage(query)]},
                stream_mode="messages",
                config={"configurable": {"thread_id": session_id}},
            ):
                # Skip tool result messages — they stream as text blocks
                # whose payload is a JSON string. Check chunk.type rather
                # than the content shape, since LangChain 1.x normalises
                # tool content to a string.
                if chunk.type == "tool":
                    continue

                if not chunk.content_blocks:
                    continue

                # Message chunk
                last_message = chunk.content_blocks[0]
                # print(f"Chunk: {last_message}")
                if last_message.get("type") == "tool_call_chunk":
                    yield {"action": "Reasoning", "content": "Reasoning..."}

                    # Writing agent action in status
                    if last_message.get("name") in tool_messages:
                        yield {
                            "action": "Reasoning",
                            "content": tool_messages.get(last_message.get("name")),
                        }

                # When agent start writes, last message will have keys type: text
                if last_message.get("type") == "text":
                    chunk_content = last_message.get("text")
                    if isinstance(chunk_content, dict):
                        continue

                    if chunk_content and "tool_call_output" not in chunk_content:
                        response += chunk_content

                        yield {
                            "action": "Generating",
                            "content": chunk_content,
                        }

            # Adding AI response to memroy
            # self.memory[session_id].append(AIMessage(response))

        except Exception as e:
            logger.error("Error: %s", e, exc_info=True)
            raise


if __name__ == "__main__":
    print("Testing agent")

    # ai_agent = setup_agent(system_prompt="You are an helpful assistant", tools=[])

    # # Example usage
    # sresponse = ai_agent.invoke(
    #     {
    #         "messages": [
    #             HumanMessage(
    #                 "What NFL team won the Super Bowl in the year Justin Bieber was born?"
    #             )
    #         ]
    #     }
    # )
    # print(sresponse["messages"][-1].content)

    agent = VideoRagAgent()

    print(agent.invoke("hi", "12345"))
