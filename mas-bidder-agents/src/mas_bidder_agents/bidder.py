from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import clean_json
from .models import Bid, ContractResult, Task
from .prompts import EVAL_HUMAN, EVAL_SYSTEM, EXEC_HUMAN, EXEC_SYSTEM


class OllamaBidderAgent:
    def __init__(self, agent_id: str, domains: list[str], llm: ChatOllama):
        self.agent_id = agent_id
        self.domains = domains
        self.llm = llm

    def evaluate(self, task: Task) -> Bid | None:
        system = EVAL_SYSTEM.format(agent_id=self.agent_id, domains=", ".join(self.domains))
        human = EVAL_HUMAN.format(task_type=task.task_type, task_desc=task.description)
        resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        try:
            data = clean_json(resp.content)
        except ValueError:
            return None
        if not data.get("can_handle", False):
            return None
        return Bid(
            agent_id=self.agent_id,
            confidence=float(data.get("confidence", 0.5)),
            cost=float(data.get("cost_dollars", 0)),
            eta_minutes=float(data.get("eta_minutes", 0)),
            reasoning=str(data.get("reasoning", "")),
        )

    def execute(self, task: Task) -> ContractResult:
        system = EXEC_SYSTEM.format(agent_id=self.agent_id, domains=", ".join(self.domains))
        human = EXEC_HUMAN.format(task_desc=task.description)
        resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        try:
            data = clean_json(resp.content)
        except ValueError:
            return ContractResult(self.agent_id, task.task_id, False, "", "JSON parse failed")
        return ContractResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            success=bool(data.get("success", False)),
            output=str(data.get("output", "")),
            error=str(data.get("error")) if data.get("error") else None,
        )
