from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from mas_bidder_agents.bidder import OllamaBidderAgent
from mas_bidder_agents.models import Task


def _mock_llm(json_str: str):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=json_str)
    return llm


def test_evaluate_can_handle():
    llm = _mock_llm(
        '{"can_handle": true, "confidence": 0.9, "cost_dollars": 10, "eta_minutes": 30, "reasoning": "I can do this"}'
    )
    agent = OllamaBidderAgent("test_agent", ["code"], llm)
    task = Task("t1", "Write code", "code")
    bid = agent.evaluate(task)
    assert bid is not None
    assert bid.agent_id == "test_agent"
    assert bid.confidence == 0.9
    assert bid.cost == 10.0
    assert bid.eta_minutes == 30.0


def test_evaluate_cannot_handle():
    llm = _mock_llm(
        '{"can_handle": false, "confidence": 0, "cost_dollars": 0, "eta_minutes": 0, "reasoning": "Not my domain"}'
    )
    agent = OllamaBidderAgent("test_agent", ["writing"], llm)
    task = Task("t1", "Write code", "code")
    assert agent.evaluate(task) is None


def test_evaluate_missing_can_handle():
    llm = _mock_llm('{"confidence": 0.8}')
    agent = OllamaBidderAgent("test_agent", ["code"], llm)
    task = Task("t1", "Write code", "code")
    assert agent.evaluate(task) is None


def test_evaluate_invalid_json():
    llm = _mock_llm("not json")
    agent = OllamaBidderAgent("test_agent", ["code"], llm)
    task = Task("t1", "Write code", "code")
    assert agent.evaluate(task) is None


def test_execute_success():
    llm = _mock_llm('{"success": true, "output": "done", "error": null}')
    agent = OllamaBidderAgent("test_agent", ["code"], llm)
    task = Task("t1", "Write code", "code")
    result = agent.execute(task)
    assert result.success is True
    assert result.output == "done"
    assert result.error is None


def test_execute_failure():
    llm = _mock_llm('{"success": false, "output": "", "error": "something broke"}')
    agent = OllamaBidderAgent("test_agent", ["code"], llm)
    task = Task("t1", "Write code", "code")
    result = agent.execute(task)
    assert result.success is False
    assert result.error == "something broke"


def test_execute_invalid_json():
    llm = _mock_llm("bad json")
    agent = OllamaBidderAgent("test_agent", ["code"], llm)
    task = Task("t1", "Write code", "code")
    result = agent.execute(task)
    assert result.success is False
    assert result.error == "JSON parse failed"
