import pytest

from mas_bidder_agents.bidder import OllamaBidderAgent
from mas_bidder_agents.models import Bid, ContractResult, Task
from mas_bidder_agents.reputation import ReputationRegistry
from mas_bidder_agents.solicitor import NoBidsException, Solicitor


class _FakeAgent(OllamaBidderAgent):
    def __init__(self, agent_id, domains, bid=None, exec_result=None):
        super().__init__(agent_id, domains, None)
        self._bid = bid
        self._exec_result = exec_result

    def evaluate(self, task):
        return self._bid

    def execute(self, task):
        return self._exec_result


def test_single_agent_wins():
    bid = Bid("a", 0.9, 10.0, 30.0, "can do")
    result = ContractResult("a", "t1", True, "done", None)
    agent = _FakeAgent("a", ["code"], bid, result)
    rep = ReputationRegistry()
    sol = Solicitor([agent], rep)

    winner, winning_bid, exec_result = sol.run_auction(Task("t1", "task", "code"))
    assert winner.agent_id == "a"
    assert winning_bid is bid
    assert exec_result.success is True
    assert rep.scores["a"]["successes"] == 1


def test_no_bids_raises():
    agent = _FakeAgent("a", ["code"], None, None)
    rep = ReputationRegistry()
    sol = Solicitor([agent], rep)
    with pytest.raises(NoBidsException):
        sol.run_auction(Task("t1", "task", "medical"))


def test_highest_utility_wins():
    cheap = Bid("cheap", 0.8, 5.0, 30.0, "cheap")
    expensive = Bid("expensive", 0.99, 80.0, 10.0, "premium")
    ok_result = ContractResult("cheap", "t1", True, "ok", None)
    good_result = ContractResult("expensive", "t1", True, "great", None)

    agents = [
        _FakeAgent("cheap", ["code"], cheap, ok_result),
        _FakeAgent("expensive", ["code"], expensive, good_result),
    ]
    rep = ReputationRegistry()
    sol = Solicitor(agents, rep)

    winner, bid, _ = sol.run_auction(Task("t1", "task", "code"))
    # cheap should win: lower cost offsets slightly lower confidence
    assert winner.agent_id == "cheap"


def test_reputation_affects_outcome():
    bid_a = Bid("a", 0.9, 50.0, 60.0, "")
    bid_b = Bid("b", 0.9, 50.0, 60.0, "")
    result = ContractResult("a", "t1", True, "", None)

    agents = [
        _FakeAgent("a", ["code"], bid_a, result),
        _FakeAgent("b", ["code"], bid_b, result),
    ]
    rep = ReputationRegistry()
    rep.record("a", False)  # a has a failure
    rep.record("a", False)  # a has two failures → penalty=1.0

    sol = Solicitor(agents, rep)
    winner, _, _ = sol.run_auction(Task("t1", "task", "code"))
    assert winner.agent_id == "b"  # b should win due to a's reputation


def test_mixed_bids_and_nones():
    bid = Bid("a", 0.9, 10.0, 30.0, "can do")
    result = ContractResult("a", "t1", True, "done", None)
    agents = [
        _FakeAgent("a", ["code"], bid, result),
        _FakeAgent("b", ["writing"], None, None),  # can't handle
    ]
    rep = ReputationRegistry()
    sol = Solicitor(agents, rep)

    winner, _, _ = sol.run_auction(Task("t1", "task", "code"))
    assert winner.agent_id == "a"
