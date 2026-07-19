from .bidder import OllamaBidderAgent
from .models import Bid, ContractResult, Task, UtilityWeights
from .reputation import ReputationRegistry
from .utility import utility_score


class NoBidsException(Exception):
    pass


class Solicitor:
    def __init__(
        self,
        agents: list[OllamaBidderAgent],
        reputation: ReputationRegistry,
        weights: UtilityWeights | None = None,
    ):
        self.agents = agents
        self.reputation = reputation
        self.weights = weights or UtilityWeights()

    def run_auction(
        self,
        task: Task,
        verbose: bool = False,
    ) -> tuple[OllamaBidderAgent, Bid, ContractResult]:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  TASK: {task.description}")
            print(f"  Type: {task.task_type}")
            print(f"{'='*60}")

        bids: list[tuple[OllamaBidderAgent, Bid]] = []

        for agent in self.agents:
            if verbose:
                print(f"  \n  {agent.agent_id}:", end="")
            bid = agent.evaluate(task)
            if bid is None:
                if verbose:
                    print(" CANNOT_HANDLE")
                continue
            adj_conf = self.reputation.adjust(agent.agent_id, bid.confidence)
            if verbose:
                print(f" bids ${bid.cost:.0f} | {bid.confidence:.0%} conf (adj: {adj_conf:.0%}) | {bid.eta_minutes:.0f}min")
                if bid.reasoning:
                    print(f"    → {bid.reasoning[:100]}")
            bids.append((agent, bid))

        if not bids:
            raise NoBidsException(f"No agent can handle: {task.description}")

        def score(item):
            a, b = item
            return utility_score(
                self.reputation.adjust(a.agent_id, b.confidence),
                b.cost,
                b.eta_minutes,
                self.weights,
            )

        bids.sort(key=score, reverse=True)
        winner_agent, winner_bid = bids[0]

        if verbose:
            print(f"\n  ★ WINNER: {winner_agent.agent_id} (utility={score(bids[0]):.2f})")

        result = winner_agent.execute(task)
        self.reputation.record(winner_agent.agent_id, result.success)

        if verbose:
            emoji = "✅" if result.success else "❌"
            print(f"  Result: {emoji} {'SUCCESS' if result.success else 'FAILURE'}")
            if result.output:
                print(f"  Output: {result.output[:400]}")
            if result.error:
                print(f"  Error: {result.error}")

        return winner_agent, winner_bid, result
