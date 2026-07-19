from .bidder import OllamaBidderAgent
from .llm import get_llm
from .models import Task
from .reputation import ReputationRegistry
from .solicitor import NoBidsException, Solicitor


def main():
    llm = get_llm()

    agents = [
        OllamaBidderAgent("coder", ["code", "debugging"], llm),
        OllamaBidderAgent("writer", ["writing", "poetry", "summarization"], llm),
        OllamaBidderAgent("ml_eng", ["ml", "data_science"], llm),
    ]

    reputation = ReputationRegistry("reputation.json")
    solicitor = Solicitor(agents, reputation)

    tasks = [
        Task("t1", "Write a Python function to sort integers using quicksort", "code"),
        Task("t2", "Write a haiku about gradient descent", "poetry"),
        Task("t3", "Fix this JS typo: function greet(name {{ return 'Hello ' + name }}", "debugging"),
    ]

    for task in tasks:
        try:
            solicitor.run_auction(task, verbose=True)
        except NoBidsException as e:
            print(f"\n  {'='*60}")
            print(f"  NO BIDS: {e}")
            print(f"  {'='*60}")


if __name__ == "__main__":
    main()
