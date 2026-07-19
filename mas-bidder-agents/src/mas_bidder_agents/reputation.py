import json
from pathlib import Path


class ReputationRegistry:
    def __init__(self, path: str | None = None):
        self.path = path
        self.scores: dict[str, dict[str, int]] = {}
        if path:
            self._load()

    def _load(self):
        p = Path(self.path)
        if p.exists() and p.stat().st_size > 0:
            with open(p) as f:
                self.scores = json.load(f)

    def _save(self):
        if self.path:
            with open(self.path, "w") as f:
                json.dump(self.scores, f, indent=2)

    def record(self, agent_id: str, success: bool) -> None:
        s = self.scores.setdefault(agent_id, {"successes": 0, "failures": 0})
        s["successes" if success else "failures"] += 1
        self._save()

    def penalty(self, agent_id: str) -> float:
        s = self.scores.get(agent_id, {"successes": 0, "failures": 0})
        total = s["successes"] + s["failures"]
        if total == 0:
            return 0.0
        return min(1.0, (s["failures"] / total) * 2.0)

    def adjust(self, agent_id: str, confidence: float) -> float:
        return confidence * (1.0 - self.penalty(agent_id))
