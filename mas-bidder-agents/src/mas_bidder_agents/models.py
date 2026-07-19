from dataclasses import dataclass, field


@dataclass
class Task:
    task_id: str
    description: str
    task_type: str
    constraints: dict = field(default_factory=lambda: {"max_cost": 100.0, "max_eta_hours": 4})


@dataclass
class Bid:
    agent_id: str
    confidence: float
    cost: float
    eta_minutes: float
    reasoning: str


@dataclass
class ContractResult:
    agent_id: str
    task_id: str
    success: bool
    output: str
    error: str | None


@dataclass
class UtilityWeights:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
