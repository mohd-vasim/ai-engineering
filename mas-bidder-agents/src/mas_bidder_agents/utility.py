from .models import UtilityWeights


def utility_score(
    confidence: float,
    cost: float,
    eta_minutes: float,
    weights: UtilityWeights | None = None,
) -> float:
    w = weights or UtilityWeights()
    return w.alpha * confidence - w.beta * cost - w.gamma * (eta_minutes / 60.0)
