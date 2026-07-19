import pytest

from mas_bidder_agents.models import UtilityWeights
from mas_bidder_agents.utility import utility_score


def test_default_weights():
    u = utility_score(confidence=0.9, cost=10.0, eta_minutes=60.0)
    assert u == pytest.approx(0.9 - 10.0 - 1.0)


def test_cost_dominated():
    weights = UtilityWeights(alpha=0.0, beta=1.0, gamma=0.0)
    u = utility_score(0.9, 50.0, 60.0, weights)
    assert u == -50.0


def test_confidence_dominated():
    weights = UtilityWeights(alpha=1.0, beta=0.0, gamma=0.0)
    assert utility_score(0.85, 100.0, 999.0, weights) == 0.85


def test_eta_dominated():
    weights = UtilityWeights(alpha=0.0, beta=0.0, gamma=1.0)
    u = utility_score(0.9, 10.0, 120.0, weights)
    assert u == -2.0


def test_zero_values():
    assert utility_score(0.0, 0.0, 0.0) == 0.0


def test_high_values():
    u = utility_score(1.0, 100.0, 600.0)
    assert u == 1.0 - 100.0 - 10.0
