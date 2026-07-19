import json as json_mod
import tempfile
from pathlib import Path

import pytest

from mas_bidder_agents.reputation import ReputationRegistry


def test_penalty_no_history():
    reg = ReputationRegistry()
    assert reg.penalty("unknown") == 0.0


def test_penalty_all_success():
    reg = ReputationRegistry()
    reg.record("a", True)
    reg.record("a", True)
    reg.record("a", True)
    assert reg.penalty("a") == 0.0


def test_penalty_all_failures():
    reg = ReputationRegistry()
    reg.record("a", False)
    reg.record("a", False)
    assert reg.penalty("a") == 1.0


def test_penalty_mixed():
    reg = ReputationRegistry()
    reg.record("a", True)
    reg.record("a", True)
    reg.record("a", False)
    # 1 failure / 3 total * 2 = 0.67
    assert reg.penalty("a") == pytest.approx(0.666, rel=1e-2)


def test_adjust_confidence():
    reg = ReputationRegistry()
    reg.record("a", False)
    assert reg.adjust("a", 0.9) == 0.0  # penalty=1.0 → 0 confidence


def test_adjust_no_penalty():
    reg = ReputationRegistry()
    reg.record("a", True)
    assert reg.adjust("a", 0.8) == 0.8


def test_persistence():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_mod.dump({}, f)
        path = f.name
    try:
        reg = ReputationRegistry(path)
        reg.record("a", True)
        reg.record("a", False)

        reg2 = ReputationRegistry(path)
        assert reg2.penalty("a") == 1.0
        assert reg2.scores["a"]["successes"] == 1
        assert reg2.scores["a"]["failures"] == 1
    finally:
        Path(path).unlink(missing_ok=True)
