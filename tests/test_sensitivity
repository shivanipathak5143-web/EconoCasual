import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from src.sensitivity import cost_sensitivity


def test_higher_cost_never_targets_more_customers():
    ite = np.random.default_rng(0).normal(1.0, 1.0, 2000)
    out = cost_sensitivity(ite, [0.0, 0.5, 1.0, 2.0, 5.0])
    assert out["n_worth_targeting"].is_monotonic_decreasing
    assert out["total_gain"].is_monotonic_decreasing


def test_zero_cost_counts_every_positive_effect():
    ite = np.array([-1.0, 0.5, 2.0, 3.0])
    out = cost_sensitivity(ite, [0.0])
    assert out.loc[0, "n_worth_targeting"] == 3


def test_budget_limits_the_number_targeted():
    ite = np.full(1000, 5.0)
    out = cost_sensitivity(ite, [1.0], budget=100)
    assert out.loc[0, "n_within_budget"] == 100


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        cost_sensitivity([1.0, 2.0], [-1.0])