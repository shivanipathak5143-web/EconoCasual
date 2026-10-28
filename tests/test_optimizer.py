
import sys
sys.path.insert(0, "/content")

import pandas as pd
import pytest

from src.optimizer import (
    greedy_budget_optimizer,
    linprog_budget_optimizer,
    milp_budget_optimizer,
    compare_strategies,
    create_allocation_export
)


@pytest.fixture
def customers():
    return pd.DataFrame({
        "customer_id": ["A", "B", "C", "D", "E"],
        "ite": [50, 40, 25, 15, 5]
    })


def test_greedy_respects_budget(customers):
    result = greedy_budget_optimizer(customers, 20, 10)

    assert result["discount_cost"].sum() <= 20
    assert len(result) == 2


def test_zero_budget(customers):
    result = greedy_budget_optimizer(customers, 0, 10)

    assert result.empty


def test_insufficient_budget(customers):
    result = greedy_budget_optimizer(customers, 9, 10)

    assert result.empty


def test_no_eligible_customers():
    df = pd.DataFrame({
        "customer_id": ["A", "B"],
        "ite": [5, 8]
    })

    result = greedy_budget_optimizer(df, 100, 10)

    assert result.empty


def test_negative_budget_raises_error(customers):
    with pytest.raises(ValueError):
        greedy_budget_optimizer(customers, -10, 10)


def test_duplicate_customers_raise_error():
    df = pd.DataFrame({
        "customer_id": ["A", "A"],
        "ite": [30, 40]
    })

    with pytest.raises(ValueError):
        greedy_budget_optimizer(df, 100, 10)


def test_linprog_respects_budget(customers):
    result = linprog_budget_optimizer(customers, 20, 10)

    assert result["allocated_cost"].sum() <= 20 + 1e-8


def test_milp_respects_budget(customers):
    result = milp_budget_optimizer(customers, 20, 10)

    assert result["discount_cost"].sum() <= 20 + 1e-8
    assert result["selected"].eq(1).all()


def test_strategy_comparison(customers):
    allocation = milp_budget_optimizer(customers, 20, 10)

    result = compare_strategies(
        customers,
        allocation,
        budget=20,
        discount_cost=10,
        n_simulations=50
    )

    assert set(result["strategy"]) == {
        "blanket", "random", "optimized"
    }


def test_allocation_export(customers):
    allocation = milp_budget_optimizer(customers, 20, 10)

    result = create_allocation_export(
        customers, allocation
    )

    assert list(result.columns) == [
        "customer_id",
        "discount_amount",
        "expected_gain",
        "segment"
    ]

from src.optimizer import greedy_optimizer as backend_greedy_optimizer


def test_backend_optimizer_returns_all_customers(customers):
    result = backend_greedy_optimizer(
        customers, budget=20, cost=10
    )

    assert len(result) == len(customers)

    assert list(result.columns) == [
        "customer_id",
        "discount_amount",
        "expected_gain",
        "segment"
    ]


def test_backend_optimizer_respects_budget(customers):
    result = backend_greedy_optimizer(
        customers, budget=20, cost=10
    )

    assert result["discount_amount"].sum() <= 20
    assert (result["discount_amount"] >= 0).all()
