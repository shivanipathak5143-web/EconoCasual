
import numpy as np
import pandas as pd

from scipy.optimize import (
    linprog,
    milp,
    LinearConstraint,
    Bounds
)


def _prepare_customers(customers, discount_cost):
    """Validate input and calculate each customer's net gain."""

    required = {"customer_id", "ite"}
    missing = required - set(customers.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if customers["customer_id"].isna().any():
        raise ValueError("Customer IDs cannot be missing")

    if customers["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found")

    if not np.isscalar(discount_cost):
        raise ValueError("Discount cost must be a scalar")

    if not np.isfinite(discount_cost) or discount_cost <= 0:
        raise ValueError("Discount cost must be positive")

    df = customers.copy()
    df["ite"] = pd.to_numeric(df["ite"], errors="raise")

    if not np.isfinite(df["ite"]).all():
        raise ValueError("ITE values must be finite")

    df["discount_cost"] = float(discount_cost)
    df["net_gain"] = df["ite"] - df["discount_cost"]

    return df


def _validate_budget(budget):
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Budget must be non-negative")


def greedy_budget_optimizer(customers, budget, discount_cost):
    """Select customers using gain per dollar spent."""

    _validate_budget(budget)
    df = _prepare_customers(customers, discount_cost)

    eligible = df[df["net_gain"] > 0].copy()

    eligible["gain_per_cost"] = (
        eligible["net_gain"] / eligible["discount_cost"]
    )

    eligible = eligible.sort_values(
        ["gain_per_cost", "net_gain", "customer_id"],
        ascending=[False, False, True],
        kind="mergesort"
    )

    selected = []
    remaining = float(budget)

    for _, customer in eligible.iterrows():
        cost = customer["discount_cost"]

        if cost <= remaining + 1e-9:
            selected.append(customer)
            remaining = max(0.0, remaining - cost)

    return pd.DataFrame(
        selected,
        columns=eligible.columns
    ).reset_index(drop=True)


def linprog_budget_optimizer(customers, budget, discount_cost):
    """Solve the continuous budget allocation problem."""

    _validate_budget(budget)
    df = _prepare_customers(customers, discount_cost)

    eligible = df[df["net_gain"] > 0].copy()

    if eligible.empty:
        return eligible.assign(
            allocation_fraction=pd.Series(dtype=float),
            allocated_cost=pd.Series(dtype=float),
            allocated_gain=pd.Series(dtype=float)
        )

    gains = eligible["net_gain"].to_numpy()
    costs = eligible["discount_cost"].to_numpy()

    result = linprog(
        c=-gains,
        A_ub=costs.reshape(1, -1),
        b_ub=[budget],
        bounds=(0, 1),
        method="highs"
    )

    if not result.success:
        raise RuntimeError(result.message)

    eligible["allocation_fraction"] = result.x
    eligible["allocated_cost"] = (
        costs * result.x
    )
    eligible["allocated_gain"] = (
        gains * result.x
    )

    return eligible[
        eligible["allocation_fraction"] > 1e-9
    ].reset_index(drop=True)


def milp_budget_optimizer(customers, budget, discount_cost):
    """Solve exact binary customer selection using SciPy MILP."""

    _validate_budget(budget)
    df = _prepare_customers(customers, discount_cost)

    eligible = df[df["net_gain"] > 0].copy()

    if eligible.empty:
        return eligible.reset_index(drop=True)

    gains = eligible["net_gain"].to_numpy()
    costs = eligible["discount_cost"].to_numpy()
    n = len(eligible)

    constraints = LinearConstraint(
        costs.reshape(1, -1),
        lb=-np.inf,
        ub=budget
    )

    result = milp(
        c=-gains,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(result.message)

    eligible["selected"] = np.rint(result.x).astype(int)

    return eligible[
        eligible["selected"] == 1
    ].reset_index(drop=True)


def compare_strategies(
    customers,
    optimized_allocation,
    budget,
    discount_cost,
    n_simulations=1000,
    random_state=42
):
    """Compare blanket, random and optimized targeting."""

    _validate_budget(budget)

    if n_simulations < 1:
        raise ValueError("At least one simulation is required")

    df = _prepare_customers(customers, discount_cost)

    blanket_spend = df["discount_cost"].sum()
    blanket_gain = df["net_gain"].sum()

    rng = np.random.default_rng(random_state)

    random_spends = []
    random_gains = []

    # Randomly select customers without exceeding the budget.
    # Negative-net-gain customers are allowed in this baseline.
    for _ in range(n_simulations):
        remaining = float(budget)
        spend = 0.0
        gain = 0.0

        for index in rng.permutation(len(df)):
            cost = df.iloc[index]["discount_cost"]

            if cost <= remaining + 1e-9:
                spend += cost
                gain += df.iloc[index]["net_gain"]
                remaining = max(0.0, remaining - cost)

        random_spends.append(spend)
        random_gains.append(gain)

    optimized_spend = optimized_allocation[
        "discount_cost"
    ].sum()

    optimized_gain = optimized_allocation[
        "net_gain"
    ].sum()

    return pd.DataFrame([
        {
            "strategy": "blanket",
            "total_spend": blanket_spend,
            "total_gain": blanket_gain
        },
        {
            "strategy": "random",
            "total_spend": np.mean(random_spends),
            "total_gain": np.mean(random_gains)
        },
        {
            "strategy": "optimized",
            "total_spend": optimized_spend,
            "total_gain": optimized_gain
        }
    ])


def create_allocation_export(
    customers,
    optimized_allocation
):
    """Create the agreed customer allocation CSV structure."""

    df = _prepare_customers(
        customers,
        optimized_allocation["discount_cost"].iloc[0]
        if not optimized_allocation.empty
        else 1.0
    )

    # Derive segments using ITE tertiles across all customers.
    ranks = df["ite"].rank(method="first")

    df["segment"] = pd.qcut(
        ranks,
        q=3,
        labels=["Low ITE", "Medium ITE", "High ITE"]
    )

    segments = df[["customer_id", "segment"]]

    allocation = optimized_allocation[
        ["customer_id", "discount_cost", "net_gain"]
    ].merge(
        segments,
        on="customer_id",
        how="left",
        validate="one_to_one"
    )

    return allocation.rename(columns={
        "discount_cost": "discount_amount",
        "net_gain": "expected_gain"
    })[
        [
            "customer_id",
            "discount_amount",
            "expected_gain",
            "segment"
        ]
    ]


def greedy_optimizer(ite_df, budget, cost):
    """
    Backend-compatible optimizer.
    Returns one allocation row for every customer.
    """

    _validate_budget(budget)

    if not np.isfinite(cost) or cost <= 0:
        raise ValueError("Discount cost must be positive")

    df = _prepare_customers(ite_df, cost)

    gain = df["net_gain"].to_numpy()

    order = np.argsort(-gain, kind="stable")

    n_affordable = min(
        len(df),
        int(budget / cost + 1e-9)
    )

    chosen = np.zeros(len(df), dtype=bool)

    for idx in order[:n_affordable]:
        if gain[idx] > 0:
            chosen[idx] = True

    ite = df["ite"].to_numpy()

    segment = np.where(
        chosen,
        "Persuadable",
        np.where(
            ite < 0,
            "Sleeping Dog",
            "Not targeted"
        )
    )

    return pd.DataFrame({
        "customer_id": df["customer_id"].to_numpy(),
        "discount_amount": np.where(chosen, cost, 0.0),
        "expected_gain": np.where(chosen, gain, 0.0),
        "segment": segment
    })
