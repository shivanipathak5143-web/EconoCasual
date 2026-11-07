"""Cost sensitivity: how many customers are worth targeting if the discount costs more or less?"""
import numpy as np
import pandas as pd


def cost_sensitivity(ite, costs, budget=None) -> pd.DataFrame:
    """
    For each assumed cost per customer, count the customers whose predicted effect (ITE) exceeds that cost.
    ite    : predicted effects, in the same unit as costs (for example dollars of revenue per customer)
    costs  : list of assumed costs to try
    budget : optional total budget; adds how many of those customers you could afford, best first
    """
    ite = np.asarray(ite, dtype=float)
    if any(c < 0 for c in costs):
        raise ValueError("costs must be >= 0")
    rows = []
    for c in costs:
        gain = ite - c
        worth = gain > 0
        row = {"cost": c,
               "n_worth_targeting": int(worth.sum()),
               "share_worth_targeting": float(worth.mean()),
               "total_gain": float(gain[worth].sum())}
        if budget is not None:
            k = len(gain) if c == 0 else int(budget / c + 1e-9)
            best = np.sort(gain[worth])[::-1][:k]
            row["n_within_budget"] = int(len(best))
            row["gain_within_budget"] = float(best.sum())
        rows.append(row)
    return pd.DataFrame(rows)