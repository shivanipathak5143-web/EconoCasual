# Budget Optimization Methodology

## Objective
Allocate a limited campaign budget to customers with positive
expected net gain based on their estimated Individual Treatment Effects.

## Net Gain
For customer i:

`Net Gain = ITE - Discount Cost`

Only customers with positive net gain are eligible for targeted
discount allocation.

## Optimization Methods

### 1. Greedy Allocation
The greedy method:
1. Calculates net gain for each customer.
2. Calculates net gain per unit of discount cost.
3. Sorts eligible customers by gain per cost.
4. Selects customers while the budget permits.

This is a simple heuristic and returns whole-customer decisions.

### 2. Linear Programming (SciPy linprog)
The LP method maximizes total net gain subject to the budget constraint.

Decision variables are continuous between zero and one.
Therefore, LP can return fractional customer allocations.

### 3. Mixed-Integer Linear Programming (SciPy milp)
The MILP method uses binary decision variables:
- 1: Select the customer.
- 0: Do not select the customer.

It provides whole-customer allocation decisions while respecting
the budget constraint.

## Backend Integration
The backend-compatible function is:

`greedy_optimizer(ite_df, budget, cost)`

It returns one row for every customer with:
- `customer_id`
- `discount_amount`
- `expected_gain`
- `segment`

Untargeted customers receive zero discount and zero expected gain.

The backend uses this function through `run_allocation()`.

## Strategy Comparison
The project compares:

1. Blanket: Offer the discount to every customer.
2. Random: Select customers without ranking by estimated effect.
3. Optimized: Select customers using estimated net gain.

The optimizer module also provides a repeated random simulation
helper. The API strategy-comparison endpoint currently uses an
expected-gain random baseline based on average ITE.

Blanket targeting is an unconstrained baseline and can spend more
than the available targeting budget.

## Validation
Unit tests check:
- Budget constraints.
- Zero and insufficient budgets.
- Large budgets.
- No eligible customers.
- Negative budgets.
- Duplicate customer IDs.
- LP and MILP allocation behavior.
- Backend-compatible output.
- Strategy comparison and allocation export.

## Assumptions and Limitations
- ITE scores must be provided by the causal modelling pipeline.
- Discount cost is treated as a fixed per-customer amount.
- The budget and discount amount are configurable inputs.
- LP may produce fractional allocations.
- Greedy is a heuristic; MILP provides binary decisions.
- Experimental budget and discount values must not be treated as
  confirmed campaign settings without team agreement.
