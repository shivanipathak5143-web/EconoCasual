# Causal Engine: data preparation, causal graph, Double ML and refutation

Owner: Shivani Pathak

This part of EconoCausal turns a marketing promotion dataset into **per-customer treatment effects (ITE)**:
how much the discount offer changes each customer's revenue. The optimizer, API and dashboard use these scores.

## What this part does

1. **Prepares the data** from the raw Kaggle file.
2. **Defines the causal model** in DoWhy: treatment, outcome and confounders.
3. **Estimates baselines**: a naive difference and a regression with controls.
4. **Trains Double ML models** (EconML) to get an ITE for every customer.
5. **Checks the model** with decile uplift tables, since the data has no known true effect.
6. **Runs refutation tests** to check the estimate is not noise.

## Dataset

Kaggle "Marketing Promotion Campaign Uplift Modelling", described by its author as a **fictional practice dataset**.
Raw columns: `recency, history, used_discount, used_bogo, zip_code, is_referral, channel, offer, conversion`.

After preparation (`data/retail_campaign.csv`):

| Item | Value |
|---|---|
| Customers | 42,613 (Discount offer or No Offer; Buy One Get One customers are dropped) |
| Treatment `discount` | 1 = got the discount offer, 0 = no offer (about 50% each) |
| Outcome `revenue` | `conversion` x an assumed value of $25 per conversion |
| Features | `recency, history, used_discount, used_bogo, is_referral`, plus yes/no columns for `zip_code` and `channel` |
| Left out of features | `offer` and `conversion` (outcome leakage) |

## Pipeline

```
data/data.csv  ->  prepare_data.py  ->  data/retail_campaign.csv
                                              |
          casual_graph.py (DoWhy) -> baselines -> double_ml.py (EconML) -> data/ite_scores_model.csv
                                              |                |
                                     evaluation.py     refutation.py
                                    (decile check)    (placebo, random cause, subset)
```

## Files

| File | Purpose |
|---|---|
| `src/prepare_data.py` | Filters offers, builds `discount` and `revenue`, converts text columns to numbers |
| `src/casual_graph.py` | DoWhy causal model, identification and linear estimate |
| `src/double_ml.py` | LinearDML and CausalForestDML (LightGBM base models), ITE prediction and saving |
| `src/evaluation.py` | Decile uplift check (replaces a comparison with a true effect) |
| `src/refutation.py` | DoWhy refuters and the same three tests around the Double ML model |
| `notebooks/casual_graph.ipynb` | Causal graph, data overview, randomization check |
| `notebooks/baselines.ipynb` | Naive and regression baselines |
| `notebooks/double_ml.ipynb` | Double ML training, decile tables, ITE scores |
| `notebooks/refutation.ipynb` | Refutation tests |
| `tests/test_pipeline.py` | pytest tests for data preparation, evaluation, refutation logic and optimizer rules |

## How to run

Python 3.10 to 3.12 is recommended. Python 3.15 failed to start the notebook kernel, so avoid it.

```bash
pip install -r requirements.txt
python src/prepare_data.py --raw data/data.csv      # creates data/retail_campaign.csv
python -m pytest tests -v
```

Then open the notebooks in order and use **Restart, then Run All**:
`casual_graph`, `baselines`, `double_ml`, `refutation`.

## Method

- **Causal graph (DoWhy):** treatment `discount`, outcome `revenue`, customer features as confounders. `identify_effect`
  checks that the effect can be estimated by adjusting for them (backdoor criterion).
- **Double ML (EconML):** one LightGBM model predicts revenue and another predicts who got the offer. The effect is
  estimated from how their residuals relate. Predictions use **5-fold cross-fitting**, so no customer is predicted by a
  model that saw them.
- **LinearDML** assumes the effect changes linearly with the features. **CausalForestDML** allows non-linear patterns.
- **Decile check:** rank customers by predicted ITE, split into 5 groups, and compare real revenue of offered and
  not-offered customers in each group. A working model shows the largest gap in the top group. Fair because offers look random.
- **Refutation tests:** placebo treatment (effect should drop to about 0), random common cause (estimate should not
  move) and data subset (estimate should stay stable).

## Results so far

| Check | Result |
|---|---|
| Conversion rate, no offer vs. discount | 10.62% vs. 18.28% |
| Naive difference in revenue | 1.9147 per customer |
| Regression with controls | 1.9015 per customer |
| Feature balance between groups | Similar averages, so offers look randomly assigned |
| Double ML ITE scores | Computed; saved to `data/ite_scores_model.csv` |
| Decile check | To fill in after running `double_ml.ipynb` |
| Refutation tests | To fill in after running `refutation.ipynb` |

The naive and adjusted estimates agree, which is what you expect when assignment is random.
Dollar figures depend on the assumed $25 per conversion.

## Output for other team members

`data/ite_scores_model.csv`, one row per customer:

```
customer_id, ite, recency, history, is_referral
```

`data/ite_scores.csv` is a separate mock file used by the dashboard charts. Do not overwrite it.

## Assumptions and limitations

- The dataset is fictional, so results show the method works, not how real customers behave.
- One conversion is worth $25 (an assumption; the data has no money values).
- There is no true effect in the data, so models are checked with decile tables and refutation tests.
- Double ML adjusts only for the features available. Hidden confounders can still bias the estimates, and
  refutation tests cannot prove they are absent.
- Most customers do not convert, so individual effects are noisy. Rely on group-level evidence.
- EconML's `LinearDML` may warn that its covariance matrix is underdetermined. Only point estimates are used from it.

## Tech stack

Python, pandas, numpy, scikit-learn, DoWhy, EconML, LightGBM, matplotlib, networkx, pytest, Jupyter, Git and GitHub.
