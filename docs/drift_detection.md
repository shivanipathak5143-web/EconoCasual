# Data Drift Detection

## Objective
The Drift Detector monitors numerical customer and campaign features
by comparing a reference dataset with newly received data.

## Method
We use the two-sample Kolmogorov-Smirnov (KS) test from SciPy.

- Null hypothesis: Both samples come from the same distribution.
- Alternative hypothesis: The distributions differ.
- Default significance level: 0.05.
- Drift is flagged when the p-value is below alpha.

## Implementation
The main function is:

`check_drift(reference_df, new_df, alpha=0.05)`

The function:
1. Validates the input DataFrames and significance level.
2. Identifies numerical features shared by both datasets.
3. Excludes customer ID and other ID columns.
4. Removes missing and non-finite observations per feature.
5. Applies a two-sample KS test to each feature.
6. Returns a feature-level report and an overall drift flag.

## Output
The report contains:
- `drift_detected`: Overall drift status.
- `significance_level`: Alpha used.
- `features_checked`: Number of numerical features tested.
- `drifted_features`: Names of features flagged.
- `feature_results`: KS statistic, p-value, valid sample counts,
  and drift status for each feature.

## API Integration
FastAPI exposes the detector through:

`POST /api/drift`

The endpoint accepts a CSV file and an optional alpha query parameter.
It compares the uploaded dataset with `data/retail_campaign.csv`.
The uploaded dataset does not overwrite the reference file.

## Validation
Tests cover:
- Identical datasets with no detected drift.
- Artificially shifted numerical features.
- Exclusion of customer IDs.
- Missing values.
- Empty datasets and mismatched numerical columns.
- Invalid significance levels.

## Limitations
- Current implementation tests numerical features only.
- KS tests are univariate and do not detect every multivariate change.
- A statistically significant difference does not automatically mean
  the change is operationally important.
- Multiple feature tests are not currently adjusted for multiplicity.
- Missing and non-finite values are excluded from individual tests.
