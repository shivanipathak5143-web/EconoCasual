
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def check_drift(reference_df, new_df, alpha=0.05):
    """
    Detect numerical feature drift using the
    two-sample Kolmogorov-Smirnov test.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Original training/reference data.
    new_df : pd.DataFrame
        Newly received data.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict
        JSON-friendly drift report.
    """

    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError("reference_df must be a DataFrame")

    if not isinstance(new_df, pd.DataFrame):
        raise TypeError("new_df must be a DataFrame")

    if reference_df.empty or new_df.empty:
        raise ValueError("Input datasets cannot be empty")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    # Identify numerical features, excluding ID columns
    ref_columns = [
        col for col in reference_df.select_dtypes(
            include=np.number
        ).columns
        if str(col).lower() != "id"
        and not str(col).lower().endswith("_id")
    ]

    new_columns = [
        col for col in new_df.select_dtypes(
            include=np.number
        ).columns
        if str(col).lower() != "id"
        and not str(col).lower().endswith("_id")
    ]

    if set(ref_columns) != set(new_columns):
        missing = sorted(set(ref_columns) - set(new_columns))
        extra = sorted(set(new_columns) - set(ref_columns))

        raise ValueError(
            f"Numeric columns must match. "
            f"Missing in new data: {missing}; "
            f"extra in new data: {extra}"
        )

    if not ref_columns:
        raise ValueError("No numerical features available")

    feature_results = []

    for column in ref_columns:

        ref_values = pd.to_numeric(
            reference_df[column], errors="coerce"
        ).to_numpy(dtype=float)

        new_values = pd.to_numeric(
            new_df[column], errors="coerce"
        ).to_numpy(dtype=float)

        # Remove missing and infinite values
        ref_values = ref_values[np.isfinite(ref_values)]
        new_values = new_values[np.isfinite(new_values)]

        if len(ref_values) < 2 or len(new_values) < 2:
            raise ValueError(
                f"Insufficient valid observations in {column}"
            )

        # Two-sample KS test
        statistic, p_value = ks_2samp(
            ref_values,
            new_values,
            alternative="two-sided"
        )

        drift_detected = bool(p_value < alpha)

        feature_results.append({
            "feature": str(column),
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "reference_count": int(len(ref_values)),
            "new_count": int(len(new_values)),
            "drift_detected": drift_detected
        })

    drifted_features = [
        result["feature"]
        for result in feature_results
        if result["drift_detected"]
    ]

    return {
        "drift_detected": bool(len(drifted_features) > 0),
        "significance_level": float(alpha),
        "features_checked": len(feature_results),
        "drifted_features": drifted_features,
        "feature_results": feature_results
    }
