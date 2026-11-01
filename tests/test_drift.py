
import numpy as np
import pandas as pd
import pytest

from src.drift import check_drift


@pytest.fixture
def reference_df():
    return pd.DataFrame({
        "customer_id": range(100),
        "recency": np.arange(100),
        "history": np.linspace(100, 500, 100),
        "revenue": np.linspace(50, 250, 100)
    })


def test_no_drift_when_data_is_identical(reference_df):
    result = check_drift(reference_df, reference_df.copy())

    assert result["drift_detected"] is False
    assert result["drifted_features"] == []


def test_detects_numeric_drift(reference_df):
    new_df = reference_df.copy()
    new_df["recency"] += 100

    result = check_drift(reference_df, new_df)

    assert result["drift_detected"] is True
    assert "recency" in result["drifted_features"]


def test_customer_id_is_ignored(reference_df):
    result = check_drift(reference_df, reference_df.copy())

    features = [
        item["feature"]
        for item in result["feature_results"]
    ]

    assert "customer_id" not in features


def test_report_contains_required_fields(reference_df):
    result = check_drift(reference_df, reference_df.copy())

    assert "drift_detected" in result
    assert "features_checked" in result
    assert "drifted_features" in result
    assert "feature_results" in result


def test_feature_report_contains_ks_metrics(reference_df):
    result = check_drift(reference_df, reference_df.copy())

    feature = result["feature_results"][0]

    assert "ks_statistic" in feature
    assert "p_value" in feature
    assert "drift_detected" in feature


def test_empty_dataset_raises_error(reference_df):
    with pytest.raises(ValueError):
        check_drift(reference_df, pd.DataFrame())


def test_mismatched_columns_raise_error(reference_df):
    new_df = reference_df.drop(columns=["revenue"])

    with pytest.raises(ValueError):
        check_drift(reference_df, new_df)


def test_invalid_alpha_raises_error(reference_df):
    with pytest.raises(ValueError):
        check_drift(reference_df, reference_df, alpha=1.5)


def test_missing_values_are_handled(reference_df):
    ref = reference_df.copy()
    new = reference_df.copy()

    ref.loc[0, "recency"] = np.nan
    new.loc[1, "recency"] = np.nan

    result = check_drift(ref, new)

    assert result["features_checked"] == 3
    assert result["drift_detected"] is False
