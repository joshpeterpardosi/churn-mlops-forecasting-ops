import pandas as pd

from churn_mlops.lib.forecast_features import build_forecast_features


def _mrr_df():
    return pd.DataFrame({
        "customerID": ["1"] * 6,
        "month": [0, 1, 2, 3, 4, 5],
        "MRR": [50.0, 50.0, 52.0, 51.0, 53.0, 0.0],
    })


def _feature_df():
    return pd.DataFrame({
        "customerID": ["1"],
        "Contract": ["Month-to-month"],
        "InternetService": ["DSL"],
        "PaymentMethod": ["Electronic check"],
        "TechSupport": ["No"],
    })


def test_lag_values_correct_for_multi_month_customer():
    result = build_forecast_features(_mrr_df(), _feature_df())
    row = result[result["month"] == 3].iloc[0]
    assert row["lag_1"] == 52.0
    assert row["lag_2"] == 50.0
    assert row["lag_3"] == 50.0
    assert row["rolling_3mo_mean"] == (52.0 + 50.0 + 50.0) / 3
    assert row["target_mrr"] == 51.0


def test_customer_with_fewer_than_four_months_contributes_no_rows():
    mrr_df = pd.DataFrame({
        "customerID": ["2", "2"],
        "month": [0, 1],
        "MRR": [30.0, 30.0],
    })
    feature_df = pd.DataFrame({
        "customerID": ["2"], "Contract": ["One year"], "InternetService": ["No"],
        "PaymentMethod": ["Mailed check"], "TechSupport": ["Yes"],
    })
    result = build_forecast_features(mrr_df, feature_df)
    assert len(result) == 0


def test_static_features_joined_correctly():
    result = build_forecast_features(_mrr_df(), _feature_df())
    row = result[result["month"] == 3].iloc[0]
    assert row["Contract"] == "Month-to-month"
    assert row["TechSupport"] == "No"


def test_churned_customer_terminal_zero_row_is_valid_target():
    result = build_forecast_features(_mrr_df(), _feature_df())
    row = result[result["month"] == 5].iloc[0]
    assert row["target_mrr"] == 0.0
